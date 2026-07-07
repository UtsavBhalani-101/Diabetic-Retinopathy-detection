# pipeline/dann/loaders_dann.py
# ============================================================
# DataLoader factories for DANN training.
# Does NOT modify pipeline/data/loaders.py.
#
# Public API
# ----------
# build_dann_source_loaders(config)
#     → train_loader, val_loader, combined_train_df
#
# build_dann_target_train_loader(config)
#     → loader  (all target datasets combined, for domain adversarial loss)
#
# build_dann_target_eval_loader(ds_name, config)
#     → loader  (single target dataset, for per-dataset QWK eval)
# ============================================================

import logging

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset, DataLoader

from pipeline.setup.utils import DATASET_REGISTRY          # read-only
from pipeline.setup.config import seed_worker, g           # read-only
from pipeline.data.dataset import (                        # read-only
    RetinopathyDataset,
    train_transformer,
    val_transformer,
)
from pipeline.dann.dataset_dann import DANNDataset

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------
# Source loaders  (4 datasets combined: APTOS, EyePACS, Messidor-1, DDR)
# ----------------------------------------------------------------

def build_dann_source_loaders(config: dict):
    """
    Build combined source train + validation DataLoaders.

    Strategy
    --------
    For each source dataset:
      1. Load the CSV
      2. Stratified 80/20 split (preserving DR grade distribution)
      3. Wrap train split in RetinopathyDataset (train_transformer) → DANNDataset(0)
      4. Wrap val  split in RetinopathyDataset (val_transformer)   — plain, no domain label
         (validation only needs class labels, no domain label needed)

    All train DANNDatasets are concatenated → single shuffled DataLoader.
    All val   datasets   are concatenated → single ordered  DataLoader.

    Parameters
    ----------
    config : dict  — must contain 'source_datasets', 'batch_size', etc.

    Returns
    -------
    train_loader      : DataLoader  — yields (image, class_label, domain_id=0)
    val_loader        : DataLoader  — yields (image, class_label)  [standard, no domain]
    combined_train_df : pd.DataFrame — concatenated source training rows
                        (used to compute class-balanced weights)
    """
    source_names = config["source_datasets"]
    seed         = config["seed"]

    train_datasets    = []
    val_datasets      = []
    combined_train_dfs = []

    for ds_name in source_names:
        reg      = DATASET_REGISTRY[ds_name]
        diag_col = reg["diagnosis_col"]

        df = pd.read_csv(reg["target_path"]).reset_index(drop=True)

        # Keep only the two columns we need to avoid stale columns across datasets
        df = df[[reg["image_col"], diag_col]].copy()

        # Add a temporary column to carry dataset name through the concat
        df["_dataset"] = ds_name

        logger.info(
            f"[{ds_name}] CSV loaded | rows={len(df)} "
            f"| class dist: {df[diag_col].value_counts().sort_index().to_dict()}"
        )

        try:
            train_df, val_df = train_test_split(
                df,
                test_size=0.2,
                random_state=seed,
                stratify=df[diag_col],
            )
        except ValueError:
            # Fallback if a class has too few samples to stratify
            logger.warning(
                f"[{ds_name}] Stratified split failed — falling back to random split"
            )
            train_df, val_df = train_test_split(df, test_size=0.2, random_state=seed)

        train_df = train_df.reset_index(drop=True)
        val_df   = val_df.reset_index(drop=True)

        logger.info(f"[{ds_name}] Split → train={len(train_df)} | val={len(val_df)}")

        # ---- train: with augmentation + domain label ----
        train_base = RetinopathyDataset(
            dataframe=train_df,
            img_path=reg["image_path"],
            img_col=reg["image_col"],
            label_col=diag_col,
            transforms=train_transformer,
            extension=reg["extension"],
            clahe_image_path=reg.get("clahe_image_path"),   # offline CLAHE dir
        )
        train_datasets.append(DANNDataset(train_base, domain_id=0))
        combined_train_dfs.append(train_df)

        # ---- val: no augmentation, no domain label ----
        val_base = RetinopathyDataset(
            dataframe=val_df,
            img_path=reg["image_path"],
            img_col=reg["image_col"],
            label_col=diag_col,
            transforms=val_transformer,
            extension=reg["extension"],
            clahe_image_path=reg.get("clahe_image_path"),   # offline CLAHE dir
        )
        val_datasets.append(val_base)

    # Combine across all source datasets
    combined_train_dataset = ConcatDataset(train_datasets)
    combined_val_dataset   = ConcatDataset(val_datasets)

    # Merge DataFrames — normalise to a common schema (image_col, diagnosis, _dataset)
    # We use a unified label column name for the combined frame
    unified_frames = []
    for ds_name, df in zip(source_names, combined_train_dfs):
        reg      = DATASET_REGISTRY[ds_name]
        diag_col = reg["diagnosis_col"]
        frame = df[[diag_col, "_dataset"]].copy()
        frame = frame.rename(columns={diag_col: "diagnosis"})
        unified_frames.append(frame)
    combined_train_df = pd.concat(unified_frames, ignore_index=True)

    logger.info(
        f"Combined source train | total={len(combined_train_dataset)} samples "
        f"| datasets={source_names}"
    )
    logger.info(
        f"Combined source val   | total={len(combined_val_dataset)} samples"
    )

    train_loader = DataLoader(
        combined_train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
        prefetch_factor=config["prefetch_factor"],
        worker_init_fn=seed_worker,
        generator=g,
    )
    val_loader = DataLoader(
        combined_val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
        prefetch_factor=config["prefetch_factor"],
    )

    logger.info(
        f"Source DataLoaders ready | batch={config['batch_size']} "
        f"| train_batches={len(train_loader)} | val_batches={len(val_loader)}"
    )

    return train_loader, val_loader, combined_train_df


# ----------------------------------------------------------------
# Target loaders  (Messidor-Grp2, Messidor-Grp3, IDRiD — no labels in loss)
# ----------------------------------------------------------------

def build_dann_target_train_loader(config: dict) -> DataLoader:
    """
    Build a combined target DataLoader for the adversarial domain loss.

    All target datasets (Messidor-Grp2, Messidor-Grp3, IDRiD) are merged
    into one loader. Domain labels are set to 1 (target) — the class_label
    is present in the batch but ignored during the DANN training loop.

    Parameters
    ----------
    config : dict  — must contain 'target_datasets', 'batch_size', etc.

    Returns
    -------
    DataLoader — yields (image, class_label, domain_id=1)
    """
    target_names = config["target_datasets"]
    target_datasets = []

    for ds_name in target_names:
        reg      = DATASET_REGISTRY[ds_name]
        diag_col = reg["diagnosis_col"]

        df = pd.read_csv(reg["target_path"]).reset_index(drop=True)
        logger.info(f"[{ds_name}] Target CSV loaded | rows={len(df)}")

        base = RetinopathyDataset(
            dataframe=df,
            img_path=reg["image_path"],
            img_col=reg["image_col"],
            label_col=diag_col,
            transforms=val_transformer,   # no augmentation on target
            extension=reg["extension"],
            clahe_image_path=reg.get("clahe_image_path"),   # offline CLAHE dir
        )
        target_datasets.append(DANNDataset(base, domain_id=1))

    combined_target = ConcatDataset(target_datasets)

    loader = DataLoader(
        combined_target,
        batch_size=config["batch_size"],
        shuffle=True,   # shuffle so target batches are mixed across datasets
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
        prefetch_factor=config["prefetch_factor"],
        worker_init_fn=seed_worker,
        generator=g,
    )

    logger.info(
        f"Target train loader | total={len(combined_target)} "
        f"| batches={len(loader)} | datasets={target_names}"
    )
    return loader


def build_dann_target_eval_loader(ds_name: str, config: dict) -> DataLoader:
    """
    Build a DataLoader for a single target dataset — used for
    per-dataset QWK evaluation after DANN training completes.

    No domain label — yields standard (image, class_label) tuples,
    same format as the baseline test loader.

    Parameters
    ----------
    ds_name : str  — must be a key in DATASET_REGISTRY.
                     Valid target datasets: 'Messidor-Grp2', 'Messidor-Grp3', 'IDRiD'
    config  : dict

    Returns
    -------
    DataLoader — yields (image, class_label)
    """
    reg      = DATASET_REGISTRY[ds_name]
    diag_col = reg["diagnosis_col"]

    df = pd.read_csv(reg["target_path"]).reset_index(drop=True)
    logger.info(f"[{ds_name}] Eval CSV loaded | rows={len(df)}")

    dataset = RetinopathyDataset(
        dataframe=df,
        img_path=reg["image_path"],
        img_col=reg["image_col"],
        label_col=diag_col,
        transforms=val_transformer,
        extension=reg["extension"],
        num_samples=config.get("test_max_samples"),
        clahe_image_path=reg.get("clahe_image_path"),       # offline CLAHE dir
    )

    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
        prefetch_factor=config["prefetch_factor"],
    )

    logger.info(
        f"[{ds_name}] Eval loader ready | samples={len(dataset)} "
        f"| batches={len(loader)}"
    )
    return loader
