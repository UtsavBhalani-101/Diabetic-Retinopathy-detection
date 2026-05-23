# pipeline/loaders.py
# ============================================================
# DataLoader factory functions:
#   - build_loaders_for_training   → train_loader, val_loader, train_df, val_df
#   - build_loader_for_testing     → single loader (full dataset or official test split)
# ============================================================

import logging

import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from .config  import DATASET_REGISTRY
from .dataset import (
    RetinopathyDataset,
    RetinopathyDatasetFromDF,
    train_transformer,
    val_transformer,
    seed_worker,
    g,
)

logger = logging.getLogger(__name__)


def build_loaders_for_training(dataset_name: str, config: dict):
    """
    Read the registry for dataset_name, perform a stratified 80/20 split
    on the CSV, and return reproducible train + val DataLoaders.

    Returns
    -------
    train_loader, val_loader, train_df, val_df
    """
    logger.info(f"[{dataset_name}] Building training loaders...")

    reg      = DATASET_REGISTRY[dataset_name]
    df       = pd.read_csv(reg["target_path"]).reset_index(drop=True)
    diag_col = reg["diagnosis_col"]

    logger.info(f"[{dataset_name}] CSV loaded: {len(df)} rows from {reg['target_path']}")
    logger.info(f"[{dataset_name}] Class distribution:\n{df[diag_col].value_counts().sort_index().to_string()}")

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=config["seed"],
        stratify=df[diag_col]
    )
    logger.info(
        f"[{dataset_name}] Stratified split → train: {len(train_df)} | val: {len(val_df)}"
    )

    train_dataset = RetinopathyDatasetFromDF(
        df=train_df,
        image_path=reg["image_path"],
        image_col=reg["image_col"],
        diagnosis_col=diag_col,
        transform=train_transformer,
        extension=reg["extension"]
    )
    val_dataset = RetinopathyDatasetFromDF(
        df=val_df,
        image_path=reg["image_path"],
        image_col=reg["image_col"],
        diagnosis_col=diag_col,
        transform=val_transformer,
        extension=reg["extension"]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
        prefetch_factor=config["prefetch_factor"],
        worker_init_fn=seed_worker,
        generator=g
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
        prefetch_factor=config["prefetch_factor"]
    )

    logger.info(
        f"[{dataset_name}] DataLoaders ready "
        f"| batch={config['batch_size']} | workers={config['num_workers']}"
    )
    return train_loader, val_loader, train_df, val_df


def build_loader_for_testing(dataset_name: str, config: dict,
                              use_test_split: bool = False):
    """
    Return a single DataLoader for a dataset (no train/val split).

    If use_test_split=True **and** the registry has 'test_target_path',
    the official held-out test images/CSV are used (e.g. IDRiD test set).

    Returns
    -------
    loader : DataLoader
    """
    reg = DATASET_REGISTRY[dataset_name]

    if use_test_split and "test_target_path" in reg:
        target_path = reg["test_target_path"]
        image_path  = reg["test_image_path"]
        logger.info(f"[{dataset_name}] Using official TEST split for evaluation")
    else:
        target_path = reg["target_path"]
        image_path  = reg["image_path"]
        logger.info(f"[{dataset_name}] Using full training CSV for evaluation (zero-shot)")

    dataset = RetinopathyDataset(
        input_path=image_path,
        target_path=target_path,
        image_col=reg["image_col"],
        diagnosis_col=reg["diagnosis_col"],
        transforms=val_transformer,
        extension=reg["extension"]
    )

    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
        prefetch_factor=config["prefetch_factor"]
    )

    logger.info(
        f"[{dataset_name}] Test loader ready | samples={len(dataset)} "
        f"| batch={config['batch_size']}"
    )
    return loader
