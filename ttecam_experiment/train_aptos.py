"""Train EfficientNetMC on APTOS 2019 for the TTECAM experiment.

Uses the existing DATASET_REGISTRY["APTOS_2019"] paths (Kaggle paths) so no
path changes are needed when running on Kaggle. WandB is intentionally excluded
— this script trains, calibrates, and saves checkpoint + temperature without
any external service dependency.

Usage (Kaggle / local venv)
---------------------------
  python -m ttecam_experiment.train_aptos
  python -m ttecam_experiment.train_aptos --epochs 15 --batch-size 64

Outputs
-------
  artifacts/weights/aptos_efficientnet.pth   (checkpoint, overwritten)
  artifacts/calibration/optimal_T.npy        (temperature scalar, overwritten)

After training, run in order:
  python -m ttecam_experiment.generate_ttecam_heatmaps
  python -m ttecam_experiment.run_ttecam_occlusion
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader

from pipeline.data.dataset import RetinopathyDataset, train_transformer, val_transformer
from pipeline.data.gpu_transforms import gpu_normalize
from pipeline.evaluation.calibration import apply_temperature, find_temperature
from pipeline.evaluation.evaluate import evaluate
from pipeline.setup.config import set_seed
from pipeline.setup.utils import DATASET_REGISTRY
from pipeline.training_loop_setup.model import EfficientNetMC


LOGGER = logging.getLogger(__name__)


# ─── Paths ──────────────────────────────────────────────────────────────────

def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


# Pull APTOS image/CSV paths from the shared registry (Kaggle-compatible paths)
_APTOS_REG = DATASET_REGISTRY["APTOS_2019"]


# ─── CLI ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs",       type=int,   default=10)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--batch-size",   type=int,   default=32,
                   help="Per-iteration batch size (no DataParallel here).")
    p.add_argument("--num-workers",  type=int,   default=0,
                   help="DataLoader workers (0 = main process, safe on Windows).")
    p.add_argument("--dropout-rate", type=float, default=0.3)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--device",       default=None,
                   help="Force a device, e.g. 'cpu' or 'cuda'. Auto-detected if omitted.")
    p.add_argument("--model-save-path",
                   default="artifacts/weights/aptos_efficientnet.pth")
    p.add_argument("--temperature-save-path",
                   default="artifacts/calibration/optimal_T.npy")
    p.add_argument("--mc-passes",    type=int,   default=30,
                   help="MC Dropout passes for post-training temperature calibration.")
    return p.parse_args()


# ─── Training helpers ────────────────────────────────────────────────────────

def _build_loaders(args: argparse.Namespace):
    """Stratified 80/20 split of APTOS train.csv → (train_loader, val_loader, train_df).

    Paths come from DATASET_REGISTRY["APTOS_2019"] — the same Kaggle-compatible
    paths used by the main pipeline. CLAHE is intentionally disabled here
    (clahe_image_path=None) since the offline preprocessed cache may not exist;
    gpu_normalize handles ImageNet normalization inline during training.
    """
    csv_path  = _APTOS_REG["target_path"]
    img_dir   = _APTOS_REG["image_path"]
    img_col   = _APTOS_REG["image_col"]       # "id_code"
    label_col = _APTOS_REG["diagnosis_col"]   # "diagnosis"
    ext       = _APTOS_REG["extension"]       # ".png"

    df = pd.read_csv(csv_path).reset_index(drop=True)
    LOGGER.info("APTOS CSV loaded: %d rows from %s", len(df), csv_path)
    LOGGER.info(
        "Class distribution:\n%s",
        df[label_col].value_counts().sort_index().to_string()
    )

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=args.seed,
        stratify=df[label_col],
    )
    LOGGER.info("Split → train: %d | val: %d", len(train_df), len(val_df))

    train_ds = RetinopathyDataset(
        dataframe=train_df,
        img_path=img_dir,
        img_col=img_col,
        label_col=label_col,
        transforms=train_transformer,
        extension=ext,
        clahe_image_path=None,   # no offline CLAHE cache needed
    )
    val_ds = RetinopathyDataset(
        dataframe=val_df,
        img_path=img_dir,
        img_col=img_col,
        label_col=label_col,
        transforms=val_transformer,
        extension=ext,
        clahe_image_path=None,
    )

    # pin_memory only if CUDA is available; prefetch_factor only if workers > 0
    loader_kwargs: dict = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    return train_loader, val_loader, train_df


def _class_weighted_loss(
    train_df: pd.DataFrame, device: torch.device
) -> torch.nn.CrossEntropyLoss:
    label_col = _APTOS_REG["diagnosis_col"]
    labels  = train_df[label_col].values
    classes = np.unique(labels)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    LOGGER.info(
        "Class weights: %s",
        {int(c): f"{w:.3f}" for c, w in zip(classes, weights)},
    )
    return torch.nn.CrossEntropyLoss(weight=torch.FloatTensor(weights).to(device))


class _NormalizedForEval(torch.nn.Module):
    """Thin wrapper: applies gpu_normalize before the model forward pass.

    evaluate() from pipeline.evaluation.evaluate does NOT normalize images —
    it expects normalized tensors. We wrap here so that raw [0,1] val images
    are normalized on-the-fly during validation, matching training behaviour.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self._model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(gpu_normalize(x))


def _mc_logits_and_labels(
    model: EfficientNetMC,
    loader: DataLoader,
    device: torch.device,
    T: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run T MC-Dropout passes on val_loader → mean logits + labels.

    Keeps dropout active (model.train()) so each forward pass is stochastic.
    Labels are collected only on the first pass to avoid duplicates.
    """
    model.train()   # enable dropout
    all_logit_runs: list[np.ndarray] = []
    labels_collected: list[int] = []

    with torch.no_grad():
        for pass_idx in range(T):
            run_logits: list[np.ndarray] = []
            for batch_idx, (images, labels) in enumerate(loader):
                images = gpu_normalize(images.to(device))
                logits = model(images)
                run_logits.append(logits.cpu().numpy())
                if pass_idx == 0:
                    labels_collected.extend(labels.numpy().tolist())
            all_logit_runs.append(np.concatenate(run_logits, axis=0))

    model.eval()
    mean_logits  = np.stack(all_logit_runs, axis=0).mean(axis=0)   # [N, C]
    labels_array = np.array(labels_collected, dtype=np.int64)
    return mean_logits, labels_array


# ─── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args   = parse_args()
    root   = _project_root()
    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    LOGGER.info("Device: %s", device)
    set_seed(args.seed)

    # ── 1. Data ──────────────────────────────────────────────────────────────
    train_loader, val_loader, train_df = _build_loaders(args)

    # ── 2. Model + criterion ─────────────────────────────────────────────────
    model = EfficientNetMC(
        num_classes=5,
        dropout_rate=args.dropout_rate,
        pretrained=True,                  # ImageNet init for faster convergence
    )
    model = model.to(device)
    criterion = _class_weighted_loss(train_df, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    LOGGER.info(
        "Model params: %d total, %d trainable",
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    # ── 3. Epoch loop ─────────────────────────────────────────────────────────
    LOGGER.info("Training for %d epochs on APTOS (Kaggle registry paths)…", args.epochs)
    os.makedirs(root / "artifacts" / "weights",     exist_ok=True)
    os.makedirs(root / "artifacts" / "calibration", exist_ok=True)

    eval_wrapper = _NormalizedForEval(model)

    for epoch in range(args.epochs):
        # ── train pass ──────────────────────────────────────────────────────
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            # Apply ImageNet normalization on-the-fly (same as the main pipeline)
            images = gpu_normalize(images.to(device))
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        # ── val pass ─────────────────────────────────────────────────────────
        # evaluate() sets model.eval() and uses no_grad internally.
        # We pass eval_wrapper so raw val images are normalized before forward.
        val_loss, val_qwk, matrix, _, _ = evaluate(
            eval_wrapper, val_loader, criterion, device
        )

        LOGGER.info(
            "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | val_QWK=%.4f",
            epoch + 1, args.epochs, train_loss, val_loss, val_qwk,
        )
        LOGGER.info("Confusion matrix:\n%s", matrix)

    # ── 4. Post-training temperature calibration ───────────────────────────────
    LOGGER.info(
        "Post-training temperature calibration with MC Dropout (T=%d)…",
        args.mc_passes,
    )
    mean_logits, val_labels = _mc_logits_and_labels(
        model, val_loader, device, T=args.mc_passes
    )
    optimal_T   = find_temperature(mean_logits, val_labels)
    cal_probs   = apply_temperature(mean_logits, optimal_T)
    final_preds = cal_probs.argmax(axis=1)
    final_qwk   = cohen_kappa_score(val_labels, final_preds, weights="quadratic")

    LOGGER.info("Optimal temperature T  = %.4f", optimal_T)
    LOGGER.info("Calibrated val QWK     = %.4f", final_qwk)

    # ── 5. Save checkpoint + temperature ──────────────────────────────────────
    model_path = root / args.model_save_path
    T_path     = root / args.temperature_save_path

    torch.save(model.state_dict(), model_path)
    np.save(T_path, np.array(optimal_T))

    LOGGER.info("Model saved  → %s", model_path)
    LOGGER.info("Optimal T    → %s  (T=%.4f)", T_path, optimal_T)
    print(f"\nDone. Next steps:\n"
          f"  1. python -m ttecam_experiment.generate_ttecam_heatmaps\n"
          f"  2. python -m ttecam_experiment.run_ttecam_occlusion\n")


if __name__ == "__main__":
    main()
