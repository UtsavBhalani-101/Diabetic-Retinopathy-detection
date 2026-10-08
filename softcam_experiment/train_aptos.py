"""Train SoftCAMEfficientNet on APTOS 2019 with ElasticNet regularized evidence maps.

SoftCAM eliminates the global average pooling layer and replaces the linear
classifier with a 1x1 convolutional class-evidence layer. The model is trained
end-to-end with an ElasticNet (L1 + L2) loss penalty directly applied to the
spatial evidence maps.

Usage (Kaggle / local venv)
---------------------------
  python -m softcam_experiment.train_aptos
  python -m softcam_experiment.train_aptos --epochs 15 --batch-size 32 --lambda1 1e-4

Outputs
-------
  artifacts/weights/aptos_softcam.pth           (checkpoint)
  artifacts/calibration/optimal_T_softcam.npy    (calibrated temperature scalar)
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
from pipeline.setup.config import set_seed
from pipeline.setup.utils import DATASET_REGISTRY
from softcam_experiment.softcam import SoftCAMEfficientNet, SoftCAMLoss

LOGGER = logging.getLogger(__name__)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=64, help="Batch size (64 recommended for 2x T4).")
    p.add_argument(
        "--num-workers",
        type=int,
        default=min(4, os.cpu_count() or 4),
        help="DataLoader worker processes (default 4 on Kaggle).",
    )
    p.add_argument("--dropout-rate", type=float, default=0.3)
    p.add_argument(
        "--lambda1",
        type=float,
        default=1e-4,
        help="Lasso (L1) sparsity penalty on evidence maps (paper recommends ~1e-4).",
    )
    p.add_argument(
        "--lambda2",
        type=float,
        default=0.0,
        help="Ridge (L2) smoothness penalty on evidence maps.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--device",
        default=None,
        help="Force device ('cpu' or 'cuda'). Auto-detected if omitted.",
    )
    p.add_argument("--image-dir", default=None, help="Override APTOS train image dir.")
    p.add_argument("--labels-csv", default=None, help="Override APTOS train CSV.")
    p.add_argument(
        "--model-save-path",
        default="artifacts/weights/aptos_softcam.pth",
    )
    p.add_argument(
        "--temperature-save-path",
        default="artifacts/calibration/optimal_T_softcam.npy",
    )
    p.add_argument(
        "--mc-passes",
        type=int,
        default=30,
        help="MC Dropout passes for post-training temperature calibration.",
    )
    return p.parse_args()


def _build_loaders(args: argparse.Namespace):
    """Stratified 80/20 split of APTOS train.csv -> (train_loader, val_loader, train_df)."""
    reg = DATASET_REGISTRY.get("APTOS_2019", {})
    csv_path = args.labels_csv or reg.get("target_path")
    img_dir = args.image_dir or reg.get("image_path")
    img_col = reg.get("image_col", "id_code")
    label_col = reg.get("diagnosis_col", "diagnosis")
    ext = reg.get("extension", ".png")

    alt_csv = "/kaggle/input/aptos2019-blindness-detection/train.csv"
    alt_img = "/kaggle/input/aptos2019-blindness-detection/train_images"
    if not args.labels_csv and csv_path and not os.path.exists(csv_path) and os.path.exists(alt_csv):
        csv_path = alt_csv
    if not args.image_dir and img_dir and not os.path.exists(img_dir) and os.path.exists(alt_img):
        img_dir = alt_img

    df = pd.read_csv(csv_path).reset_index(drop=True)
    LOGGER.info("APTOS CSV loaded: %d rows from %s", len(df), csv_path)
    LOGGER.info("APTOS image directory: %s", img_dir)
    LOGGER.info(
        "Class distribution:\n%s",
        df[label_col].value_counts().sort_index().to_string(),
    )

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=args.seed,
        stratify=df[label_col],
    )
    LOGGER.info("Split -> train: %d | val: %d", len(train_df), len(val_df))

    train_ds = RetinopathyDataset(
        dataframe=train_df,
        img_path=img_dir,
        img_col=img_col,
        label_col=label_col,
        transforms=train_transformer,
        extension=ext,
        clahe_image_path=None,
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

    loader_kwargs: dict = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, train_df


def _class_weighted_loss(
    train_df: pd.DataFrame, device: torch.device, label_col: str = "diagnosis"
) -> torch.nn.CrossEntropyLoss:
    if label_col not in train_df.columns:
        reg = DATASET_REGISTRY.get("APTOS_2019", {})
        label_col = reg.get("diagnosis_col", "diagnosis")
    labels = train_df[label_col].values
    classes = np.unique(labels)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    LOGGER.info(
        "Class weights: %s",
        {int(c): f"{w:.3f}" for c, w in zip(classes, weights)},
    )
    return torch.nn.CrossEntropyLoss(weight=torch.FloatTensor(weights).to(device))


def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.CrossEntropyLoss,
    device: torch.device,
) -> tuple[float, float, np.ndarray]:
    """Evaluate model on validation loader."""
    model.eval()
    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for images, labels in loader:
            images = gpu_normalize(images.to(device))
            labels = labels.to(device)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(images)
                loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    val_loss = total_loss / len(loader)
    val_qwk = cohen_kappa_score(all_labels, all_preds, weights="quadratic")
    from sklearn.metrics import confusion_matrix

    matrix = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2, 3, 4])
    return val_loss, float(val_qwk), matrix


def _mc_logits_and_labels(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    T: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run T stochastic forward passes with dropout active -> mean logits + labels."""
    model.train()  # Keep dropout enabled
    all_logit_runs: list[np.ndarray] = []
    labels_collected: list[int] = []

    with torch.no_grad():
        for pass_idx in range(T):
            run_logits: list[np.ndarray] = []
            for images, labels in loader:
                images = gpu_normalize(images.to(device))
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    logits = model(images)
                run_logits.append(logits.float().cpu().numpy())
                if pass_idx == 0:
                    labels_collected.extend(labels.numpy().tolist())
            all_logit_runs.append(np.concatenate(run_logits, axis=0))

    model.eval()
    mean_logits = np.stack(all_logit_runs, axis=0).mean(axis=0)
    labels_array = np.array(labels_collected, dtype=np.int64)
    return mean_logits, labels_array


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = parse_args()
    root = _project_root()
    device = torch.device(
        args.device
        if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    n_gpus = torch.cuda.device_count()
    LOGGER.info("Device: %s | Total GPUs available: %d", device, n_gpus)
    if n_gpus > 0:
        for i in range(n_gpus):
            LOGGER.info("  GPU %d: %s", i, torch.cuda.get_device_name(i))
    set_seed(args.seed)

    # 1. Data
    train_loader, val_loader, train_df = _build_loaders(args)

    # 2. Model & Regularized SoftCAM Loss
    base_model = SoftCAMEfficientNet(
        num_classes=5,
        dropout_rate=args.dropout_rate,
        pretrained=True,
    ).to(device)

    # Multi-GPU support (e.g. 2x T4 on Kaggle)
    if n_gpus > 1:
        LOGGER.info("Enabling torch.nn.DataParallel across %d GPUs!", n_gpus)
        model = torch.nn.DataParallel(base_model)
    else:
        model = base_model

    # cuDNN auto-tuner for fixed 224x224 tensor shapes
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    base_criterion = _class_weighted_loss(train_df, device)
    softcam_loss_fn = SoftCAMLoss(
        base_criterion=base_criterion,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

    LOGGER.info(
        "SoftCAM params: %d total, %d trainable",
        sum(p.numel() for p in base_model.parameters()),
        sum(p.numel() for p in base_model.parameters() if p.requires_grad),
    )
    LOGGER.info(
        "Regularization: lambda1 (L1 Lasso)=%.2e | lambda2 (L2 Ridge)=%.2e | AMP=%s",
        args.lambda1,
        args.lambda2,
        device.type == "cuda",
    )

    # 3. Training Loop
    LOGGER.info("Starting SoftCAM training for %d epochs...", args.epochs)
    os.makedirs(root / "artifacts" / "weights", exist_ok=True)
    os.makedirs(root / "artifacts" / "calibration", exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        running_total_loss = 0.0
        running_ce_loss = 0.0
        running_l1_reg = 0.0

        for images, labels in train_loader:
            images = gpu_normalize(images.to(device))
            labels = labels.to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits, class_maps = model(images, return_maps=True)
                total_loss, ce_loss, l1_reg, _ = softcam_loss_fn(logits, labels, class_maps)

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_total_loss += total_loss.item()
            running_ce_loss += ce_loss.item()
            running_l1_reg += l1_reg.item()

        n_batches = len(train_loader)
        epoch_loss = running_total_loss / n_batches
        epoch_ce = running_ce_loss / n_batches
        epoch_l1 = running_l1_reg / n_batches

        # Validation
        val_loss, val_qwk, matrix = _evaluate(model, val_loader, base_criterion, device)

        LOGGER.info(
            "Epoch %d/%d | train_loss=%.4f (CE=%.4f, L1_reg=%.4f) | val_loss=%.4f | val_QWK=%.4f",
            epoch + 1,
            args.epochs,
            epoch_loss,
            epoch_ce,
            epoch_l1,
            val_loss,
            val_qwk,
        )
        LOGGER.info("Confusion matrix:\n%s", matrix)

    # 4. Post-training Temperature Calibration
    LOGGER.info(
        "Post-training temperature calibration with MC Dropout (T=%d)...",
        args.mc_passes,
    )
    mean_logits, val_labels = _mc_logits_and_labels(
        model, val_loader, device, T=args.mc_passes
    )
    optimal_T = find_temperature(mean_logits, val_labels)
    cal_probs = apply_temperature(mean_logits, optimal_T)
    final_preds = cal_probs.argmax(axis=1)
    final_qwk = cohen_kappa_score(val_labels, final_preds, weights="quadratic")

    LOGGER.info("Optimal temperature T = %.4f", optimal_T)
    LOGGER.info("Calibrated val QWK    = %.4f", final_qwk)

    # 5. Save Model and Temperature (unwrap DataParallel)
    model_path = root / args.model_save_path
    T_path = root / args.temperature_save_path

    raw_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
    torch.save(raw_to_save.state_dict(), model_path)
    np.save(T_path, np.array(optimal_T))

    LOGGER.info("SoftCAM model saved -> %s", model_path)
    LOGGER.info("Optimal T saved     -> %s (T=%.4f)", T_path, optimal_T)
    print(
        f"\nSoftCAM training complete. Next steps:\n"
        f"  1. python -m softcam_experiment.generate_softcam_heatmaps\n"
        f"  2. python -m softcam_experiment.run_softcam_occlusion\n"
    )


if __name__ == "__main__":
    main()
