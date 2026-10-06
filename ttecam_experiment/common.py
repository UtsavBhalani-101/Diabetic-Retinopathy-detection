"""Shared helpers for standalone TTECAM experiments."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from torch.utils.data import DataLoader

from pipeline.data.dataset import RetinopathyDataset, val_transformer
from pipeline.evaluation.calibration import apply_temperature
from pipeline.evaluation.evaluate import compute_uncertainty_signals, mc_evaluate_full
from pipeline.setup.config import (
    BASE_CONFIG,
    UNCERTAINTY_ENTROPY_THRESHOLD,
    UNCERTAINTY_MARGIN_THRESHOLD,
    UNCERTAINTY_MC_STD_THRESHOLD,
    set_seed,
)
from pipeline.training_loop_setup.model import EfficientNetMC


LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def local_idrid_paths(root: Path) -> dict[str, str]:
    base = root / "datasets" / "IDRiD" / "B. Disease Grading"
    return {
        "test_image_path": str(
            base / "1. Original Images" / "b. Testing Set"
        ),
        "test_target_path": str(
            base / "2. Groundtruths" / "b. IDRiD_Disease Grading_Testing Labels.csv"
        ),
        "image_col": "Image name",
        "diagnosis_col": "Retinopathy grade",
        "extension": ".jpg",
        "num_classes": 5,
    }


def build_idrid_test_dataset(args: argparse.Namespace) -> RetinopathyDataset:
    paths = local_idrid_paths(project_root())
    image_path = args.image_dir or paths["test_image_path"]
    target_path = args.labels_csv or paths["test_target_path"]

    return RetinopathyDataset(
        img_path=image_path,
        target_path=target_path,
        img_col=paths["image_col"],
        label_col=paths["diagnosis_col"],
        transforms=val_transformer,
        extension=paths["extension"],
        num_samples=args.max_samples,
        clahe_image_path=None,
    )


def build_loader(dataset: RetinopathyDataset, args: argparse.Namespace) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def load_model(args: argparse.Namespace, device: torch.device) -> EfficientNetMC:
    model = EfficientNetMC(
        num_classes=5,
        dropout_rate=args.dropout_rate,
        pretrained=False,
    )
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def load_temperature(args: argparse.Namespace) -> float:
    if args.temperature is not None:
        return float(args.temperature)
    return float(np.load(args.temperature_path))


def evaluate_condition(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    mc_passes: int,
    temperature: float,
    baseline_true_probs: np.ndarray | None = None,
) -> dict[str, Any]:
    mean_probs, uncertainties, labels, logits = mc_evaluate_full(
        model, loader, device, T=mc_passes
    )
    calibrated_probs = apply_temperature(logits, temperature)
    preds = calibrated_probs.argmax(axis=1)
    confidences = calibrated_probs.max(axis=1)
    true_probs = calibrated_probs[np.arange(len(labels)), labels]
    qwk = cohen_kappa_score(labels, preds, weights="quadratic")
    matrix = confusion_matrix(labels, preds, labels=[0, 1, 2, 3, 4])
    entropy, margin, mc_uncertainty = compute_uncertainty_signals(
        calibrated_probs, uncertainties
    )
    uncertain_mask = (
        (entropy > UNCERTAINTY_ENTROPY_THRESHOLD)
        | (margin < UNCERTAINTY_MARGIN_THRESHOLD)
        | (mc_uncertainty > UNCERTAINTY_MC_STD_THRESHOLD)
    )
    correct_mask = preds == labels

    result: dict[str, Any] = {
        "qwk": float(qwk),
        "mean_confidence": float(confidences.mean()),
        "mean_entropy": float(entropy.mean()),
        "mean_margin": float(margin.mean()),
        "mean_mc_uncertainty": float(mc_uncertainty.mean()),
        "uncertain_fraction": float(uncertain_mask.mean()),
        "quadrant_certain_wrong": int((~uncertain_mask & ~correct_mask).sum()),
        "quadrant_certain_right": int((~uncertain_mask & correct_mask).sum()),
        "quadrant_uncertain_wrong": int((uncertain_mask & ~correct_mask).sum()),
        "quadrant_uncertain_right": int((uncertain_mask & correct_mask).sum()),
        "confusion_matrix": matrix.tolist(),
        "labels": labels.tolist(),
        "predictions": preds.tolist(),
        "true_class_probs": true_probs.tolist(),
    }

    if baseline_true_probs is not None:
        baseline_true_probs = np.asarray(baseline_true_probs, dtype=np.float32)
        drops = baseline_true_probs - true_probs
        result["mean_prob_drop_vs_baseline"] = float(drops.mean())
        result["positive_drop_fraction"] = float((drops > 0).mean())

    return result


def write_results_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "condition",
        "top_k_percent",
        "seed",
        "qwk",
        "mean_confidence",
        "mean_prob_drop_vs_baseline",
        "positive_drop_fraction",
        "uncertain_fraction",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model-path",
        default=BASE_CONFIG["model_save_path"],
        help="Path to the pre-DANN EfficientNetMC checkpoint.",
    )
    parser.add_argument(
        "--temperature-path",
        default=BASE_CONFIG["optimal_T_save_path"],
        help="Path to saved temperature scalar .npy.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Override the saved temperature scalar.",
    )
    parser.add_argument("--image-dir", default=None, help="Override IDRiD test image dir.")
    parser.add_argument("--labels-csv", default=None, help="Override IDRiD test CSV.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--dropout-rate", type=float, default=BASE_CONFIG["dropout_rate"])
    parser.add_argument("--seed", type=int, default=BASE_CONFIG["seed"])


def select_device(device_name: str | None) -> torch.device:
    if device_name:
        return torch.device(device_name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def initialize(args: argparse.Namespace) -> torch.device:
    configure_logging()
    set_seed(args.seed)
    device = select_device(args.device)
    LOGGER.info("Using device: %s", device)
    return device
