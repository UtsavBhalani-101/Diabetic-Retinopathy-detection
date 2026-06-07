import numpy as np
import os
import torch
import logging
import wandb
from scipy.spatial.distance import mahalanobis

logger = logging.getLogger(__name__)


def calculate_mahalanobis_distance(model, loader, device, num_classes,
                                   per_class_mean, global_inv_cov,
                                   save_dir="artifacts/mahalanobis"):
    """
    Compute Mahalanobis distance of every test sample to each class-conditional
    Gaussian fitted on training features, then log per-class and global
    summary statistics to wandb.

    Parameters
    ----------
    model : nn.Module
        Must expose ``model.base(images)`` which returns feature vectors.
    loader : DataLoader
        Test / validation loader yielding (images, labels).
    device : torch.device
    num_classes : int
        Number of classes (e.g. 5 for DR grading).
    per_class_mean : dict[int, np.ndarray]
        ``{class_idx: mean_feature_vector}`` fitted on training data.
        Each value has shape ``[D]`` where D is the feature dimension.
    global_inv_cov : np.ndarray
        Single global inverse covariance matrix fitted on all training data.
        Has shape ``[D, D]``.
    save_dir : str
        Directory where per-class ``.npy`` distance files and labels are saved.
        Defaults to ``artifacts/mahalanobis``.

    Returns
    -------
    per_class_distances : dict[int, np.ndarray]
        ``{class_idx: array_of_distances}`` — Mahalanobis distance of every
        test sample to that class's distribution.
    all_labels : np.ndarray
        Ground-truth labels for every test sample, in loader order.
    """

    class_names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]

    # ── 1. Extract features from all test samples (single pass) ──────────
    model.eval()
    all_features = []
    all_labels = []

    logger.info("Extracting features from test set...")
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            features = model.base(images)           # [B, D]
            all_features.append(features.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_features = np.vstack(all_features)          # [N, D]
    all_labels = np.array(all_labels)
    n_samples = all_features.shape[0]

    logger.info(f"Extracted features for {n_samples} samples, "
                f"feature dim = {all_features.shape[1]}")

    # ── 2. Per-class Mahalanobis distances ───────────────────────────────
    per_class_distances = {}

    for c in range(num_classes):
        logger.info(f"Computing Mahalanobis distance to class {c} "
                    f"({class_names[c]})...")

        mean_c = per_class_mean[c]

        distances = np.empty(n_samples)
        for i in range(n_samples):
            # sqrt((x - μ_c)^T  Σ^{-1}  (x - μ_c))
            distances[i] = mahalanobis(all_features[i], mean_c, global_inv_cov)

        per_class_distances[c] = distances

    # ── 3. Report per-class stats ────────────────────────────────────────
    print("\n--- Per-Class Mahalanobis Distance Summary ---")
    wandb_payload = {}

    for c in range(num_classes):
        d = per_class_distances[c]
        min_d, max_d, avg_d = float(d.min()), float(d.max()), float(d.mean())

        tag = class_names[c]
        print(f"  [{tag}]  min={min_d:.4f}  max={max_d:.4f}  avg={avg_d:.4f}")
        logger.info(f"  Class {c} ({tag}) | "
                    f"min={min_d:.4f}  max={max_d:.4f}  avg={avg_d:.4f}")

        wandb_payload[f"Mahalanobis/{tag}/Min"] = round(min_d, 4)
        wandb_payload[f"Mahalanobis/{tag}/Max"] = round(max_d, 4)
        wandb_payload[f"Mahalanobis/{tag}/Avg"] = round(avg_d, 4)

    # ── 4. Global stats (across all classes) ─────────────────────────────
    all_dists = np.concatenate(list(per_class_distances.values()))
    global_min = float(all_dists.min())
    global_max = float(all_dists.max())
    global_avg = float(all_dists.mean())

    print(f"\n  [Global]  min={global_min:.4f}  max={global_max:.4f}  "
          f"avg={global_avg:.4f}")

    wandb_payload["Mahalanobis/Global/Min"] = round(global_min, 4)
    wandb_payload["Mahalanobis/Global/Max"] = round(global_max, 4)
    wandb_payload["Mahalanobis/Global/Avg"] = round(global_avg, 4)

    wandb.log(wandb_payload)

    # ── 5. Save per-class distances and labels as .npy ───────────────────
    os.makedirs(save_dir, exist_ok=True)

    for c in range(num_classes):
        path = os.path.join(save_dir, f"distances_class_{c}.npy")
        np.save(path, per_class_distances[c])
        logger.info(f"Saved class {c} ({class_names[c]}) distances → {path}")

    labels_path = os.path.join(save_dir, "labels.npy")
    np.save(labels_path, all_labels)
    logger.info(f"Saved test labels → {labels_path}")

    return per_class_distances, all_labels