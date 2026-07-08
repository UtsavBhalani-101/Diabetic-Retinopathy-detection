import numpy as np
import os
import torch
import logging
import wandb

logger = logging.getLogger(__name__)


def calculate_cosine_similarity(model, loader, device, num_classes,
                                per_class_mean,
                                save_dir="artifacts/cosine_similarity"):
    """
    Compute Cosine Similarity of every test sample to each class centroid
    fitted on training features, then log per-class and global
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
    save_dir : str
        Directory where per-class ``.npy`` similarity files and labels are saved.
        Defaults to ``artifacts/cosine_similarity``.

    Returns
    -------
    per_class_similarities : dict[int, np.ndarray]
        ``{class_idx: array_of_similarities}`` — Cosine similarity of every
        test sample to that class's centroid.
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
            if hasattr(model, "get_features"):
                features = model.get_features(images)
            else:
                features = model.base(images)           # [B, D]
            all_features.append(features.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_features = np.vstack(all_features)          # [N, D]
    all_labels = np.array(all_labels)
    n_samples = all_features.shape[0]

    logger.info(f"Extracted features for {n_samples} samples, "
                f"feature dim = {all_features.shape[1]}")

    # ── 2. Per-class Cosine Similarities ───────────────────────────────
    per_class_similarities = {}

    for c in range(num_classes):
        logger.info(f"Computing Cosine similarity to class {c} "
                    f"({class_names[c]})...")

        mu = per_class_mean[c]
        # cosine similarity = dot(x, mu) / (|x| * |mu|)
        dot = all_features @ mu  # [N]
        norms = np.linalg.norm(all_features, axis=1) * np.linalg.norm(mu)
        # Avoid division by zero
        norms = np.clip(norms, a_min=1e-8, a_max=None)
        similarities = dot / norms

        per_class_similarities[c] = similarities

    # ── 3. Report per-class stats ────────────────────────────────────────
    print("\n--- Per-Class Cosine Similarity Summary ---")
    wandb_payload = {}

    for c in range(num_classes):
        s = per_class_similarities[c]
        min_s, max_s, avg_s = float(s.min()), float(s.max()), float(s.mean())

        tag = class_names[c]
        print(f"  [{tag}]  min={min_s:.4f}  max={max_s:.4f}  avg={avg_s:.4f}")
        logger.info(f"  Class {c} ({tag}) | "
                    f"min={min_s:.4f}  max={max_s:.4f}  avg={avg_s:.4f}")

        wandb_payload[f"CosineSimilarity/{tag}/Min"] = round(min_s, 4)
        wandb_payload[f"CosineSimilarity/{tag}/Max"] = round(max_s, 4)
        wandb_payload[f"CosineSimilarity/{tag}/Avg"] = round(avg_s, 4)

    # ── 4. Global stats (across all classes) ─────────────────────────────
    all_sims = np.concatenate(list(per_class_similarities.values()))
    global_min = float(all_sims.min())
    global_max = float(all_sims.max())
    global_avg = float(all_sims.mean())

    print(f"\n  [Global]  min={global_min:.4f}  max={global_max:.4f}  "
          f"avg={global_avg:.4f}")

    wandb_payload["CosineSimilarity/Global/Min"] = round(global_min, 4)
    wandb_payload["CosineSimilarity/Global/Max"] = round(global_max, 4)
    wandb_payload["CosineSimilarity/Global/Avg"] = round(global_avg, 4)

    wandb.log(wandb_payload)

    # ── 5. Save per-class similarities and labels as .npy ───────────────────
    os.makedirs(save_dir, exist_ok=True)

    for c in range(num_classes):
        path = os.path.join(save_dir, f"similarities_class_{c}.npy")
        np.save(path, per_class_similarities[c])
        logger.info(f"Saved class {c} ({class_names[c]}) similarities → {path}")

    labels_path = os.path.join(save_dir, "labels.npy")
    np.save(labels_path, all_labels)
    logger.info(f"Saved test labels → {labels_path}")

    return per_class_similarities, all_labels
