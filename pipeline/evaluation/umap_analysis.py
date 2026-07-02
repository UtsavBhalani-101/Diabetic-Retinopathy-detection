# pipeline/evaluation/umap_analysis.py
# ============================================================
# UMAP feature-space visualisation:
#   - extract_features : pull 1280-dim EfficientNet embeddings
#                        from model.base() for a full DataLoader
#   - plot_umap        : fit UMAP on those features, save a
#                        colour-coded scatter plot (by DR grade)
#   - compare_umaps    : side-by-side 3-panel figure for the
#                        occlusion experiment (original vs top-10
#                        vs top-30 occluded)
#
# Feature source
# --------------
# We embed via model.base(images) — the 1280-dim global average-
# pooled output of EfficientNet-B0 BEFORE the dropout + classifier
# head.  This is the same feature space used by cosine_similarity.py
# and the per-class centroid computation in train.py, ensuring all
# analyses are comparable.
# ============================================================

import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb

logger = logging.getLogger(__name__)

# DR grade colour palette (one colour per class, colour-blind friendly)
_CLASS_COLOURS = [
    "#4C72B0",  # 0 — No DR      (blue)
    "#55A868",  # 1 — Mild        (green)
    "#C44E52",  # 2 — Moderate    (red)
    "#8172B2",  # 3 — Severe      (purple)
    "#CCB974",  # 4 — Proliferative (yellow-brown)
]
_CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]


# ──────────────────────────────────────────────────────────────
# 1.  Feature extraction
# ──────────────────────────────────────────────────────────────

def extract_features(
    model: torch.nn.Module,
    loader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run a single forward pass through model.base() for every batch
    in `loader` and collect the resulting feature vectors.

    Parameters
    ----------
    model  : EfficientNetMC (must expose .base attribute)
    loader : DataLoader yielding (images, labels)
    device : torch.device

    Returns
    -------
    features : np.ndarray  shape [N, 1280]
    labels   : np.ndarray  shape [N]  integer DR grades
    """
    model.eval()
    all_features: list[np.ndarray] = []
    all_labels:   list[int]        = []

    logger.info("extract_features() | starting feature extraction...")

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(device)
            features = model.base(images)              # [B, 1280]
            all_features.append(features.cpu().numpy())
            all_labels.extend(labels.numpy())

            if (batch_idx + 1) % max(1, len(loader) // 4) == 0:
                logger.debug(
                    f"extract_features() | batch {batch_idx + 1}/{len(loader)}"
                )

    features = np.vstack(all_features)   # [N, 1280]
    labels   = np.array(all_labels)      # [N]

    logger.info(
        f"extract_features() | done | N={len(labels)} samples | "
        f"feature dim={features.shape[1]}"
    )
    return features, labels


# ──────────────────────────────────────────────────────────────
# 2.  Single UMAP plot
# ──────────────────────────────────────────────────────────────

def plot_umap(
    features:    np.ndarray,
    labels:      np.ndarray,
    title:       str,
    save_path:   str,
    n_neighbors: int = 15,
    min_dist:    float = 0.1,
) -> np.ndarray:
    """
    Fit a 2-D UMAP on `features` and save a scatter plot coloured
    by DR grade label.

    Parameters
    ----------
    features    : [N, D] feature matrix (e.g. 1280-dim embeddings)
    labels      : [N]    integer labels 0-4
    title       : plot title string
    save_path   : absolute path to save the .png file
    n_neighbors : UMAP hyperparameter (controls local vs global structure)
    min_dist    : UMAP hyperparameter (controls cluster tightness)

    Returns
    -------
    embedding : [N, 2] 2-D UMAP coordinates (useful for compare_umaps)
    """
    try:
        import umap  # umap-learn
    except ImportError:
        raise ImportError(
            "umap-learn is required: pip install umap-learn"
        )

    logger.info(
        f"plot_umap() | fitting UMAP | N={len(labels)} | title='{title}'"
    )

    reducer   = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=42,
        verbose=False,
    )
    embedding = reducer.fit_transform(features)   # [N, 2]

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 7))

    for c in range(5):
        mask = (labels == c)
        if mask.sum() == 0:
            continue
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=_CLASS_COLOURS[c],
            label=f"{c} — {_CLASS_NAMES[c]} (n={mask.sum()})",
            alpha=0.75,
            s=30,
            edgecolors="none",
        )

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.legend(loc="best", fontsize=9, markerscale=1.5)
    ax.grid(True, linewidth=0.4, alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

    logger.info(f"plot_umap() | saved → {save_path}")
    return embedding


# ──────────────────────────────────────────────────────────────
# 3.  Side-by-side 3-panel comparison figure
# ──────────────────────────────────────────────────────────────

def compare_umaps(
    embeddings:  list[np.ndarray],
    labels_list: list[np.ndarray],
    titles:      list[str],
    save_path:   str,
) -> None:
    """
    Create a side-by-side 3-panel UMAP figure for the occlusion
    experiment (original / top-10 / top-30 occluded).

    Parameters
    ----------
    embeddings  : list of 3 arrays, each [N, 2] (pre-computed UMAP coords)
    labels_list : list of 3 label arrays, each [N]
    titles      : list of 3 title strings
    save_path   : absolute path to save the comparison .png
    """
    assert len(embeddings) == 3, "compare_umaps expects exactly 3 embeddings"

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    for ax, embedding, labels, title in zip(axes, embeddings, labels_list, titles):
        for c in range(5):
            mask = (labels == c)
            if mask.sum() == 0:
                continue
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                c=_CLASS_COLOURS[c],
                label=f"{c} — {_CLASS_NAMES[c]}",
                alpha=0.75,
                s=25,
                edgecolors="none",
            )
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.legend(loc="best", fontsize=7, markerscale=1.3)
        ax.grid(True, linewidth=0.3, alpha=0.5)

    plt.suptitle(
        "Feature Space Shift Under GradCAM Occlusion (IDRiD)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"compare_umaps() | 3-panel figure saved → {save_path}")

    # Log to W&B
    try:
        wandb.log({"umap/comparison_3panel": wandb.Image(save_path)})
        logger.info("compare_umaps() | logged to W&B")
    except Exception as exc:
        logger.warning(f"compare_umaps() | W&B log failed: {exc}")
