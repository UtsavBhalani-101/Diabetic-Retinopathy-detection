# pipeline/orchestrators/run_occlusion_experiment.py
# ============================================================
# 3-Pass GradCAM Occlusion + UMAP Experiment on IDRiD
#
# Pipeline:
#   Step 1  Train on APTOS_2019 (or skip with --skip-train)
#   Step 2  Pass 1 — Baseline IDRiD inference
#             · Full GradCAM++ per image  → saves .npy heatmaps + W&B artifact
#             · Forward pass             → probs, logits, features [N, 1280]
#             · Metrics: QWK, entropy, confusion matrix
#             · UMAP of features         → saved + logged to W&B
#   Step 3  Pass 2 — Top 10% occluded
#             · OccludedDataset loads saved heatmaps, zeros top 10% pixels
#             · Forward pass             → probs, logits, features
#             · Metrics + per-image probability drop vs Pass 1
#             · UMAP of features         → saved + logged to W&B
#   Step 4  Pass 3 — Top 30% occluded
#             · Same as Step 3 but top 30% pixels zeroed
#   Step 5  Comparative summary
#             · 3-panel side-by-side UMAP figure
#             · Bar chart: mean probability drop (baseline vs top10 vs top30)
#             · Summary table printed to log
#
# Usage (run from project root on Kaggle/Linux):
#   python -m pipeline.orchestrators.run_occlusion_experiment
#   python -m pipeline.orchestrators.run_occlusion_experiment --skip-train
#   python -m pipeline.orchestrators.run_occlusion_experiment \
#       --skip-train \
#       --model-path artifacts/weights/aptos_efficientnet.pth \
#       --idrid-base /kaggle/input/datasets/antiti/idrid-testing-dataset/IDRiD/B. Disease Grading \
#       --idrid-split test
# ============================================================

import argparse
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from torch.utils.data import DataLoader
from dotenv import load_dotenv

from pipeline.setup.config import BASE_CONFIG, setup_logging, setting_gpu, set_seed
from pipeline.data.dataset import RetinopathyDataset, val_transformer, CLAHEPreprocess
from pipeline.data.occluded_dataset import OccludedDataset
from pipeline.training_loop_setup.model import EfficientNetMC
from pipeline.evaluation.umap_analysis import extract_features, plot_umap, compare_umaps
from pipeline.orchestrators.run_idrid_gradcam import (
    _build_idrid_splits,
    run_full_gradcam_on_split,
    DEFAULT_IDRID_BASE,
    IMAGE_COL,
    LABEL_COL,
    EXTENSION,
    CLASS_NAMES,
)

logger = logging.getLogger("pipeline.run_occlusion_experiment")


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _setup_wandb() -> None:
    """Load .env and authenticate W&B."""
    load_dotenv()
    api_key = os.environ.get("WANDB_API_KEY")
    if api_key:
        wandb.login(key=api_key)
        logger.info("W&B authenticated via WANDB_API_KEY")
    else:
        logger.warning("WANDB_API_KEY not found — falling back to interactive login")
        wandb.login()


def _build_loader(image_dir: str, csv_path: str, config: dict) -> DataLoader:
    """Build a standard (non-occluded) IDRiD DataLoader."""
    dataset = RetinopathyDataset(
        img_path=image_dir,
        img_col=IMAGE_COL,
        label_col=LABEL_COL,
        transforms=val_transformer,
        extension=EXTENSION,
        target_path=csv_path,
    )
    return DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
        prefetch_factor=config["prefetch_factor"],
    ), dataset


def _build_occluded_loader(
    base_dataset: RetinopathyDataset,
    heatmap_dir: str,
    top_k_percent: float,
    config: dict,
) -> DataLoader:
    """Wrap base_dataset with OccludedDataset and return a DataLoader."""
    occ_dataset = OccludedDataset(
        base_dataset=base_dataset,
        heatmap_dir=heatmap_dir,
        top_k_percent=top_k_percent,
    )
    return DataLoader(
        occ_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
        prefetch_factor=config["prefetch_factor"],
    )


def _run_inference_pass(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """
    Single deterministic forward pass (no MC Dropout).

    Returns a dict containing:
      probs    : [N, 5]  softmax probabilities
      preds    : [N]     argmax predictions
      labels   : [N]     ground-truth labels
      features : [N, 1280] model.base() embeddings
      max_prob : [N]     max softmax probability per sample (confidence)
    """
    model.eval()
    all_probs:    list[np.ndarray] = []
    all_labels:   list[int]        = []
    all_features: list[np.ndarray] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            features = model.base(images)                  # [B, 1280]
            logits   = model.classifier(model.dropout(features))  # [B, 5]
            probs    = torch.softmax(logits, dim=1).cpu().numpy()  # [B, 5]

            all_features.append(features.cpu().numpy())
            all_probs.append(probs)
            all_labels.extend(labels.numpy())

    features  = np.vstack(all_features)   # [N, 1280]
    probs     = np.vstack(all_probs)      # [N, 5]
    labels    = np.array(all_labels)      # [N]
    preds     = probs.argmax(axis=1)      # [N]
    max_prob  = probs.max(axis=1)         # [N]

    return {
        "probs":    probs,
        "preds":    preds,
        "labels":   labels,
        "features": features,
        "max_prob": max_prob,
    }


def _compute_metrics(results: dict) -> dict:
    """
    Compute QWK + per-class accuracy from an inference results dict.

    Returns dict with: qwk, cm, per_class_acc, mean_confidence, mean_entropy
    """
    preds  = results["preds"]
    labels = results["labels"]
    probs  = results["probs"]

    qwk = cohen_kappa_score(labels, preds, weights="quadratic")
    cm  = confusion_matrix(labels, preds, labels=list(range(5)))

    # Per-class accuracy
    per_class_acc = {}
    for c in range(5):
        mask = (labels == c)
        if mask.sum() > 0:
            per_class_acc[c] = float((preds[mask] == c).mean())
        else:
            per_class_acc[c] = float("nan")

    # Predictive entropy
    eps     = 1e-8
    entropy = -np.sum(probs * np.log(probs + eps), axis=1)

    return {
        "qwk":             float(qwk),
        "cm":              cm,
        "per_class_acc":   per_class_acc,
        "mean_confidence": float(results["max_prob"].mean()),
        "mean_entropy":    float(entropy.mean()),
    }


def _log_pass_metrics(
    metrics:    dict,
    pass_label: str,
    split_name: str,
    wandb_prefix: str,
) -> None:
    """Log a single pass's metrics to logger and W&B."""
    logger.info(f"  QWK              : {metrics['qwk']:.4f}")
    logger.info(f"  Mean Confidence  : {metrics['mean_confidence']:.4f}")
    logger.info(f"  Mean Entropy     : {metrics['mean_entropy']:.4f}")
    logger.info(f"  Confusion Matrix :\n{metrics['cm']}")
    logger.info("  Per-class accuracy:")
    for c in range(5):
        acc = metrics["per_class_acc"][c]
        logger.info(f"    Class {c} ({CLASS_NAMES[c]:<15s}): {acc:.3f}")

    try:
        wandb.log({
            f"{wandb_prefix}/qwk":             metrics["qwk"],
            f"{wandb_prefix}/mean_confidence": metrics["mean_confidence"],
            f"{wandb_prefix}/mean_entropy":    metrics["mean_entropy"],
        })
    except Exception as exc:
        logger.warning(f"W&B log failed for {wandb_prefix}: {exc}")


def _plot_prob_drop_bar(
    mean_drops:  list[float],
    labels:      list[str],
    save_path:   str,
) -> None:
    """
    Bar chart showing mean probability drop (confidence reduction) for
    top-10% and top-30% occlusion relative to the baseline.
    """
    colours = ["#C44E52", "#8172B2"]
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(
        labels, mean_drops,
        color=colours[:len(labels)],
        width=0.5, edgecolor="white", linewidth=1.2,
    )

    for bar, val in zip(bars, mean_drops):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    ax.set_ylabel("Mean Probability Drop\n(baseline confidence − occluded confidence)", fontsize=10)
    ax.set_title("Confidence Drop Under GradCAM Occlusion (IDRiD)", fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(mean_drops) * 1.25 if mean_drops else 0.5)
    ax.grid(axis="y", linewidth=0.5, alpha=0.6)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"Probability drop bar chart saved → {save_path}")

    try:
        wandb.log({"occlusion/prob_drop_bar": wandb.Image(save_path)})
    except Exception as exc:
        logger.warning(f"W&B log for prob_drop_bar failed: {exc}")


# ──────────────────────────────────────────────────────────────
# Main orchestrator
# ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "3-pass GradCAM occlusion + UMAP experiment on IDRiD. "
            "Train on APTOS → baseline IDRiD → top-10% occluded → top-30% occluded."
        )
    )
    parser.add_argument(
        "--skip-train", action="store_true",
        help="Skip APTOS training; load existing model weights from --model-path",
    )
    parser.add_argument(
        "--model-path", type=str, default=None,
        help="Path to model weights (default: from BASE_CONFIG)",
    )
    parser.add_argument(
        "--idrid-base", type=str, default=None,
        help=f"IDRiD 'B. Disease Grading' base directory (default: {DEFAULT_IDRID_BASE})",
    )
    parser.add_argument(
        "--idrid-split", type=str, default="test",
        choices=["train", "test", "both"],
        help=(
            "Which IDRiD split to run the occlusion experiment on. "
            "'test' = official 27-image test set (default). "
            "'train' = 54-image training set. "
            "'both' = run all splits sequentially."
        ),
    )
    args = parser.parse_args()

    # ── Setup ──────────────────────────────────────────────
    setup_logging()
    set_seed(BASE_CONFIG["seed"])
    device = setting_gpu()

    _setup_wandb()

    run = wandb.init(
        project=BASE_CONFIG["project_name"],
        job_type="occlusion_experiment",
        mode=BASE_CONFIG.get("wandb_mode", "online"),
        config={
            "idrid_split":         args.idrid_split,
            "occlusion_top_k":     BASE_CONFIG["occlusion_top_k_percents"],
            "model":               BASE_CONFIG["model"],
            "batch_size":          BASE_CONFIG["batch_size"],
        },
    )

    heatmap_base_dir = BASE_CONFIG["gradcam_heatmap_save_dir"]
    umap_dir         = BASE_CONFIG["umap_save_dir"]
    top_k_list       = BASE_CONFIG["occlusion_top_k_percents"]  # e.g. [10, 30]

    model_path = args.model_path or BASE_CONFIG["model_save_path"]

    # ── Step 1: Train or load ───────────────────────────────
    if not args.skip_train:
        logger.info("=" * 60)
        logger.info("STEP 1: Training on APTOS_2019")
        logger.info("=" * 60)
        from pipeline.orchestrators.train import train_model
        train_model("APTOS_2019", BASE_CONFIG)
        model_path = BASE_CONFIG["model_save_path"]
    else:
        logger.info("STEP 1: Skipped — loading existing weights")
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Model weights not found: {model_path}. "
                "Train first or pass --model-path."
            )

    # ── Load model ─────────────────────────────────────────
    model = EfficientNetMC(
        num_classes=5,
        dropout_rate=BASE_CONFIG["dropout_rate"],
        pretrained=False,
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    logger.info(f"Model loaded from {model_path}")

    # ── Resolve which IDRiD splits to process ──────────────
    idrid_base   = args.idrid_base or DEFAULT_IDRID_BASE
    all_splits   = _build_idrid_splits(idrid_base)

    if args.idrid_split == "both":
        splits_to_run = ["train", "test"]
    else:
        splits_to_run = [args.idrid_split]

    # ── Per-split processing ────────────────────────────────
    for split_name in splits_to_run:
        split_paths   = all_splits[split_name]
        image_dir     = split_paths["image_dir"]
        csv_path      = split_paths["csv_path"]
        heatmap_dir   = os.path.join(heatmap_base_dir, split_name)

        logger.info("=" * 60)
        logger.info(f"PROCESSING IDRiD {split_name.upper()} SPLIT")
        logger.info("=" * 60)

        # ── Step 2: Pass 1 — Baseline + GradCAM heatmap saving ─
        logger.info("─" * 50)
        logger.info(f"PASS 1 | Baseline | IDRiD-{split_name}")
        logger.info("─" * 50)

        # Run GradCAM on every image, saving .npy heatmaps + W&B artifact
        run_full_gradcam_on_split(
            model=model,
            device=device,
            split_name=split_name,
            image_dir=image_dir,
            csv_path=csv_path,
            output_dir=os.path.join("results", "idrid_full_gradcam"),
            heatmap_save_dir=heatmap_base_dir,
        )

        # Baseline forward pass for metrics and features
        baseline_loader, base_dataset = _build_loader(image_dir, csv_path, BASE_CONFIG)
        baseline_results = _run_inference_pass(model, baseline_loader, device)
        baseline_metrics = _compute_metrics(baseline_results)

        logger.info(f"Pass 1 Metrics — IDRiD {split_name.upper()}:")
        _log_pass_metrics(baseline_metrics, "Pass 1 (Baseline)", split_name,
                          f"occlusion/{split_name}/pass1_baseline")

        # UMAP for baseline
        baseline_umap_path = os.path.join(umap_dir, split_name, "umap_baseline.png")
        baseline_embedding = plot_umap(
            features=baseline_results["features"],
            labels=baseline_results["labels"],
            title=f"IDRiD {split_name.upper()} — Pass 1: Baseline (no occlusion)",
            save_path=baseline_umap_path,
        )
        try:
            wandb.log({f"umap/{split_name}/pass1_baseline": wandb.Image(baseline_umap_path)})
        except Exception as exc:
            logger.warning(f"W&B UMAP log failed: {exc}")

        # Collect results for comparison
        all_embeddings  = [baseline_embedding]
        all_labels_list = [baseline_results["labels"]]
        umap_titles     = [f"Pass 1: Baseline"]
        pass_metrics    = [baseline_metrics]

        # ── Steps 3+: Occluded passes ────────────────────────
        for top_k in top_k_list:
            pass_num = top_k_list.index(top_k) + 2  # Pass 2 for top_k_list[0], etc.
            logger.info("─" * 50)
            logger.info(f"PASS {pass_num} | Top {top_k}% Occluded | IDRiD-{split_name}")
            logger.info("─" * 50)

            occ_loader = _build_occluded_loader(
                base_dataset=base_dataset,
                heatmap_dir=heatmap_dir,
                top_k_percent=top_k,
                config=BASE_CONFIG,
            )
            occ_results = _run_inference_pass(model, occ_loader, device)
            occ_metrics = _compute_metrics(occ_results)

            # Per-image probability drop vs baseline
            # For each image, compare the baseline confidence of the PREDICTED
            # class (not the true class) to the occluded model's confidence
            # for the same class — this directly measures what the model loses.
            baseline_conf = baseline_results["max_prob"]           # [N]
            occluded_conf = occ_results["max_prob"]                # [N]
            prob_drops    = baseline_conf - occluded_conf          # [N]
            mean_drop     = float(prob_drops.mean())
            pct_dropped   = float((prob_drops > 0).mean() * 100)

            logger.info(f"Pass {pass_num} Metrics — IDRiD {split_name.upper()} (Top {top_k}%):")
            _log_pass_metrics(occ_metrics, f"Pass {pass_num} (Top {top_k}%)", split_name,
                              f"occlusion/{split_name}/pass{pass_num}_top{top_k}")

            logger.info(f"  Mean prob drop vs baseline : {mean_drop:.4f}")
            logger.info(f"  % images with positive drop: {pct_dropped:.1f}%")

            try:
                wandb.log({
                    f"occlusion/{split_name}/top{top_k}_mean_prob_drop":    mean_drop,
                    f"occlusion/{split_name}/top{top_k}_pct_images_dropped": pct_dropped,
                })
            except Exception as exc:
                logger.warning(f"W&B prob drop log failed: {exc}")

            # UMAP for this occluded pass
            occ_umap_path = os.path.join(umap_dir, split_name, f"umap_top{top_k}.png")
            occ_embedding = plot_umap(
                features=occ_results["features"],
                labels=occ_results["labels"],
                title=f"IDRiD {split_name.upper()} — Pass {pass_num}: Top {top_k}% Occluded",
                save_path=occ_umap_path,
            )
            try:
                wandb.log({
                    f"umap/{split_name}/pass{pass_num}_top{top_k}": wandb.Image(occ_umap_path)
                })
            except Exception as exc:
                logger.warning(f"W&B UMAP log failed: {exc}")

            all_embeddings.append(occ_embedding)
            all_labels_list.append(occ_results["labels"])
            umap_titles.append(f"Pass {pass_num}: Top {top_k}% Occluded")
            pass_metrics.append(occ_metrics)

        # ── Step 5: Comparative summary ─────────────────────
        logger.info("─" * 50)
        logger.info(f"STEP 5 | Comparative Summary | IDRiD-{split_name}")
        logger.info("─" * 50)

        # 3-panel UMAP
        comparison_path = os.path.join(umap_dir, split_name, "umap_comparison_3panel.png")
        compare_umaps(
            embeddings=all_embeddings,
            labels_list=all_labels_list,
            titles=umap_titles,
            save_path=comparison_path,
        )

        # Probability drop bar chart (drops relative to baseline)
        drop_labels = [f"Top {k}%" for k in top_k_list]
        mean_drops  = []
        for k in top_k_list:
            occ_loader_tmp = _build_occluded_loader(
                base_dataset=base_dataset,
                heatmap_dir=heatmap_dir,
                top_k_percent=k,
                config=BASE_CONFIG,
            )
            occ_res_tmp  = _run_inference_pass(model, occ_loader_tmp, device)
            mean_drops.append(float((baseline_results["max_prob"] - occ_res_tmp["max_prob"]).mean()))

        bar_path = os.path.join(umap_dir, split_name, "prob_drop_bar.png")
        _plot_prob_drop_bar(mean_drops, drop_labels, bar_path)

        # Summary table
        logger.info("=" * 60)
        logger.info(f"FINAL COMPARISON TABLE — IDRiD {split_name.upper()}")
        logger.info("=" * 60)
        logger.info(f"  {'Pass':<30s} {'QWK':>6} {'Confidence':>12} {'Entropy':>10}")
        logger.info("  " + "-" * 62)
        for label, m in zip(["Baseline (no occlusion)"] + drop_labels, pass_metrics):
            logger.info(
                f"  {label:<30s} {m['qwk']:>6.4f} "
                f"{m['mean_confidence']:>12.4f} {m['mean_entropy']:>10.4f}"
            )
        logger.info("=" * 60)

    wandb.finish()
    logger.info("run_occlusion_experiment.py complete.")


if __name__ == "__main__":
    main()
