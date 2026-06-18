# pipeline/orchestrators/run_idrid_gradcam.py
# ============================================================
# Standalone orchestrator: Train on APTOS → Full GradCAM on IDRiD
#
# For every image in the IDRiD dataset (train + test splits),
# generates a GradCAM++ heatmap overlay and organizes results
# into success/ and failure/ folders based on prediction accuracy.
#
# Usage (module form — run from project root):
#   python -m pipeline.orchestrators.run_idrid_gradcam
#   python -m pipeline.orchestrators.run_idrid_gradcam --skip-train
#   python -m pipeline.orchestrators.run_idrid_gradcam --skip-train --model-path artifacts/weights/my_model.pth
# ============================================================

import argparse
import csv
import logging
import os
import time

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from pipeline.setup.config import BASE_CONFIG, setup_logging
from pipeline.data.dataset import RetinopathyDataset, val_transformer, CLAHEPreprocess
from pipeline.training_loop_setup.model import EfficientNetMC

logger = logging.getLogger("pipeline.run_idrid_gradcam")


# ----------------------------------------------------------------
# Constants
# ----------------------------------------------------------------

CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]

# Hardcoded local IDRiD paths (relative to project root)
IDRID_BASE = os.path.join("datasets", "IDRiD", "B. Disease Grading")

IDRID_SPLITS = {
    "train": {
        "image_dir": os.path.join(IDRID_BASE, "1. Original Images", "a. Training Set"),
        "csv_path":  os.path.join(IDRID_BASE, "2. Groundtruths", "a. IDRiD_Disease Grading_Training Labels.csv"),
    },
    "test": {
        "image_dir": os.path.join(IDRID_BASE, "1. Original Images", "b. Testing Set"),
        "csv_path":  os.path.join(IDRID_BASE, "2. Groundtruths", "b. IDRiD_Disease Grading_Testing Labels.csv"),
    },
}

# IDRiD column names (from the CSV headers)
IMAGE_COL = "Image name"
LABEL_COL = "Retinopathy grade"
EXTENSION = ".jpg"

OUTPUT_DIR = os.path.join("results", "idrid_full_gradcam")


# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------

def get_rgb_img(img_path: str) -> np.ndarray:
    """
    Load an image, apply CLAHE preprocessing, resize to 224×224,
    and return as float32 numpy array in [0, 1] range.

    This produces the base RGB image that show_cam_on_image() needs
    for the GradCAM overlay. Must match the preprocessing the model
    expects so the heatmap aligns with what the model actually sees.
    """
    img = Image.open(img_path).convert("RGB")
    clahe = CLAHEPreprocess(clip_limit=2.0, tile_grid_size=(8, 8))
    img = clahe(img)
    img = img.resize((224, 224))
    return np.array(img).astype(np.float32) / 255.0


def annotate_image(visualization: np.ndarray,
                   true_label: int, pred_label: int) -> np.ndarray:
    """
    Burn prediction vs actual text onto the top of a GradCAM
    visualization image.

    Draws a black banner at the top with:
      Line 1:  True: X (ClassName) | Pred: Y (ClassName)
      Line 2:  ✓ CORRECT  or  ✗ WRONG (in green/red)

    This makes every saved image self-documenting — you can tell
    at a glance what the model predicted vs the ground truth without
    needing to parse the filename.
    """
    img = Image.fromarray(visualization)
    draw = ImageDraw.Draw(img)

    true_name = CLASS_NAMES[true_label]
    pred_name = CLASS_NAMES[pred_label]
    correct = (true_label == pred_label)

    text_line1 = f"True: {true_label} ({true_name}) | Pred: {pred_label} ({pred_name})"
    text_line2 = "CORRECT" if correct else "WRONG"

    # Try system font, fall back to PIL default
    try:
        font = ImageFont.truetype("arial.ttf", 13)
    except (IOError, OSError):
        font = ImageFont.load_default()

    # Black banner at the top of the image
    banner_height = 36
    draw.rectangle([(0, 0), (img.width, banner_height)], fill=(0, 0, 0))

    # Line 1: prediction info in white
    draw.text((4, 2), text_line1, fill=(255, 255, 255), font=font)

    # Line 2: verdict in green (correct) or red (wrong)
    verdict_color = (0, 255, 0) if correct else (255, 60, 60)
    draw.text((4, 18), text_line2, fill=verdict_color, font=font)

    return np.array(img)


# ----------------------------------------------------------------
# Core: Full GradCAM on a single split
# ----------------------------------------------------------------

def run_full_gradcam_on_split(
    model: torch.nn.Module,
    device: torch.device,
    split_name: str,
    image_dir: str,
    csv_path: str,
    output_dir: str,
) -> list[dict]:
    """
    Run GradCAM++ on EVERY image in a single IDRiD split.

    Two-pass approach:
      Pass 1 (batched):  Fast forward pass with DataLoader to collect
                         all predictions and ground-truth labels.
      Pass 2 (per-image): Individual GradCAM computation, overlay,
                          annotation, and save to success/failure folder.

    This two-pass design is necessary because GradCAM requires
    per-image gradient computation (can't be meaningfully batched),
    but we want the predictions first so we know the folder destination
    before generating the expensive heatmap.

    Returns a list of result dicts for the summary CSV.
    """
    logger.info(f"[IDRiD-{split_name}] Loading dataset: {csv_path}")
    logger.info(f"[IDRiD-{split_name}] Image directory: {image_dir}")

    # Validate paths exist
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    dataset = RetinopathyDataset(
        img_path=image_dir,
        img_col=IMAGE_COL,
        label_col=LABEL_COL,
        transforms=val_transformer,
        extension=EXTENSION,
        target_path=csv_path,
    )

    loader = DataLoader(
        dataset, batch_size=32, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    n_images = len(dataset)
    logger.info(f"[IDRiD-{split_name}] Total images to process: {n_images}")

    # ------------------------------------------------------------------
    # Pass 1: Batch forward pass → collect all predictions
    # ------------------------------------------------------------------
    # We run a standard eval-mode forward pass (no MC Dropout) to get
    # deterministic predictions. MC Dropout is unnecessary here because
    # we only need the argmax prediction for the success/failure split,
    # and GradCAM targets the predicted class anyway.
    # ------------------------------------------------------------------
    logger.info(f"[IDRiD-{split_name}] Pass 1: Batch inference for predictions...")
    all_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    n_correct = int((all_preds == all_labels).sum())
    n_wrong = int((all_preds != all_labels).sum())
    accuracy = n_correct / n_images * 100

    logger.info(
        f"[IDRiD-{split_name}] Predictions: "
        f"{n_correct} correct ({accuracy:.1f}%), {n_wrong} wrong"
    )

    # Per-class breakdown
    for c in range(5):
        mask = (all_labels == c)
        if mask.sum() > 0:
            class_correct = int((all_preds[mask] == c).sum())
            class_total = int(mask.sum())
            logger.info(
                f"  Class {c} ({CLASS_NAMES[c]}): "
                f"{class_correct}/{class_total} correct "
                f"({class_correct / class_total * 100:.1f}%)"
            )

    # ------------------------------------------------------------------
    # Pass 2: Per-image GradCAM → annotate → save to success/failure
    # ------------------------------------------------------------------
    # GradCAM requires gradients w.r.t. a specific target class for each
    # image individually. We target the model's PREDICTED class (not the
    # true class) because we want to see WHERE the model is looking when
    # it makes its decision — this reveals whether it's focusing on
    # clinically relevant regions even when wrong.
    # ------------------------------------------------------------------
    logger.info(f"[IDRiD-{split_name}] Pass 2: Per-image GradCAM generation...")

    # GradCAM++ setup — target the last block of EfficientNet's feature extractor
    target_layers = [model.base.blocks[-1]]
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers)

    # Create output directories
    success_dir = os.path.join(output_dir, split_name, "success")
    failure_dir = os.path.join(output_dir, split_name, "failure")
    os.makedirs(success_dir, exist_ok=True)
    os.makedirs(failure_dir, exist_ok=True)

    results = []
    start_time = time.time()

    for idx in range(n_images):
        row = dataset.df.iloc[idx]
        image_id = str(row[IMAGE_COL])
        true_label = int(all_labels[idx])
        pred_label = int(all_preds[idx])
        correct = (true_label == pred_label)

        img_path = os.path.join(image_dir, f"{image_id}{EXTENSION}")

        # --- RGB base image for GradCAM overlay ---
        rgb_img = get_rgb_img(img_path)

        # --- Input tensor (with full val transforms including normalization) ---
        pil_img = Image.open(img_path).convert("RGB")
        input_tensor = val_transformer(pil_img).unsqueeze(0).to(device)

        # --- GradCAM++ heatmap ---
        # Target the predicted class so we see what the model is attending to
        targets = [ClassifierOutputTarget(pred_label)]
        model.eval()
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]  # [H, W]

        # --- Overlay heatmap on original image ---
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        # --- Burn text annotation onto the image ---
        annotated = annotate_image(visualization, true_label, pred_label)

        # --- Build filename and save ---
        true_name_clean = CLASS_NAMES[true_label].replace(" ", "")
        pred_name_clean = CLASS_NAMES[pred_label].replace(" ", "")

        if correct:
            out_name = f"{image_id}_True{true_label}_Pred{pred_label}_{true_name_clean}.jpg"
            out_path = os.path.join(success_dir, out_name)
        else:
            # Failure filenames show the confusion: TrueClass→PredClass
            out_name = (
                f"{image_id}_True{true_label}_Pred{pred_label}_"
                f"{true_name_clean}_to_{pred_name_clean}.jpg"
            )
            out_path = os.path.join(failure_dir, out_name)

        Image.fromarray(annotated).save(out_path, quality=95)

        # Collect result for summary CSV
        results.append({
            "image_name": image_id,
            "split": split_name,
            "true_label": true_label,
            "pred_label": pred_label,
            "true_class": CLASS_NAMES[true_label],
            "pred_class": CLASS_NAMES[pred_label],
            "correct": correct,
        })

        # Progress logging every 25 images + final
        if (idx + 1) % 25 == 0 or (idx + 1) == n_images:
            elapsed = time.time() - start_time
            rate = (idx + 1) / elapsed
            remaining = (n_images - idx - 1) / rate if rate > 0 else 0
            logger.info(
                f"  [{split_name}] {idx + 1}/{n_images} images done "
                f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)"
            )

    logger.info(
        f"[IDRiD-{split_name}] Complete! "
        f"Successes: {success_dir} ({n_correct} images) | "
        f"Failures: {failure_dir} ({n_wrong} images)"
    )

    return results


# ----------------------------------------------------------------
# Summary CSV
# ----------------------------------------------------------------

def write_summary_csv(all_results: list[dict], output_dir: str) -> str:
    """
    Write a combined summary CSV with one row per image across all splits.

    Columns: image_name, split, true_label, pred_label, true_class,
             pred_class, correct

    This CSV makes it easy to filter/sort results in Excel or pandas
    without having to traverse the folder structure.
    """
    csv_path = os.path.join(output_dir, "summary.csv")
    os.makedirs(output_dir, exist_ok=True)

    fieldnames = [
        "image_name", "split", "true_label", "pred_label",
        "true_class", "pred_class", "correct",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    logger.info(f"Summary CSV saved → {csv_path} ({len(all_results)} rows)")
    return csv_path


# ----------------------------------------------------------------
# Final summary printout
# ----------------------------------------------------------------

def print_final_summary(all_results: list[dict]) -> None:
    """
    Print a comprehensive final summary to the log:
    overall accuracy, per-split accuracy, per-class accuracy,
    and a confusion breakdown.
    """
    total = len(all_results)
    total_correct = sum(1 for r in all_results if r["correct"])

    logger.info("=" * 60)
    logger.info("FINAL SUMMARY — IDRiD Full GradCAM Analysis")
    logger.info("=" * 60)
    logger.info(f"Total images processed: {total}")
    logger.info(f"Overall accuracy: {total_correct}/{total} ({total_correct / total * 100:.1f}%)")
    logger.info("")

    # Per-split
    for split in ["train", "test"]:
        split_results = [r for r in all_results if r["split"] == split]
        if not split_results:
            continue
        split_correct = sum(1 for r in split_results if r["correct"])
        split_total = len(split_results)
        logger.info(
            f"  {split.upper()} split: {split_correct}/{split_total} "
            f"({split_correct / split_total * 100:.1f}%)"
        )

    logger.info("")

    # Per-class
    logger.info("Per-class accuracy:")
    for c in range(5):
        class_results = [r for r in all_results if r["true_label"] == c]
        if not class_results:
            continue
        class_correct = sum(1 for r in class_results if r["correct"])
        class_total = len(class_results)
        logger.info(
            f"  Class {c} ({CLASS_NAMES[c]:>15s}): "
            f"{class_correct}/{class_total} ({class_correct / class_total * 100:.1f}%)"
        )

    logger.info("")

    # Common misclassifications
    misclassifications = [r for r in all_results if not r["correct"]]
    if misclassifications:
        # Count (true → pred) pairs
        confusion_pairs: dict[tuple[int, int], int] = {}
        for r in misclassifications:
            pair = (r["true_label"], r["pred_label"])
            confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1

        sorted_pairs = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)
        logger.info("Top misclassification patterns:")
        for (true_l, pred_l), count in sorted_pairs[:10]:
            logger.info(
                f"  {CLASS_NAMES[true_l]} → {CLASS_NAMES[pred_l]}: "
                f"{count} images"
            )

    logger.info("=" * 60)


# ----------------------------------------------------------------
# Main orchestrator
# ----------------------------------------------------------------

def main() -> None:
    """
    Full pipeline:
      1. (Optional) Train on APTOS_2019
      2. Load trained model
      3. Run GradCAM on every IDRiD image (train + test splits)
      4. Save to success/failure folders with annotations
      5. Generate summary CSV + print final report
    """
    parser = argparse.ArgumentParser(
        description="Train on APTOS → Full GradCAM analysis on IDRiD dataset"
    )
    parser.add_argument(
        "--skip-train", action="store_true",
        help="Skip APTOS training and load existing model weights"
    )
    parser.add_argument(
        "--model-path", type=str, default=None,
        help="Path to model weights (default: from BASE_CONFIG)"
    )
    parser.add_argument(
        "--output-dir", type=str, default=OUTPUT_DIR,
        help=f"Output directory for GradCAM results (default: {OUTPUT_DIR})"
    )
    args = parser.parse_args()

    # ---- Setup ----
    setup_logging()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # ---- Step 1: Train or load model ----
    model_path = args.model_path or BASE_CONFIG["model_save_path"]

    if not args.skip_train:
        logger.info("=" * 60)
        logger.info("STEP 1: Training on APTOS_2019")
        logger.info("=" * 60)

        # Import here to avoid circular imports and unnecessary wandb setup
        # when --skip-train is used
        from pipeline.orchestrators.train import train_model, setup_wandb

        setup_wandb()
        optimal_T, _ = train_model("APTOS_2019", BASE_CONFIG)
        model_path = BASE_CONFIG["model_save_path"]
        logger.info(f"Training complete | optimal_T={optimal_T:.4f}")
    else:
        logger.info("=" * 60)
        logger.info("STEP 1: SKIPPED (--skip-train) — loading existing weights")
        logger.info("=" * 60)
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"Model weights not found at '{model_path}'. "
                f"Train first or provide --model-path."
            )

    # ---- Step 2: Load model ----
    logger.info(f"Loading model from: {model_path}")
    model = EfficientNetMC(
        num_classes=5,
        dropout_rate=BASE_CONFIG["dropout_rate"],
        pretrained=False,  # loading from disk, no ImageNet download needed
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    logger.info("Model loaded and set to eval mode")

    # ---- Step 3: GradCAM on both IDRiD splits ----
    all_results = []

    for split_name, split_paths in IDRID_SPLITS.items():
        logger.info("=" * 60)
        logger.info(f"STEP 3: GradCAM on IDRiD {split_name.upper()} split")
        logger.info("=" * 60)

        split_results = run_full_gradcam_on_split(
            model=model,
            device=device,
            split_name=split_name,
            image_dir=split_paths["image_dir"],
            csv_path=split_paths["csv_path"],
            output_dir=args.output_dir,
        )
        all_results.extend(split_results)

    # ---- Step 4: Summary CSV ----
    write_summary_csv(all_results, args.output_dir)

    # ---- Step 5: Final report ----
    print_final_summary(all_results)

    logger.info(f"All GradCAM results saved to: {args.output_dir}")
    logger.info("Done!")


if __name__ == "__main__":
    main()
