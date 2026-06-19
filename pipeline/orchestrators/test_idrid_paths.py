# pipeline/orchestrators/test_idrid_paths.py
# ============================================================
# Lightweight test: Validate IDRiD dataset paths & loading
#
# Checks CSV files exist, columns are correct, images are readable.
# No model, no GPU, no training needed.
#
# Usage:  python -m pipeline.orchestrators.test_idrid_paths
# ============================================================

import logging
import os
import sys

import pandas as pd
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------
# Hardcoded IDRiD paths (same as run_idrid_gradcam.py)
# ----------------------------------------------------------------
IDRID_BASE = "/kaggle/input/datasets/antiti/idrid-testing-dataset/IDRiD/B. Disease Grading"
SPLITS = {
    "train": {
        "image_dir": f"{IDRID_BASE}/1. Original Images/a. Training Set",
        "csv_path":  f"{IDRID_BASE}/2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv",
    },
    "test": {
        "image_dir": f"{IDRID_BASE}/1. Original Images/b. Testing Set",
        "csv_path":  f"{IDRID_BASE}/2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv",
    },
}

IMAGE_COL = "Image name"
LABEL_COL = "Retinopathy grade"
EXTENSION = ".jpg"
CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]


def check_split(split_name: str, image_dir: str, csv_path: str) -> bool:
    """Validate a single IDRiD split. Returns True if all checks pass."""
    passed = True
    logger.info(f"--- Checking {split_name.upper()} split ---")

    # 1. CSV existence
    if os.path.isfile(csv_path):
        logger.info(f"  ✓ CSV found: {csv_path}")
    else:
        logger.error(f"  ✗ CSV NOT FOUND: {csv_path}")
        parent = os.path.dirname(csv_path)
        if os.path.isdir(parent):
            logger.error(f"    Files in {parent}: {os.listdir(parent)}")
        else:
            logger.error(f"    Parent dir also missing: {parent}")
        return False

    # 2. CSV columns & label distribution
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"  ✓ CSV loaded: {len(df)} rows, columns = {list(df.columns)}")

        if IMAGE_COL not in df.columns:
            logger.error(f"  ✗ Missing column '{IMAGE_COL}'. Available: {list(df.columns)}")
            passed = False
        if LABEL_COL not in df.columns:
            logger.error(f"  ✗ Missing column '{LABEL_COL}'. Available: {list(df.columns)}")
            passed = False
        if not passed:
            return False

        label_counts = df[LABEL_COL].value_counts().sort_index()
        logger.info(f"  ✓ Label distribution:")
        for label, count in label_counts.items():
            name = CLASS_NAMES[label] if 0 <= label < len(CLASS_NAMES) else "UNKNOWN"
            logger.info(f"      Class {label} ({name}): {count} images")
    except Exception as e:
        logger.error(f"  ✗ Failed to read CSV: {e}")
        return False

    # 3. Image directory
    if os.path.isdir(image_dir):
        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        logger.info(f"  ✓ Image dir found: {image_dir} ({len(image_files)} image files)")
    else:
        logger.error(f"  ✗ Image dir NOT FOUND: {image_dir}")
        return False

    # 4. Sample 5 images — check they're loadable
    sample_df = df.head(5)
    ok, fail = 0, 0
    for _, row in sample_df.iterrows():
        img_path = os.path.join(image_dir, f"{row[IMAGE_COL]}{EXTENSION}")
        if os.path.isfile(img_path):
            try:
                img = Image.open(img_path)
                img.verify()
                ok += 1
            except Exception as e:
                logger.warning(f"  ⚠ Image corrupt: {img_path} ({e})")
                fail += 1
        else:
            logger.warning(f"  ⚠ Image missing: {img_path}")
            fail += 1

    if fail == 0:
        logger.info(f"  ✓ Sample images: {ok}/{len(sample_df)} OK")
    else:
        logger.warning(f"  ⚠ Sample images: {ok} OK, {fail} FAILED")
        passed = False

    return passed


def main() -> None:
    logger.info("=" * 60)
    logger.info("IDRiD Dataset Path Validation")
    logger.info("=" * 60)
    logger.info(f"CWD: {os.getcwd()}")
    logger.info(f"IDRiD base: {IDRID_BASE}")
    logger.info(f"Base exists: {os.path.isdir(IDRID_BASE)}")
    logger.info("")

    all_passed = True
    for split_name, paths in SPLITS.items():
        if not check_split(split_name, paths["image_dir"], paths["csv_path"]):
            all_passed = False
        logger.info("")

    logger.info("=" * 60)
    if all_passed:
        logger.info("RESULT: ✓ All checks PASSED — IDRiD dataset is ready!")
    else:
        logger.error("RESULT: ✗ Some checks FAILED — see errors above")
        sys.exit(1)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
