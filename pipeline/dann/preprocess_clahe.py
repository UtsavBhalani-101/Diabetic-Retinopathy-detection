# pipeline/dann/preprocess_clahe.py
# ============================================================
# Offline CLAHE preprocessing — run this ONCE before training.
#
# Reads raw images from DATASET_REGISTRY paths, applies CLAHE,
# and saves the results to /kaggle/working/clahe_preprocessed/<ds_name>/.
#
# After this script completes:
#   - pipeline/data/dataset.py auto-detects the preprocessed dirs
#     and skips all on-the-fly CLAHE during training.
#
# Runtime estimate (Kaggle CPU, 2 cores):
#   APTOS_2019       (~3,662 images) : ~1–2 min
#   Messidor-Grp1    (~  400 images) : ~15 sec
#   Messidor-Grp2    (~  395 images) : ~15 sec
#   Messidor-Grp3    (~  400 images) : ~15 sec
#   DDR-China        (~6,266 images) : ~2–3 min
#   EyePACS-Resized  (~35,126 images): ~15–20 min  ← largest
#   Total            (~46k images)   : ~20–25 min once
#                                    → saves hours per training run
# ============================================================

import os
import glob
import logging
from pathlib import Path

import cv2
from tqdm import tqdm

from pipeline.setup.utils import DATASET_REGISTRY

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Output root ───────────────────────────────────────────────────────────────
# /kaggle/working is writable and persists for the session.
# After preprocessing, save it as a Kaggle Dataset so future sessions can
# mount it directly and skip this step entirely.
OUTPUT_BASE = "/kaggle/working/clahe_preprocessed"

# ── CLAHE parameters — must match the old CLAHEPreprocess settings ────────────
CLAHE_CLIP_LIMIT  = 2.0
CLAHE_TILE_GRID   = (8, 8)
JPEG_SAVE_QUALITY = 95      # for .jpg/.jpeg output

# ── Which datasets to preprocess ─────────────────────────────────────────────
DATASETS_TO_PROCESS = [
    "APTOS_2019",
    "EyePACS-Resized",
    "Messidor-Grp1",
    "Messidor-Grp2",
    "Messidor-Grp3",
    "DDR-China",
]

# Supported raw image extensions to glob when registry extension is empty ""
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def apply_clahe(img_bgr, clahe_obj):
    """
    Apply CLAHE to the L channel of LAB colour space.

    Parameters
    ----------
    img_bgr  : np.ndarray  BGR image as returned by cv2.imread
    clahe_obj: pre-created cv2.CLAHE (reused across all images — not recreated)

    Returns
    -------
    np.ndarray  BGR image, CLAHE-applied, ready for cv2.imwrite
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    lab     = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l2      = clahe_obj.apply(l)
    lab2    = cv2.merge((l2, a, b))
    img2    = cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)
    return cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)


def preprocess_dataset(ds_name: str, clahe_obj) -> dict:
    """
    Process all images for one dataset and save them to disk.

    Resumes automatically if interrupted — files that already exist
    in the output directory are skipped without re-processing.

    Returns
    -------
    dict with keys: processed, skipped, failed
    """
    reg       = DATASET_REGISTRY[ds_name]
    src_dir   = reg["image_path"]
    out_dir   = os.path.join(OUTPUT_BASE, ds_name)
    extension = reg.get("extension", "")    # e.g. ".png", ".jpeg", ""

    os.makedirs(out_dir, exist_ok=True)

    # ── Collect source files ──────────────────────────────────────────────────
    if extension:
        # Registry specifies a fixed extension — scan exactly that
        files  = glob.glob(os.path.join(src_dir, f"*{extension}"))
        files += glob.glob(os.path.join(src_dir, f"*{extension.upper()}"))
    else:
        # No extension in registry — scan all supported image types
        files = []
        for ext in SUPPORTED_EXTENSIONS:
            files += glob.glob(os.path.join(src_dir, f"*{ext}"))
            files += glob.glob(os.path.join(src_dir, f"*{ext.upper()}"))

    files = list(set(files))   # remove duplicates from upper/lower variants

    if not files:
        logger.warning(f"[{ds_name}] No images found in: {src_dir}")
        return {"processed": 0, "skipped": 0, "failed": 0}

    logger.info(f"[{ds_name}] {len(files):,} images  →  {out_dir}")

    processed = skipped = failed = 0

    for img_path in tqdm(files, desc=f"  {ds_name}", unit="img"):
        # Keep the same extension as the source file so the DataLoader can
        # find it using the same filename stem it reads from the CSV.
        src_ext  = Path(img_path).suffix.lower()
        out_name = Path(img_path).stem + src_ext

        if src_ext in {".jpg", ".jpeg"}:
            save_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_SAVE_QUALITY]
        elif src_ext == ".png":
            save_params = [cv2.IMWRITE_PNG_COMPRESSION, 3]   # fast compression
        else:
            save_params = []   # default params for tif/bmp

        out_path = os.path.join(out_dir, out_name)

        # Skip already-processed files so re-runs are fast
        if os.path.exists(out_path):
            skipped += 1
            continue

        img = cv2.imread(img_path)
        if img is None:
            logger.warning(f"    [SKIP] Cannot read: {img_path}")
            failed += 1
            continue

        try:
            img_clahe = apply_clahe(img, clahe_obj)
            cv2.imwrite(out_path, img_clahe, save_params)
            processed += 1
        except Exception as exc:
            logger.error(f"    [ERROR] {img_path}: {exc}")
            failed += 1

    logger.info(
        f"[{ds_name}] processed={processed:,}  "
        f"skipped(existing)={skipped:,}  failed={failed:,}"
    )
    return {"processed": processed, "skipped": skipped, "failed": failed}


def main():
    logger.info("=" * 60)
    logger.info("Offline CLAHE Preprocessing")
    logger.info(f"Output root : {OUTPUT_BASE}")
    logger.info(f"Datasets    : {DATASETS_TO_PROCESS}")
    logger.info("=" * 60)

    # ── Create the CLAHE object ONCE — reused across every image ─────────────
    clahe_obj = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_GRID,
    )

    totals = {"processed": 0, "skipped": 0, "failed": 0}

    for ds_name in DATASETS_TO_PROCESS:
        if ds_name not in DATASET_REGISTRY:
            logger.warning(f"[{ds_name}] Not found in DATASET_REGISTRY — skipping")
            continue
        counts = preprocess_dataset(ds_name, clahe_obj)
        for k in totals:
            totals[k] += counts[k]

    logger.info("=" * 60)
    logger.info("Preprocessing complete!")
    logger.info(
        f"  Total processed : {totals['processed']:,}\n"
        f"  Total skipped   : {totals['skipped']:,}  (already existed)\n"
        f"  Total failed    : {totals['failed']:,}"
    )
    logger.info("")
    logger.info("NEXT STEPS:")
    logger.info(f"  1. Spot-check a few images in {OUTPUT_BASE}/<ds_name>/")
    logger.info("  2. Training will now auto-detect and use these dirs —")
    logger.info("     no further changes needed.")
    logger.info("  3. (Optional) Save the output as a Kaggle Dataset so")
    logger.info("     future sessions can mount it directly (skip this step).")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
