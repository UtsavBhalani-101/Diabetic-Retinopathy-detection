# pipeline/dann/preprocess_clahe.py
# ============================================================
# Offline CLAHE preprocessing — run this ONCE before training.
#
# Same workflow as before:
#   1. Run this script once → CLAHE-processed images saved to disk
#   2. Train normally — RetinopathyDataset auto-detects the preprocessed
#      dir and reads from it, skipping all CLAHE at training time
#
# Change from previous version:
#   OLD: OpenCV  cv2.CLAHE, CPU, one image at a time
#   NEW: Kornia  equalize_clahe, GPU tensor op, per-image [1,C,H,W] batch
#
# Why GPU even for single images?
#   A CUDA kernel for a 2K retinal image (~2000×2000×3 = 12M pixels) takes
#   ~3–8 ms on a T4/P100. The same image takes ~80–200 ms on CPU with OpenCV.
#   That's a 20–50× speedup per image, and we have ~46k images across all datasets.
#
# Runtime estimate (Kaggle P100/T4):
#   APTOS_2019       (~3,662 images) : ~30–60 sec
#   Messidor-Grp1/2/3 (~400 each)   : ~5–10 sec each
#   DDR-China        (~6,266 images) : ~1–2 min
#   EyePACS-Resized  (~35,126 images): ~5–8 min   ← was 15–20 min on CPU
#   Total            (~46k images)   : ~8–12 min  (was ~25 min on CPU)
#
# After completion:
#   - Preprocessed images are in /kaggle/working/clahe_preprocessed/<ds_name>/
#   - RetinopathyDataset picks them up automatically via clahe_image_path
#   - (Optional) Save the output folder as a Kaggle Dataset so future
#     sessions can mount it directly, skipping this step entirely.
# ============================================================

import os
import glob
import logging
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import kornia.enhance

from pipeline.setup.utils import DATASET_REGISTRY

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Output root ───────────────────────────────────────────────────────────────
OUTPUT_BASE = "/kaggle/working/clahe_preprocessed"

# ── CLAHE parameters ─────────────────────────────────────────────────────────
# NOTE: "visually equivalent to OpenCV clipLimit=2.0" is NOT independently
# verified — public reports (Kornia GitHub discussions) show Kornia's
# equalize_clahe does not numerically match OpenCV's CLAHE even with scaled
# parameters, and Kornia's own docs only claim its LUT approach "uses the
# same approach as OpenCV," with an explicit note that this may change
# between versions. Spot-check a handful of outputs against the old
# CLAHEPreprocess (OpenCV) results before trusting this preprocessing
# is equivalent to what your baseline pipeline used — this matters for
# any downstream claim that DANN's input distribution is comparable to
# the APTOS-trained baseline's.
CLAHE_CLIP_LIMIT  = 40.0
CLAHE_TILE_GRID   = (8, 8)

# JPEG quality for saving (ignored for .png)
JPEG_SAVE_QUALITY = 95

# ── Which datasets to preprocess ─────────────────────────────────────────────
DATASETS_TO_PROCESS = [
    "APTOS_2019",
    "EyePACS-Resized",
    "Messidor-Grp1",
    "Messidor-Grp2",
    "Messidor-Grp3",
    "DDR-China",
    "IDRiD",         # target domain — needed for domain-adversarial training
]

# Supported extensions to scan when registry extension is "" (unspecified)
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


# ── GPU CLAHE helper ─────────────────────────────────────────────────────────

def apply_clahe_gpu(
    img_path: str,
    device: torch.device,
    clip_limit: float = CLAHE_CLIP_LIMIT,
    grid_size: tuple  = CLAHE_TILE_GRID,
) -> np.ndarray:
    """
    Load one image, apply CLAHE on GPU using Kornia, return uint8 RGB numpy array.

    Steps
    -----
    1. PIL.Image.open → RGB numpy  [H, W, 3]  uint8
    2. numpy → float32 tensor  [1, 3, H, W]  range [0.0, 1.0]  → GPU
    3. kornia.enhance.equalize_clahe  (CUDA kernel, parallel over all pixels)
    4. GPU tensor → CPU numpy  [H, W, 3]  uint8
    """
    # 1. Load
    img_np = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0

    # 2. → GPU tensor [1, C, H, W]
    img_tensor = (
        torch.from_numpy(img_np)        # [H, W, 3]
        .permute(2, 0, 1)               # [3, H, W]
        .unsqueeze(0)                   # [1, 3, H, W]
        .to(device)
    )

    # 3. Kornia CLAHE on GPU
    # slow_and_differentiable=False → fast non-differentiable CUDA path
    img_clahe = kornia.enhance.equalize_clahe(
        img_tensor,
        clip_limit=clip_limit,
        grid_size=grid_size,
        slow_and_differentiable=False,
    )

    # 4. → CPU numpy uint8 [H, W, 3]
    img_out = (
        img_clahe[0]                    # [3, H, W]
        .permute(1, 2, 0)               # [H, W, 3]
        .cpu()
        .numpy()
    )
    return (img_out * 255.0).clip(0, 255).astype(np.uint8)


# ── Per-dataset processing ───────────────────────────────────────────────────

def preprocess_dataset(ds_name: str, device: torch.device) -> dict:
    """
    Process all images for one dataset and save them to OUTPUT_BASE/<ds_name>/.

    Skips files that already exist so the script can be safely re-run
    after an interrupt without re-doing completed work.

    Returns
    -------
    dict with keys: processed, skipped, failed
    """
    reg       = DATASET_REGISTRY[ds_name]
    src_dir   = reg["image_path"]
    out_dir   = os.path.join(OUTPUT_BASE, ds_name)
    extension = reg.get("extension", "")

    os.makedirs(out_dir, exist_ok=True)

    # ── Collect source image paths ────────────────────────────────────────────
    if extension:
        files  = glob.glob(os.path.join(src_dir, f"*{extension}"))
        files += glob.glob(os.path.join(src_dir, f"*{extension.upper()}"))
    else:
        files = []
        for ext in SUPPORTED_EXTENSIONS:
            files += glob.glob(os.path.join(src_dir, f"*{ext}"))
            files += glob.glob(os.path.join(src_dir, f"*{ext.upper()}"))

    files = list(set(files))   # deduplicate upper/lower variants

    if not files:
        logger.warning(f"[{ds_name}] No images found in: {src_dir}")
        return {"processed": 0, "skipped": 0, "failed": 0}

    logger.info(f"[{ds_name}] {len(files):,} images  →  {out_dir}")

    processed = skipped = failed = 0

    for img_path in tqdm(files, desc=f"  {ds_name}", unit="img"):
        # Keep the same extension so DataLoader can find the file by stem
        src_ext  = Path(img_path).suffix.lower()
        out_name = Path(img_path).stem + src_ext
        out_path = os.path.join(out_dir, out_name)

        # Skip already-processed files (safe to interrupt and resume)
        if os.path.exists(out_path):
            skipped += 1
            continue

        try:
            img_rgb = apply_clahe_gpu(img_path, device)

            # Save using PIL — handles all extensions uniformly
            out_img = Image.fromarray(img_rgb)

            if src_ext in {".jpg", ".jpeg"}:
                out_img.save(out_path, quality=JPEG_SAVE_QUALITY, subsampling=0)
            elif src_ext == ".png":
                out_img.save(out_path, compress_level=3)   # fast compression
            else:
                out_img.save(out_path)

            processed += 1

        except Exception as exc:
            logger.error(f"    [ERROR] {img_path}: {exc}")
            failed += 1

    logger.info(
        f"[{ds_name}] done  |  processed={processed:,}  "
        f"skipped(existing)={skipped:,}  failed={failed:,}"
    )
    return {"processed": processed, "skipped": skipped, "failed": failed}


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("=" * 60)
    logger.info("Offline CLAHE Preprocessing  [Kornia / GPU]")
    logger.info(f"Device      : {device}" + (
        f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else
        "  ⚠ No GPU found — falling back to CPU (will be slower)"
    ))
    logger.info(f"Output root : {OUTPUT_BASE}")
    logger.info(f"Datasets    : {DATASETS_TO_PROCESS}")
    logger.info(f"clip_limit  : {CLAHE_CLIP_LIMIT}  |  grid_size : {CLAHE_TILE_GRID}")
    logger.info("=" * 60)

    if device.type == "cpu":
        logger.warning(
            "Running on CPU. Kornia CLAHE is still correct on CPU but slower. "
            "Enable GPU in the Kaggle notebook (Settings → Accelerator → GPU) "
            "for the full speedup."
        )

    totals = {"processed": 0, "skipped": 0, "failed": 0}

    for ds_name in DATASETS_TO_PROCESS:
        if ds_name not in DATASET_REGISTRY:
            logger.warning(f"[{ds_name}] Not found in DATASET_REGISTRY — skipping")
            continue
        counts = preprocess_dataset(ds_name, device)
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
    logger.info(f"  1. Spot-check a few output images in {OUTPUT_BASE}/<ds_name>/")
    logger.info("  2. Training will auto-detect and use these images — no other changes needed.")
    logger.info("  3. (Optional) Save the output as a Kaggle Dataset so future sessions")
    logger.info("     can mount it at /kaggle/input/... and skip this step entirely.")
    logger.info("     To do that, update _CLAHE_ROOT in pipeline/setup/utils.py to point")
    logger.info("     to the mounted input path.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
