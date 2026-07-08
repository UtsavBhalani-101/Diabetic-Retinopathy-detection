# pipeline/config.py
# ============================================================
# Central configuration:
#   - setup_logging()   : configures console + file handlers once
#   - BASE_CONFIG       : all hyperparameters consumed by train/test
#   - DATASET_REGISTRY  : re-exported from root utils.py
#   - Setting device (GPU/CPU) and reproducibility
# ============================================================

import sys
import os
import logging
import datetime
import numpy as np
import torch
import random

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------
# Logging setup — call once at the start of run.py (or any entry point)
# ----------------------------------------------------------------


def setup_logging(log_dir: str = "artifacts/logs", level: int = logging.INFO) -> None:
    """
    Configure the root logger with:
      - A StreamHandler that prints to stdout with timestamps
      - A FileHandler that writes to  <log_dir>/pipeline_<timestamp>.log

    Call this ONCE at the very start of run.py (or your notebook).
    All subsequent  logging.getLogger(__name__)  calls in every pipeline
    module will inherit these handlers automatically.
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"pipeline_{timestamp}.log")

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Force stdout to UTF-8 on Windows (avoids CP1252 UnicodeEncodeError)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(fmt)

    # File handler (append mode — keeps the full run history)
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(level)

    # Avoid adding duplicate handlers if called more than once
    if not root.handlers:
        root.addHandler(ch)
        root.addHandler(fh)
    else:
        root.handlers.clear()
        root.addHandler(ch)
        root.addHandler(fh)

    logging.info(f"Logging initialised → console + {log_file}")


# ----------------------------------------------------------------
# BASE_CONFIG — edit here to change any hyperparameter globally.
# ----------------------------------------------------------------
BASE_CONFIG: dict = {
    # ---- wandb ----
    "project_name": "aptos-blindness-detection",
    # ---- model ----
    "model": "efficientnet_b0",
    "image_size": (224, 224),
    "dropout_rate": 0.3,
    "num_classes": 5,
    # ---- training ----
    "epochs": 10,
    "optimizer": "adam",
    "lr": 1e-4,
    "loss": "weighted_cross_entropy",
    # batch_size is the TOTAL batch across ALL GPUs.
    # DataParallel splits this evenly: 256 total / 2 GPUs = 128 per GPU.
    # Each Kaggle T4 has 15 GiB VRAM — 128 samples of EfficientNet-B0 at
    # 224x224 uses ~4-5 GiB, well within budget. Raise to 512 if VRAM allows.
    "batch_size": 256,
    "seed": 42,
    # ---- calibration / uncertainty ----
    "calibration_measure": "ECE",
    "calibration_fix": "temp_scaling",
    "mc_dropout_passes": 30,
    # ---- augmentations (logged to wandb; actual transforms live in dataset.py) ----
    "augmentations": [
        "horizontal_flip",
        "vertical_flip",
        "rotation_360",
        "color_jitter",
    ],
    "color_jitter": {
        "brightness": 0.2,
        "contrast": 0.2,
        "saturation": 0.1,
        "hue": 0.05,
    },
    # ---- dataloader ----
    # Rule of thumb: 4 workers per GPU.  2 Kaggle GPUs → 8 workers.
    # This ensures the CPU pipeline never starves the GPUs.
    "num_workers": 8,
    "pin_memory": True,
    "prefetch_factor": 4,
    # ---- artifact save paths ----
    "model_save_path": "artifacts/weights/aptos_efficientnet.pth",
    "optimal_T_save_path": "artifacts/calibration/optimal_T.npy",
    "calib_plot_train_path": "artifacts/calibration/plots/calibration_train.png",
    "class_centroids_save_path": "artifacts/centroids/mean.npy",
    # "mahalanobis_inv_cov_save_path": "artifacts/mahalanobis/inv_cov.npy",
    "test_max_samples": 2000,
    # ---- GradCAM occlusion experiment paths ----
    "gradcam_heatmap_save_dir": "artifacts/gradcam_heatmaps/idrid",
    "umap_save_dir":            "artifacts/umap",
    "occlusion_top_k_percents": [10, 30],
}


# ----------------------------------------------------------------
# Uncertainty Thresholds
# ----------------------------------------------------------------
# These thresholds define when predictions should be referred to a specialist
# due to low confidence or high fragility.
#
# 1. Predictive Entropy: Measures the spread of probability distributions.
#    Since max possible entropy for 5 classes is log(5) ≈ 1.61, a threshold
#    of 1.0 flags cases where probability is highly distributed/not peaky.
#
# 2. Predictive Margin: The gap between the top-1 and top-2 predicted classes.
#    A margin < 0.3 flags cases where the model is highly indecisive between
#    two competing classes (e.g. 0.45 vs 0.35 probability).
#
# 3. MC Dropout Standard Deviation: Measures prediction fragility under network
#    perturbations. A standard deviation > 0.05 flags samples whose predictions
#    are unstable across stochastic forward passes.
UNCERTAINTY_ENTROPY_THRESHOLD: float = 1.0
UNCERTAINTY_MARGIN_THRESHOLD: float = 0.3
UNCERTAINTY_MC_STD_THRESHOLD: float = 0.05

# ----------------------------------------------------------------
# Setting GPU and seed for reproducibility
# ----------------------------------------------------------------


# ^ ------------------------------- setting gpu ----------------------------------


def setting_gpu() -> torch.device:
    """
    Detect all available GPUs, log their names and VRAM, and return
    a torch.device pointing at cuda:0 (DataParallel uses cuda:0 as the
    primary device and scatters batches to all other visible GPUs).
    """
    if not torch.cuda.is_available():
        logger.warning("No GPU found — running on CPU (will be slow for training)")
        return torch.device("cpu")

    n_gpus = torch.cuda.device_count()
    device  = torch.device("cuda")   # defaults to cuda:0; DP handles the rest

    logger.info(f"Device selected : {device}")
    logger.info(f"GPUs available  : {n_gpus}")
    for i in range(n_gpus):
        props = torch.cuda.get_device_properties(i)
        vram_gib = props.total_memory / (1024 ** 3)
        logger.info(
            f"  GPU {i}: {props.name} | "
            f"VRAM={vram_gib:.1f} GiB | "
            f"SM count={props.multi_processor_count}"
        )

    if n_gpus > 1:
        logger.info(
            f"DataParallel will split each batch across all {n_gpus} GPUs "
            f"({n_gpus} x {torch.cuda.get_device_name(0)})"
        )
    return device


# ^ ------------------------------- reproducibility ----------------------------------


def set_seed(seed: int = 42) -> None:
    """
    Fix all random seeds for reproducibility.

    cuDNN settings:
      - deterministic=False : allows cuDNN to pick the fastest conv algorithm
        per input shape.  Setting True forces a single deterministic kernel
        which is significantly slower and blocks multi-GPU optimisations.
      - benchmark=True      : cuDNN auto-tunes the best algorithm for your
        exact input size on first run, then caches it.  This is the main
        lever for squeezing out GPU throughput with a fixed image size.

    Note: with benchmark=True there is a small amount of non-determinism in
    conv layers across runs (different kernel choices across platforms), but
    the seed still makes Python/NumPy/torch ops fully reproducible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Allow cuDNN to auto-select the fastest kernel — critical for GPU throughput
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark     = True
    logger.debug(f"Random seed fixed: {seed}")


# Worker-level seed fn — passed as worker_init_fn to DataLoader
def seed_worker(worker_id: int) -> None:  # noqa: ARG001
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# Shared Generator for DataLoader reproducibility
g = torch.Generator()
g.manual_seed(42)
