# pipeline/config.py
# ============================================================
# Central configuration:
#   - setup_logging()   : configures console + file handlers once
#   - BASE_CONFIG       : all hyperparameters consumed by train/test
#   - DATASET_REGISTRY  : re-exported from root utils.py
# ============================================================

import sys
import os
import logging
import datetime
import numpy as np
import torch 
import random

# Make the project root importable when this package is run from any CWD
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils import DATASET_REGISTRY  # noqa: E402


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
    "model":        "efficientnet_b0",
    "image_size":   (224, 224),
    "dropout_rate": 0.3,
    "num_classes":  5,

    # ---- training ----
    "epochs":    10,
    "optimizer": "adam",
    "lr":        1e-4,
    "loss":      "weighted_cross_entropy",
    "batch_size": 32,
    "seed":       42,

    # ---- calibration / uncertainty ----
    "calibration_measure": "ECE",
    "calibration_fix":     "temp_scaling",
    "mc_dropout_passes":   30,

    # ---- augmentations (logged to wandb; actual transforms live in dataset.py) ----
    "augmentations": [
        "horizontal_flip", "vertical_flip",
        "rotation_360", "color_jitter"
    ],
    "color_jitter": {
        "brightness": 0.2, "contrast": 0.2,
        "saturation": 0.1, "hue": 0.05
    },

    # ---- dataloader ----
    "num_workers":    4,
    "pin_memory":     True,
    "prefetch_factor": 2,

    # ---- artifact save paths ----
    "model_save_path":       "artifacts/weights/aptos_efficientnet.pth",
    "optimal_T_save_path":   "artifacts/calibration/optimal_T.npy",
    "calib_plot_train_path": "artifacts/calibration/plots/calibration_train.png",
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

# ^ ------------------------------- setting gpu ----------------------------------

def setting_gpu() -> torch.device:
    """Detect GPU, log name, and return a torch.device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device selected: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("No GPU found — running on CPU (will be slow for training)")
    return device


# ^ ------------------------------- reproducibility ----------------------------------

def set_seed(seed: int = 42) -> None:
    """Fix all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.debug(f"Random seed fixed: {seed}")


# Worker-level seed fn — passed as worker_init_fn to DataLoader
def seed_worker(worker_id: int) -> None:  # noqa: ARG001
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# Shared Generator for DataLoader reproducibility
g = torch.Generator()
g.manual_seed(42)
