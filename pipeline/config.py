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

# Make the project root importable when this package is run from any CWD
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils import DATASET_REGISTRY  # noqa: E402


# ----------------------------------------------------------------
# Logging setup — call once at the start of run.py (or any entry point)
# ----------------------------------------------------------------

def setup_logging(log_dir: str = "artifacts", level: int = logging.INFO) -> None:
    """
    Configure the root logger with:
      - A StreamHandler that prints to stdout with timestamps
      - A FileHandler that appends to  <log_dir>/pipeline.log

    Call this ONCE at the very start of run.py (or your notebook).
    All subsequent  logging.getLogger(__name__)  calls in every pipeline
    module will inherit these handlers automatically.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "pipeline.log")

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
    "model_save_path":       "artifacts/aptos_efficientnet.pth",
    "optimal_T_save_path":   "artifacts/optimal_T.npy",
    "calib_plot_train_path": "artifacts/calibration_train.png",
}
