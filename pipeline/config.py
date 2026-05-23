# pipeline/config.py
# ============================================================
# Central configuration:
#   - BASE_CONFIG  : all hyper-parameters consumed by train_model / test_model
#   - DATASET_REGISTRY re-exported from root utils.py so every pipeline
#     module only needs to import from here (single source of truth)
# ============================================================

import sys
import os

# Make the project root importable when this package is run from any CWD
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils import DATASET_REGISTRY  # noqa: E402  (imported after sys.path fix)

# ----------------------------------------------------------------
# BASE_CONFIG — edit here to change any hyper-parameter globally.
# Individual keys can be overridden when calling train_model /
# test_model by passing a modified copy of this dict.
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
