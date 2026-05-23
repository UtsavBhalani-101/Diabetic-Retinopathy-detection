# pipeline/__init__.py
# Public API — exposes the most commonly used entry-points.
#
# IMPORTANT: sys.path is patched here FIRST so that `utils.py` at the
# project root is importable regardless of how the package is invoked
# (python -m pipeline.run, direct import, Kaggle notebook, etc.).

import sys as _sys
import os as _os

_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

from .config import BASE_CONFIG, DATASET_REGISTRY  # noqa: F401

try:
    from .train import train_model   # noqa: F401
    from .test  import test_model    # noqa: F401
    __all__ = ["train_model", "test_model", "BASE_CONFIG", "DATASET_REGISTRY"]
except ImportError:
    # wandb (or another optional dep) not installed — train/test unavailable
    __all__ = ["BASE_CONFIG", "DATASET_REGISTRY"]
