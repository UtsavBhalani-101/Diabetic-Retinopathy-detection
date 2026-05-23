# data_layer.py
# ============================================================
# BACKWARD-COMPATIBILITY SHIM
#
# This file has been refactored into the pipeline/ package.
# All logic now lives in:
#
#   pipeline/config.py       — BASE_CONFIG + DATASET_REGISTRY
#   pipeline/dataset.py      — transforms, seed, dataset classes
#   pipeline/loaders.py      — build_loaders_for_training / _testing
#   pipeline/model.py        — EfficientNetMC, get_loss_criterion
#   pipeline/evaluate.py     — evaluate, mc_evaluate_full,
#                              compute_uncertainty_signals
#   pipeline/calibration.py  — per_class_calibration,
#                              find_temperature, apply_temperature
#   pipeline/train.py        — train_model
#   pipeline/test.py         — test_model
#   pipeline/run.py          — orchestrator + __main__
#
# To run the full pipeline:
#   python -m pipeline.run
#
# ============================================================

# Re-export everything so any existing `from data_layer import X` still works
from pipeline.config      import BASE_CONFIG, DATASET_REGISTRY   # noqa: F401
from pipeline.dataset     import (                                # noqa: F401
    setting_gpu, set_seed, seed_worker, g,
    val_transformer, train_transformer,
    RetinopathyDataset, RetinopathyDatasetFromDF,
)
from pipeline.loaders     import (                                # noqa: F401
    build_loaders_for_training, build_loader_for_testing,
)
from pipeline.model       import EfficientNetMC, get_loss_criterion  # noqa: F401
from pipeline.evaluate    import (                                # noqa: F401
    evaluate, mc_evaluate_full, compute_uncertainty_signals,
)
from pipeline.calibration import (                                # noqa: F401
    per_class_calibration, find_temperature, apply_temperature,
)
from pipeline.train       import train_model                      # noqa: F401
from pipeline.test        import test_model                       # noqa: F401
