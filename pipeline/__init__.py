# pipeline/__init__.py
# Public API — import the most commonly used entry-points so callers
# can simply do:  from pipeline import train_model, test_model
from .train import train_model
from .test  import test_model
from .config import BASE_CONFIG, DATASET_REGISTRY

__all__ = ["train_model", "test_model", "BASE_CONFIG", "DATASET_REGISTRY"]
