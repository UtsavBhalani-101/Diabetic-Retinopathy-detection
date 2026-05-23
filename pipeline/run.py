# pipeline/run.py
# ============================================================
# Entry point for the full pipeline.
#
# Usage:
#   python -m pipeline.run          (from project root)
#   python pipeline/run.py          (also works)
# ============================================================

import logging

from .config import BASE_CONFIG, setup_logging
from .train  import train_model
from .test   import test_model

logger = logging.getLogger(__name__)


def orchestrator(config: dict | None = None) -> None:
    """
    Run the full APTOS → external-dataset pipeline.

    Parameters
    ----------
    config : optional override dict; defaults to BASE_CONFIG.
             Only the keys you specify are overridden — the rest keep defaults.
             Example: orchestrator({"epochs": 20, "lr": 5e-5})
    """
    cfg = {**BASE_CONFIG, **(config or {})}

    logger.info("=" * 60)
    logger.info("STEP 1: Training on APTOS_2019")
    logger.info("=" * 60)

    optimal_T = train_model("APTOS_2019", cfg)

    logger.info(f"Training complete | optimal_T={optimal_T:.4f}")
    logger.info("Starting zero-shot evaluation on external datasets...")

    # (dataset_name, use_test_split)
    # use_test_split=True → use the registry's test_target_path
    # Only IDRiD has an official held-out test set registered
    external_datasets = [
        ("IDRiD",           True),
        ("DDR-China",       False),
        ("Messidor-Grp1",   False),
        ("Messidor-Grp2",   False),
        ("Messidor-Grp3",   False),
        ("EyePACS-Resized", False),
    ]

    for ds_name, use_test in external_datasets:
        logger.info("=" * 60)
        logger.info(f"STEP 2: Testing on {ds_name} | use_test_split={use_test}")
        logger.info("=" * 60)
        test_model(
            dataset_name=ds_name,
            model_path=cfg["model_save_path"],
            optimal_T=optimal_T,
            config=cfg,
            use_test_split=use_test
        )

    logger.info("All done.")


if __name__ == "__main__":
    setup_logging()          # timestamps → console + artifacts/pipeline.log
    orchestrator()
