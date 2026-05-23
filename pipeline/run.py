# pipeline/run.py
# ============================================================
# Entry point for the full pipeline.
#
# Usage:
#   python -m pipeline.run          (from project root)
#   python pipeline/run.py          (also works)
#
# Steps:
#   1. Train on APTOS_2019  → saves model + optimal_T → returns optimal_T
#   2. Zero-shot test on every external dataset in the list below
#      (IDRiD uses its official test CSV; others use their full training CSV)
# ============================================================

from .config import BASE_CONFIG
from .train  import train_model
from .test   import test_model


def orchestrator(config: dict | None = None) -> None:
    """
    Run the full APTOS → external-dataset pipeline.

    Parameters
    ----------
    config : optional override dict; defaults to BASE_CONFIG from pipeline/config.py.
             Only the keys you specify are overridden — the rest keep their defaults.
    """
    # Merge caller overrides on top of defaults
    cfg = {**BASE_CONFIG, **(config or {})}

    # ----------------------------------------------------------------
    # Step 1 — Train on APTOS_2019
    # ----------------------------------------------------------------
    print("=" * 60)
    print("STEP 1: Training on APTOS_2019")
    print("=" * 60)
    optimal_T = train_model("APTOS_2019", cfg)

    # ----------------------------------------------------------------
    # Step 2 — Zero-shot evaluation on external datasets
    #
    # Tuple format: (dataset_name, use_test_split)
    #   use_test_split=True  → use the registry's test_target_path
    #                          (only IDRiD has an official test set)
    #   use_test_split=False → evaluate on the full training CSV
    # ----------------------------------------------------------------
    external_datasets = [
        ("IDRiD",           True),   # has official held-out test CSV
        ("DDR-China",       False),
        ("Messidor-Grp1",   False),
        ("Messidor-Grp2",   False),
        ("Messidor-Grp3",   False),
        ("EyePACS-Resized", False),
    ]

    for ds_name, use_test in external_datasets:
        print("\n" + "=" * 60)
        print(f"STEP 2: Testing on {ds_name}")
        print("=" * 60)
        test_model(
            dataset_name=ds_name,
            model_path=cfg["model_save_path"],
            optimal_T=optimal_T,
            config=cfg,
            use_test_split=use_test
        )

    print("\nAll done.")


if __name__ == "__main__":
    # Run with all defaults — edit BASE_CONFIG in pipeline/config.py
    # or pass overrides:
    #   orchestrator({"epochs": 20, "lr": 5e-5})
    orchestrator()
