# pipeline/test.py
# ============================================================
# Zero-shot evaluation of a trained APTOS model on any external dataset.
#
# Responsibilities:
#   1. Load model weights from disk  (no training, no saving)
#   2. wandb init (job_type="test")
#   3. Build test DataLoader from registry
#   4. MC Dropout (T=30) → apply calibration T from APTOS training
#   5. Triage + four-quadrant analysis + calibration plot
#   6. wandb log all test metrics  |  wandb finish
# ============================================================

import logging
import os
import numpy as np
import torch
import wandb
import datetime

from dotenv import load_dotenv
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from pipeline.setup.utils import DATASET_REGISTRY
from pipeline.setup.config import (
    BASE_CONFIG,
    UNCERTAINTY_ENTROPY_THRESHOLD,
    UNCERTAINTY_MARGIN_THRESHOLD,
    UNCERTAINTY_MC_STD_THRESHOLD
)
from pipeline.setup.config import setting_gpu
from pipeline.data.loaders import build_loader_for_testing
from pipeline.training_loop_setup.model import EfficientNetMC
from pipeline.evaluation.evaluate import mc_evaluate_full, compute_uncertainty_signals
from pipeline.evaluation.cosine_similarity import calculate_cosine_similarity
from pipeline.evaluation.calibration import (
    apply_temperature,
    per_class_calibration,
    triage_sample
)

logger = logging.getLogger(__name__)

def setup_wandb() -> None:
    """
    Load .env and authenticate wandb using the API key.
    Falls back to interactive login if no key is found.
    """
    load_dotenv()  # reads .env at project root → os.environ
    api_key = os.environ.get("WANDB_API_KEY")
    if api_key:
        wandb.login(key=api_key)
        logger.info("wandb authenticated via WANDB_API_KEY from .env")
    else:
        logger.warning("WANDB_API_KEY not found in .env — falling back to interactive login")
        wandb.login()



def test_model(dataset_name: str, model_path: str, optimal_T: float,
               config: dict, use_test_split: bool = False) -> None:
    """
    Zero-shot evaluation on an external dataset.

    Parameters
    ----------
    dataset_name   : key in DATASET_REGISTRY (e.g. "IDRiD")
    model_path     : path to saved .pth weights from train_model()
    optimal_T      : temperature scalar returned by train_model()
    config         : same config dict used at training time
    use_test_split : True → use registry's test_target_path if it exists
    """
    logger.info(f"{'='*60}")
    logger.info(f"test_model() | dataset={dataset_name} | T={optimal_T:.4f}")
    logger.info(f"{'='*60}")

    device = setting_gpu()

    reg         = DATASET_REGISTRY[dataset_name]
    num_classes = reg["num_classes"]

    # ------------------------------------------------------------------
    # 1.  Load model weights
    # ------------------------------------------------------------------
    model = EfficientNetMC(
        num_classes=num_classes,
        dropout_rate=config["dropout_rate"],
        pretrained=False   # weights loaded from disk; don't download ImageNet
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    logger.info(f"[{dataset_name}] Model loaded from {model_path}")

    # ------------------------------------------------------------------
    # 2.  wandb init (separate run, job_type="test")
    # ------------------------------------------------------------------
    # wandb authentication is handled centrally in run_pipeline.setup_wandb()

    run = wandb.init(  # noqa: F841
        project=config["project_name"],
        job_type="test",
        mode=config.get("wandb_mode", "online"),
        config={
            "dataset":           dataset_name,
            "model":             config["model"],
            "model_path":        model_path,
            "optimal_T":         optimal_T,
            "mc_dropout_passes": config["mc_dropout_passes"],
            "batch_size":        config["batch_size"],
            "use_test_split":    use_test_split,
            "device":            (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available() else "cpu"
            ),
        }
    )
    logger.info(f"[{dataset_name}] wandb run initialised | job=test")

    # ------------------------------------------------------------------
    # 3.  Test loader
    # ------------------------------------------------------------------
    loader = build_loader_for_testing(
        dataset_name, config, use_test_split=use_test_split
    )

    # ------------------------------------------------------------------
    # 4.  MC Dropout (T=30)
    # ------------------------------------------------------------------
    logger.info(f"[{dataset_name}] MC Dropout | T={config['mc_dropout_passes']}")
    all_mean_probs, all_uncertainties, all_labels_arr, all_logits = mc_evaluate_full(
        model, loader, device, T=config["mc_dropout_passes"]
    )

    # ------------------------------------------------------------------
    # 5.  Apply calibration temperature from APTOS training
    # ------------------------------------------------------------------
    logger.info(f"[{dataset_name}] Applying temperature T={optimal_T:.4f}")
    calibrated_probs = apply_temperature(all_logits, optimal_T)
    final_preds      = calibrated_probs.argmax(axis=1)

    qwk = cohen_kappa_score(all_labels_arr, final_preds, weights="quadratic")
    cm  = confusion_matrix(all_labels_arr, final_preds)
    logger.info(f"[{dataset_name}] QWK: {qwk:.4f}")
    logger.info(f"[{dataset_name}] Confusion Matrix:\n{cm}")

    # ------------------------------------------------------------------
    # 6.  Uncertainty signals + triage printout
    # ------------------------------------------------------------------
    entropy, margin, mc_uncertainty = compute_uncertainty_signals(
        calibrated_probs, all_uncertainties
    )

    logger.info(f"[{dataset_name}] Mean entropy      : {entropy.mean():.4f}")
    logger.info(f"[{dataset_name}] Mean margin       : {margin.mean():.4f}")
    logger.info(f"[{dataset_name}] Uncertain fraction: {(entropy > UNCERTAINTY_ENTROPY_THRESHOLD).mean():.4f}")

    logger.info(f"[{dataset_name}] Triage Summary (first 20 samples):")
    for i in range(min(20, len(final_preds))):
        pred = final_preds[i]
        true = all_labels_arr[i]
        flag = triage_sample(
            pred, entropy[i], margin[i], mc_uncertainty[i]
        )
        logger.info(
            f"  Sample {i:3d} | True:{true} Pred:{pred} "
            f"| H={entropy[i]:.3f} M={margin[i]:.3f} "
            f"MC={mc_uncertainty[i]:.3f} | {flag}"
        )

    # ------------------------------------------------------------------
    # 7.  Four-quadrant breakdown
    # ------------------------------------------------------------------
    uncertain_mask = (
        (entropy > UNCERTAINTY_ENTROPY_THRESHOLD) |
        (margin < UNCERTAINTY_MARGIN_THRESHOLD) |
        (mc_uncertainty > UNCERTAINTY_MC_STD_THRESHOLD)
    )
    correct_mask   = (final_preds == all_labels_arr)
    logger.info(f"[{dataset_name}] --- Four Quadrant Uncertainty Breakdown ---")
    logger.info(f"  Certain + Wrong (dangerous): {(~uncertain_mask & ~correct_mask).sum()}")
    logger.info(f"  Certain + Right (ideal)     : {(~uncertain_mask &  correct_mask).sum()}")
    logger.info(f"  Uncertain + Wrong (caught)  : {( uncertain_mask & ~correct_mask).sum()}")
    logger.info(f"  Uncertain + Right (over-ref): {( uncertain_mask &  correct_mask).sum()}")

    os.makedirs("artifacts/calibration/plots", exist_ok=True)
    calib_path = f"artifacts/calibration/plots/calibration_{dataset_name.replace(' ', '_')}.png"
    per_class_calibration(calibrated_probs, all_labels_arr, save_path=calib_path)

    # Backup timestamped plot
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_calib_path = f"artifacts/calibration/plots/calibration_{dataset_name.replace(' ', '_')}_{timestamp}.png"
    per_class_calibration(calibrated_probs, all_labels_arr, save_path=backup_calib_path)

    # ------------------------------------------------------------------
    # 9.  Cosine Similarity to Centroids (per-class)
    # ------------------------------------------------------------------

    per_class_mean_from_train = np.load(BASE_CONFIG["class_centroids_save_path"], allow_pickle=True).item()

    cosine_sim_save_dir = f"artifacts/cosine_similarity/{dataset_name.replace(' ', '_')}"
    per_class_similarities, sim_labels = calculate_cosine_similarity(
        model, loader, device, num_classes,
        per_class_mean=per_class_mean_from_train,
        save_dir=cosine_sim_save_dir
    )



    # ------------------------------------------------------------------
    # 10.  wandb log
    # ------------------------------------------------------------------
    wandb.log({
        "test_qwk":                 float(qwk),
        "test_mean_entropy":        float(entropy.mean()),
        "test_mean_margin":         float(margin.mean()),
        "test_mean_mc_uncertainty": float(mc_uncertainty.mean()),
        "test_uncertain_fraction":  float(uncertain_mask.mean()),
        "quadrant_certain_wrong":   int((~uncertain_mask & ~correct_mask).sum()),
        "quadrant_certain_right":   int((~uncertain_mask &  correct_mask).sum()),
        "quadrant_uncertain_wrong": int(( uncertain_mask & ~correct_mask).sum()),
        "quadrant_uncertain_right": int(( uncertain_mask &  correct_mask).sum()),
        "test_confusion_matrix":    wandb.plot.confusion_matrix(
            preds=final_preds.tolist(),
            y_true=all_labels_arr.tolist(),
            class_names=["0", "1", "2", "3", "4"]
        ),
        "test_calibration_plot":    wandb.Image(calib_path),
    })

    wandb.finish()
    logger.info(f"[{dataset_name}] test_model() complete.")

if __name__ == "__main__":
    setup_wandb()

    # Read paths from config — same keys that train_model() uses to save
    model_path = BASE_CONFIG["model_save_path"]
    optimal_T  = float(np.load(BASE_CONFIG["optimal_T_save_path"]))

    datasets = ["DDR-China", "EyePACS-Resized"]
    for dataset in datasets:
        test_model(dataset_name=dataset, model_path=model_path,
                   optimal_T=optimal_T, config=BASE_CONFIG, use_test_split=False)