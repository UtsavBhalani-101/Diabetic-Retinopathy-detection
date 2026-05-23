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

import os

import torch
import wandb
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from .config      import DATASET_REGISTRY
from .dataset     import setting_gpu
from .loaders     import build_loader_for_testing
from .model       import EfficientNetMC
from .evaluate    import mc_evaluate_full, compute_uncertainty_signals
from .calibration import apply_temperature, per_class_calibration


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
                     (e.g. IDRiD has an official held-out test set)
    """
    device = setting_gpu()

    reg         = DATASET_REGISTRY[dataset_name]
    num_classes = reg["num_classes"]

    # ------------------------------------------------------------------
    # 1.  Load model weights — pretrained=False because we supply weights
    # ------------------------------------------------------------------
    model = EfficientNetMC(
        num_classes=num_classes,
        dropout_rate=config["dropout_rate"]
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    print(f"\n[{dataset_name}] Model loaded from {model_path}")

    # ------------------------------------------------------------------
    # 2.  wandb init (separate run, job_type="test")
    # ------------------------------------------------------------------
    wandb.login()
    run = wandb.init(  # noqa: F841
        project=config["project_name"],
        job_type="test",
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

    # ------------------------------------------------------------------
    # 3.  Test loader
    # ------------------------------------------------------------------
    loader = build_loader_for_testing(
        dataset_name, config, use_test_split=use_test_split
    )

    # ------------------------------------------------------------------
    # 4.  MC Dropout (T=30)
    # ------------------------------------------------------------------
    print(f"\n[{dataset_name}] Running MC Dropout (T={config['mc_dropout_passes']})...")
    all_mean_probs, all_uncertainties, all_labels_arr, all_logits = mc_evaluate_full(
        model, loader, device, T=config["mc_dropout_passes"]
    )

    # ------------------------------------------------------------------
    # 5.  Apply calibration temperature from APTOS training
    # ------------------------------------------------------------------
    calibrated_probs = apply_temperature(all_logits, optimal_T)
    final_preds      = calibrated_probs.argmax(axis=1)

    qwk = cohen_kappa_score(all_labels_arr, final_preds, weights="quadratic")
    cm  = confusion_matrix(all_labels_arr, final_preds)
    print(f"\nQWK on {dataset_name}: {qwk:.4f}")
    print("Confusion Matrix:\n", cm)

    # ------------------------------------------------------------------
    # 6.  Uncertainty signals + triage printout
    # ------------------------------------------------------------------
    entropy, margin, mc_uncertainty = compute_uncertainty_signals(
        calibrated_probs, all_uncertainties
    )

    print(f"\nMean entropy      : {entropy.mean():.4f}")
    print(f"Mean margin       : {margin.mean():.4f}")
    print(f"Uncertain fraction: {(entropy > 1.0).mean():.4f}")

    print(f"\nTriage Summary — {dataset_name} (first 20 samples):")
    for i in range(min(20, len(final_preds))):
        pred = final_preds[i]
        true = all_labels_arr[i]
        if entropy[i] > 1.0 or margin[i] < 0.3 or mc_uncertainty[i] > 0.05:
            flag = "UNCERTAIN - refer to specialist"
        elif pred >= 3:
            flag = "HIGH SEVERITY - urgent review"
        else:
            flag = "ROUTINE"
        print(
            f"Sample {i:3d} | True: {true} | Pred: {pred} "
            f"| Entropy: {entropy[i]:.3f} | Margin: {margin[i]:.3f} "
            f"| MC std: {mc_uncertainty[i]:.3f} | {flag}"
        )

    # ------------------------------------------------------------------
    # 7.  Four-quadrant uncertainty breakdown
    # ------------------------------------------------------------------
    uncertain_mask = (entropy > 1.0) | (margin < 0.3) | (mc_uncertainty > 0.05)
    correct_mask   = (final_preds == all_labels_arr)
    print("\n--- Four Quadrant Uncertainty Breakdown ---")
    print("Certain + Wrong (dangerous):", (~uncertain_mask & ~correct_mask).sum())
    print("Certain + Right (ideal)     :", (~uncertain_mask &  correct_mask).sum())
    print("Uncertain + Wrong (caught)  :", ( uncertain_mask & ~correct_mask).sum())
    print("Uncertain + Right (over-ref):", ( uncertain_mask &  correct_mask).sum())

    # ------------------------------------------------------------------
    # 8.  Calibration plot
    # ------------------------------------------------------------------
    os.makedirs("artifacts", exist_ok=True)
    calib_path = f"artifacts/calibration_{dataset_name.replace(' ', '_')}.png"
    print(f"\nGenerating calibration plot → {calib_path}")
    per_class_calibration(calibrated_probs, all_labels_arr, save_path=calib_path)

    # ------------------------------------------------------------------
    # 9.  wandb log
    # ------------------------------------------------------------------
    wandb.log({
        "test_qwk":                float(qwk),
        "test_mean_entropy":       float(entropy.mean()),
        "test_mean_margin":        float(margin.mean()),
        "test_mean_mc_uncertainty": float(mc_uncertainty.mean()),
        "test_uncertain_fraction": float(uncertain_mask.mean()),
        "quadrant_certain_wrong":  int((~uncertain_mask & ~correct_mask).sum()),
        "quadrant_certain_right":  int((~uncertain_mask &  correct_mask).sum()),
        "quadrant_uncertain_wrong": int(( uncertain_mask & ~correct_mask).sum()),
        "quadrant_uncertain_right": int(( uncertain_mask &  correct_mask).sum()),
        "test_confusion_matrix":   wandb.plot.confusion_matrix(
            preds=final_preds.tolist(),
            y_true=all_labels_arr.tolist(),
            class_names=["0", "1", "2", "3", "4"]
        ),
        "test_calibration_plot":   wandb.Image(calib_path),
    })

    wandb.finish()
    print(f"\n[{dataset_name}] Test run complete.")
