# pipeline/train.py
# ============================================================
# Full training pipeline for a single dataset (APTOS_2019).
#
# Responsibilities:
#   1. Build train / val loaders (stratified 80/20 split)
#   2. Init model, optimizer, class-weighted criterion
#   3. wandb init with full config (dynamic fields filled in here)
#   4. Epoch loop: train → val → lightweight MC dropout (T=10)
#   5. Post-loop: full MC dropout (T=30) → temperature scaling
#      → post-calibration triage / stats / calibration plot only
#   6. Save model weights + optimal_T to artifacts/
#   7. wandb finish  |  return optimal_T
# ============================================================

import os

import numpy as np
import torch
import wandb
from sklearn.metrics import cohen_kappa_score

from .config       import DATASET_REGISTRY
from .dataset      import setting_gpu, set_seed
from .loaders      import build_loaders_for_training
from .model        import EfficientNetMC, get_loss_criterion
from .evaluate     import evaluate, mc_evaluate_full, compute_uncertainty_signals
from .calibration  import find_temperature, apply_temperature, per_class_calibration


def train_model(dataset_name: str, config: dict) -> float:
    """
    Train EfficientNetMC on dataset_name and return the optimal temperature T.

    Model weights and optimal_T are saved to disk (artifacts/) only here.
    Pass the returned optimal_T directly to test_model().

    Parameters
    ----------
    dataset_name : key in DATASET_REGISTRY (e.g. "APTOS_2019")
    config       : hyperparameter dict — use BASE_CONFIG from pipeline.config
                   and override any key before passing

    Returns
    -------
    optimal_T : float
    """
    device = setting_gpu()
    set_seed(config["seed"])

    # ------------------------------------------------------------------
    # 1.  Build loaders (stratified 80/20 split on the CSV)
    # ------------------------------------------------------------------
    train_loader, val_loader, train_df, val_df = build_loaders_for_training(
        dataset_name, config
    )

    reg         = DATASET_REGISTRY[dataset_name]
    diag_col    = reg["diagnosis_col"]
    num_classes = reg["num_classes"]

    # ------------------------------------------------------------------
    # 2.  Model, optimizer, criterion
    # ------------------------------------------------------------------
    criterion     = get_loss_criterion(train_df, diag_col, device)
    weights_tensor = criterion.weight.cpu().numpy()

    model = EfficientNetMC(
        num_classes=num_classes,
        dropout_rate=config["dropout_rate"]
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    # ------------------------------------------------------------------
    # 3.  wandb init — all static + dynamic fields in one dict
    # ------------------------------------------------------------------
    wandb.login()
    run_config = {
        "dataset":             dataset_name,
        "model":               config["model"],
        "image_size":          config["image_size"],
        "epochs":              config["epochs"],
        "optimizer":           config["optimizer"],
        "lr":                  config["lr"],
        "loss":                config["loss"],
        "class_weights":       weights_tensor.tolist(),
        "calibration_measure": config["calibration_measure"],
        "calibration_fix":     config["calibration_fix"],
        "batch_size":          config["batch_size"],
        "dropout_rate":        config["dropout_rate"],
        "mc_dropout_passes":   config["mc_dropout_passes"],
        "augmentations":       config["augmentations"],
        "color_jitter":        config["color_jitter"],
        "seed":                config["seed"],
        "train_samples":       len(train_df),
        "val_samples":         len(val_df),
        "class_distribution":  train_df[diag_col].value_counts().sort_index().to_dict(),
        "num_classes":         num_classes,
        "num_workers":         config["num_workers"],
        "pin_memory":          config["pin_memory"],
        "prefetch_factor":     config["prefetch_factor"],
        "device":              (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available() else "cpu"
        ),
    }
    run = wandb.init(  # noqa: F841
        project=config["project_name"],
        config=run_config,
        job_type="train"
    )

    # ------------------------------------------------------------------
    # 4.  Epoch loop
    # ------------------------------------------------------------------
    print("\nTraining loop start...")
    os.makedirs("artifacts", exist_ok=True)

    for epoch in range(config["epochs"]):

        # ---- train ----
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            output_logits = model(images)
            loss = criterion(output_logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        print(f"\nEpoch {epoch + 1} | Train Loss: {train_loss:.4f}")

        # ---- standard validation ----
        val_loss, val_qwk, matrix, val_preds, val_labels = evaluate(
            model, val_loader, criterion, device
        )
        print(f"Epoch {epoch + 1} | Val Loss: {val_loss:.4f} | Val QWK: {val_qwk:.4f}")
        print("Confusion Matrix:\n", matrix)

        # ---- lightweight MC dropout (T=10) — epoch-level uncertainty tracking ----
        mean_prob_loop, uncertainty_loop, _, _ = mc_evaluate_full(
            model, val_loader, device, T=10
        )
        entropy_loop, margin_loop, mc_unc_loop = compute_uncertainty_signals(
            mean_prob_loop, uncertainty_loop
        )
        uncertain_frac = float(
            ((entropy_loop > 1.0) | (margin_loop < 0.3) | (mc_unc_loop > 0.05)).mean()
        )

        wandb.log({
            "epoch":              epoch + 1,
            "train_loss":         train_loss,
            "val_loss":           val_loss,
            "val_qwk":            val_qwk,
            "mean_entropy":       float(entropy_loop.mean()),
            "mean_margin":        float(margin_loop.mean()),
            "mean_mc_uncertainty": float(mc_unc_loop.mean()),
            "uncertain_fraction": uncertain_frac,
            "learning_rate":      optimizer.param_groups[0]["lr"],
            "confusion_matrix":   wandb.plot.confusion_matrix(
                preds=val_preds,
                y_true=val_labels,
                class_names=["0", "1", "2", "3", "4"]
            ),
        })

    print("\nTraining complete.")

    # ------------------------------------------------------------------
    # 5a.  Post-training full MC dropout (T=30)
    # ------------------------------------------------------------------
    print(f"\nRunning MC Dropout over full val set (T={config['mc_dropout_passes']})...")
    all_mean_probs, all_uncertainties, all_labels_arr, all_logits = mc_evaluate_full(
        model, val_loader, device, T=config["mc_dropout_passes"]
    )

    # ------------------------------------------------------------------
    # 5b.  Temperature scaling FIRST — no pre-calibration reporting
    # ------------------------------------------------------------------
    print("\nApplying temperature calibration...")
    optimal_T        = find_temperature(all_logits, all_labels_arr)
    calibrated_probs = apply_temperature(all_logits, optimal_T)
    final_preds      = calibrated_probs.argmax(axis=1)

    # ------------------------------------------------------------------
    # 5c.  Post-calibration uncertainty signals + triage
    # ------------------------------------------------------------------
    cal_entropy, cal_margin, cal_mc_unc = compute_uncertainty_signals(
        calibrated_probs, all_uncertainties
    )

    print("\nTriage Summary — post-calibration (first 20 samples):")
    for i in range(min(20, len(final_preds))):
        pred = final_preds[i]
        true = all_labels_arr[i]
        if cal_entropy[i] > 1.0 or cal_margin[i] < 0.3 or cal_mc_unc[i] > 0.05:
            flag = "UNCERTAIN - refer to specialist"
        elif pred >= 3:
            flag = "HIGH SEVERITY - urgent review"
        else:
            flag = "ROUTINE"
        print(
            f"Sample {i:3d} | True: {true} | Pred: {pred} "
            f"| Entropy: {cal_entropy[i]:.3f} | Margin: {cal_margin[i]:.3f} "
            f"| MC std: {cal_mc_unc[i]:.3f} | {flag}"
        )

    cal_uncertain_mask = (
        (cal_entropy > 1.0) | (cal_margin < 0.3) | (cal_mc_unc > 0.05)
    )
    correct_mask = (final_preds == all_labels_arr)

    print(f"\nUncertain fraction (calibrated): {cal_uncertain_mask.mean():.3f}")
    print(f"Calibrated Mean entropy        : {cal_entropy.mean():.4f}")
    print(f"Calibrated Mean margin         : {cal_margin.mean():.4f}")
    print(f"Calibrated Mean MC std         : {cal_mc_unc.mean():.4f}")
    print("\n--- Four Quadrant Uncertainty Breakdown ---")
    print("Certain + Wrong (dangerous):", (~cal_uncertain_mask & ~correct_mask).sum())
    print("Certain + Right (ideal)     :", (~cal_uncertain_mask & correct_mask).sum())
    print("Uncertain + Wrong (caught)  :", ( cal_uncertain_mask & ~correct_mask).sum())
    print("Uncertain + Right (over-ref):", ( cal_uncertain_mask &  correct_mask).sum())

    # ------------------------------------------------------------------
    # 5d.  Calibration plot (post-scaling only)
    # ------------------------------------------------------------------
    calib_path = config.get("calib_plot_train_path", "artifacts/calibration_train.png")
    print("\nGenerating calibration plot (post-scaling)...")
    per_class_calibration(calibrated_probs, all_labels_arr, save_path=calib_path)

    final_qwk = cohen_kappa_score(all_labels_arr, final_preds, weights="quadratic")

    wandb.log({
        "optimal_T":               float(optimal_T),
        "final_val_qwk":           float(final_qwk),
        "final_mean_entropy":      float(cal_entropy.mean()),
        "final_mean_margin":       float(cal_margin.mean()),
        "final_mean_mc_uncertainty": float(cal_mc_unc.mean()),
        "final_uncertain_fraction": float(cal_uncertain_mask.mean()),
        "quadrant_certain_wrong":  int((~cal_uncertain_mask & ~correct_mask).sum()),
        "quadrant_certain_right":  int((~cal_uncertain_mask &  correct_mask).sum()),
        "quadrant_uncertain_wrong": int(( cal_uncertain_mask & ~correct_mask).sum()),
        "quadrant_uncertain_right": int(( cal_uncertain_mask &  correct_mask).sum()),
        "calibration_plot":        wandb.Image(calib_path),
    })

    # ------------------------------------------------------------------
    # 6.  Save model + optimal_T  (training only — never in test_model)
    # ------------------------------------------------------------------
    model_path = config.get("model_save_path",     "artifacts/aptos_efficientnet.pth")
    T_path     = config.get("optimal_T_save_path", "artifacts/optimal_T.npy")

    torch.save(model.state_dict(), model_path)
    np.save(T_path, np.array(optimal_T))
    print(f"\nModel saved     → {model_path}")
    print(f"Optimal T saved → {T_path}  (T = {optimal_T:.4f})")

    wandb.finish()
    print("\nTraining run complete.")
    return optimal_T
