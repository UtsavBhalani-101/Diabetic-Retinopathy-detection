# pipeline/train.py
# ============================================================
# Full training pipeline for a single dataset (APTOS_2019).
# ============================================================

import logging
import os

import numpy as np
import torch
import wandb
import datetime

from dotenv import load_dotenv
from sklearn.metrics import cohen_kappa_score

from pipeline.setup.utils import DATASET_REGISTRY
from pipeline.setup.config import (
    BASE_CONFIG,
    UNCERTAINTY_ENTROPY_THRESHOLD,
    UNCERTAINTY_MARGIN_THRESHOLD,
    UNCERTAINTY_MC_STD_THRESHOLD
)
from pipeline.setup.config import setting_gpu, set_seed
from pipeline.data.loaders import build_loaders_for_training
from pipeline.training_loop_setup.model import EfficientNetMC, get_loss_criterion
from pipeline.evaluation.evaluate import evaluate, mc_evaluate_full, compute_uncertainty_signals
from pipeline.evaluation.calibration import (
    find_temperature,
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
    logger.info(f"{'='*60}")
    logger.info(f"train_model() | dataset={dataset_name} | starting")
    logger.info(f"{'='*60}")

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
    criterion      = get_loss_criterion(train_df, diag_col, device)
    weights_tensor = criterion.weight.cpu().numpy()

    model = EfficientNetMC(
        num_classes=num_classes,
        dropout_rate=config["dropout_rate"],
        pretrained=config.get("pretrained", True)
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model params | total={total_params:,} | trainable={trainable_params:,}")
    logger.info(f"Optimizer | Adam | lr={config['lr']}")

    # ------------------------------------------------------------------
    # 3.  wandb init — all static + dynamic fields in one dict
    # ------------------------------------------------------------------
    # wandb authentication is handled centrally in run_pipeline.setup_wandb()

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
        job_type="train",
        mode=config.get("wandb_mode", "online")
    )
    logger.info(f"wandb run initialised | project={config['project_name']} | job=train")

    # ------------------------------------------------------------------
    # 4.  Epoch loop
    # ------------------------------------------------------------------
    logger.info(f"Training loop start | epochs={config['epochs']}")
    os.makedirs("artifacts/weights", exist_ok=True)
    os.makedirs("artifacts/calibration/plots", exist_ok=True)

    for epoch in range(config["epochs"]):
        logger.info(f"--- Epoch {epoch + 1}/{config['epochs']} ---")

        # ---- train ----
        model.train()
        running_loss = 0.0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            output_logits = model(images)
            loss = criterion(output_logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # log every 10 batches at DEBUG
            if (batch_idx + 1) % 10 == 0:
                logger.debug(
                    f"  Batch {batch_idx + 1}/{len(train_loader)} "
                    f"| loss={loss.item():.4f}"
                )

        train_loss = running_loss / len(train_loader)
        logger.info(f"Epoch {epoch + 1} | Train Loss: {train_loss:.4f}")

        # ---- standard validation ----
        val_loss, val_qwk, matrix, val_preds, val_labels = evaluate(
            model, val_loader, criterion, device
        )
        logger.info(
            f"Epoch {epoch + 1} | Val Loss: {val_loss:.4f} | Val QWK: {val_qwk:.4f}"
        )
        logger.info(f"Confusion Matrix:\n{matrix}")

        # ---- lightweight MC dropout (T=10) epoch-level uncertainty tracking ----
        logger.debug(f"Epoch {epoch + 1} | Running MC dropout (T=10) for uncertainty tracking...")
        mean_prob_loop, uncertainty_loop, _, _ = mc_evaluate_full(
            model, val_loader, device, T=10
        )
        entropy_loop, margin_loop, mc_unc_loop = compute_uncertainty_signals(
            mean_prob_loop, uncertainty_loop
        )
        uncertain_frac = float(
            (
                (entropy_loop > UNCERTAINTY_ENTROPY_THRESHOLD) |
                (margin_loop < UNCERTAINTY_MARGIN_THRESHOLD) |
                (mc_unc_loop > UNCERTAINTY_MC_STD_THRESHOLD)
            ).mean()
        )
        logger.info(
            f"Epoch {epoch + 1} | MC uncertainty | "
            f"entropy={entropy_loop.mean():.4f} | margin={margin_loop.mean():.4f} "
            f"| uncertain_frac={uncertain_frac:.3f}"
        )

        wandb.log({
            "epoch":               epoch + 1,
            "train_loss":          train_loss,
            "val_loss":            val_loss,
            "val_qwk":             val_qwk,
            "mean_entropy":        float(entropy_loop.mean()),
            "mean_margin":         float(margin_loop.mean()),
            "mean_mc_uncertainty": float(mc_unc_loop.mean()),
            "uncertain_fraction":  uncertain_frac,
            "learning_rate":       optimizer.param_groups[0]["lr"],
            "confusion_matrix":    wandb.plot.confusion_matrix(
                preds=val_preds,
                y_true=val_labels,
                class_names=["0", "1", "2", "3", "4"]
            ),
        })

    logger.info("Training loop complete.")

    # ------------------------------------------------------------------
    # 5a.  Post-training full MC dropout (T=30)
    # ------------------------------------------------------------------
    logger.info(f"Post-training MC Dropout | T={config['mc_dropout_passes']}")
    all_mean_probs, all_uncertainties, all_labels_arr, all_logits = mc_evaluate_full(
        model, val_loader, device, T=config["mc_dropout_passes"]
    )

    # ------------------------------------------------------------------
    # 5b.  Temperature scaling — FIRST, before any reporting
    # ------------------------------------------------------------------
    logger.info("Applying temperature calibration...")
    optimal_T        = find_temperature(all_logits, all_labels_arr)
    calibrated_probs = apply_temperature(all_logits, optimal_T)
    final_preds      = calibrated_probs.argmax(axis=1)

    # ------------------------------------------------------------------
    # 5c.  Post-calibration uncertainty signals + triage
    # ------------------------------------------------------------------
    cal_entropy, cal_margin, cal_mc_unc = compute_uncertainty_signals(
        calibrated_probs, all_uncertainties
    )

    logger.info("Triage Summary — post-calibration (first 20 samples):")
    for i in range(min(20, len(final_preds))):
        pred = final_preds[i]
        true = all_labels_arr[i]
        flag = triage_sample(
            pred, cal_entropy[i], cal_margin[i], cal_mc_unc[i]
        )
        logger.info(
            f"  Sample {i:3d} | True:{true} Pred:{pred} "
            f"| H={cal_entropy[i]:.3f} M={cal_margin[i]:.3f} "
            f"MC={cal_mc_unc[i]:.3f} | {flag}"
        )

    cal_uncertain_mask = (
        (cal_entropy > UNCERTAINTY_ENTROPY_THRESHOLD) |
        (cal_margin < UNCERTAINTY_MARGIN_THRESHOLD) |
        (cal_mc_unc > UNCERTAINTY_MC_STD_THRESHOLD)
    )
    correct_mask = (final_preds == all_labels_arr)

    logger.info(f"Uncertain fraction (calibrated): {cal_uncertain_mask.mean():.3f}")
    logger.info(f"Calibrated Mean entropy        : {cal_entropy.mean():.4f}")
    logger.info(f"Calibrated Mean margin         : {cal_margin.mean():.4f}")
    logger.info(f"Calibrated Mean MC std         : {cal_mc_unc.mean():.4f}")
    logger.info("--- Four Quadrant Uncertainty Breakdown ---")
    logger.info(f"  Certain + Wrong (dangerous): {(~cal_uncertain_mask & ~correct_mask).sum()}")
    logger.info(f"  Certain + Right (ideal)     : {(~cal_uncertain_mask & correct_mask).sum()}")
    logger.info(f"  Uncertain + Wrong (caught)  : {( cal_uncertain_mask & ~correct_mask).sum()}")
    logger.info(f"  Uncertain + Right (over-ref): {( cal_uncertain_mask &  correct_mask).sum()}")

    # ------------------------------------------------------------------
    # 5d.  Calibration plot (post-scaling only)
    # ------------------------------------------------------------------
    calib_path = config.get("calib_plot_train_path", "artifacts/calibration/plots/calibration_train.png")
    per_class_calibration(calibrated_probs, all_labels_arr, save_path=calib_path)

    # Backup timestamped plot
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_calib_path = os.path.join(os.path.dirname(calib_path), f"calibration_train_{timestamp}.png")
    per_class_calibration(calibrated_probs, all_labels_arr, save_path=backup_calib_path)

    final_qwk = cohen_kappa_score(all_labels_arr, final_preds, weights="quadratic")
    logger.info(f"Final calibrated val QWK: {final_qwk:.4f}")

    wandb.log({
        "optimal_T":                float(optimal_T),
        "final_val_qwk":            float(final_qwk),
        "final_mean_entropy":       float(cal_entropy.mean()),
        "final_mean_margin":        float(cal_margin.mean()),
        "final_mean_mc_uncertainty": float(cal_mc_unc.mean()),
        "final_uncertain_fraction": float(cal_uncertain_mask.mean()),
        "quadrant_certain_wrong":   int((~cal_uncertain_mask & ~correct_mask).sum()),
        "quadrant_certain_right":   int((~cal_uncertain_mask &  correct_mask).sum()),
        "quadrant_uncertain_wrong": int(( cal_uncertain_mask & ~correct_mask).sum()),
        "quadrant_uncertain_right": int(( cal_uncertain_mask &  correct_mask).sum()),
        "calibration_plot":         wandb.Image(calib_path),
    })

    # ------------------------------------------------------------------
    # 6.  Save model + optimal_T (training only — never in test_model)
    # ------------------------------------------------------------------
    model_path = config.get("model_save_path",     "artifacts/weights/aptos_efficientnet.pth")
    T_path     = config.get("optimal_T_save_path", "artifacts/calibration/optimal_T.npy")

    torch.save(model.state_dict(), model_path)
    np.save(T_path, np.array(optimal_T))

    # Backup timestamped weight and T parameters
    backup_model_path = os.path.join(os.path.dirname(model_path), f"aptos_efficientnet_{timestamp}.pth")
    backup_T_path = os.path.join(os.path.dirname(T_path), f"optimal_T_{timestamp}.npy")
    torch.save(model.state_dict(), backup_model_path)
    np.save(backup_T_path, np.array(optimal_T))

    logger.info(f"Model saved     → {model_path} (Backup: {backup_model_path})")
    logger.info(f"Optimal T saved → {T_path} (Backup: {backup_T_path}) (T={optimal_T:.4f})")

    wandb.finish()
    logger.info("train_model() complete.")
    return optimal_T

if __name__ == "__main__":
    setup_wandb()
    train_model(dataset_name="APTOS_2019", config=BASE_CONFIG)