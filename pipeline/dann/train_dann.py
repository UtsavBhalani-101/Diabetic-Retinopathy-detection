# pipeline/dann/train_dann.py
# ============================================================
# DANN training loop for multi-source → multi-target DR grading.
#
# Source domains (labelled):  APTOS, EyePACS, Messidor-Grp1, DDR-China
# Target domains (unlabelled during training): Messidor-Grp2, Messidor-Grp3, IDRiD
#
# Loss per batch:
#   L_total = L_class(source) + λ · L_domain(source + target)
#
# After training:
#   - MC Dropout uncertainty (T=30)
#   - Temperature scaling calibration
#   - Per-target-dataset QWK evaluation
#   - Model + optimal_T saved to artifacts/dann/
# ============================================================

import datetime
import logging
import os

import numpy as np
import torch
import torch.nn as nn
import wandb
from itertools import cycle
from sklearn.metrics import cohen_kappa_score

from pipeline.setup.config import (                              # read-only
    UNCERTAINTY_ENTROPY_THRESHOLD,
    UNCERTAINTY_MARGIN_THRESHOLD,
    UNCERTAINTY_MC_STD_THRESHOLD,
    setting_gpu,
    set_seed,
)
from pipeline.evaluation.calibration import (                   # read-only
    find_temperature,
    apply_temperature,
    per_class_calibration,
    triage_sample,
)

from pipeline.dann.model_dann import (
    DANNEfficientNet,
    get_dann_loss_criterion,
    compute_lambda,
)
from pipeline.dann.loaders_dann import (
    build_dann_source_loaders,
    build_dann_target_train_loader,
    build_dann_target_eval_loader,
)
from pipeline.data.gpu_transforms import gpu_normalize   # ImageNet normalize only — CLAHE is offline (see preprocess_clahe.py)

logger = logging.getLogger(__name__)


# ============================================================
# Internal evaluation helpers
# (DANN model returns a tuple — these replace pipeline.evaluation.evaluate
#  without modifying it)
# ============================================================

def _evaluate_dann(model: DANNEfficientNet,
                   loader,
                   criterion: nn.CrossEntropyLoss,
                   device: torch.device):
    """
    Standard (single-pass) evaluation on a source-only val loader.

    The val loader yields (image, class_label) — standard RetinopathyDataset,
    no domain label. Only class_logits from the label head are used.

    Returns
    -------
    val_loss : float
    val_qwk  : float
    matrix   : np.ndarray  confusion matrix
    preds    : list[int]
    labels   : list[int]
    """
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            images = gpu_normalize(images)   # normalize only — CLAHE already applied offline
            labels = labels.to(device)

            class_logits = model.predict_class(images)   # [B, 5], no domain head
            loss = criterion(class_logits, labels)
            running_loss += loss.item()

            preds = class_logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    val_loss = running_loss / len(loader)
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    val_qwk = cohen_kappa_score(all_labels, all_preds, weights="quadratic")

    from sklearn.metrics import confusion_matrix
    matrix = confusion_matrix(all_labels, all_preds,
                               labels=list(range(5)))

    return val_loss, val_qwk, matrix, all_preds.tolist(), all_labels.tolist()


def _mc_evaluate_dann(model: DANNEfficientNet,
                      loader,
                      device: torch.device,
                      T: int = 30):
    """
    MC Dropout evaluation using the label head only.

    Forces dropout layers to train() mode (stochastic) while running T
    forward passes. Mirrors the logic in pipeline.evaluation.evaluate.mc_evaluate_full
    but uses model.predict_class() to avoid unpacking the domain output.

    Returns
    -------
    mean_probs    : np.ndarray [N, 5]   — averaged softmax probabilities
    uncertainties : np.ndarray [N, 5]   — std of softmax probs across T passes
    labels_arr    : np.ndarray [N]      — true class labels
    logits_arr    : np.ndarray [N, 5]   — logits from final pass (for temp scaling)
    """
    model.eval()
    # Enable dropout during inference (MC Dropout)
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

    all_prob_runs = []   # T × [N, 5]
    all_labels    = []
    last_logits   = []

    with torch.no_grad():
        for t in range(T):
            batch_probs  = []
            batch_logits = []
            labels_collected = []

            for images, labels in loader:
                images = images.to(device)
                images = gpu_normalize(images)   # normalize only — CLAHE already applied offline
                logits = model.predict_class(images)    # [B, 5]
                probs  = torch.softmax(logits, dim=1)   # [B, 5]
                batch_probs.append(probs.cpu().numpy())
                batch_logits.append(logits.cpu().numpy())
                if t == 0:
                    labels_collected.extend(labels.numpy())

            all_prob_runs.append(np.vstack(batch_probs))
            if t == 0:
                all_labels  = labels_collected
                last_logits = np.vstack(batch_logits)

    # Stack → [T, N, 5]
    stacked = np.stack(all_prob_runs, axis=0)
    mean_probs    = stacked.mean(axis=0)   # [N, 5]
    uncertainties = stacked.std(axis=0)    # [N, 5]

    return (
        mean_probs,
        uncertainties,
        np.array(all_labels),
        last_logits,
    )


def _compute_uncertainty_signals(mean_probs: np.ndarray,
                                 uncertainties: np.ndarray):
    """
    Compute entropy, margin, and MC std from MC Dropout outputs.
    Mirrors pipeline.evaluation.evaluate.compute_uncertainty_signals.
    """
    # Predictive entropy: H = -Σ p log(p+ε)
    eps     = 1e-8
    entropy = -(mean_probs * np.log(mean_probs + eps)).sum(axis=1)  # [N]

    # Predictive margin: difference between top-1 and top-2 probabilities
    sorted_p = np.sort(mean_probs, axis=1)[:, ::-1]                 # [N, 5] desc
    margin   = sorted_p[:, 0] - sorted_p[:, 1]                      # [N]

    # MC std: mean std across classes
    mc_std = uncertainties.mean(axis=1)                              # [N]

    return entropy, margin, mc_std


# ============================================================
# Main DANN training function
# ============================================================

def train_dann(config: dict) -> float:
    """
    Full DANN training pipeline.

    Parameters
    ----------
    config : dict — use DANN_CONFIG from pipeline.dann.config_dann

    Returns
    -------
    optimal_T : float — temperature scaling parameter for post-hoc calibration
    """
    logger.info("=" * 60)
    logger.info("train_dann() | DANN training starting")
    logger.info(f"Source datasets : {config['source_datasets']}")
    logger.info(f"Target datasets : {config['target_datasets']}")
    logger.info("=" * 60)

    device = setting_gpu()
    set_seed(config["seed"])

    dann_epochs = config.get("dann_epochs", config["epochs"])

    # ------------------------------------------------------------------
    # 1. Build DataLoaders
    # ------------------------------------------------------------------
    logger.info("Building source loaders (train + val)...")
    src_train_loader, src_val_loader, combined_train_df = build_dann_source_loaders(config)

    logger.info("Building target training loader...")
    tgt_train_loader = build_dann_target_train_loader(config)

    # ------------------------------------------------------------------
    # 2. Model, criterion, optimizer
    # ------------------------------------------------------------------
    criterion = get_dann_loss_criterion(
        combined_train_df,
        diagnosis_col="diagnosis",   # unified column from loaders_dann
        device=device,
    )

    # DESIGN DECISION (confirmed): Cold start from ImageNet weights.
    # DANN trains the full backbone from scratch on the combined 4-dataset
    # source pool — aptos_efficientnet.pth is intentionally NOT loaded.
    # Rationale: avoids confounding the domain-adaptation experiment with
    # APTOS-specific priors already baked into the backbone; gives DANN a
    # clean slate to learn source-invariant features via adversarial training.
    model = DANNEfficientNet(
        num_classes=config["num_classes"],
        dropout_rate=config["dropout_rate"],
        domain_hidden=config["domain_hidden_dim"],
        domain_dropout=config["domain_dropout"],
        pretrained=config.get("pretrained", True),
    ).to(device)

    # Single optimizer for all parameters (backbone + both heads)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    # Unweighted CE for the domain head (binary, balanced by construction)
    domain_criterion = nn.CrossEntropyLoss()

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model params | total={total_params:,} | trainable={trainable_params:,}")

    # ------------------------------------------------------------------
    # 3. wandb init
    # ------------------------------------------------------------------
    run_config = {
        "model":             config["model"],
        "dann_epochs":       dann_epochs,
        "dann_lambda_max":   config["dann_lambda_max"],
        "domain_hidden_dim": config["domain_hidden_dim"],
        "domain_dropout":    config["domain_dropout"],
        "source_datasets":   config["source_datasets"],
        "target_datasets":   config["target_datasets"],
        "batch_size":        config["batch_size"],
        "lr":                config["lr"],
        "dropout_rate":      config["dropout_rate"],
        "mc_dropout_passes": config["mc_dropout_passes"],
        "seed":              config["seed"],
        "total_source_train": len(src_train_loader.dataset),
        "total_target_train": len(tgt_train_loader.dataset),
        "device": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available() else "cpu"
        ),
    }
    wandb.init(
        project=config["project_name"],
        config=run_config,
        job_type="dann_train",
        mode=config.get("wandb_mode", "online"),
    )
    logger.info(f"wandb run initialised | project={config['project_name']}")

    # ------------------------------------------------------------------
    # 4. Training loop
    # ------------------------------------------------------------------
    total_steps = dann_epochs * len(src_train_loader)
    global_step = 0

    os.makedirs(os.path.dirname(config["model_save_path"]),     exist_ok=True)
    os.makedirs(os.path.dirname(config["optimal_T_save_path"]), exist_ok=True)
    os.makedirs(os.path.dirname(config["calib_plot_train_path"]), exist_ok=True)

    logger.info(f"DANN training loop | epochs={dann_epochs} | total_steps={total_steps}")

    for epoch in range(dann_epochs):
        logger.info(f"--- Epoch {epoch + 1}/{dann_epochs} ---")
        model.train()

        # Cycle the target loader so it always has batches available,
        # even though target (~800 samples) << source (~105k samples).
        tgt_iter = cycle(tgt_train_loader)

        epoch_class_loss  = 0.0
        epoch_domain_loss = 0.0
        epoch_total_loss  = 0.0
        domain_correct    = 0
        domain_total      = 0

        for batch_idx, (src_imgs, src_labels, _) in enumerate(src_train_loader):
            # Fetch one target batch (cycles if exhausted)
            tgt_imgs, _, _ = next(tgt_iter)

            src_imgs   = src_imgs.to(device)
            src_imgs   = gpu_normalize(src_imgs)   # normalize only — CLAHE already applied offline
            src_labels = src_labels.to(device)
            tgt_imgs   = tgt_imgs.to(device)
            tgt_imgs   = gpu_normalize(tgt_imgs)   # normalize only — CLAHE already applied offline

            # Compute current λ
            p   = global_step / max(total_steps - 1, 1)
            lam = compute_lambda(p, config["dann_lambda_max"])

            optimizer.zero_grad()

            # ---- Forward: source ----
            src_class_logits, src_dom_logits = model(src_imgs, alpha=lam)

            # ---- Forward: target (only domain head matters) ----
            _, tgt_dom_logits = model(tgt_imgs, alpha=lam)

            # ---- Classification loss (source only) ----
            class_loss = criterion(src_class_logits, src_labels)

            # ---- Domain loss (source=0, target=1) ----
            src_dom_labels = torch.zeros(
                src_imgs.size(0), dtype=torch.long, device=device
            )
            tgt_dom_labels = torch.ones(
                tgt_imgs.size(0), dtype=torch.long, device=device
            )
            dom_logits = torch.cat([src_dom_logits, tgt_dom_logits], dim=0)
            dom_labels = torch.cat([src_dom_labels, tgt_dom_labels], dim=0)
            domain_loss = domain_criterion(dom_logits, dom_labels)

            # ---- Total loss ----
            # NOTE: lam is NOT reapplied here. The GRL already scales the
            # backbone-bound gradient by -lam (passed as `alpha=lam` above).
            # Multiplying domain_loss by lam again here would double-apply
            # the schedule for the backbone (lam^2 effective) while the
            # domain head itself only sees a single lam — an asymmetry not
            # in the original Ganin et al. (2016) formulation. Loss stays
            # unweighted; λ lives in the GRL only.
            total_loss = class_loss + domain_loss

            total_loss.backward()
            optimizer.step()
            global_step += 1

            # Accumulate for epoch logging
            epoch_class_loss  += class_loss.item()
            epoch_domain_loss += domain_loss.item()
            epoch_total_loss  += total_loss.item()

            # Domain discriminator accuracy (detached — just for monitoring)
            with torch.no_grad():
                dom_preds    = dom_logits.argmax(dim=1)
                domain_correct += (dom_preds == dom_labels).sum().item()
                domain_total   += dom_labels.size(0)

            if (batch_idx + 1) % 50 == 0:
                logger.debug(
                    f"  Batch {batch_idx + 1}/{len(src_train_loader)} "
                    f"| class_loss={class_loss.item():.4f} "
                    f"| domain_loss={domain_loss.item():.4f} "
                    f"| λ={lam:.4f}"
                )

        # ---- Epoch-level summary ----
        n_batches        = len(src_train_loader)
        avg_class_loss   = epoch_class_loss  / n_batches
        avg_domain_loss  = epoch_domain_loss / n_batches
        avg_total_loss   = epoch_total_loss  / n_batches
        domain_disc_acc  = domain_correct / domain_total if domain_total > 0 else 0.0

        logger.info(
            f"Epoch {epoch + 1} | "
            f"class_loss={avg_class_loss:.4f} | "
            f"domain_loss={avg_domain_loss:.4f} | "
            f"λ={lam:.4f} | "
            f"disc_acc={domain_disc_acc:.3f}"
        )

        # ---- Source validation ----
        val_loss, val_qwk, matrix, val_preds, val_labels_list = _evaluate_dann(
            model, src_val_loader, criterion, device
        )
        logger.info(
            f"Epoch {epoch + 1} | Val Loss={val_loss:.4f} | Val QWK={val_qwk:.4f}"
        )
        logger.info(f"Confusion Matrix:\n{matrix}")

        # ---- Lightweight MC Dropout (T=10) per epoch ----
        logger.debug(f"Epoch {epoch + 1} | MC Dropout (T=10)...")
        mean_p, unc, _, _ = _mc_evaluate_dann(model, src_val_loader, device, T=10)
        entropy, margin, mc_std = _compute_uncertainty_signals(mean_p, unc)
        uncertain_frac = float(
            (
                (entropy > UNCERTAINTY_ENTROPY_THRESHOLD) |
                (margin  < UNCERTAINTY_MARGIN_THRESHOLD)  |
                (mc_std  > UNCERTAINTY_MC_STD_THRESHOLD)
            ).mean()
        )

        logger.info(
            f"Epoch {epoch + 1} | MC uncertainty | "
            f"entropy={entropy.mean():.4f} | margin={margin.mean():.4f} "
            f"| uncertain_frac={uncertain_frac:.3f}"
        )

        wandb.log({
            "epoch":                epoch + 1,
            "lambda":               lam,
            "train_class_loss":     avg_class_loss,
            "train_domain_loss":    avg_domain_loss,
            "train_total_loss":     avg_total_loss,
            "domain_disc_acc":      domain_disc_acc,
            "src_val_loss":         val_loss,
            "src_val_qwk":          val_qwk,
            "mean_entropy":         float(entropy.mean()),
            "mean_margin":          float(margin.mean()),
            "mean_mc_uncertainty":  float(mc_std.mean()),
            "uncertain_fraction":   uncertain_frac,
            "learning_rate":        optimizer.param_groups[0]["lr"],
        })

    logger.info("DANN training loop complete.")

    # ------------------------------------------------------------------
    # 5a. Post-training full MC Dropout (T=30) on source val
    # ------------------------------------------------------------------
    logger.info(f"Post-training MC Dropout | T={config['mc_dropout_passes']}")
    all_mean_probs, all_uncertainties, all_labels_arr, all_logits = _mc_evaluate_dann(
        model, src_val_loader, device, T=config["mc_dropout_passes"]
    )

    # ------------------------------------------------------------------
    # 5b. Temperature scaling calibration
    # ------------------------------------------------------------------
    logger.info("Applying temperature calibration...")
    optimal_T        = find_temperature(all_logits, all_labels_arr)
    calibrated_probs = apply_temperature(all_logits, optimal_T)
    final_preds      = calibrated_probs.argmax(axis=1)

    # ------------------------------------------------------------------
    # 5c. Post-calibration uncertainty + triage
    # ------------------------------------------------------------------
    cal_entropy, cal_margin, cal_mc_unc = _compute_uncertainty_signals(
        calibrated_probs, all_uncertainties
    )

    logger.info("Triage Summary — post-calibration (first 20 samples):")
    for i in range(min(20, len(final_preds))):
        pred = final_preds[i]
        true = all_labels_arr[i]
        flag = triage_sample(pred, cal_entropy[i], cal_margin[i], cal_mc_unc[i])
        logger.info(
            f"  Sample {i:3d} | True:{true} Pred:{pred} "
            f"| H={cal_entropy[i]:.3f} M={cal_margin[i]:.3f} "
            f"MC={cal_mc_unc[i]:.3f} | {flag}"
        )

    cal_uncertain_mask = (
        (cal_entropy > UNCERTAINTY_ENTROPY_THRESHOLD) |
        (cal_margin  < UNCERTAINTY_MARGIN_THRESHOLD)  |
        (cal_mc_unc  > UNCERTAINTY_MC_STD_THRESHOLD)
    )
    correct_mask = (final_preds == all_labels_arr)

    logger.info(f"Uncertain fraction (calibrated): {cal_uncertain_mask.mean():.3f}")
    logger.info(f"Calibrated Mean entropy        : {cal_entropy.mean():.4f}")
    logger.info(f"Calibrated Mean margin         : {cal_margin.mean():.4f}")
    logger.info(f"Calibrated Mean MC std         : {cal_mc_unc.mean():.4f}")
    logger.info("--- Four Quadrant Uncertainty Breakdown ---")
    logger.info(f"  Certain + Wrong (dangerous): {(~cal_uncertain_mask & ~correct_mask).sum()}")
    logger.info(f"  Certain + Right (ideal)     : {(~cal_uncertain_mask &  correct_mask).sum()}")
    logger.info(f"  Uncertain + Wrong (caught)  : {( cal_uncertain_mask & ~correct_mask).sum()}")
    logger.info(f"  Uncertain + Right (over-ref): {( cal_uncertain_mask &  correct_mask).sum()}")

    # ------------------------------------------------------------------
    # 5d. Calibration plot (source val, post-temperature scaling)
    # ------------------------------------------------------------------
    calib_path = config["calib_plot_train_path"]
    per_class_calibration(calibrated_probs, all_labels_arr, save_path=calib_path)

    timestamp      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_calib   = os.path.join(
        os.path.dirname(calib_path), f"calibration_dann_{timestamp}.png"
    )
    per_class_calibration(calibrated_probs, all_labels_arr, save_path=backup_calib)

    final_qwk = cohen_kappa_score(all_labels_arr, final_preds, weights="quadratic")
    logger.info(f"Final calibrated source val QWK: {final_qwk:.4f}")

    wandb.log({
        "optimal_T":                    float(optimal_T),
        "final_src_val_qwk":            float(final_qwk),
        "final_mean_entropy":           float(cal_entropy.mean()),
        "final_mean_margin":            float(cal_margin.mean()),
        "final_mean_mc_uncertainty":    float(cal_mc_unc.mean()),
        "final_uncertain_fraction":     float(cal_uncertain_mask.mean()),
        "quadrant_certain_wrong":       int((~cal_uncertain_mask & ~correct_mask).sum()),
        "quadrant_certain_right":       int((~cal_uncertain_mask &  correct_mask).sum()),
        "quadrant_uncertain_wrong":     int(( cal_uncertain_mask & ~correct_mask).sum()),
        "quadrant_uncertain_right":     int(( cal_uncertain_mask &  correct_mask).sum()),
        "calibration_plot":             wandb.Image(calib_path),
    })

    # ------------------------------------------------------------------
    # 6. Save model + optimal_T
    # ------------------------------------------------------------------
    model_path = config["model_save_path"]
    T_path     = config["optimal_T_save_path"]

    torch.save(model.state_dict(), model_path)
    np.save(T_path, np.array(optimal_T))

    backup_model = os.path.join(
        os.path.dirname(model_path), f"dann_efficientnet_{timestamp}.pth"
    )
    backup_T = os.path.join(
        os.path.dirname(T_path), f"optimal_T_{timestamp}.npy"
    )
    torch.save(model.state_dict(), backup_model)
    np.save(backup_T, np.array(optimal_T))

    logger.info(f"Model saved     → {model_path}  (backup: {backup_model})")
    logger.info(f"Optimal T saved → {T_path}  (T={optimal_T:.4f})")

    # ------------------------------------------------------------------
    # 7. Per-target-dataset evaluation (zero-shot generalisation)
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Per-target evaluation — zero-shot generalisation")
    logger.info("=" * 60)

    for ds_name in config["target_datasets"]:
        logger.info(f"Evaluating on target: {ds_name}")
        tgt_eval_loader = build_dann_target_eval_loader(ds_name, config)

        tgt_val_loss, tgt_qwk, tgt_matrix, tgt_preds, tgt_labels = _evaluate_dann(
            model, tgt_eval_loader, criterion, device
        )
        logger.info(
            f"[{ds_name}] QWK={tgt_qwk:.4f} | Loss={tgt_val_loss:.4f}"
        )
        logger.info(f"[{ds_name}] Confusion Matrix:\n{tgt_matrix}")

        # Domain adaptation summary: compare domain disc acc trajectory
        wandb.log({
            f"target_{ds_name}_qwk":  tgt_qwk,
            f"target_{ds_name}_loss": tgt_val_loss,
        })

    # ------------------------------------------------------------------
    # 8. Save per-class centroids (for Cosine Similarity / UMAP analysis)
    # ------------------------------------------------------------------
    logger.info("Computing per-class feature centroids from source val set...")
    model.eval()
    train_features    = []
    train_labels_list = []

    with torch.no_grad():
        for images, labels in src_val_loader:
            images = images.to(device)
            images = gpu_normalize(images)   # normalize only — CLAHE already applied offline
            feats  = model.get_features(images)   # [B, 1280]
            train_features.append(feats.cpu().numpy())
            train_labels_list.extend(labels.numpy())

    train_features   = np.vstack(train_features)
    train_labels_arr = np.array(train_labels_list)
    num_classes      = config["num_classes"]
    class_names      = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]

    per_class_mean = {}
    for c in range(num_classes):
        class_feats        = train_features[train_labels_arr == c]
        per_class_mean[c]  = class_feats.mean(axis=0)
        logger.info(
            f"  Class {c} ({class_names[c]}): {class_feats.shape[0]} samples | "
            f"mean norm={np.linalg.norm(per_class_mean[c]):.4f}"
        )

    centroid_path = config.get(
        "class_centroids_save_path", "artifacts/dann/centroids/mean.npy"
    )
    os.makedirs(os.path.dirname(centroid_path), exist_ok=True)
    np.save(centroid_path, np.array(per_class_mean, dtype=object))
    logger.info(f"Per-class centroids saved → {centroid_path}")

    wandb.finish()
    logger.info("train_dann() complete.")

    return optimal_T
