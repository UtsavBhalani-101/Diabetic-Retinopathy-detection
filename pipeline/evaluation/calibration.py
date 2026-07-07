# pipeline/calibration.py
# ============================================================
# Post-hoc calibration:
#   - per_class_calibration : reliability diagrams + ECE per class
#   - find_temperature      : optimize T to minimise ECE on val logits
#   - apply_temperature     : scale logits by T and return softmax probs
# ============================================================

import logging
import os

import numpy as np
import matplotlib.pyplot as plt

from pipeline.setup.config import (
    UNCERTAINTY_ENTROPY_THRESHOLD,
    UNCERTAINTY_MARGIN_THRESHOLD,
    UNCERTAINTY_MC_STD_THRESHOLD,
)

logger = logging.getLogger(__name__)


def per_class_calibration(mean_probs: np.ndarray, labels: np.ndarray,
                           n_classes: int = 5,
                           save_path: str = "artifacts/calibration.png") -> None:
    """
    Plot one reliability diagram per class and print ECE for each.
    A perfectly calibrated model's bars sit on the diagonal dashed line.
    ECE = weighted mean of |confidence − accuracy| across bins.
    """
    logger.info(f"per_class_calibration() | n_classes={n_classes} | save_path={save_path}")
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

    fig, axes = plt.subplots(1, n_classes, figsize=(4 * n_classes, 4))
    class_names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]

    for c in range(n_classes):
        probs_c       = mean_probs[:, c]
        binary_labels = (labels == c).astype(int)

        n_bins     = 10
        bin_edges  = np.linspace(0, 1, n_bins + 1)
        bin_acc    = []
        bin_conf   = []
        bin_counts = []

        for i in range(n_bins):
            mask = (probs_c >= bin_edges[i]) & (probs_c < bin_edges[i + 1])
            if mask.sum() > 0:
                bin_acc.append(binary_labels[mask].mean())
                bin_conf.append(probs_c[mask].mean())
                bin_counts.append(mask.sum())

        bin_acc    = np.array(bin_acc)
        bin_conf   = np.array(bin_conf)
        bin_counts = np.array(bin_counts)

        ece = (
            np.sum(bin_counts * np.abs(bin_conf - bin_acc)) / bin_counts.sum()
            if bin_counts.sum() > 0 else 0.0
        )

        ax = axes[c]
        ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
        ax.bar(bin_conf, bin_acc, width=0.08, alpha=0.7, label="Model")
        ax.set_title(f"{class_names[c]}\nECE = {ece:.3f}")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7)

        logger.info(f"  Class {c} ({class_names[c]}) ECE: {ece:.4f}")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)   # avoid display blocking in non-interactive environments
    logger.info(f"Calibration plot saved → {save_path}")


def find_temperature(logits: np.ndarray, labels: np.ndarray,
                     n_classes: int = 5) -> float:
    """
    Find the scalar temperature T that minimises the average ECE across
    all classes on the validation set.

    Uses scipy's bounded scalar optimiser over T in [0.1, 10.0].

    Parameters
    ----------
    logits   : [N, C]  mean logits from MC Dropout passes
    labels   : [N]     ground-truth integer labels

    Returns
    -------
    optimal_T : float
    """
    from scipy.optimize import minimize_scalar
    from scipy.special import softmax as scipy_softmax

    logger.info("find_temperature() | searching optimal T in [0.1, 10.0]...")

    def ece_given_T(T: float) -> float:
        scaled_probs = scipy_softmax(logits / T, axis=1)
        total_ece    = 0.0

        for c in range(n_classes):
            probs_c       = scaled_probs[:, c]
            binary_labels = (labels == c).astype(int)

            n_bins     = 10
            bin_edges  = np.linspace(0, 1, n_bins + 1)
            bin_counts = []
            bin_gaps   = []

            for i in range(n_bins):
                mask = (probs_c >= bin_edges[i]) & (probs_c < bin_edges[i + 1])
                if mask.sum() > 0:
                    bin_counts.append(mask.sum())
                    bin_gaps.append(abs(probs_c[mask].mean() - binary_labels[mask].mean()))

            if bin_counts:
                bc = np.array(bin_counts)
                bg = np.array(bin_gaps)
                total_ece += np.sum(bc * bg) / bc.sum()

        return total_ece / n_classes

    result    = minimize_scalar(ece_given_T, bounds=(0.1, 10.0), method="bounded")
    optimal_T = result.x

    logger.info(f"find_temperature() | optimal T = {optimal_T:.4f}")
    logger.info(f"find_temperature() | ECE before scaling (T=1.0): {ece_given_T(1.0):.4f}")
    logger.info(f"find_temperature() | ECE after  scaling (T={optimal_T:.2f}): {ece_given_T(optimal_T):.4f}")
    return optimal_T


def apply_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    """
    Divide logits by T and apply softmax to get calibrated probabilities.

    Parameters
    ----------
    logits : [N, C]  mean logits from MC Dropout
    T      : float   temperature (> 1 softens, < 1 sharpens)

    Returns
    -------
    calibrated_probs : [N, C]
    """
    from scipy.special import softmax as scipy_softmax
    calibrated = scipy_softmax(logits / T, axis=1)
    logger.debug(f"apply_temperature() | T={T:.4f} applied to {logits.shape[0]} samples")
    return calibrated


def triage_sample(pred: int, entropy: float, margin: float, mc_std: float,
                  entropy_thresh: float = UNCERTAINTY_ENTROPY_THRESHOLD,
                  margin_thresh: float = UNCERTAINTY_MARGIN_THRESHOLD,
                  mc_std_thresh: float = UNCERTAINTY_MC_STD_THRESHOLD) -> str:
    """
    Categorize a single sample prediction as UNCERTAIN, HIGH SEVERITY, or ROUTINE.

    Parameters
    ----------
    pred : int
        Predicted class index (0-4).
    entropy : float
        Predictive entropy of the softmax probabilities.
    margin : float
        Top-2 probability gap.
    mc_std : float
        Mean MC standard deviation across passes.
    entropy_thresh : float
        Threshold above which a sample is flagged as uncertain.
    margin_thresh : float
        Threshold below which a sample is flagged as uncertain.
    mc_std_thresh : float
        Threshold above which a sample is flagged as uncertain.

    Returns
    -------
    triage_label : str
        One of:
        - "UNCERTAIN - refer to specialist"
        - "HIGH SEVERITY - urgent review"
        - "ROUTINE"
    """
    if entropy > entropy_thresh or margin < margin_thresh or mc_std > mc_std_thresh:
        return "UNCERTAIN - refer to specialist"
    elif pred >= 3:
        return "HIGH SEVERITY - urgent review"
    else:
        return "ROUTINE"


if __name__ == "__main__":
    from pipeline.setup.config import setup_logging
    setup_logging(level=logging.DEBUG)

    logger.info("Starting synthetic test for calibration.py...")
    
    np.random.seed(42)
    N = 1000
    C = 5

    # 1. Generate synthetic ground-truth labels
    labels = np.random.randint(0, C, size=N)

    # 2. Generate uncalibrated logits
    # One-hot true label targets, scaled and offset with noise to simulate model output
    one_hot = np.eye(C)[labels]
    logits = one_hot * 1.5 + np.random.randn(N, C) * 0.8

    # 3. Compute uncalibrated softmax probabilities
    from scipy.special import softmax as scipy_softmax
    mean_probs = scipy_softmax(logits, axis=1)

    logger.info(f"Generated synthetic data: logits shape {logits.shape}, labels shape {labels.shape}")

    # 4. Find optimal temperature T
    optimal_T = find_temperature(logits, labels, n_classes=C)
    logger.info(f"Found optimal Temperature: {optimal_T:.4f}")

    # 5. Apply temperature scaling
    calibrated_probs = apply_temperature(logits, optimal_T)
    logger.info(f"Calibrated probabilities shape: {calibrated_probs.shape}")

    # 6. Generate per-class calibration plots (uncalibrated vs calibrated)
    per_class_calibration(mean_probs, labels, n_classes=C, save_path="artifacts/test_calibration_uncalibrated.png")
    per_class_calibration(calibrated_probs, labels, n_classes=C, save_path="artifacts/test_calibration_calibrated.png")

    # 7. Test triage_sample function
    # Test cases:
    # - Case 1: Low uncertainty, low predicted class index (Routine)
    # - Case 2: High uncertainty (e.g. high entropy), low predicted class (Uncertain)
    # - Case 3: Low uncertainty, high predicted class index >= 3 (High Severity)
    # - Case 4: High uncertainty (e.g. high MC std), high predicted class (Uncertain)
    t_routine = triage_sample(pred=1, entropy=0.2, margin=0.8, mc_std=0.01)
    t_uncertain_ent = triage_sample(pred=1, entropy=1.5, margin=0.1, mc_std=0.01)
    t_high_sev = triage_sample(pred=4, entropy=0.1, margin=0.9, mc_std=0.02)
    t_uncertain_std = triage_sample(pred=4, entropy=0.2, margin=0.8, mc_std=0.09)

    logger.info("Triage sample tests:")
    logger.info(f"  - Predict class 1, Low uncertainty -> Expected: ROUTINE, Got: {t_routine}")
    logger.info(f"  - Predict class 1, High entropy -> Expected: UNCERTAIN - refer to specialist, Got: {t_uncertain_ent}")
    logger.info(f"  - Predict class 4, High MC std -> Expected: UNCERTAIN - refer to specialist, Got: {t_uncertain_std}")
    logger.info(f"  - Predict class 4, Low uncertainty -> Expected: HIGH SEVERITY - urgent review, Got: {t_high_sev}")
    
    logger.info("calibration.py tests completed successfully!")


