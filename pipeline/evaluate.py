# pipeline/evaluate.py
# ============================================================
# Inference and uncertainty measurement:
#   - evaluate                  : standard val loop (loss + QWK + confusion matrix)
#   - mc_evaluate_full          : MC Dropout forward passes → probs + logits
#   - compute_uncertainty_signals: entropy, margin, MC std from MC output
# ============================================================

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import cohen_kappa_score, confusion_matrix


def evaluate(model, loader, criterion, device):
    """
    Standard validation loop.

    Returns
    -------
    epoch_loss  : float
    qwk         : float   (quadratic weighted kappa)
    matrix      : ndarray (confusion matrix)
    all_preds   : list[int]
    all_labels  : list[int]
    """
    model.eval()
    all_preds  = []
    all_labels = []
    running_loss = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            output_logits = model(images)
            loss = criterion(output_logits, labels)
            running_loss += loss.item()

            preds = torch.argmax(output_logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader)
    qwk    = cohen_kappa_score(all_labels, all_preds, weights="quadratic")
    matrix = confusion_matrix(all_labels, all_preds)

    print(f"Val Loss: {epoch_loss:.4f} | Val QWK: {qwk:.4f}")
    return epoch_loss, qwk, matrix, all_preds, all_labels


def mc_evaluate_full(model, loader, device, T: int = 30):
    """
    MC Dropout forward passes.

    Sets the model to eval() but forces all Dropout layers to train() so
    weights are randomly dropped on each of the T passes.  Returns both
    averaged softmax probs and averaged logits (needed for temperature scaling).

    Returns
    -------
    all_mean_probs    : ndarray [N, C]
    all_uncertainties : ndarray [N]   (mean std across classes)
    all_labels        : ndarray [N]
    all_logits        : ndarray [N, C] (averaged raw logits across T passes)
    """
    all_mean_probs    = []
    all_uncertainties = []
    all_labels        = []
    all_logits        = []

    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()   # keep dropout active during inference

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            passes       = []
            logit_passes = []

            for _ in range(T):
                logits = model(images)
                probs  = torch.softmax(logits, dim=1).cpu().numpy()
                passes.append(probs)
                logit_passes.append(logits.cpu().numpy())

            passes       = np.array(passes)        # [T, batch, C]
            logit_passes = np.array(logit_passes)  # [T, batch, C]

            mean_probs   = passes.mean(axis=0)
            mean_logits  = logit_passes.mean(axis=0)
            uncertainty  = passes.std(axis=0).mean(axis=1)

            all_mean_probs.append(mean_probs)
            all_uncertainties.append(uncertainty)
            all_labels.extend(labels.numpy())
            all_logits.append(mean_logits)

    all_mean_probs    = np.vstack(all_mean_probs)
    all_uncertainties = np.concatenate(all_uncertainties)
    all_labels        = np.array(all_labels)
    all_logits        = np.vstack(all_logits)   # [N, C]

    return all_mean_probs, all_uncertainties, all_labels, all_logits


def compute_uncertainty_signals(mean_probs: np.ndarray,
                                uncertainties: np.ndarray):
    """
    Derive three complementary uncertainty signals from MC Dropout output.

    Parameters
    ----------
    mean_probs    : [N, C]  averaged softmax probs across T passes
    uncertainties : [N]     per-sample mean std across classes and passes

    Returns
    -------
    entropy       : [N]  predictive entropy  (higher → more uncertain)
    margin        : [N]  top-2 probability gap (lower → more uncertain)
    mc_uncertainty: [N]  raw MC std passed through unchanged
    """
    eps     = 1e-8
    entropy = -np.sum(mean_probs * np.log(mean_probs + eps), axis=1)

    sorted_probs  = np.sort(mean_probs, axis=1)[:, ::-1]
    margin        = sorted_probs[:, 0] - sorted_probs[:, 1]

    mc_uncertainty = uncertainties  # already per-sample mean std

    return entropy, margin, mc_uncertainty
