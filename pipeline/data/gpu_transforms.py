# pipeline/data/gpu_transforms.py
# ============================================================
# GPU-accelerated normalization.
#
# CLAHE is applied OFFLINE ONCE via pipeline/dann/preprocess_clahe.py.
# RetinopathyDataset auto-detects the preprocessed directory
# (clahe_image_path) and reads pre-processed images directly.
#
# This function therefore does ONLY ImageNet normalization on GPU —
# it must NOT re-apply CLAHE, or images get equalized twice
# (over-contrasted, degraded, effectively different distribution
# than what preprocess_clahe.py produced and validated).
#
# If clahe_image_path was NOT found for a given dataset (see the
# warning logged by RetinopathyDataset), CLAHEPreprocess in
# pipeline/data/dataset.py's transform pipeline is the correct
# fallback — NOT this function. Do not add CLAHE back here unless
# you also remove the offline preprocessing step entirely.
#
# Usage in training loop
# ----------------------
#   from pipeline.data.gpu_transforms import gpu_normalize
#
#   imgs = imgs.to(device)
#   imgs = gpu_normalize(imgs)
#   logits = model(imgs)
# ============================================================

import torch

# ImageNet mean/std — same values as the old torchvision.Normalize
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225])

# Cache per-device tensors so we don't call .to(device) on every forward step
_mean_cache: dict = {}
_std_cache:  dict = {}


def _get_norm_params(device: torch.device):
    """Return [1,3,1,1] mean and std tensors on `device`, created once and cached."""
    if device not in _mean_cache:
        _mean_cache[device] = _IMAGENET_MEAN.to(device).view(1, 3, 1, 1)
        _std_cache[device]  = _IMAGENET_STD.to(device).view(1, 3, 1, 1)
    return _mean_cache[device], _std_cache[device]


def gpu_normalize(imgs: torch.Tensor) -> torch.Tensor:
    """
    ImageNet-normalize a batch on GPU. CLAHE is NOT applied here —
    it is handled offline by preprocess_clahe.py before training starts.

    Parameters
    ----------
    imgs : torch.Tensor
           Shape [B, C, H, W], dtype float32, values in [0.0, 1.0].
           This is exactly what torchvision.transforms.ToTensor produces.

    Returns
    -------
    torch.Tensor
        Shape [B, C, H, W], float32, ImageNet-normalized. Ready for model input.
    """
    mean, std = _get_norm_params(imgs.device)
    return (imgs - mean) / std
