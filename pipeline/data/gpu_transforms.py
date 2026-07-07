# pipeline/data/gpu_transforms.py
# ============================================================
# GPU-accelerated CLAHE + normalization using Kornia.
#
# Replaces the old CPU pipeline:
#   CLAHEPreprocess (OpenCV, in DataLoader worker)
#   → ToTensor
#   → torchvision.Normalize
#
# New CPU pipeline (lightweight, workers stay fast):
#   Resize → augmentations → ToTensor → [0, 1] float32
#
# New GPU step (called once per batch after .to(device)):
#   kornia.enhance.equalize_clahe → (x - mean) / std
#
# Usage in training loop
# ----------------------
#   from pipeline.data.gpu_transforms import gpu_clahe_normalize
#
#   imgs = imgs.to(device)
#   imgs = gpu_clahe_normalize(imgs)   # ← CLAHE + normalize on GPU
#   logits = model(imgs)
#
# Why this solves the bottleneck
# --------------------------------
# Before: 4 CPU workers each ran OpenCV CLAHE (~80–200 ms/image).
#         CPU was at 100%, GPU was starved and idle.
# After:  CPU workers only Resize + augment + ToTensor (~5–10 ms/image).
#         GPU runs Kornia CLAHE on the whole [B, C, H, W] batch in one
#         CUDA kernel (~3–8 ms total for the batch), then immediately
#         runs the forward pass. CPU and GPU now work in parallel.
# ============================================================

import torch
import kornia.enhance

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


def gpu_clahe_normalize(
    imgs: torch.Tensor,
    clip_limit: float = 40.0,
    grid_size: tuple  = (8, 8),
) -> torch.Tensor:
    """
    Apply CLAHE on GPU via Kornia, then ImageNet-normalize the batch.

    Call this once per batch right after .to(device) and before model forward.

    Parameters
    ----------
    imgs       : torch.Tensor
                 Shape [B, C, H, W], dtype float32, values in [0.0, 1.0].
                 This is exactly what torchvision.transforms.ToTensor produces.
    clip_limit : float
                 Kornia CLAHE clip limit (default 40.0). This is Kornia's scale —
                 visually equivalent to OpenCV clipLimit=2.0 on retinal images.
    grid_size  : tuple (rows, cols)
                 CLAHE tile grid. (8, 8) matches the old CLAHEPreprocess setting.

    Returns
    -------
    torch.Tensor
        Shape [B, C, H, W], float32, ImageNet-normalized. Ready for model input.
    """
    # ── Step 1: CLAHE on GPU ────────────────────────────────────────────────
    # equalize_clahe works on [0, 1] float tensors.
    # slow_and_differentiable=False → fast, non-differentiable CUDA path.
    # We don't need gradients through CLAHE (it's preprocessing, not a learned op).
    imgs = kornia.enhance.equalize_clahe(
        imgs,
        clip_limit=clip_limit,
        grid_size=grid_size,
        slow_and_differentiable=False,
    )

    # ── Step 2: ImageNet normalization — (x - mean) / std ──────────────────
    # Same operation as torchvision.Normalize but applied on-GPU in a single
    # vectorised op across the whole batch.
    mean, std = _get_norm_params(imgs.device)
    return (imgs - mean) / std
