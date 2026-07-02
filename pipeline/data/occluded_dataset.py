# pipeline/data/occluded_dataset.py
# ============================================================
# OccludedDataset:
#   Wraps an existing RetinopathyDataset and applies a GradCAM-based
#   percentile occlusion mask to each image tensor before returning it.
#
#   Used by the 3-pass occlusion experiment:
#     Pass 1  — base RetinopathyDataset (no occlusion)
#     Pass 2  — OccludedDataset(base, top_k_percent=10)
#     Pass 3  — OccludedDataset(base, top_k_percent=30)
#
# Masking logic
# -------------
# For a given image index `idx`:
#   1. Load the pre-saved GradCAM++ heatmap: {heatmap_dir}/{idx}.npy
#      Shape: (H, W), float32 in [0, 1]  (H=W=224 for EfficientNet-B0)
#   2. Compute threshold = np.percentile(heatmap, 100 - top_k_percent)
#      e.g. top_k_percent=10 → threshold = 90th percentile value
#   3. Build boolean mask: pixels where heatmap >= threshold  → True
#      This guarantees EXACTLY top_k_percent % of pixels are masked.
#   4. Zero out those pixel positions in all 3 channels of the
#      already-normalized image tensor.
#      Setting values to 0.0 in normalized space = replacing with the
#      ImageNet mean colour (since val_transformer normalises with
#      mean=[0.485, 0.456, 0.406]), which is the least disruptive
#      replacement for pixels the model won't use.
# ============================================================

import logging
import os

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class OccludedDataset(Dataset):
    """
    Wraps a RetinopathyDataset (or any Dataset returning (image_tensor, label))
    and applies a GradCAM-based top-K percentile occlusion mask per image.

    Parameters
    ----------
    base_dataset : Dataset
        The underlying dataset.  Must return (image_tensor, label) where
        image_tensor is shape (C, H, W) and already normalized.
    heatmap_dir : str
        Directory containing pre-saved .npy heatmap files named by index:
        ``{heatmap_dir}/{idx}.npy``  (e.g. ``artifacts/gradcam_heatmaps/idrid/train/0.npy``)
    top_k_percent : float
        Percentage of pixels to occlude (ranked from highest activation).
        E.g. 10 → occlude the top 10% brightest pixels in the heatmap.
    """

    def __init__(self, base_dataset: Dataset, heatmap_dir: str,
                 top_k_percent: float):
        if not 0 < top_k_percent < 100:
            raise ValueError(
                f"top_k_percent must be in (0, 100), got {top_k_percent}"
            )
        if not os.path.isdir(heatmap_dir):
            raise FileNotFoundError(
                f"Heatmap directory not found: {heatmap_dir}. "
                f"Run run_idrid_gradcam.py with --heatmap-dir first."
            )

        self.base_dataset  = base_dataset
        self.heatmap_dir   = heatmap_dir
        self.top_k_percent = top_k_percent
        self._percentile   = 100.0 - top_k_percent  # e.g. 10% → 90th percentile

        logger.info(
            f"OccludedDataset | n={len(base_dataset)} | "
            f"top_k={top_k_percent}% | threshold=p{self._percentile:.0f} | "
            f"heatmap_dir={heatmap_dir}"
        )

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        # 1. Get the normalized image tensor and label from the base dataset
        image, label = self.base_dataset[idx]          # (C, H, W), label

        # 2. Load the corresponding GradCAM++ heatmap
        npy_path = os.path.join(self.heatmap_dir, f"{idx}.npy")
        if not os.path.isfile(npy_path):
            raise FileNotFoundError(
                f"Heatmap not found for index {idx}: {npy_path}. "
                f"Make sure run_idrid_gradcam.py was run with --heatmap-dir."
            )
        heatmap = np.load(npy_path).astype(np.float32)  # (H, W) in [0, 1]

        # 3. Compute per-image threshold so exactly top_k_percent% are masked
        threshold = float(np.percentile(heatmap, self._percentile))

        # 4. Build boolean mask: True where heatmap >= threshold (hot pixels)
        mask = torch.from_numpy(heatmap >= threshold)   # (H, W) bool

        # 5. Zero out all 3 channels at hot pixel locations
        #    image shape: (C, H, W)  — clone to avoid mutating cached tensors
        image = image.clone()
        image[:, mask] = 0.0   # 0.0 in normalized space ≈ ImageNet mean colour

        return image, label
