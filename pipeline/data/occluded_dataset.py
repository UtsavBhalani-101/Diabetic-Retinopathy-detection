# pipeline/data/occluded_dataset.py
# ============================================================
# Two occlusion dataset wrappers for the GradCAM experiment:
#
#   OccludedDataset       — GradCAM-guided occlusion
#     Masks the top-K% highest-activation pixels from a pre-saved
#     GradCAM++ heatmap.  Used for Passes 2 & 3.
#
#   RandomOccludedDataset — Random occlusion (control baseline)
#     Masks a randomly selected K% of pixels with no relation to
#     the model's attention.  Used for Passes 4 & 5.
#     Comparing these to GradCAM passes at the same K tells you
#     whether GradCAM is highlighting truly important regions:
#       - GradCAM drop >> Random drop  →  GradCAM hotspot is meaningful
#       - GradCAM drop ≈  Random drop  →  model is using distributed features
#
# Masking logic (both classes)
# ----------------------------
# Occluded pixels are set to 0.0 in the already-normalized tensor.
# In the ImageNet-normalized space (mean=[0.485,0.456,0.406],
# std=[0.229,0.224,0.225]), 0.0 corresponds to replacing the pixel
# with a value approximately equal to the dataset mean — the least
# disruptive substitution for pixels the model should ignore.
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


class RandomOccludedDataset(Dataset):
    """
    Wraps a RetinopathyDataset (or any Dataset returning (image_tensor, label))
    and applies a **randomly selected** K% pixel occlusion mask per image.

    This serves as the control baseline for the GradCAM occlusion experiment.
    By occluding the same fraction of pixels as the GradCAM passes but at
    random locations, we can measure whether GradCAM-guided occlusion causes
    a disproportionately large confidence drop — which would confirm that
    GradCAM is highlighting genuinely important regions rather than just any
    arbitrary region.

    Key interpretation:
      - GradCAM drop >> Random drop  →  hotspot is meaningfully important
      - GradCAM drop ≈  Random drop  →  model relies on distributed features;
                                        GradCAM may be less reliable as an
                                        explanation

    Parameters
    ----------
    base_dataset : Dataset
        The underlying dataset.  Must return (image_tensor, label) where
        image_tensor is shape (C, H, W) and already normalized.
    top_k_percent : float
        Percentage of pixels to occlude at random.
        E.g. 10 → randomly select and zero out 10% of all pixels.
    base_seed : int, optional
        Base random seed.  The actual seed used per image is
        ``base_seed + idx`` so the mask for each image is deterministic
        across runs but different for every image.  Default: 42.
    """

    def __init__(self, base_dataset: Dataset, top_k_percent: float,
                 base_seed: int = 42):
        if not 0 < top_k_percent < 100:
            raise ValueError(
                f"top_k_percent must be in (0, 100), got {top_k_percent}"
            )

        self.base_dataset  = base_dataset
        self.top_k_percent = top_k_percent
        self.base_seed     = base_seed

        logger.info(
            f"RandomOccludedDataset | n={len(base_dataset)} | "
            f"top_k={top_k_percent}% | base_seed={base_seed}"
        )

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        # 1. Get the normalized image tensor and label
        image, label = self.base_dataset[idx]   # (C, H, W)

        _, H, W = image.shape
        n_pixels = H * W

        # 2. Compute how many pixels to occlude
        n_occlude = int(round(n_pixels * self.top_k_percent / 100.0))

        # 3. Pick random pixel positions — deterministic per image via idx seed
        #    Using base_seed + idx ensures:
        #      · Same mask every time this image is loaded (reproducible)
        #      · Different mask for every image (no spatial bias)
        rng = np.random.default_rng(seed=self.base_seed + idx)
        flat_indices = rng.choice(n_pixels, size=n_occlude, replace=False)

        # 4. Convert flat indices to (row, col) positions
        rows = flat_indices // W
        cols = flat_indices  % W

        # 5. Zero out all 3 channels at those locations
        image = image.clone()
        image[:, rows, cols] = 0.0   # 0.0 in normalized space ≈ ImageNet mean

        return image, label
