"""Standalone occlusion datasets for raw [0, 1] image tensors."""

from __future__ import annotations

import os

import numpy as np
import torch
from torch.utils.data import Dataset


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)


def _mask_with_imagenet_mean(image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Replace selected raw pixels with ImageNet mean color."""
    image = image.clone()
    fill = IMAGENET_MEAN.to(image.device).view(3, 1)
    image[:, mask] = fill
    return image


class HeatmapOccludedDataset(Dataset):
    """Occlude top-k heatmap pixels in a base dataset returning raw tensors."""

    def __init__(self, base_dataset: Dataset, heatmap_dir: str, top_k_percent: float):
        if not 0 < top_k_percent < 100:
            raise ValueError(f"top_k_percent must be in (0, 100), got {top_k_percent}")
        if not os.path.isdir(heatmap_dir):
            raise FileNotFoundError(f"Heatmap directory not found: {heatmap_dir}")
        self.base_dataset = base_dataset
        self.heatmap_dir = heatmap_dir
        self.top_k_percent = top_k_percent
        self._percentile = 100.0 - top_k_percent

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        image, label = self.base_dataset[idx]
        npy_path = os.path.join(self.heatmap_dir, f"{idx}.npy")
        if not os.path.isfile(npy_path):
            raise FileNotFoundError(f"Heatmap not found for index {idx}: {npy_path}")

        heatmap = np.load(npy_path).astype(np.float32)
        threshold = float(np.percentile(heatmap, self._percentile))
        mask = torch.from_numpy(heatmap >= threshold)
        return _mask_with_imagenet_mean(image, mask), label


class RandomMeanOccludedDataset(Dataset):
    """Randomly occlude k percent of raw pixels with ImageNet mean color."""

    def __init__(self, base_dataset: Dataset, top_k_percent: float, base_seed: int = 42):
        if not 0 < top_k_percent < 100:
            raise ValueError(f"top_k_percent must be in (0, 100), got {top_k_percent}")
        self.base_dataset = base_dataset
        self.top_k_percent = top_k_percent
        self.base_seed = base_seed

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        image, label = self.base_dataset[idx]
        _, height, width = image.shape
        n_pixels = height * width
        n_occlude = int(round(n_pixels * self.top_k_percent / 100.0))

        rng = np.random.default_rng(seed=self.base_seed + idx)
        flat_indices = rng.choice(n_pixels, size=n_occlude, replace=False)
        rows = flat_indices // width
        cols = flat_indices % width

        mask = torch.zeros((height, width), dtype=torch.bool)
        mask[rows, cols] = True
        return _mask_with_imagenet_mean(image, mask), label
