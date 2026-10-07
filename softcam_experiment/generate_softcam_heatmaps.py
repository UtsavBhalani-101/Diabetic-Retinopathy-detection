"""Generate SoftCAM heatmaps for the IDRiD official test split in a single forward pass."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from pipeline.data.gpu_transforms import gpu_normalize
from softcam_experiment.common import (
    add_common_args,
    build_idrid_test_dataset,
    build_loader,
    initialize,
    load_softcam_model,
    project_root,
)


def colorize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    """Apply a smooth colormap without adding matplotlib as a runtime dependency."""
    h = np.clip(heatmap.astype(np.float32), 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * h - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * h - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * h - 1.0), 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def save_overlay(image_tensor: torch.Tensor, heatmap: np.ndarray, path: Path) -> None:
    """Save a blended heatmap overlay for visual sanity checks."""
    image = image_tensor.detach().cpu().permute(1, 2, 0).numpy()
    image = np.clip(image, 0.0, 1.0)
    heatmap_rgb = colorize_heatmap(heatmap)
    overlay = np.clip((0.45 * image) + (0.55 * heatmap_rgb), 0.0, 1.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((overlay * 255).astype(np.uint8)).save(path)


def save_heatmap_png(heatmap: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    heatmap_rgb = colorize_heatmap(heatmap)
    Image.fromarray((heatmap_rgb * 255).astype(np.uint8)).save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--heatmap-dir",
        default="artifacts/softcam_heatmaps/idrid/test",
        help="Directory for {idx}.npy heatmaps.",
    )
    parser.add_argument(
        "--overlay-dir",
        default="artifacts/softcam_overlays/idrid/test",
        help="Directory for visual overlay PNGs.",
    )
    parser.add_argument(
        "--heatmap-png-dir",
        default="artifacts/softcam_heatmap_pngs/idrid/test",
        help="Directory for standalone heatmap PNG previews.",
    )
    parser.add_argument(
        "--overlay-limit",
        type=int,
        default=24,
        help="Maximum number of visual overlays to save.",
    )
    parser.add_argument(
        "--class-source",
        choices=["predicted", "true"],
        default="predicted",
        help="Class whose SoftCAM evidence map is saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = initialize(args)
    root = project_root()

    heatmap_dir = root / args.heatmap_dir
    overlay_dir = root / args.overlay_dir
    heatmap_png_dir = root / args.heatmap_png_dir
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    heatmap_png_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_idrid_test_dataset(args)
    loader = build_loader(dataset, args)
    model = load_softcam_model(args, device)
    model.eval()

    offset = 0
    metadata_rows = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            normalized_images = gpu_normalize(images)
            class_indices = labels if args.class_source == "true" else None

            # SoftCAM extracts heatmaps and predictions in a single forward pass
            heatmaps, logits, selected_classes = model.heatmaps(
                normalized_images,
                class_indices=class_indices,
                output_size=(images.shape[-2], images.shape[-1]),
            )
            preds = logits.argmax(dim=1)

            for batch_idx in range(images.shape[0]):
                global_idx = offset + batch_idx
                heatmap = heatmaps[batch_idx].detach().cpu().numpy().astype(np.float32)
                np.save(heatmap_dir / f"{global_idx}.npy", heatmap)

                if global_idx < args.overlay_limit:
                    save_overlay(
                        images[batch_idx],
                        heatmap,
                        overlay_dir / f"{global_idx}.png",
                    )
                    save_heatmap_png(
                        heatmap,
                        heatmap_png_dir / f"{global_idx}.png",
                    )

                metadata_rows.append(
                    {
                        "idx": global_idx,
                        "true": int(labels[batch_idx].item()),
                        "pred": int(preds[batch_idx].item()),
                        "mapped_class": int(selected_classes[batch_idx].item()),
                        "heatmap_min": float(heatmap.min()),
                        "heatmap_max": float(heatmap.max()),
                        "heatmap_mean": float(heatmap.mean()),
                        "heatmap_std": float(heatmap.std()),
                    }
                )
            offset += images.shape[0]

    metadata_path = heatmap_dir / "metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "idx",
                "true",
                "pred",
                "mapped_class",
                "heatmap_min",
                "heatmap_max",
                "heatmap_mean",
                "heatmap_std",
            ],
        )
        writer.writeheader()
        writer.writerows(metadata_rows)

    print(f"Saved {offset} SoftCAM heatmaps to {heatmap_dir}")
    print(f"Saved overlay previews to {overlay_dir}")
    print(f"Saved heatmap PNG previews to {heatmap_png_dir}")
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
