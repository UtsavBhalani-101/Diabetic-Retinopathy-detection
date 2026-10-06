"""Generate TTECAM heatmaps for the IDRiD official test split."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ttecam_experiment.common import (
    add_common_args,
    build_idrid_test_dataset,
    build_loader,
    initialize,
    load_model,
    project_root,
)
from ttecam_experiment.ttecam import TTECAM


def save_overlay(image_tensor: torch.Tensor, heatmap: np.ndarray, path: Path) -> None:
    """Save a simple red heatmap overlay for visual sanity checks."""
    image = image_tensor.detach().cpu().permute(1, 2, 0).numpy()
    image = np.clip(image, 0.0, 1.0)
    red = np.zeros_like(image)
    red[..., 0] = heatmap
    overlay = np.clip((0.65 * image) + (0.35 * red), 0.0, 1.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((overlay * 255).astype(np.uint8)).save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--heatmap-dir",
        default="artifacts/ttecam_heatmaps/idrid/test",
        help="Directory for {idx}.npy heatmaps.",
    )
    parser.add_argument(
        "--overlay-dir",
        default="artifacts/ttecam_overlays/idrid/test",
        help="Directory for visual overlay PNGs.",
    )
    parser.add_argument(
        "--overlay-limit",
        type=int,
        default=24,
        help="Maximum number of overlays to save.",
    )
    parser.add_argument(
        "--class-source",
        choices=["predicted", "true"],
        default="predicted",
        help="Class whose TTECAM map is saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = initialize(args)
    root = project_root()
    heatmap_dir = root / args.heatmap_dir
    overlay_dir = root / args.overlay_dir
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_idrid_test_dataset(args)
    loader = build_loader(dataset, args)
    model = load_model(args, device)
    ttecam = TTECAM(model).to(device)
    ttecam.eval()

    offset = 0
    metadata_rows = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            class_indices = labels if args.class_source == "true" else None
            heatmaps, logits, selected_classes = ttecam.heatmaps(
                images,
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

                metadata_rows.append(
                    {
                        "idx": global_idx,
                        "true": int(labels[batch_idx].item()),
                        "pred": int(preds[batch_idx].item()),
                        "mapped_class": int(selected_classes[batch_idx].item()),
                    }
                )
            offset += images.shape[0]

    metadata_path = heatmap_dir / "metadata.csv"
    with metadata_path.open("w", encoding="utf-8") as f:
        f.write("idx,true,pred,mapped_class\n")
        for row in metadata_rows:
            f.write(
                f"{row['idx']},{row['true']},{row['pred']},{row['mapped_class']}\n"
            )

    print(f"Saved {offset} TTECAM heatmaps to {heatmap_dir}")
    print(f"Saved overlay previews to {overlay_dir}")
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
