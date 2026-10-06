"""TTECAM implementation for the existing EfficientNetMC model.

TTECAM converts the trained linear classifier into an equivalent 1x1
convolution at test time. The convolution produces one spatial activation map
per class, and global average pooling of those maps recovers class logits.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TTECAM(nn.Module):
    """Test-time class activation map wrapper for EfficientNetMC."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.base = model.base
        self.num_classes = model.classifier.out_features
        in_channels = model.classifier.in_features

        self.class_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=self.num_classes,
            kernel_size=1,
            bias=model.classifier.bias is not None,
        )
        self._copy_classifier_weights()

    def _copy_classifier_weights(self) -> None:
        with torch.no_grad():
            weight = self.model.classifier.weight.detach().view(
                self.num_classes, -1, 1, 1
            )
            self.class_conv.weight.copy_(weight)
            if self.model.classifier.bias is not None:
                self.class_conv.bias.copy_(self.model.classifier.bias.detach())

    def forward_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Return class activation maps with shape [B, C, h, w]."""
        features = self.base.forward_features(x)
        return self.class_conv(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits recovered by spatially averaging class maps."""
        class_maps = self.forward_maps(x)
        return class_maps.mean(dim=(2, 3))

    @torch.no_grad()
    def heatmaps(
        self,
        x: torch.Tensor,
        class_indices: torch.Tensor | None = None,
        output_size: tuple[int, int] = (224, 224),
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate normalized heatmaps for selected classes.

        If ``class_indices`` is omitted, heatmaps are generated for the model's
        predicted class per image. Returns ``(heatmaps, logits, classes)``.
        """
        self.eval()
        class_maps = self.forward_maps(x)
        logits = class_maps.mean(dim=(2, 3))

        if class_indices is None:
            class_indices = logits.argmax(dim=1)

        batch_indices = torch.arange(x.shape[0], device=x.device)
        selected_maps = class_maps[batch_indices, class_indices]
        selected_maps = F.relu(selected_maps)
        selected_maps = F.interpolate(
            selected_maps.unsqueeze(1),
            size=output_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        flat = selected_maps.flatten(start_dim=1)
        mins = flat.min(dim=1).values.view(-1, 1, 1)
        maxs = flat.max(dim=1).values.view(-1, 1, 1)
        heatmaps = (selected_maps - mins) / (maxs - mins + 1e-8)
        return heatmaps, logits, class_indices
