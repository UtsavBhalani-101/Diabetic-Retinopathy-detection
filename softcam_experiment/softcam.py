"""SoftCAM architecture and ElasticNet loss formulation.

SoftCAM removes the global average pooling (GAP) layer and replaces the linear
classification head with a 1x1 convolutional class-evidence layer. The evidence
maps directly produce the class logits via spatial average pooling, enabling
inherent self-explainability and allowing ElasticNet regularization to be
applied directly to the explanation maps during training.
"""

from __future__ import annotations

import logging
from typing import Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

LOGGER = logging.getLogger(__name__)


class SoftCAMEfficientNet(nn.Module):
    """EfficientNet-B0 with a 1x1 convolutional class-evidence layer.

    Parameters
    ----------
    num_classes : int
        Number of output categories (e.g. 5 for DR grading).
    dropout_rate : float
        Dropout probability applied to convolutional features before evidence mapping.
    pretrained : bool
        Whether to initialize the convolutional backbone with ImageNet weights.
    """

    def __init__(
        self,
        num_classes: int = 5,
        dropout_rate: float = 0.3,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        # Feature extractor without global pooling or classification head
        self.base = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            num_classes=0,
        )
        in_channels = self.base.num_features  # 1280 for efficientnet_b0

        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()

        # 1x1 convolution mapping high-dimensional features to class-evidence maps
        self.evidence_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=num_classes,
            kernel_size=1,
            stride=1,
            bias=True,
        )

        LOGGER.info(
            "SoftCAMEfficientNet initialized | classes=%d | in_channels=%d | dropout=%.2f | pretrained=%s",
            num_classes,
            in_channels,
            dropout_rate,
            pretrained,
        )

    def forward_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Extract spatial class-evidence maps A of shape [B, C, H, W]."""
        features = self.base.forward_features(x)
        features = self.dropout(features)
        class_maps = self.evidence_conv(features)
        return class_maps

    def forward(
        self,
        x: torch.Tensor,
        return_maps: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Compute logits by spatial average pooling of class-evidence maps.

        If return_maps is True, returns (logits, class_maps).
        """
        class_maps = self.forward_maps(x)
        # Average pooling over spatial dimensions H and W
        logits = class_maps.mean(dim=(2, 3))

        if return_maps:
            return logits, class_maps
        return logits

    @torch.no_grad()
    def heatmaps(
        self,
        x: torch.Tensor,
        class_indices: torch.Tensor | None = None,
        output_size: Tuple[int, int] = (224, 224),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate normalized [0, 1] heatmaps for target classes in a single forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input images of shape [B, 3, H, W], normalized with ImageNet stats.
        class_indices : torch.Tensor, optional
            Target class index per sample. If None, uses model's predicted class.
        output_size : Tuple[int, int]
            Target spatial resolution for upsampling (e.g. (224, 224)).

        Returns
        -------
        heatmaps : torch.Tensor
            Normalized heatmaps of shape [B, H_out, W_out] in [0, 1].
        logits : torch.Tensor
            Class logits of shape [B, C].
        class_indices : torch.Tensor
            Selected class indices of shape [B].
        """
        self.eval()
        class_maps = self.forward_maps(x)
        logits = class_maps.mean(dim=(2, 3))

        if class_indices is None:
            class_indices = logits.argmax(dim=1)

        batch_indices = torch.arange(x.shape[0], device=x.device)
        selected_maps = class_maps[batch_indices, class_indices]

        # Follow SoftCAM paper: positive evidence corresponds to disease features
        selected_maps = F.relu(selected_maps)

        # Bilinear interpolation to the target resolution
        selected_maps = F.interpolate(
            selected_maps.unsqueeze(1),
            size=output_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        # Per-image min-max normalization to [0, 1]
        flat = selected_maps.flatten(start_dim=1)
        mins = flat.min(dim=1).values.view(-1, 1, 1)
        maxs = flat.max(dim=1).values.view(-1, 1, 1)
        heatmaps = (selected_maps - mins) / (maxs - mins + 1e-8)

        return heatmaps, logits, class_indices


class SoftCAMLoss(nn.Module):
    """ElasticNet regularized loss for SoftCAM training.

    Loss = CrossEntropy(logits, targets) + lambda1 * L1(A) + lambda2 * L2(A)

    Parameters
    ----------
    base_criterion : nn.Module
        The base classification loss (e.g. class-weighted CrossEntropyLoss).
    lambda1 : float
        Lasso (L1) penalty strength to enforce sparsity in evidence maps.
    lambda2 : float
        Ridge (L2) penalty strength to smooth evidence maps.
    """

    def __init__(
        self,
        base_criterion: nn.Module,
        lambda1: float = 0.0,
        lambda2: float = 0.0,
    ) -> None:
        super().__init__()
        self.base_criterion = base_criterion
        self.lambda1 = float(lambda1)
        self.lambda2 = float(lambda2)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        class_maps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the combined loss and return individual components."""
        ce_loss = self.base_criterion(logits, targets)

        # L1 (Lasso) penalty: promotes sparsity
        l1_reg = class_maps.abs().mean()

        # L2 (Ridge) penalty: promotes smoothness
        l2_reg = (class_maps**2).mean()

        total_loss = ce_loss + (self.lambda1 * l1_reg) + (self.lambda2 * l2_reg)
        return total_loss, ce_loss, l1_reg, l2_reg
