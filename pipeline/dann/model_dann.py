# pipeline/dann/model_dann.py
# ============================================================
# DANNEfficientNet — EfficientNet-B0 backbone extended with:
#   - Label classifier  : Dropout → Linear(1280, 5)   [DR grades 0-4]
#   - Domain classifier : GRL → MLP(1280→512→2)       [source / target]
#
# Helper:
#   compute_lambda(p)  : λ schedule from Ganin et al. (2016)
#   get_dann_loss_criterion : class-weighted CrossEntropyLoss for source data
# ============================================================

import logging
import math

import numpy as np
import torch
import torch.nn as nn
import timm
from sklearn.utils.class_weight import compute_class_weight
from torch import Tensor

from pipeline.dann.grad_reverse import GradientReversalLayer

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------
# λ schedule
# ----------------------------------------------------------------

def compute_lambda(p: float, max_lambda: float = 1.0) -> float:
    """
    Ganin et al. (2016) λ schedule.

    Parameters
    ----------
    p          : float in [0, 1] — training progress (global_step / total_steps)
    max_lambda : float           — ceiling value (default 1.0)

    Returns
    -------
    λ ∈ [0, max_lambda]

    At p=0  → λ ≈ 0   (no domain adversarial pressure at the start)
    At p=1  → λ = max_lambda (full adversarial pressure at the end)
    """
    return max_lambda * (2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)


# ----------------------------------------------------------------
# DANN model
# ----------------------------------------------------------------

class DANNEfficientNet(nn.Module):
    """
    EfficientNet-B0 with two prediction heads:

    1. Label head (classification)
       Input  : feature vector [B, 1280]
       Output : class logits   [B, num_classes]   — DR grade 0-4
       Used   : on SOURCE images only (ground-truth labels available)

    2. Domain head (adversarial discriminator)
       Input  : reversed features via GRL [B, 1280]
       Output : domain logits [B, 2]              — 0=source, 1=target
       Used   : on ALL images (source + target)

    The GRL negates gradients flowing back to the feature extractor
    from the domain head, so the backbone learns domain-invariant features.

    Parameters
    ----------
    num_classes    : int   number of DR grades (5)
    dropout_rate   : float dropout before the label head
    domain_hidden  : int   hidden size of domain discriminator MLP
    domain_dropout : float dropout inside domain discriminator
    pretrained     : bool  load ImageNet weights
    """

    def __init__(
        self,
        num_classes: int   = 5,
        dropout_rate: float = 0.3,
        domain_hidden: int  = 512,
        domain_dropout: float = 0.3,
        pretrained: bool    = True,
    ):
        super().__init__()

        # ---- Shared feature extractor ----
        self.feature_extractor = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            num_classes=0          # remove timm's built-in head → raw features
        )
        feat_dim = self.feature_extractor.num_features  # 1280 for EfficientNet-B0

        # ---- Label classification head ----
        self.label_classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(feat_dim, num_classes),
        )

        # ---- Gradient reversal layer ----
        self.grl = GradientReversalLayer()

        # ---- Domain discrimination head ----
        # Binary: 0 = source domain, 1 = target domain
        self.domain_classifier = nn.Sequential(
            nn.Linear(feat_dim, domain_hidden),
            nn.BatchNorm1d(domain_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=domain_dropout),
            nn.Linear(domain_hidden, 2),
        )

        logger.info(
            f"DANNEfficientNet | feat_dim={feat_dim} | num_classes={num_classes} "
            f"| dropout={dropout_rate} | domain_hidden={domain_hidden} "
            f"| domain_dropout={domain_dropout} | pretrained={pretrained}"
        )

    def forward(self, x: Tensor, alpha: float = 1.0):
        """
        Parameters
        ----------
        x     : image tensor [B, C, H, W]
        alpha : current λ value for the GRL (0 at start, 1 at end)

        Returns
        -------
        class_logits  : [B, num_classes]  — used for DR grading (source only)
        domain_logits : [B, 2]            — used for domain alignment (all images)
        """
        feat = self.feature_extractor(x)                    # [B, 1280]
        class_logits  = self.label_classifier(feat)         # [B, 5]
        reversed_feat = self.grl(feat, alpha)               # identity fwd, −λ bwd
        domain_logits = self.domain_classifier(reversed_feat)  # [B, 2]
        return class_logits, domain_logits

    def get_features(self, x: Tensor) -> Tensor:
        """
        Return raw backbone features — used for:
          - UMAP visualisation
          - MC Dropout inference (via predict_class)
          - Cosine similarity / Mahalanobis OOD detection
        """
        return self.feature_extractor(x)

    def predict_class(self, x: Tensor) -> Tensor:
        """
        Return only class logits (no domain head).
        Convenience method for evaluation / MC Dropout inference
        where the domain branch is not needed.
        """
        feat = self.feature_extractor(x)
        return self.label_classifier(feat)


# ----------------------------------------------------------------
# Loss criterion factory
# ----------------------------------------------------------------

def get_dann_loss_criterion(
    combined_df,
    diagnosis_col: str,
    device: torch.device,
) -> nn.CrossEntropyLoss:
    """
    Compute class-balanced weights from the combined source training DataFrame
    and return a weighted CrossEntropyLoss.

    Uses the same approach as the baseline pipeline so the grading loss is
    consistent regardless of how many source datasets are combined.

    Parameters
    ----------
    combined_df   : pd.DataFrame — concatenated source training DataFrames
    diagnosis_col : str          — column name holding DR grade labels
    device        : torch.device

    Returns
    -------
    nn.CrossEntropyLoss with class weights on device
    """
    labels         = combined_df[diagnosis_col].values
    unique_classes = np.unique(labels)
    weights        = compute_class_weight(
        class_weight="balanced",
        classes=unique_classes,
        y=labels,
    )
    class_weights = torch.FloatTensor(weights).to(device)

    logger.info(
        f"DANN loss criterion | classes={unique_classes.tolist()} "
        f"| weights={[f'{w:.3f}' for w in weights]}"
    )
    return nn.CrossEntropyLoss(weight=class_weights)
