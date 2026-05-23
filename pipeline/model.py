# pipeline/model.py
# ============================================================
# Model architecture and loss criterion:
#   - EfficientNetMC    : EfficientNet-B0 with MC Dropout head
#   - get_loss_criterion: class-balanced CrossEntropyLoss
# ============================================================

import numpy as np
import torch
import torch.nn as nn
import timm
from sklearn.utils.class_weight import compute_class_weight


class EfficientNetMC(nn.Module):
    """
    EfficientNet-B0 with a single MC Dropout layer before the linear head.

    Setting dropout_rate > 0 and calling mc_evaluate_full() with the
    dropout layers forced to train() mode gives Bayesian-style uncertainty
    estimates at inference time (MC Dropout).
    """

    def __init__(self, num_classes: int, dropout_rate: float = 0.3):
        super().__init__()
        # Remove the default classifier head (num_classes=0) so we can
        # insert our own Dropout → Linear
        self.base = timm.create_model(
            "efficientnet_b0", pretrained=True, num_classes=0
        )
        self.dropout    = nn.Dropout(p=dropout_rate)
        self.classifier = nn.Linear(self.base.num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.base(x)
        features = self.dropout(features)
        return self.classifier(features)


def get_loss_criterion(df, diagnosis_col: str, device: torch.device) -> nn.CrossEntropyLoss:
    """
    Compute class-balanced weights from the training DataFrame and return
    a weighted CrossEntropyLoss.

    Balancing is critical for APTOS where class 0 (No DR) is ~5× more
    frequent than class 3 (Severe).
    """
    labels         = df[diagnosis_col].values
    unique_classes = np.unique(labels)
    weights        = compute_class_weight(
        class_weight="balanced",
        classes=unique_classes,
        y=labels
    )
    class_weights = torch.FloatTensor(weights).to(device)
    return nn.CrossEntropyLoss(weight=class_weights)
