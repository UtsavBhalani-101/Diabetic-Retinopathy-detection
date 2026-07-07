# pipeline/dann/grad_reverse.py
# ============================================================
# Gradient Reversal Layer (GRL) — Ganin et al., 2016
#
# Forward  : identity  (x → x)
# Backward : negation  (∂L/∂x → −λ · ∂L/∂x)
#
# This single operation makes the feature extractor simultaneously:
#   - minimise the classification loss (label branch)
#   - maximise the domain loss (i.e. confuse the discriminator)
# ============================================================

import torch
import torch.nn as nn
from torch import Tensor


class GradReverseFunction(torch.autograd.Function):
    """
    Custom autograd Function implementing the Gradient Reversal Layer.

    During the forward pass it acts as an identity.
    During the backward pass it multiplies the incoming gradient by -alpha,
    effectively reversing the gradient direction for the feature extractor.
    """

    @staticmethod
    def forward(ctx, x: Tensor, alpha: float) -> Tensor:
        # Save alpha as a tensor so it can be retrieved in backward.
        ctx.save_for_backward(torch.tensor(alpha, dtype=torch.float32))
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        (alpha,) = ctx.saved_tensors
        # Negate and scale the gradient — this is the reversal.
        # Return None for alpha because it has no gradient (it is a float).
        return -alpha.item() * grad_output, None


class GradientReversalLayer(nn.Module):
    """
    Thin nn.Module wrapper around GradReverseFunction.

    Parameters
    ----------
    alpha : float
        Scaling factor for gradient negation (λ in the paper).
        Passed at call time so it can be scheduled externally.

    Usage
    -----
        grl = GradientReversalLayer()
        reversed_feat = grl(features, alpha=current_lambda)
    """

    def forward(self, x: Tensor, alpha: float) -> Tensor:
        return GradReverseFunction.apply(x, alpha)
