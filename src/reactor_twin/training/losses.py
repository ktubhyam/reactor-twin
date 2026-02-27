"""Multi-objective loss functions for physics-constrained training."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class MultiObjectiveLoss(nn.Module):
    """Weighted multi-objective loss for physics-informed Neural DEs.

    Combines:
    1. Data-fitting loss (MSE between predictions and targets)
    2. Physics loss (conservation laws, thermodynamics)
    3. Constraint loss (positivity, mass balance, etc.)
    4. Regularization loss (weight decay, smoothness)

    Attributes:
        weights: Dictionary of loss component weights.
        constraints: Optional list of physics constraints to evaluate.
    """

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        constraints: list[Any] | None = None,
    ):
        """Initialize multi-objective loss.

        Args:
            weights: Dictionary mapping loss names to weights.
                Default: {'data': 1.0}.
            constraints: List of AbstractConstraint objects.
        """
        super().__init__()
        self.weights = weights or {"data": 1.0}
        self.constraints = constraints or []

        logger.info(f"Initialized MultiObjectiveLoss: weights={self.weights}")

    def data_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Mean squared error between predictions and targets.

        Args:
            predictions: Model predictions, shape (batch, time, state_dim).
            targets: Ground truth, shape (batch, time, state_dim).

        Returns:
            Scalar MSE loss.
        """
        return torch.mean((predictions - targets) ** 2)

    def physics_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Physics-based loss (conservation laws, energy, etc.).

        This is a placeholder - specific physics losses should be defined
        based on the reactor system.

        Args:
            predictions: Model predictions, shape (batch, time, state_dim).
            targets: Ground truth (for reference).

        Returns:
            Scalar physics loss.
        """
        # Physics loss delegates to constraints. Override this method in subclasses
        # to add system-specific physics losses (e.g., energy/mass balance residuals).
        return torch.tensor(0.0, device=predictions.device, requires_grad=False)

    def constraint_loss(
        self,
        predictions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute constraint violation penalties.

        Args:
            predictions: Model predictions, shape (batch, time, state_dim).

        Returns:
            Dictionary mapping constraint names to violation penalties.
        """
        constraint_losses = {}

        for constraint in self.constraints:
            _, violation = constraint(predictions)
            if isinstance(violation, dict):
                # ConstraintPipeline returns dict
                constraint_losses.update(violation)
            else:
                # Single constraint returns scalar or multi-element tensor
                viol_sum = violation.sum()
                if viol_sum.item() > 0:
                    constraint_losses[constraint.name] = viol_sum

        return constraint_losses

    def regularization_loss(
        self,
        model: nn.Module,
    ) -> torch.Tensor:
        """L2 regularization on model parameters.

        Args:
            model: Neural network model.

        Returns:
            Scalar L2 penalty.
        """
        l2_penalty = torch.tensor(0.0, device=next(model.parameters()).device)
        for param in model.parameters():
            l2_penalty = l2_penalty + torch.norm(param) ** 2
        return l2_penalty

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        model: nn.Module | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute total weighted loss.

        Args:
            predictions: Model predictions, shape (batch, time, state_dim).
            targets: Ground truth, shape (batch, time, state_dim).
            model: Neural network (optional, for regularization).

        Returns:
            Dictionary with keys:
                - 'total': Total weighted loss (scalar)
                - Individual loss components (data, physics, constraints, etc.)
        """
        losses: dict[str, torch.Tensor] = {}

        # 1. Data-fitting loss
        data_loss_val = self.data_loss(predictions, targets)
        losses["data"] = data_loss_val

        # 2. Physics loss
        physics_loss_val = self.physics_loss(predictions, targets)
        losses["physics"] = physics_loss_val

        # 3. Constraint losses
        constraint_losses = self.constraint_loss(predictions)
        losses.update(constraint_losses)

        # 4. Regularization (optional)
        if model is not None and "regularization" in self.weights:
            reg_loss_val = self.regularization_loss(model)
            losses["regularization"] = reg_loss_val

        # Compute total weighted loss
        total_loss = torch.tensor(0.0, device=predictions.device)
        for loss_name, loss_value in losses.items():
            weight = self.weights.get(loss_name, 0.0)
            total_loss = total_loss + weight * loss_value

        losses["total"] = total_loss

        return losses

    def update_weights(self, new_weights: dict[str, float]) -> None:
        """Update loss weights (for curriculum learning).

        Args:
            new_weights: Dictionary of new weights to merge in.
        """
        self.weights.update(new_weights)
        logger.debug(f"Updated loss weights: {self.weights}")


__all__ = ["MultiObjectiveLoss"]
