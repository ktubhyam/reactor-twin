"""Positivity constraint for concentrations and other non-negative quantities."""

from __future__ import annotations

import logging

import torch

from reactor_twin.physics.constraints import AbstractConstraint
from reactor_twin.utils.registry import CONSTRAINT_REGISTRY

logger = logging.getLogger(__name__)


@CONSTRAINT_REGISTRY.register("positivity")
class PositivityConstraint(AbstractConstraint):
    """Enforce non-negativity for physical quantities (concentrations, temperature).

    Strategies:
    - Hard mode: Apply softplus or ReLU projection.
    - Soft mode: Add penalty for negative values.

    Attributes:
        indices: Indices of state variables to constrain (default: all).
        method: 'softplus', 'relu', or 'square' for hard projection.
        epsilon: Small constant to avoid exact zeros (for log-stability).
    """

    def __init__(
        self,
        name: str = "positivity",
        mode: str = "hard",
        weight: float = 1.0,
        indices: list[int] | None = None,
        method: str = "softplus",
        epsilon: float = 1e-8,
    ):
        """Initialize positivity constraint.

        Args:
            name: Constraint identifier.
            mode: 'hard' or 'soft'.
            weight: Weight for soft constraint penalty.
            indices: State indices to constrain. If None, applies to all.
            method: Projection method ('softplus', 'relu', 'square').
            epsilon: Small offset for numerical stability.
        """
        super().__init__(name, mode, weight)
        self.indices = indices
        self.method = method
        self.epsilon = epsilon

        if method not in ("softplus", "relu", "square"):
            raise ValueError(
                f"method must be 'softplus', 'relu', or 'square', got '{method}'"
            )

        logger.debug(
            f"Initialized PositivityConstraint: method={method}, epsilon={epsilon}"
        )

    def project(self, z: torch.Tensor) -> torch.Tensor:
        """Project state to non-negative values (hard mode).

        Args:
            z: State tensor, shape (batch, state_dim) or (batch, time, state_dim).

        Returns:
            Non-negative state with same shape.
        """
        z_positive = z.clone()

        # Select indices to constrain
        if self.indices is not None:
            z_subset = z_positive[..., self.indices]
        else:
            z_subset = z_positive

        # Apply projection
        if self.method == "softplus":
            z_constrained = torch.nn.functional.softplus(z_subset) + self.epsilon
        elif self.method == "relu":
            z_constrained = torch.nn.functional.relu(z_subset) + self.epsilon
        elif self.method == "square":
            z_constrained = z_subset ** 2 + self.epsilon
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Update constrained subset
        if self.indices is not None:
            z_positive[..., self.indices] = z_constrained
        else:
            z_positive = z_constrained

        return z_positive

    def compute_violation(self, z: torch.Tensor) -> torch.Tensor:
        """Compute penalty for negative values (soft mode).

        Args:
            z: State tensor, shape (batch, state_dim) or (batch, time, state_dim).

        Returns:
            Scalar penalty (sum of squared negative values).
        """
        # Select indices to constrain
        if self.indices is not None:
            z_subset = z[..., self.indices]
        else:
            z_subset = z

        # Penalty: sum of squared negative parts
        negative_part = torch.nn.functional.relu(-z_subset)
        penalty = torch.mean(negative_part ** 2)

        return penalty


__all__ = ["PositivityConstraint"]
