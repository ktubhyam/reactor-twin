"""Base abstract class and implementations for physics constraints."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import torch
from torch import nn

from reactor_twin.exceptions import ValidationError

logger = logging.getLogger(__name__)


class AbstractConstraint(nn.Module, ABC):
    """Abstract base class for physics constraints.

    Constraints can be applied as:
    - Hard constraints: Architectural projection onto constraint manifold.
    - Soft constraints: Penalty terms added to loss function.

    Attributes:
        name: Constraint identifier.
        mode: 'hard' or 'soft'.
        weight: Weight for soft constraint penalty.
    """

    def __init__(self, name: str, mode: str = "hard", weight: float = 1.0):
        """Initialize constraint.

        Args:
            name: Constraint identifier.
            mode: 'hard' for projection, 'soft' for penalty.
            weight: Weight for soft constraint (ignored in hard mode).
        """
        super().__init__()
        if mode not in ("hard", "soft"):
            raise ValidationError(f"mode must be 'hard' or 'soft', got '{mode}'")
        self.name = name
        self.mode = mode
        self.weight = weight
        logger.debug(f"Initialized {self.__class__.__name__}: mode={mode}, weight={weight}")

    @abstractmethod
    def project(self, z: torch.Tensor) -> torch.Tensor:
        """Project state onto constraint manifold (hard constraints).

        Args:
            z: State tensor, shape (batch, state_dim) or (batch, time, state_dim).

        Returns:
            Constrained state with same shape as input.
        """
        raise NotImplementedError("Subclasses must implement project()")

    @abstractmethod
    def compute_violation(self, z: torch.Tensor) -> torch.Tensor:
        """Compute constraint violation (soft constraints).

        Args:
            z: State tensor, shape (batch, state_dim) or (batch, time, state_dim).

        Returns:
            Violation penalty, scalar tensor.
        """
        raise NotImplementedError("Subclasses must implement compute_violation()")

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply constraint.

        Args:
            z: Input state.

        Returns:
            Tuple of (constrained_state, violation_penalty).
        """
        if self.mode == "hard":
            z_constrained = self.project(z)
            violation = torch.tensor(0.0, device=z.device)
        else:  # soft
            z_constrained = z
            violation = self.weight * self.compute_violation(z)

        return z_constrained, violation

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', mode='{self.mode}', weight={self.weight})"
        )


class ConstraintPipeline(nn.Module):
    """Compose multiple constraints into a pipeline.

    Example:
        >>> pipeline = ConstraintPipeline([
        ...     MassBalanceConstraint(mode='hard'),
        ...     PositivityConstraint(mode='hard'),
        ...     EnergyBalanceConstraint(mode='soft', weight=0.1),
        ... ])
        >>> z_constrained, total_violation = pipeline(z)
    """

    def __init__(self, constraints: list[AbstractConstraint]):
        """Initialize constraint pipeline.

        Args:
            constraints: List of constraint objects.
        """
        super().__init__()
        self.constraints = nn.ModuleList(constraints)
        logger.info(
            f"Initialized ConstraintPipeline with {len(constraints)} constraints: "
            f"{[c.name for c in constraints]}"
        )

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Apply all constraints sequentially.

        Args:
            z: Input state.

        Returns:
            Tuple of (constrained_state, violations_dict).
            violations_dict maps constraint name to violation penalty.
        """
        violations: dict[str, torch.Tensor] = {}
        z_current = z

        for constraint in self.constraints:
            z_current, violation = constraint(z_current)
            if violation.item() > 0:
                violations[constraint.name] = violation

        return z_current, violations

    def __repr__(self) -> str:
        """String representation."""
        constraint_names = [c.name for c in self.constraints]
        return f"ConstraintPipeline({constraint_names})"


__all__ = ["AbstractConstraint", "ConstraintPipeline"]
