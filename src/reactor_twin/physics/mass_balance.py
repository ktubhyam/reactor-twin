"""Mass balance constraint for chemical species conservation."""

from __future__ import annotations

import logging

import torch

from reactor_twin.physics.constraints import AbstractConstraint
from reactor_twin.utils.registry import CONSTRAINT_REGISTRY

logger = logging.getLogger(__name__)


@CONSTRAINT_REGISTRY.register("mass_balance")
class MassBalanceConstraint(AbstractConstraint):
    """Enforce mass/mole balance via stoichiometric projection.

    For a system with stoichiometric matrix nu (num_reactions x num_species),
    the rate of change must satisfy: dC/dt = nu^T * r where r is reaction rates.

    Hard mode: Project predicted dC/dt onto the stoichiometric subspace.
    Soft mode: Penalize deviation from stoichiometric consistency.

    Attributes:
        stoich_matrix: Stoichiometric coefficients, shape (num_reactions, num_species).
        initial_mass: Initial total mass for checking conservation.
    """

    def __init__(
        self,
        name: str = "mass_balance",
        mode: str = "hard",
        weight: float = 1.0,
        stoich_matrix: torch.Tensor | None = None,
        check_total_mass: bool = True,
    ):
        """Initialize mass balance constraint.

        Args:
            name: Constraint identifier.
            mode: 'hard' or 'soft'.
            weight: Weight for soft constraint penalty.
            stoich_matrix: Stoichiometric matrix (reactions x species).
                If None, assumes all species are conserved independently.
            check_total_mass: If True, check total mass is conserved.
        """
        super().__init__(name, mode, weight)
        self.stoich_matrix = stoich_matrix
        self.check_total_mass = check_total_mass
        self.initial_mass: torch.Tensor | None = None

        if stoich_matrix is not None:
            # Compute null space basis for stoichiometric subspace
            self._compute_projection_matrix()

        logger.debug(
            f"Initialized MassBalanceConstraint: "
            f"stoich_shape={stoich_matrix.shape if stoich_matrix is not None else None}, "
            f"check_total_mass={check_total_mass}"
        )

    def _compute_projection_matrix(self) -> None:
        """Compute projection matrix onto stoichiometric subspace.

        For stoichiometric matrix S (reactions x species), the projection
        matrix P = S^T (S S^T)^-1 S ensures dC/dt lies in range(S^T).
        """
        if self.stoich_matrix is None:
            return

        S = self.stoich_matrix  # (num_reactions, num_species)

        # Projection matrix: P = S^T (S S^T)^-1 S
        # This projects onto the column space of S^T (valid stoichiometric directions)
        SST = S @ S.T  # (num_reactions, num_reactions)
        SST_inv = torch.linalg.pinv(SST)  # Pseudo-inverse for numerical stability
        self.projection_matrix = S.T @ SST_inv @ S  # (num_species, num_species)

        logger.debug(f"Computed projection matrix: shape={self.projection_matrix.shape}")

    def project(self, z: torch.Tensor) -> torch.Tensor:
        """Project concentrations to satisfy mass balance (hard mode).

        For time-dependent trajectories, we don't project the concentrations
        directly, but rather ensure their *rates of change* respect stoichiometry.
        This is handled in the ODE function itself.

        For static concentrations, we can project to the nearest point that
        conserves total mass.

        Args:
            z: State tensor, shape (batch, state_dim) or (batch, time, state_dim).

        Returns:
            Projected state (in this case, we primarily enforce via ODE dynamics).
        """
        if self.check_total_mass:
            # Store initial mass for conservation checking
            if self.initial_mass is None:
                self.initial_mass = z.sum(dim=-1, keepdim=True)

            # Project to conserve total mass
            current_mass = z.sum(dim=-1, keepdim=True)
            z_projected = z * (self.initial_mass / (current_mass + 1e-8))

            return z_projected

        # If no total mass check, return as-is
        # (mass balance is enforced in the ODE dynamics)
        return z

    def compute_violation(self, z: torch.Tensor) -> torch.Tensor:
        """Compute mass balance violation penalty (soft mode).

        Penalizes deviation from total mass conservation.

        Args:
            z: State tensor, shape (batch, state_dim) or (batch, time, state_dim).

        Returns:
            Scalar penalty for mass balance violation.
        """
        if self.initial_mass is None:
            # First call - store initial mass
            self.initial_mass = z[..., 0, :].sum(dim=-1, keepdim=True)  # (batch, 1)

        # Compute mass at all time points
        if z.ndim == 3:  # (batch, time, state_dim)
            current_mass = z.sum(dim=-1)  # (batch, time)
            target_mass = self.initial_mass.expand_as(current_mass)
        else:  # (batch, state_dim)
            current_mass = z.sum(dim=-1)  # (batch,)
            target_mass = self.initial_mass.squeeze(-1)

        # Mean squared deviation from initial mass
        violation = torch.mean((current_mass - target_mass) ** 2)

        return violation

    def reset(self) -> None:
        """Reset stored initial mass (call between batches)."""
        self.initial_mass = None


__all__ = ["MassBalanceConstraint"]
