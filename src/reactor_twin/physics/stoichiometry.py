"""Stoichiometric consistency constraint."""

from __future__ import annotations

import logging
from typing import cast

import torch

from reactor_twin.physics.constraints import AbstractConstraint
from reactor_twin.utils.registry import CONSTRAINT_REGISTRY

logger = logging.getLogger(__name__)


@CONSTRAINT_REGISTRY.register("stoichiometry")
class StoichiometricConstraint(AbstractConstraint):
    """Enforce stoichiometric consistency: predict rates, not species.

    Key insight: Instead of predicting dC/dt directly (which may violate
    stoichiometry), predict reaction rates r, then compute dC/dt = nu^T * r.

    Hard mode: Architectural - ODE function outputs rates, applies stoichiometry.
    Soft mode: Penalize deviation from stoichiometric relationships.

    Attributes:
        stoich_matrix: Stoichiometric matrix nu (num_reactions x num_species).
        num_reactions: Number of independent reactions.
        num_species: Number of chemical species.
    """

    def __init__(
        self,
        name: str = "stoichiometry",
        mode: str = "hard",
        weight: float = 1.0,
        stoich_matrix: torch.Tensor | None = None,
    ):
        """Initialize stoichiometric constraint.

        Args:
            name: Constraint identifier.
            mode: 'hard' or 'soft'.
            weight: Weight for soft constraint penalty.
            stoich_matrix: Stoichiometric matrix (num_reactions x num_species).
                rows = reactions, columns = species, entries = stoichiometric coefficients.
        """
        super().__init__(name, mode, weight)

        if stoich_matrix is None:
            raise ValueError("stoich_matrix is required for StoichiometricConstraint")

        self.stoich_matrix = stoich_matrix  # (num_reactions, num_species)
        self.num_reactions, self.num_species = stoich_matrix.shape

        logger.debug(
            f"Initialized StoichiometricConstraint: "
            f"reactions={self.num_reactions}, species={self.num_species}"
        )

    def project(self, z: torch.Tensor) -> torch.Tensor:
        """Project state derivatives onto stoichiometric subspace (hard mode).

        For hard constraints, this is typically done in the ODE function:
        1. Neural net predicts reaction rates r (num_reactions,)
        2. Compute dC/dt = nu^T * r (num_species,)

        This method projects post-hoc for compatibility.

        Args:
            z: State or state derivative, shape (batch, state_dim).

        Returns:
            Projected onto column space of stoichiometric matrix.
        """
        # Project onto range(nu^T) using pseudo-inverse
        # z_projected = nu^T * (nu nu^T)^-1 * nu * z

        nu = self.stoich_matrix.to(z.device)  # (reactions, species)
        nu_T = nu.T  # (species, reactions)

        # Compute projection matrix: P = nu^T (nu nu^T)^-1 nu
        nu_nu_T = nu @ nu_T  # (reactions, reactions)
        nu_nu_T_inv = torch.linalg.pinv(nu_nu_T)
        P = nu_T @ nu_nu_T_inv @ nu  # (species, species)

        # Apply projection
        if z.ndim == 2:  # (batch, state_dim)
            z_projected = z @ P.T
        elif z.ndim == 3:  # (batch, time, state_dim)
            z_projected = torch.einsum("bts,sr->btr", z, P)
        else:
            raise ValueError(f"Unsupported z shape: {z.shape}")

        return cast(torch.Tensor, z_projected)

    def compute_violation(self, z: torch.Tensor) -> torch.Tensor:
        """Compute stoichiometric violation penalty (soft mode).

        Penalizes the component of dC/dt that lies outside the stoichiometric subspace.

        Args:
            z: State derivative dC/dt, shape (batch, state_dim) or (batch, time, state_dim).

        Returns:
            Scalar penalty for stoichiometric violation.
        """
        # Project onto stoichiometric subspace
        z_projected = self.project(z)

        # Penalize the difference (component orthogonal to stoichiometric subspace)
        violation = torch.mean((z - z_projected) ** 2)

        return violation

    def forward_stoichiometry(self, rates: torch.Tensor) -> torch.Tensor:
        """Compute species rates from reaction rates: dC/dt = nu^T * r.

        This is the primary way to enforce hard stoichiometric constraints.

        Args:
            rates: Reaction rates, shape (batch, num_reactions) or (batch, time, num_reactions).

        Returns:
            Species rates dC/dt, shape (batch, num_species) or (batch, time, num_species).
        """
        nu_T = self.stoich_matrix.T.to(rates.device)  # (species, reactions)

        if rates.ndim == 2:  # (batch, reactions)
            dC_dt = rates @ nu_T.T  # (batch, species)
        elif rates.ndim == 3:  # (batch, time, reactions)
            dC_dt = torch.einsum("btr,sr->bts", rates, nu_T)
        else:
            raise ValueError(f"Unsupported rates shape: {rates.shape}")

        return dC_dt


__all__ = ["StoichiometricConstraint"]
