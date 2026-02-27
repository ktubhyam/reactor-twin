"""Thermodynamic consistency constraints (Gibbs, entropy, equilibrium)."""

from __future__ import annotations

import logging

import torch

from reactor_twin.physics.constraints import AbstractConstraint
from reactor_twin.utils.registry import CONSTRAINT_REGISTRY

logger = logging.getLogger(__name__)

from reactor_twin.utils.constants import R_GAS


@CONSTRAINT_REGISTRY.register("thermodynamics")
class ThermodynamicConstraint(AbstractConstraint):
    """Enforce thermodynamic consistency.

    Constraints:
    1. Second law: Entropy production >= 0 (dS/dt >= 0)
    2. Gibbs free energy: Decreases for spontaneous reactions (dG/dt <= 0)
    3. Equilibrium: Reactions stop at equilibrium (K_eq satisfied)

    Hard mode: Not applicable (thermodynamics is emergent from kinetics).
    Soft mode: Penalize violations of second law and equilibrium.

    Attributes:
        check_entropy: If True, penalize entropy decrease.
        check_gibbs: If True, penalize Gibbs energy increase.
        equilibrium_constants: K_eq for each reaction, shape (num_reactions,).
    """

    def __init__(
        self,
        name: str = "thermodynamics",
        mode: str = "soft",
        weight: float = 1.0,
        check_entropy: bool = True,
        check_gibbs: bool = True,
        equilibrium_constants: torch.Tensor | None = None,
        temperature: float = 298.15,
    ):
        """Initialize thermodynamic constraint.

        Args:
            name: Constraint identifier.
            mode: Only 'soft' mode supported (thermodynamics is emergent).
            weight: Weight for soft constraint penalty.
            check_entropy: Penalize entropy decrease (2nd law violation).
            check_gibbs: Penalize Gibbs energy increase.
            equilibrium_constants: K_eq for reversible reactions.
            temperature: System temperature (K) for equilibrium calculations.
        """
        super().__init__(name, mode, weight)
        self.check_entropy = check_entropy
        self.check_gibbs = check_gibbs
        self.equilibrium_constants = equilibrium_constants
        self.temperature = temperature

        if mode == "hard":
            raise ValueError(
                "Hard thermodynamic constraints not supported. "
                "Thermodynamics emerges from correct kinetics."
            )

        logger.debug(
            f"Initialized ThermodynamicConstraint: "
            f"check_entropy={check_entropy}, check_gibbs={check_gibbs}"
        )

    def project(self, z: torch.Tensor) -> torch.Tensor:
        """Not applicable for thermodynamic constraints."""
        raise NotImplementedError(
            "Hard thermodynamic constraints not supported. Use soft mode."
        )

    def compute_violation(self, z: torch.Tensor) -> torch.Tensor:
        """Compute thermodynamic violation penalty (soft mode).

        Penalizes:
        1. Entropy decrease (if check_entropy=True)
        2. Gibbs energy increase (if check_gibbs=True)
        3. Deviation from equilibrium (if equilibrium_constants provided)

        Args:
            z: State tensor, shape (batch, time, state_dim).
                Assumed: [...concentrations..., temperature].

        Returns:
            Scalar penalty for thermodynamic violations.
        """
        if z.ndim != 3:
            return torch.tensor(0.0, device=z.device)

        violation = torch.tensor(0.0, device=z.device)

        # Extract concentrations and temperature
        C = z[..., :-1]  # (batch, time, num_species)
        T = z[..., -1:]  # (batch, time, 1)

        # 1. Check entropy monotonicity (simplified: S ≈ -R Σ(C_i ln C_i))
        if self.check_entropy:
            # Avoid log(0) with small epsilon
            C_safe = torch.clamp(C, min=1e-8)
            entropy = -R_GAS * torch.sum(C_safe * torch.log(C_safe), dim=-1)  # (batch, time)

            # Entropy should not decrease
            entropy_change = torch.diff(entropy, dim=1)  # (batch, time-1)
            entropy_violations = torch.relu(-entropy_change)  # Penalize decreases
            violation = violation + torch.mean(entropy_violations)

        # 2. Check Gibbs energy monotonicity (G = H - TS, simplified)
        if self.check_gibbs:
            # Simplified Gibbs: G ≈ Σ(C_i * μ_i) where μ_i is chemical potential
            # For simplicity, use G ≈ Σ(C_i) * T (arbitrary units)
            total_concentration = C.sum(dim=-1)  # (batch, time)
            gibbs = total_concentration * T.squeeze(-1)  # (batch, time)

            # Gibbs should not increase for spontaneous processes
            gibbs_change = torch.diff(gibbs, dim=1)  # (batch, time-1)
            gibbs_violations = torch.relu(gibbs_change)  # Penalize increases
            violation = violation + torch.mean(gibbs_violations)

        # 3. Check equilibrium condition (K_eq = Π(C_i^nu_i))
        if self.equilibrium_constants is not None:
            # At equilibrium: Q = K_eq where Q is reaction quotient
            # Q = Π(C_i^nu_i) for each reaction
            # For now, just a placeholder - requires stoichiometry matrix
            logger.warning("Equilibrium constraint not yet implemented")

        return violation


__all__ = ["ThermodynamicConstraint"]
