"""Energy balance constraint for thermochemical systems."""

from __future__ import annotations

import logging

import torch

from reactor_twin.physics.constraints import AbstractConstraint
from reactor_twin.utils.registry import CONSTRAINT_REGISTRY

logger = logging.getLogger(__name__)


@CONSTRAINT_REGISTRY.register("energy_balance")
class EnergyBalanceConstraint(AbstractConstraint):
    """Enforce energy conservation for reactive systems.

    Energy balance: ρ Cp V dT/dt = Σ(ΔH_rxn * r_j * V) + Q_heat + F Cp (T_in - T)

    Hard mode: Compute temperature from energy balance (not predicted).
    Soft mode: Penalize deviation from energy balance equation.

    Attributes:
        heat_capacity: Cp (J/(mol*K)) for species.
        heats_of_reaction: ΔH_rxn (J/mol) for each reaction, shape (num_reactions,).
        stoich_matrix: Stoichiometric matrix for computing reaction extents.
    """

    def __init__(
        self,
        name: str = "energy_balance",
        mode: str = "soft",  # Hard mode requires coupling with reactor dynamics
        weight: float = 1.0,
        heat_capacity: float | None = None,
        heats_of_reaction: torch.Tensor | None = None,
        stoich_matrix: torch.Tensor | None = None,
    ):
        """Initialize energy balance constraint.

        Args:
            name: Constraint identifier.
            mode: 'hard' or 'soft'. Hard mode computes T from energy balance.
            weight: Weight for soft constraint penalty.
            heat_capacity: Molar heat capacity Cp (J/(mol*K)).
            heats_of_reaction: Heat of reaction for each reaction (J/mol).
            stoich_matrix: Stoichiometric matrix (reactions x species).
        """
        super().__init__(name, mode, weight)
        self.heat_capacity = heat_capacity
        self.heats_of_reaction = heats_of_reaction
        self.stoich_matrix = stoich_matrix

        logger.debug(f"Initialized EnergyBalanceConstraint: mode={mode}, Cp={heat_capacity}")

    def project(self, z: torch.Tensor) -> torch.Tensor:
        """Compute temperature from energy balance (hard mode).

        This requires access to reaction rates, which is not available
        from state alone. Hard energy balance is typically enforced in
        the ODE function itself, not via post-hoc projection.

        Args:
            z: State tensor, shape (batch, state_dim) or (batch, time, state_dim).

        Returns:
            State with temperature computed from energy balance.
        """
        # Hard energy balance projection is not feasible from state alone
        # because it requires reaction rates, heat transfer coefficients, etc.
        # It must be enforced within the ODE function via HybridODEFunc.
        # Fall back to soft mode penalty for standalone usage.
        logger.warning(
            "Hard energy balance is not supported as standalone projection. "
            "Use soft mode, or enforce via HybridODEFunc. Returning state unchanged."
        )
        return z

    def compute_violation(self, z: torch.Tensor) -> torch.Tensor:
        """Compute energy balance violation penalty (soft mode).

        For a trajectory, we penalize large changes in total energy that
        are not explained by reaction enthalpy.

        Args:
            z: State tensor, shape (batch, time, state_dim).
                Last dimension assumed to be [...concentrations..., temperature].

        Returns:
            Scalar penalty for energy balance violation.
        """
        if z.ndim != 3:
            # Energy balance violation only defined for trajectories
            return torch.tensor(0.0, device=z.device)

        # Extract concentrations and temperature
        # Assume last state variable is temperature
        C = z[..., :-1]  # (batch, time, num_species)
        T = z[..., -1]  # (batch, time)

        if (
            self.heat_capacity is not None
            and self.heats_of_reaction is not None
            and self.stoich_matrix is not None
        ):
            # Estimate reaction extents from concentration changes via stoichiometry
            delta_C = torch.diff(C, dim=1)  # (batch, time-1, n_species)
            S = self.stoich_matrix.to(z.device)  # (n_reactions, n_species)
            ST_pinv = torch.linalg.pinv(S.T)  # (n_reactions, n_species)
            delta_xi = torch.einsum("rn,btn->btr", ST_pinv, delta_C)  # (batch, time-1, n_reactions)
            H_rxn = self.heats_of_reaction.to(z.device)  # (n_reactions,)
            # Expected ΔT = -ΔH_rxn · Δξ / Cp (adiabatic energy balance)
            delta_T_expected = -torch.einsum("r,btr->bt", H_rxn, delta_xi) / self.heat_capacity
            delta_T_actual = torch.diff(T, dim=1)  # (batch, time-1)
            violation = torch.mean((delta_T_actual - delta_T_expected) ** 2)
        elif self.heat_capacity is not None:
            # Only Cp available: penalize changes inconsistent with concentration changes
            # ΔE_total = Cp * ΔT * Σ(C_i) — penalize imbalance between ΔT and Σ(ΔC_i)
            delta_T = torch.diff(T, dim=1)  # (batch, time-1)
            delta_C_total = torch.diff(C.sum(dim=-1), dim=1)  # (batch, time-1)
            # Heuristic: Cp * ΔT should scale with Σ(ΔC_i); penalize mismatch
            violation = torch.mean((self.heat_capacity * delta_T + delta_C_total) ** 2)
        else:
            # Without Cp, just penalize large temperature swings
            temp_change = torch.diff(T, dim=1)  # (batch, time-1)
            violation = torch.mean(temp_change**2)

        return violation


__all__ = ["EnergyBalanceConstraint"]
