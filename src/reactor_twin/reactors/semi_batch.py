"""Semi-batch reactor implementation."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from reactor_twin.reactors.base import AbstractReactor
from reactor_twin.reactors.kinetics.base import AbstractKinetics
from reactor_twin.utils.registry import REACTOR_REGISTRY

logger = logging.getLogger(__name__)


@REACTOR_REGISTRY.register("semi_batch")
class SemiBatchReactor(AbstractReactor):
    """Semi-batch reactor with continuous feed and no outflow.

    A reactor with continuous inflow of reactants but no outflow (batch-like).
    Volume increases over time due to feed. Common in pharmaceutical and
    specialty chemical manufacturing.

    State variables:
        - C_i: Concentrations of species i (mol/L)
        - V: Volume (L)
        - T: Temperature (K) (optional, for non-isothermal)

    Mass balance: dC_i/dt = Σ(nu_ij * r_j) + (F_in/V) * (C_in,i - C_i)
    Volume balance: dV/dt = F_in
    Energy balance: ρ Cp V dT/dt = Σ(ΔH_j * r_j * V) + F_in * ρ * Cp * (T_in - T) + Q_heat

    Attributes:
        kinetics: Reaction kinetics model.
        isothermal: If True, ignore energy balance.
        feed_rate: Volumetric flow rate F_in (L/min).
        feed_concentrations: Inlet concentrations C_in (mol/L).
    """

    def __init__(
        self,
        name: str,
        num_species: int,
        params: dict[str, Any],
        kinetics: AbstractKinetics | None = None,
        isothermal: bool = True,
    ):
        """Initialize semi-batch reactor.

        Args:
            name: Reactor identifier.
            num_species: Number of chemical species.
            params: Reactor parameters (V0, T0, F_in, C_in, rho, Cp, etc.).
            kinetics: Reaction kinetics model.
            isothermal: If True, operate at constant temperature.
        """
        self.kinetics = kinetics
        self.isothermal = isothermal
        super().__init__(name, num_species, params)

        # Validate required parameters
        required = ["V", "T", "F_in", "C_in"]
        for key in required:
            if key not in params:
                raise ValueError(f"Missing required parameter: {key}")

        if not isothermal:
            required_thermo = ["rho", "Cp", "T_in"]
            for key in required_thermo:
                if key not in params:
                    raise ValueError(f"Non-isothermal semi-batch reactor requires parameter: {key}")

    def _compute_state_dim(self) -> int:
        """Compute state dimension.

        Returns:
            num_species + 1 (volume) + 1 (temperature) if non-isothermal,
            else num_species + 1 (volume).
        """
        dim = self.num_species + 1  # Always include volume
        if not self.isothermal:
            dim += 1  # Temperature
        return dim

    def ode_rhs(
        self,
        t: float,
        y: np.ndarray,
        u: np.ndarray | None = None,
    ) -> np.ndarray:
        """Semi-batch reactor ODE right-hand side.

        Mass balance: dC_i/dt = Σ(nu_ij * r_j) + (F_in/V) * (C_in,i - C_i)
        Volume balance: dV/dt = F_in
        Energy balance (non-isothermal): dT/dt = [Σ(ΔH_j * r_j) + F_in * ρ * Cp * (T_in - T)/V] / (ρ * Cp)

        Args:
            t: Time (scalar).
            y: State vector [C_1, ..., C_n, V] or [C_1, ..., C_n, V, T].
            u: Control inputs (e.g., F_in, T_in, Q).

        Returns:
            Time derivative dy/dt.
        """
        # Extract states
        idx = 0
        C = y[idx : idx + self.num_species]
        idx += self.num_species
        V = y[idx]
        idx += 1

        if not self.isothermal:
            T = y[idx]
        else:
            T = self.params["T"]

        # Get control inputs (allow dynamic feed rate/temp)
        F_in = u[0] if u is not None and len(u) > 0 else self.params["F_in"]
        C_in = np.array(self.params["C_in"])

        if not self.isothermal:
            T_in = u[1] if u is not None and len(u) > 1 else self.params["T_in"]
            Q_ext = u[2] if u is not None and len(u) > 2 else 0.0
        else:
            Q_ext = None

        # Compute reaction rates
        if self.kinetics is not None:
            rates = self.kinetics.compute_rates(C, T)
        else:
            rates = np.zeros(self.num_species)

        # Mass balances: dC/dt = rates + (F_in/V) * (C_in - C)
        # Note: dC/dt also affected by dilution from volume change
        # Total derivative: d(C*V)/dt = rates*V + F_in*C_in
        # Expanding: C*dV/dt + V*dC/dt = rates*V + F_in*C_in
        # Since dV/dt = F_in: C*F_in + V*dC/dt = rates*V + F_in*C_in
        # Therefore: dC/dt = rates + (F_in/V) * (C_in - C)
        dC_dt = rates + (F_in / V) * (C_in - C)

        # Volume balance: dV/dt = F_in
        dV_dt = F_in

        # Energy balance
        if not self.isothermal:
            rho = self.params["rho"]
            Cp = self.params["Cp"]

            # Heat of reaction: Q_rxn = sum(dH_j * r_j) where r_j are per-reaction rates
            dH_rxn = self.params.get("dH_rxn")
            if dH_rxn is not None and self.kinetics is not None:
                dH_rxn_arr = np.array(dH_rxn)
                rxn_rates = self.kinetics.compute_reaction_rates(C, T)
                heat_of_reaction = np.dot(dH_rxn_arr, rxn_rates) * V  # J/min
            else:
                heat_of_reaction = 0.0

            # Feed enthalpy change: F_in * rho * Cp * (T_in - T) / (rho * Cp * V)
            # Simplifies to: F_in * (T_in - T) / V
            feed_enthalpy_term = F_in * (T_in - T) / V

            dT_dt = heat_of_reaction / (rho * Cp) + feed_enthalpy_term + Q_ext / (rho * Cp * V)
        else:
            dT_dt = None

        # Assemble derivative
        dy_dt = [dC_dt, np.array([dV_dt])]
        if dT_dt is not None:
            dy_dt.append(np.array([dT_dt]))

        return np.concatenate(dy_dt)

    def get_initial_state(self) -> np.ndarray:
        """Get initial conditions.

        Returns:
            Initial state [C_1, ..., C_n, V] or [C_1, ..., C_n, V, T].
        """
        C0 = np.array(self.params.get("C_initial", np.zeros(self.num_species)))
        V0 = self.params.get("V_initial", self.params["V"])

        state = [C0, np.array([V0])]

        if not self.isothermal:
            T0 = self.params.get("T_initial", self.params["T"])
            state.append(np.array([T0]))

        return np.concatenate(state)

    def get_state_labels(self) -> list[str]:
        """Get state variable labels.

        Returns:
            List of labels.
        """
        labels = [f"C_{i}" for i in range(self.num_species)]
        labels.append("V")

        if not self.isothermal:
            labels.append("T")

        return labels

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> SemiBatchReactor:
        """Deserialize semi-batch reactor from configuration.

        Args:
            config: Configuration dictionary.

        Returns:
            SemiBatchReactor instance.
        """
        return cls(
            name=config["name"],
            num_species=config["num_species"],
            params=config["params"],
            isothermal=config.get("isothermal", True),
        )


__all__ = ["SemiBatchReactor"]
