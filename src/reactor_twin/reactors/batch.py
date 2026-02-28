"""Batch reactor implementation."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import numpy.typing as npt

from reactor_twin.exceptions import ConfigurationError
from reactor_twin.reactors.base import AbstractReactor
from reactor_twin.reactors.kinetics.base import AbstractKinetics
from reactor_twin.utils.registry import REACTOR_REGISTRY

logger = logging.getLogger(__name__)


@REACTOR_REGISTRY.register("batch")
class BatchReactor(AbstractReactor):
    """Batch reactor with time-varying volume (for gas-phase reactions).

    A closed system with no inflow or outflow. For liquid-phase reactions,
    volume is typically constant. For gas-phase reactions, volume can change
    with pressure/temperature.

    State variables:
        - C_i: Concentrations of species i (mol/L)
        - V: Volume (L) (optional, for gas-phase)
        - T: Temperature (K) (optional, for non-isothermal)

    Mass balance: dC_i/dt = Σ(nu_ij * r_j)
    Energy balance: ρ Cp V dT/dt = Σ(ΔH_j * r_j * V) + Q_heat

    Attributes:
        kinetics: Reaction kinetics model.
        isothermal: If True, ignore energy balance.
        constant_volume: If True, volume is fixed.
    """

    def __init__(
        self,
        name: str,
        num_species: int,
        params: dict[str, Any],
        kinetics: AbstractKinetics | None = None,
        isothermal: bool = True,
        constant_volume: bool = True,
    ):
        """Initialize batch reactor.

        Args:
            name: Reactor identifier.
            num_species: Number of chemical species.
            params: Reactor parameters (V0, T0, rho, Cp, etc.).
            kinetics: Reaction kinetics model.
            isothermal: If True, operate at constant temperature.
            constant_volume: If True, volume is constant.
        """
        self.kinetics = kinetics
        self.isothermal = isothermal
        self.constant_volume = constant_volume
        super().__init__(name, num_species, params)

        # Validate required parameters
        required = ["V", "T"]
        for key in required:
            if key not in params:
                raise ConfigurationError(f"Missing required parameter: {key}")

        if not isothermal:
            required_thermo = ["rho", "Cp"]
            for key in required_thermo:
                if key not in params:
                    raise ConfigurationError(
                        f"Non-isothermal batch reactor requires parameter: {key}"
                    )

    def _compute_state_dim(self) -> int:
        """Compute state dimension.

        Returns:
            num_species + 1 (temperature) if non-isothermal,
            num_species + 1 (volume) if variable volume,
            num_species + 2 (volume + temperature) if both,
            else num_species.
        """
        dim = self.num_species
        if not self.constant_volume:
            dim += 1  # Volume
        if not self.isothermal:
            dim += 1  # Temperature
        return dim

    def ode_rhs(
        self,
        t: float,
        y: npt.NDArray[Any],
        u: npt.NDArray[Any] | None = None,
    ) -> npt.NDArray[Any]:
        """Batch reactor ODE right-hand side.

        Mass balance: dC_i/dt = Σ(nu_ij * r_j)
        Energy balance (non-isothermal): dT/dt = Σ(ΔH_j * r_j) / (rho * Cp) + Q / (rho * Cp * V)

        Args:
            t: Time (scalar).
            y: State vector [C_1, ..., C_n] or [C_1, ..., C_n, T] or [C_1, ..., C_n, V, T].
            u: Control inputs (e.g., external heating Q).

        Returns:
            Time derivative dy/dt.
        """
        # Extract states
        idx = 0
        C = y[idx : idx + self.num_species]
        idx += self.num_species

        if not self.constant_volume:
            V = y[idx]
            idx += 1
        else:
            V = self.params["V"]

        if not self.isothermal:
            T = y[idx]
        else:
            T = self.params["T"]

        # Compute reaction rates
        if self.kinetics is not None:
            rates = self.kinetics.compute_rates(C, T)
        else:
            rates = np.zeros(self.num_species)

        # Mass balances: dC/dt = sum(nu_ij * r_j)
        dC_dt = rates

        # Volume change (for gas-phase reactions, ideal gas law)
        if not self.constant_volume:
            # For ideal gas at constant P, T: V ~ n_total
            # dV/dt = V * (dn/dt) / n_total
            dn_dt = rates.sum()  # Net mole change rate
            C_total = max(C.sum(), 1e-10)  # Guard against division by zero
            dV_dt = dn_dt * V / C_total
        else:
            dV_dt = None

        # Energy balance
        if not self.isothermal:
            rho = self.params["rho"]
            Cp = self.params["Cp"]
            Q_ext = u[0] if u is not None else 0.0  # External heating (W)

            # Heat of reaction: Q_rxn = sum(dH_j * r_j) where r_j are per-reaction rates
            dH_rxn = self.params.get("dH_rxn")
            if dH_rxn is not None and self.kinetics is not None:
                dH_rxn_arr = np.array(dH_rxn)
                rxn_rates = self.kinetics.compute_reaction_rates(C, T)
                heat_of_reaction = np.dot(dH_rxn_arr, rxn_rates) * V  # J/min
            else:
                heat_of_reaction = 0.0

            dT_dt = heat_of_reaction / (rho * Cp) + Q_ext / (rho * Cp * V)
        else:
            dT_dt = None

        # Assemble derivative
        dy_dt = [dC_dt]
        if dV_dt is not None:
            dy_dt.append(np.array([dV_dt]))
        if dT_dt is not None:
            dy_dt.append(np.array([dT_dt]))

        return np.concatenate(dy_dt)

    def get_initial_state(self) -> npt.NDArray[Any]:
        """Get initial conditions.

        Returns:
            Initial state [C_1, ..., C_n] or [C_1, ..., C_n, T] or [C_1, ..., C_n, V, T].
        """
        C0 = np.array(self.params.get("C_initial", np.zeros(self.num_species)))

        state = [C0]

        if not self.constant_volume:
            V0 = self.params.get("V_initial", self.params["V"])
            state.append(np.array([V0]))

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

        if not self.constant_volume:
            labels.append("V")

        if not self.isothermal:
            labels.append("T")

        return labels

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> BatchReactor:
        """Deserialize batch reactor from configuration.

        Args:
            config: Configuration dictionary.

        Returns:
            BatchReactor instance.
        """
        return cls(
            name=config["name"],
            num_species=config["num_species"],
            params=config["params"],
            isothermal=config.get("isothermal", True),
            constant_volume=config.get("constant_volume", True),
        )


__all__ = ["BatchReactor"]
