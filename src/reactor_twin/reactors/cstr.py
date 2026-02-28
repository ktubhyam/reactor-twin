"""Continuous Stirred-Tank Reactor (CSTR) implementation."""

from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from reactor_twin.exceptions import ConfigurationError
from reactor_twin.reactors.base import AbstractReactor
from reactor_twin.reactors.kinetics.base import AbstractKinetics
from reactor_twin.utils.registry import REACTOR_REGISTRY

logger = logging.getLogger(__name__)


@REACTOR_REGISTRY.register("cstr")
class CSTRReactor(AbstractReactor):
    """Continuous Stirred-Tank Reactor (CSTR).

    Models a well-mixed reactor with continuous inlet and outlet streams.

    State variables:
        - C_i: Concentrations of species i (mol/L)
        - T: Temperature (K)

    Parameters:
        V: Reactor volume (L)
        F: Volumetric flow rate (L/min)
        rho: Density (g/L)
        Cp: Heat capacity (J/(g*K))
        UA: Heat transfer coefficient * area (J/(min*K))
        T_coolant: Coolant temperature (K)
        C_feed: Feed concentrations (mol/L)
        T_feed: Feed temperature (K)

    Attributes:
        kinetics: Reaction kinetics model.
        isothermal: If True, ignore energy balance.
    """

    def __init__(
        self,
        name: str,
        num_species: int,
        params: dict[str, Any],
        kinetics: AbstractKinetics | None = None,
        isothermal: bool = False,
    ):
        """Initialize CSTR reactor.

        Args:
            name: Reactor identifier.
            num_species: Number of chemical species.
            params: Reactor parameters (V, F, rho, Cp, UA, etc.).
            kinetics: Reaction kinetics model. Defaults to None.
            isothermal: If True, operate at constant temperature.
        """
        self.kinetics = kinetics
        self.isothermal = isothermal
        super().__init__(name, num_species, params)

        # Validate required parameters
        required = ["V", "F", "C_feed", "T_feed"]
        for key in required:
            if key not in params:
                raise ConfigurationError(f"Missing required parameter: {key}")

        if not isothermal:
            required_thermo = ["rho", "Cp", "UA", "T_coolant"]
            for key in required_thermo:
                if key not in params:
                    raise ConfigurationError(f"Non-isothermal CSTR requires parameter: {key}")

    def _compute_state_dim(self) -> int:
        """Compute state dimension.

        Returns:
            num_species + 1 if non-isothermal, else num_species.
        """
        return self.num_species + (0 if self.isothermal else 1)

    def ode_rhs(
        self,
        t: float,
        y: npt.NDArray[Any],
        u: npt.NDArray[Any] | None = None,
    ) -> npt.NDArray[Any]:
        """CSTR ODE right-hand side.

        Mass balance: dC_i/dt = (F/V)*(C_feed_i - C_i) + sum_j(nu_ij * r_j)
        Energy balance: dT/dt = (F/V)*(T_feed - T) + sum_j(dH_j * r_j) / (rho*Cp*V)
                                + UA*(T_coolant - T) / (rho*Cp*V)

        Args:
            t: Time (scalar).
            y: State vector [C_1, ..., C_n, T] or [C_1, ..., C_n].
            u: Control inputs (optional, e.g., F, T_coolant).

        Returns:
            Time derivative dy/dt.
        """
        # Extract states
        C = y[: self.num_species]  # Concentrations
        T = y[self.num_species] if not self.isothermal else self.params["T_feed"]

        # Extract parameters
        V = self.params["V"]
        F = self.params["F"] if u is None else u[0]  # Flow can be control
        C_feed = np.array(self.params["C_feed"])
        T_feed = self.params["T_feed"]

        # Compute reaction rates
        if self.kinetics is not None:
            rates = self.kinetics.compute_rates(C, T)
        else:
            rates = np.zeros(self.num_species)

        # Mass balances
        dC_dt = (F / V) * (C_feed - C) + rates

        if self.isothermal:
            return cast(npt.NDArray[Any], dC_dt)

        # Energy balance
        rho = self.params["rho"]
        Cp = self.params["Cp"]
        UA = self.params["UA"]
        T_coolant = self.params["T_coolant"]

        # Heat of reaction: Q_rxn = sum(dH_j * r_j) where r_j are per-reaction rates
        dH_rxn = self.params.get("dH_rxn")
        if dH_rxn is not None and self.kinetics is not None:
            dH_rxn = np.array(dH_rxn)
            rxn_rates = self.kinetics.compute_reaction_rates(C, T)
            heat_of_reaction = np.dot(dH_rxn, rxn_rates) * V  # J/min
        else:
            heat_of_reaction = 0.0

        dT_dt = (
            (F / V) * (T_feed - T)
            + heat_of_reaction / (rho * Cp * V)
            + (UA / (rho * Cp * V)) * (T_coolant - T)
        )

        return cast(npt.NDArray[Any], np.concatenate([dC_dt, [dT_dt]]))

    def get_initial_state(self) -> npt.NDArray[Any]:
        """Get initial conditions.

        Returns:
            Initial state [C_1, ..., C_n, T] or [C_1, ..., C_n].
        """
        C0 = np.array(self.params.get("C_initial", self.params["C_feed"]))
        if self.isothermal:
            return C0
        T0 = self.params.get("T_initial", self.params["T_feed"])
        return np.concatenate([C0, [T0]])

    def get_state_labels(self) -> list[str]:
        """Get state variable labels.

        Returns:
            List of labels like ['C_A', 'C_B', 'T'].
        """
        labels = [f"C_{i}" for i in range(self.num_species)]
        if not self.isothermal:
            labels.append("T")
        return labels

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> CSTRReactor:
        """Deserialize CSTR from configuration dictionary.

        Args:
            config: Configuration dictionary.

        Returns:
            CSTR instance.
        """
        return cls(
            name=config["name"],
            num_species=config["num_species"],
            params=config["params"],
            isothermal=config.get("isothermal", False),
        )


__all__ = ["CSTRReactor"]
