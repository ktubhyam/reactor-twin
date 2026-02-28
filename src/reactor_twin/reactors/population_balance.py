"""Population Balance (Crystallization) reactor using Method of Moments."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from reactor_twin.exceptions import ConfigurationError
from reactor_twin.reactors.base import AbstractReactor
from reactor_twin.utils.registry import REACTOR_REGISTRY

logger = logging.getLogger(__name__)


@REACTOR_REGISTRY.register("population_balance")
class PopulationBalanceReactor(AbstractReactor):
    """Crystallization reactor modelled via Method of Moments.

    Tracks solute concentration *C* and the first *N* moments of the
    crystal size distribution (mu_0 .. mu_{N-1}).

    State: [C, mu_0, mu_1, ..., mu_{N-1}, (T)]

    Physics:
        Supersaturation : S   = (C - C_sat) / C_sat
        Growth rate     : G   = kg * max(S, 0)^g
        Nucleation rate : B0  = kb * max(S, 0)^b
        Moment ODEs     : dmu_0/dt  = B0
                          dmu_k/dt  = k * G * mu_{k-1}   (k >= 1)
        Mass balance    : dC/dt = -3 * kv * rho_crystal * G * mu_2

    Derived quantities (not part of state):
        mean_size = mu_1 / mu_0
        CV = sqrt(mu_2 * mu_0 / mu_1^2 - 1)

    Required parameters:
        V:             Reactor volume (L)
        C_sat:         Saturation concentration (mol/L)
        kg:            Growth-rate constant
        g:             Growth-rate exponent
        kb:            Nucleation-rate constant
        b:             Nucleation-rate exponent
        shape_factor:  Crystal shape factor kv (dimensionless)
        rho_crystal:   Crystal density (g/L)

    Optional:
        num_moments:   Number of moments to track (default 4: mu_0..mu_3)

    Attributes:
        num_moments: Number of distribution moments tracked.
        isothermal: If True, ignore energy balance.
    """

    def __init__(
        self,
        name: str,
        num_species: int,
        params: dict[str, Any],
        isothermal: bool = True,
        num_moments: int = 4,
    ):
        self.isothermal = isothermal
        self.num_moments = num_moments
        super().__init__(name, num_species, params)

        required = ["V", "C_sat", "kg", "g", "kb", "b", "shape_factor", "rho_crystal"]
        for key in required:
            if key not in params:
                raise ConfigurationError(f"Missing required parameter: {key}")

        if not isothermal:
            for key in ["rho", "Cp"]:
                if key not in params:
                    raise ConfigurationError(
                        f"Non-isothermal population balance reactor requires parameter: {key}"
                    )

    def _compute_state_dim(self) -> int:
        # 1 (concentration) + num_moments + optional temperature
        dim = 1 + self.num_moments
        if not self.isothermal:
            dim += 1
        return dim

    def _supersaturation(self, C: float) -> float:
        C_sat = self.params["C_sat"]
        return (C - C_sat) / C_sat

    def _growth_rate(self, S: float) -> float:
        kg = self.params["kg"]
        g = self.params["g"]
        return kg * max(S, 0.0) ** g

    def _nucleation_rate(self, S: float) -> float:
        kb = self.params["kb"]
        b = self.params["b"]
        return kb * max(S, 0.0) ** b

    def ode_rhs(
        self,
        t: float,
        y: np.ndarray,
        u: np.ndarray | None = None,
    ) -> np.ndarray:
        # Extract state
        C = y[0]
        mu = y[1 : 1 + self.num_moments]

        S = self._supersaturation(C)
        G = self._growth_rate(S)
        B0 = self._nucleation_rate(S)

        kv = self.params["shape_factor"]
        rho_c = self.params["rho_crystal"]

        # Moment ODEs
        dmu_dt = np.zeros(self.num_moments)
        dmu_dt[0] = B0
        for k in range(1, self.num_moments):
            dmu_dt[k] = k * G * mu[k - 1]

        # Mass balance: solute consumed by crystal growth
        # Ensure mu_2 is available (need at least 3 moments)
        if self.num_moments >= 3:
            mu_2 = mu[2]
        else:
            mu_2 = 0.0
        dC_dt = -3.0 * kv * rho_c * G * mu_2

        derivatives = [np.array([dC_dt]), dmu_dt]

        if not self.isothermal:
            # Simplified: no heat source, adiabatic
            derivatives.append(np.array([0.0]))

        return np.concatenate(derivatives)

    def get_initial_state(self) -> np.ndarray:
        C0 = self.params.get("C_initial", self.params.get("C_sat", 1.0) * 1.5)
        mu0 = np.array(self.params.get("mu_initial", np.zeros(self.num_moments)))
        state = [np.array([C0]), mu0]
        if not self.isothermal:
            T0 = self.params.get("T_initial", 298.15)
            state.append(np.array([T0]))
        return np.concatenate(state)

    def get_state_labels(self) -> list[str]:
        labels = ["C"]
        labels += [f"mu_{k}" for k in range(self.num_moments)]
        if not self.isothermal:
            labels.append("T")
        return labels

    def mean_size(self, y: np.ndarray) -> float:
        """Compute mean crystal size mu_1 / mu_0."""
        mu_0 = y[1]
        mu_1 = y[2] if self.num_moments >= 2 else 0.0
        if mu_0 <= 0:
            return 0.0
        return mu_1 / mu_0

    def coefficient_of_variation(self, y: np.ndarray) -> float:
        """Compute CV = sqrt(mu_2*mu_0/mu_1^2 - 1)."""
        if self.num_moments < 3:
            return 0.0
        mu_0, mu_1, mu_2 = y[1], y[2], y[3]
        if mu_1 <= 0 or mu_0 <= 0:
            return 0.0
        ratio = mu_2 * mu_0 / (mu_1**2)
        if ratio < 1.0:
            return 0.0
        return float(np.sqrt(ratio - 1.0))

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> PopulationBalanceReactor:
        return cls(
            name=config["name"],
            num_species=config["num_species"],
            params=config["params"],
            isothermal=config.get("isothermal", True),
            num_moments=config.get("num_moments", 4),
        )


__all__ = ["PopulationBalanceReactor"]
