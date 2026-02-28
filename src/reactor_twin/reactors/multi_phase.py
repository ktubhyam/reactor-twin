"""Gas-Liquid (Multi-Phase) reactor implementation."""

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


@REACTOR_REGISTRY.register("multi_phase")
class MultiPhaseReactor(AbstractReactor):
    """Gas-Liquid multi-phase reactor.

    Models a stirred tank with separate gas and liquid phases connected
    by inter-phase mass transfer (Henry's law + kLa model).

    State variables:
        - C_L_i: Liquid-phase concentrations (mol/L), i = 0..n_liquid-1
        - C_G_j: Gas-phase concentrations (mol/L), j = 0..n_gas-1
        - T: Temperature (K) (optional, for non-isothermal)

    Physics:
        Henry's law:       C_L_eq_j = C_G_j / H_j
        Mass transfer:     N_j = kLa * (C_L_eq_j - C_L_j)
        Liquid balance:    dC_L_i/dt = (F_L/V_L)*(C_L_feed_i - C_L_i) + r_i
                                       + kLa*(C_eq_i - C_L_i)  [for transferred species]
        Gas balance:       dC_G_j/dt = (F_G/V_G)*(C_G_feed_j - C_G_j)
                                       - (kLa*V_L/V_G)*(C_eq_j - C_L_j)

    Parameters:
        V_L: Liquid volume (L)
        V_G: Gas volume (L)
        F_L: Liquid volumetric flow rate (L/min)
        F_G: Gas volumetric flow rate (L/min)
        kLa: Volumetric mass-transfer coefficient (1/min)
        H: Henry's law constants, one per gas species (dimensionless C_G/C_L_eq)
        C_L_feed: Liquid feed concentrations (mol/L)
        C_G_feed: Gas feed concentrations (mol/L)
        T_feed: Feed temperature (K)
        gas_species_indices: Indices in the liquid-phase vector that participate
                             in gas-liquid transfer (maps to gas species 0..m-1)

    Attributes:
        kinetics: Reaction kinetics model (acts on liquid phase).
        isothermal: If True, ignore energy balance.
        num_gas_species: Number of gas-phase species.
        num_liquid_species: Same as num_species (total liquid species).
    """

    def __init__(
        self,
        name: str,
        num_species: int,
        params: dict[str, Any],
        kinetics: AbstractKinetics | None = None,
        isothermal: bool = True,
    ):
        self.kinetics = kinetics
        self.isothermal = isothermal

        # Validate required params BEFORE super().__init__()
        required = [
            "V_L",
            "V_G",
            "F_L",
            "F_G",
            "kLa",
            "H",
            "C_L_feed",
            "C_G_feed",
            "T_feed",
            "gas_species_indices",
        ]
        for key in required:
            if key not in params:
                raise ConfigurationError(f"Missing required parameter: {key}")

        if not isothermal:
            for key in ["rho", "Cp", "UA", "T_coolant"]:
                if key not in params:
                    raise ConfigurationError(
                        f"Non-isothermal multi-phase reactor requires parameter: {key}"
                    )

        # Pre-compute before super().__init__ calls _compute_state_dim
        gas_indices = params.get("gas_species_indices", [])
        self.num_gas_species = len(gas_indices)

        super().__init__(name, num_species, params)

    def _compute_state_dim(self) -> int:
        dim = self.num_species + self.num_gas_species
        if not self.isothermal:
            dim += 1
        return dim

    def ode_rhs(
        self,
        t: float,
        y: npt.NDArray[Any],
        u: npt.NDArray[Any] | None = None,
    ) -> npt.NDArray[Any]:
        n_liq = self.num_species
        n_gas = self.num_gas_species

        # Extract states
        C_L = y[:n_liq]
        C_G = y[n_liq : n_liq + n_gas]
        T = y[n_liq + n_gas] if not self.isothermal else self.params["T_feed"]

        # Parameters
        V_L = self.params["V_L"]
        V_G = self.params["V_G"]
        F_L = self.params["F_L"]
        F_G = self.params["F_G"]
        kLa = self.params["kLa"]
        H = np.array(self.params["H"])
        C_L_feed = np.array(self.params["C_L_feed"])
        C_G_feed = np.array(self.params["C_G_feed"])
        gas_idx = np.array(self.params["gas_species_indices"])

        # Reaction rates (liquid phase only)
        if self.kinetics is not None:
            rates = self.kinetics.compute_rates(C_L, T)
        else:
            rates = np.zeros(n_liq)

        # Liquid balance: dilution + reaction
        dC_L_dt = (F_L / V_L) * (C_L_feed - C_L) + rates

        # Mass transfer for gas-transferring species
        C_L_eq = C_G / H  # equilibrium liquid conc
        transfer = kLa * (C_L_eq - C_L[gas_idx])
        dC_L_dt[gas_idx] += transfer

        # Gas balance
        dC_G_dt = (F_G / V_G) * (C_G_feed - C_G) - (kLa * V_L / V_G) * (C_L_eq - C_L[gas_idx])

        derivatives = [dC_L_dt, dC_G_dt]

        if not self.isothermal:
            rho = self.params["rho"]
            Cp = self.params["Cp"]
            UA = self.params["UA"]
            T_coolant = self.params["T_coolant"]
            T_feed = self.params["T_feed"]
            dT_dt = (F_L / V_L) * (T_feed - T) + (UA / (rho * Cp * V_L)) * (T_coolant - T)
            derivatives.append(np.array([dT_dt]))

        return np.concatenate(derivatives)

    def get_initial_state(self) -> npt.NDArray[Any]:
        C_L0 = np.array(self.params.get("C_L_initial", self.params["C_L_feed"]))
        C_G0 = np.array(self.params.get("C_G_initial", self.params["C_G_feed"]))
        state = [C_L0, C_G0]
        if not self.isothermal:
            T0 = self.params.get("T_initial", self.params["T_feed"])
            state.append(np.array([T0]))
        return np.concatenate(state)

    def get_state_labels(self) -> list[str]:
        labels = [f"C_L_{i}" for i in range(self.num_species)]
        labels += [f"C_G_{j}" for j in range(self.num_gas_species)]
        if not self.isothermal:
            labels.append("T")
        return labels

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> MultiPhaseReactor:
        return cls(
            name=config["name"],
            num_species=config["num_species"],
            params=config["params"],
            isothermal=config.get("isothermal", True),
        )


__all__ = ["MultiPhaseReactor"]
