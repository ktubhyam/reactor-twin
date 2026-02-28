"""Fluidized Bed reactor implementation (Kunii-Levenspiel two-phase model)."""

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


@REACTOR_REGISTRY.register("fluidized_bed")
class FluidizedBedReactor(AbstractReactor):
    """Fluidized bed reactor using the Kunii-Levenspiel two-phase model.

    Splits the bed into a bubble phase and an emulsion (dense) phase
    with inter-phase mass transfer.

    State variables:
        - C_b_i: Bubble-phase concentrations (mol/L), i = 0..n-1
        - C_e_i: Emulsion-phase concentrations (mol/L), i = 0..n-1
        - T: Temperature (K) (optional, for non-isothermal)

    Physics (Kunii-Levenspiel):
        Bubble rise velocity: u_b = u_0 - u_mf + 0.711 * sqrt(g * d_b)
        Bubble fraction:      delta = (u_0 - u_mf) / u_b
        Bubble volume:        V_b = delta * A_bed * H_bed
        Emulsion volume:      V_e = (1 - delta) * epsilon_mf * A_bed * H_bed
        Bubble:   dC_b_i/dt = (u_0*A_bed/V_b)*(C_feed_i - C_b_i)
                              - K_be*(C_b_i - C_e_i) + gamma_b*r_b_i
        Emulsion: dC_e_i/dt = K_be*(V_b/V_e)*(C_b_i - C_e_i) + r_e_i

    Parameters:
        u_mf: Minimum fluidization velocity (m/s)
        u_0: Superficial gas velocity (m/s)
        epsilon_mf: Voidage at minimum fluidization (0 < epsilon_mf < 1)
        d_b: Bubble diameter (m), must be > 0
        H_bed: Bed height (m)
        A_bed: Bed cross-sectional area (m^2)
        K_be: Bubble-emulsion interchange coefficient (1/s), must be >= 0
        C_feed: Feed concentrations (mol/L)
        T_feed: Feed temperature (K)
        gamma_b: Fraction of catalyst in bubble phase (default 0.01)

    Attributes:
        kinetics: Reaction kinetics model.
        isothermal: If True, ignore energy balance.
    """

    GRAVITY = 9.81  # m/s^2

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

        # Copy params to avoid mutating caller's dict
        params = dict(params)

        # Validate required params BEFORE super().__init__()
        required = [
            "u_mf",
            "u_0",
            "epsilon_mf",
            "d_b",
            "H_bed",
            "A_bed",
            "K_be",
            "C_feed",
            "T_feed",
        ]
        for key in required:
            if key not in params:
                raise ConfigurationError(f"Missing required parameter: {key}")

        # Validate physics constraints
        if params["u_0"] <= params["u_mf"]:
            raise ConfigurationError(
                f"u_0 ({params['u_0']}) must be greater than u_mf ({params['u_mf']}) "
                "for fluidization"
            )

        if params["d_b"] <= 0:
            raise ConfigurationError(f"d_b must be > 0, got {params['d_b']}")

        eps_mf = params["epsilon_mf"]
        if not (0 < eps_mf < 1):
            raise ConfigurationError(
                f"epsilon_mf must be between 0 and 1 (exclusive), got {eps_mf}"
            )

        if params["K_be"] < 0:
            raise ConfigurationError(f"K_be must be >= 0, got {params['K_be']}")

        if not isothermal:
            for key in ["rho", "Cp"]:
                if key not in params:
                    raise ConfigurationError(
                        f"Non-isothermal fluidized bed requires parameter: {key}"
                    )

        # Set default gamma_b
        if "gamma_b" not in params:
            params["gamma_b"] = 0.01

        super().__init__(name, num_species, params)

    def _compute_state_dim(self) -> int:
        # bubble + emulsion concentrations + optional T
        dim = 2 * self.num_species
        if not self.isothermal:
            dim += 1
        return dim

    def _bubble_rise_velocity(self) -> float:
        """Compute bubble rise velocity u_b (m/s)."""
        u_0 = self.params["u_0"]
        u_mf = self.params["u_mf"]
        d_b = self.params["d_b"]
        return cast(float, u_0 - u_mf + 0.711 * np.sqrt(self.GRAVITY * d_b))

    def _bubble_fraction(self) -> float:
        """Compute bubble volume fraction delta."""
        u_0 = self.params["u_0"]
        u_mf = self.params["u_mf"]
        u_b = self._bubble_rise_velocity()
        return cast(float, (u_0 - u_mf) / u_b)

    def _phase_volumes(self) -> tuple[float, float]:
        """Compute bubble and emulsion phase volumes.

        Returns:
            (V_b, V_e) in same units as A_bed * H_bed.
        """
        delta = self._bubble_fraction()
        A = self.params["A_bed"]
        H = self.params["H_bed"]
        eps_mf = self.params["epsilon_mf"]

        V_b = max(delta * A * H, 1e-12)
        V_e = max((1.0 - delta) * eps_mf * A * H, 1e-12)
        return V_b, V_e

    def ode_rhs(
        self,
        t: float,
        y: npt.NDArray[Any],
        u: npt.NDArray[Any] | None = None,
    ) -> npt.NDArray[Any]:
        n = self.num_species

        # Extract states
        C_b = y[:n]
        C_e = y[n : 2 * n]
        T = y[2 * n] if not self.isothermal else self.params["T_feed"]

        # Parameters
        u_0 = self.params["u_0"]
        A_bed = self.params["A_bed"]
        K_be = self.params["K_be"]
        gamma_b = self.params["gamma_b"]
        C_feed = np.array(self.params["C_feed"])

        V_b, V_e = self._phase_volumes()

        # Reaction rates — bubble phase uses C_b, emulsion uses C_e
        if self.kinetics is not None:
            rates_b = self.kinetics.compute_rates(C_b, T)
            rates_e = self.kinetics.compute_rates(C_e, T)
        else:
            rates_b = np.zeros(n)
            rates_e = np.zeros(n)

        # Bubble phase
        dC_b_dt = (u_0 * A_bed / V_b) * (C_feed - C_b) - K_be * (C_b - C_e) + gamma_b * rates_b

        # Emulsion phase
        dC_e_dt = K_be * (V_b / V_e) * (C_b - C_e) + rates_e

        derivatives = [dC_b_dt, dC_e_dt]

        if not self.isothermal:
            rho = self.params["rho"]
            Cp = self.params["Cp"]
            T_feed = self.params["T_feed"]
            V_total = V_b + V_e

            # Convective energy balance
            dT_dt = (u_0 * A_bed * rho * Cp * (T_feed - T)) / (rho * Cp * V_total)

            # Heat of reaction contribution
            if "dH_rxn" in self.params and self.kinetics is not None:
                dH_rxn = np.array(self.params["dH_rxn"])
                dT_dt += np.sum(dH_rxn * rates_e) / (rho * Cp)

            derivatives.append(np.array([dT_dt]))

        return np.concatenate(derivatives)

    def validate_state(self, y: npt.NDArray[Any]) -> bool:
        """Validate physical constraints on both phases.

        Args:
            y: State vector, shape (state_dim,).

        Returns:
            True if state is physically valid, False otherwise.
        """
        n = self.num_species
        C_b = y[:n]
        C_e = y[n : 2 * n]
        return bool(np.all(C_b >= 0) and np.all(C_e >= 0))

    def get_initial_state(self) -> npt.NDArray[Any]:
        C_feed = np.array(self.params["C_feed"])
        C_b0 = np.array(self.params.get("C_b_initial", C_feed))
        C_e0 = np.array(self.params.get("C_e_initial", C_feed))
        state = [C_b0, C_e0]
        if not self.isothermal:
            T0 = self.params.get("T_initial", self.params["T_feed"])
            state.append(np.array([T0]))
        return np.concatenate(state)

    def get_state_labels(self) -> list[str]:
        labels = [f"C_b_{i}" for i in range(self.num_species)]
        labels += [f"C_e_{i}" for i in range(self.num_species)]
        if not self.isothermal:
            labels.append("T")
        return labels

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> FluidizedBedReactor:
        return cls(
            name=config["name"],
            num_species=config["num_species"],
            params=config["params"],
            isothermal=config.get("isothermal", True),
        )


__all__ = ["FluidizedBedReactor"]
