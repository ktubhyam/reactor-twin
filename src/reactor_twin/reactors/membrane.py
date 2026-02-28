"""Membrane reactor implementation with selective permeation."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from reactor_twin.exceptions import ConfigurationError
from reactor_twin.reactors.base import AbstractReactor
from reactor_twin.reactors.kinetics.base import AbstractKinetics
from reactor_twin.utils.registry import REACTOR_REGISTRY

logger = logging.getLogger(__name__)


@REACTOR_REGISTRY.register("membrane")
class MembraneReactor(AbstractReactor):
    """Membrane reactor with selective species permeation.

    Models a reactor with retentate and permeate sides separated by a
    selective membrane.  Species transfer through the membrane follows
    either linear permeation or Sievert's law (for H2).

    State variables:
        - C_ret_i: Retentate-side concentrations (mol/L), i = 0..n_ret-1
        - C_perm_j: Permeate-side concentrations (mol/L), j = 0..n_perm-1
        - T: Temperature (K) (optional, for non-isothermal)

    Physics:
        Linear permeation: J_i = Q_i * A_membrane * (C_ret_i - C_perm_i)
        Sievert's law (H2): J = Q * A_membrane * (sqrt(C_ret) - sqrt(C_perm))
        Retentate:  dC_ret_i/dt = (F_ret/V_ret)*(C_feed_i - C_ret_i) + r_i - J_i/V_ret
        Permeate:   dC_perm_j/dt = (F_perm/V_perm)*(0 - C_perm_j) + J_j/V_perm

    Parameters:
        V_ret: Retentate volume (L)
        V_perm: Permeate volume (L)
        F_ret: Retentate volumetric flow rate (L/min)
        F_perm: Permeate volumetric flow rate (L/min)
        A_membrane: Membrane area (m^2)
        Q: Permeability coefficients for permeating species
        permeating_species_indices: Indices in retentate that permeate
        permeation_law: 'linear' or 'sievert'
        C_ret_feed: Retentate feed concentrations (mol/L)
        T_feed: Feed temperature (K)

    Attributes:
        kinetics: Reaction kinetics model (acts on retentate phase).
        isothermal: If True, ignore energy balance.
        num_permeating: Number of species that permeate through membrane.
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

        # Copy params to avoid mutating caller's dict
        params = dict(params)

        # Validate required params BEFORE super().__init__()
        required = [
            "V_ret",
            "V_perm",
            "F_ret",
            "F_perm",
            "A_membrane",
            "Q",
            "permeating_species_indices",
            "permeation_law",
            "C_ret_feed",
            "T_feed",
        ]
        for key in required:
            if key not in params:
                raise ConfigurationError(f"Missing required parameter: {key}")

        law = params["permeation_law"]
        if law not in ("linear", "sievert"):
            raise ConfigurationError(f"permeation_law must be 'linear' or 'sievert', got '{law}'")

        perm_indices = params["permeating_species_indices"]
        Q = params["Q"]

        # Validate Q length matches permeating species
        if len(Q) != len(perm_indices):
            raise ConfigurationError(
                f"len(Q)={len(Q)} must match len(permeating_species_indices)={len(perm_indices)}"
            )

        # Validate species indices are in range
        for idx in perm_indices:
            if not (0 <= idx < num_species):
                raise ConfigurationError(
                    f"permeating_species_indices contains {idx}, must be in [0, {num_species})"
                )

        # Validate C_ret_feed length
        C_ret_feed = params["C_ret_feed"]
        if len(C_ret_feed) != num_species:
            raise ConfigurationError(
                f"len(C_ret_feed)={len(C_ret_feed)} must match num_species={num_species}"
            )

        self.num_permeating = len(perm_indices)

        if not isothermal:
            for key in ["rho", "Cp", "UA", "T_coolant"]:
                if key not in params:
                    raise ConfigurationError(
                        f"Non-isothermal membrane reactor requires parameter: {key}"
                    )

        super().__init__(name, num_species, params)

    def _compute_state_dim(self) -> int:
        # retentate concentrations + permeate concentrations + optional T
        dim = self.num_species + self.num_permeating
        if not self.isothermal:
            dim += 1
        return dim

    def _compute_flux(self, C_ret_perm: np.ndarray, C_perm: np.ndarray) -> np.ndarray:
        """Compute permeation flux through membrane.

        Args:
            C_ret_perm: Retentate concentrations of permeating species.
            C_perm: Permeate concentrations.

        Returns:
            Flux array J, shape (num_permeating,).
        """
        A = self.params["A_membrane"]
        Q = np.array(self.params["Q"])
        law = self.params["permeation_law"]

        if law == "sievert":
            # Sievert's law: J = Q * A * (sqrt(C_ret) - sqrt(C_perm))
            J = Q * A * (np.sqrt(np.maximum(C_ret_perm, 0.0)) - np.sqrt(np.maximum(C_perm, 0.0)))
        else:
            # Linear: J = Q * A * (C_ret - C_perm)
            J = Q * A * (np.maximum(C_ret_perm, 0.0) - np.maximum(C_perm, 0.0))

        return J

    def ode_rhs(
        self,
        t: float,
        y: np.ndarray,
        u: np.ndarray | None = None,
    ) -> np.ndarray:
        n_ret = self.num_species
        n_perm = self.num_permeating

        # Extract states
        C_ret = y[:n_ret]
        C_perm = y[n_ret : n_ret + n_perm]
        T = y[n_ret + n_perm] if not self.isothermal else self.params["T_feed"]

        # Parameters
        V_ret = self.params["V_ret"]
        V_perm = self.params["V_perm"]
        F_ret = self.params["F_ret"]
        F_perm = self.params["F_perm"]
        C_ret_feed = np.array(self.params["C_ret_feed"])
        perm_idx = np.array(self.params["permeating_species_indices"])

        # Reaction rates (retentate phase only)
        if self.kinetics is not None:
            rates = self.kinetics.compute_rates(C_ret, T)
        else:
            rates = np.zeros(n_ret)

        # Retentate balance: dilution + reaction
        dC_ret_dt = (F_ret / V_ret) * (C_ret_feed - C_ret) + rates

        # Permeation flux
        J = self._compute_flux(C_ret[perm_idx], C_perm)

        # Subtract flux from retentate (permeating species)
        dC_ret_dt[perm_idx] -= J / V_ret

        # Permeate balance: sweep flow + incoming flux
        dC_perm_dt = (F_perm / V_perm) * (0.0 - C_perm) + J / V_perm

        derivatives = [dC_ret_dt, dC_perm_dt]

        if not self.isothermal:
            rho = self.params["rho"]
            Cp = self.params["Cp"]
            UA = self.params["UA"]
            T_coolant = self.params["T_coolant"]
            T_feed = self.params["T_feed"]
            dT_dt = (F_ret / V_ret) * (T_feed - T) + (UA / (rho * Cp * V_ret)) * (T_coolant - T)

            # Heat of reaction contribution
            if "dH_rxn" in self.params and self.kinetics is not None:
                dH_rxn = np.array(self.params["dH_rxn"])
                dT_dt += np.sum(dH_rxn * rates) / (rho * Cp * V_ret)

            derivatives.append(np.array([dT_dt]))

        return np.concatenate(derivatives)

    def validate_state(self, y: np.ndarray) -> bool:
        """Validate physical constraints on retentate and permeate phases.

        Args:
            y: State vector, shape (state_dim,).

        Returns:
            True if state is physically valid, False otherwise.
        """
        n_ret = self.num_species
        n_perm = self.num_permeating
        C_ret = y[:n_ret]
        C_perm = y[n_ret : n_ret + n_perm]
        return bool(np.all(C_ret >= 0) and np.all(C_perm >= 0))

    def get_initial_state(self) -> np.ndarray:
        C_ret0 = np.array(self.params.get("C_ret_initial", self.params["C_ret_feed"]))
        C_perm0 = np.array(self.params.get("C_perm_initial", np.zeros(self.num_permeating)))
        state = [C_ret0, C_perm0]
        if not self.isothermal:
            T0 = self.params.get("T_initial", self.params["T_feed"])
            state.append(np.array([T0]))
        return np.concatenate(state)

    def get_state_labels(self) -> list[str]:
        labels = [f"C_ret_{i}" for i in range(self.num_species)]
        labels += [f"C_perm_{j}" for j in range(self.num_permeating)]
        if not self.isothermal:
            labels.append("T")
        return labels

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> MembraneReactor:
        return cls(
            name=config["name"],
            num_species=config["num_species"],
            params=config["params"],
            isothermal=config.get("isothermal", True),
        )


__all__ = ["MembraneReactor"]
