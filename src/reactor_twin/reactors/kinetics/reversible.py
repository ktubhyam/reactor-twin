"""Reversible reaction kinetics with equilibrium constraints."""

from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from reactor_twin.reactors.kinetics.base import AbstractKinetics
from reactor_twin.utils.registry import KINETICS_REGISTRY

logger = logging.getLogger(__name__)


@KINETICS_REGISTRY.register("reversible")
class ReversibleKinetics(AbstractKinetics):
    """Reversible reaction kinetics with forward and reverse rates.

    For equilibrium-limited reactions: A + B ⇌ C + D

    Rate law: r = k_f * Π(C_reactants^alpha) - k_r * Π(C_products^beta)

    At equilibrium: r = 0
        k_f / k_r = K_eq = Π(C_products_eq^nu_p) / Π(C_reactants_eq^nu_r)

    Temperature dependence (van 't Hoff):
        K_eq(T) = K_eq0 * exp(-ΔH_rxn / R * (1/T - 1/T0))
        k_f(T) = A_f * exp(-E_a_f / (R*T))
        k_r(T) = k_f(T) / K_eq(T)

    Or specify both rate constants independently:
        k_f(T) = A_f * exp(-E_a_f / (R*T))
        k_r(T) = A_r * exp(-E_a_r / (R*T))

    Parameters:
        k_f: Forward rate constants, shape (num_reactions,).
        k_r: Reverse rate constants, shape (num_reactions,).
        orders_f: Forward reaction orders, shape (num_reactions, num_species).
        orders_r: Reverse reaction orders, shape (num_reactions, num_species).
        stoich: Stoichiometric coefficients (net), shape (num_reactions, num_species).
        A_f: Forward pre-exponential (optional), shape (num_reactions,).
        E_a_f: Forward activation energy (optional), shape (num_reactions,).
        A_r: Reverse pre-exponential (optional), shape (num_reactions,).
        E_a_r: Reverse activation energy (optional), shape (num_reactions,).
        K_eq: Equilibrium constants (alternative to k_r), shape (num_reactions,).
    """

    def __init__(self, name: str, num_reactions: int, params: dict[str, Any]):
        """Initialize reversible kinetics.

        Args:
            name: Kinetics identifier.
            num_reactions: Number of reactions.
            params: Dictionary with keys 'k_f', 'k_r' OR 'K_eq', 'orders_f',
                   'orders_r', 'stoich', optionally 'A_f', 'E_a_f', 'A_r', 'E_a_r'.
        """
        super().__init__(name, num_reactions, params)

        # Validate parameters
        required = ["k_f", "orders_f", "orders_r", "stoich"]
        for key in required:
            if key not in params:
                raise ValueError(f"Missing required parameter: {key}")

        self.k_f = np.array(params["k_f"])
        self.orders_f = np.array(params["orders_f"])
        self.orders_r = np.array(params["orders_r"])
        self.stoich = np.array(params["stoich"])

        # Reverse rate: either k_r or K_eq (will compute k_r = k_f / K_eq)
        if "k_r" in params:
            self.k_r = np.array(params["k_r"])
            self.K_eq = None
        elif "K_eq" in params:
            self.K_eq = np.array(params["K_eq"])
            self.k_r = self.k_f / self.K_eq
        else:
            raise ValueError("Must provide either 'k_r' or 'K_eq'")

        # Temperature dependence (optional)
        self.A_f = params.get("A_f")
        self.E_a_f = params.get("E_a_f")
        self.A_r = params.get("A_r")
        self.E_a_r = params.get("E_a_r")

        if self.A_f is not None:
            self.A_f = np.array(self.A_f)
        if self.E_a_f is not None:
            self.E_a_f = np.array(self.E_a_f)
        if self.A_r is not None:
            self.A_r = np.array(self.A_r)
        if self.E_a_r is not None:
            self.E_a_r = np.array(self.E_a_r)

        # Validate shapes
        assert self.k_f.shape == (num_reactions,), "k_f shape mismatch"
        assert self.k_r.shape == (num_reactions,), "k_r shape mismatch"
        assert self.orders_f.shape[0] == num_reactions, "orders_f shape mismatch"
        assert self.orders_r.shape[0] == num_reactions, "orders_r shape mismatch"
        assert self.stoich.shape[0] == num_reactions, "stoich shape mismatch"

    def compute_rates(
        self,
        concentrations: npt.NDArray[Any],
        temperature: float,
    ) -> npt.NDArray[Any]:
        """Compute reaction rates for reversible reactions.

        Args:
            concentrations: Species concentrations, shape (num_species,).
            temperature: Temperature in Kelvin.

        Returns:
            Net production rates for each species, shape (num_species,).
        """
        from reactor_twin.utils.constants import R_GAS

        # Forward rate constants (with temperature dependence if provided)
        if self.A_f is not None and self.E_a_f is not None:
            k_f_T = self.A_f * np.exp(-self.E_a_f / (R_GAS * temperature))
        else:
            k_f_T = self.k_f

        # Reverse rate constants (with temperature dependence if provided)
        if self.A_r is not None and self.E_a_r is not None:
            k_r_T = self.A_r * np.exp(-self.E_a_r / (R_GAS * temperature))
        else:
            k_r_T = self.k_r

        # Compute reaction rates: r_j = r_f - r_r
        reaction_rates = np.zeros(self.num_reactions)

        for j in range(self.num_reactions):
            # Forward rate: r_f = k_f * Π(C_i^alpha_i)
            r_forward = k_f_T[j]
            for i in range(len(concentrations)):
                if self.orders_f[j, i] != 0:
                    C_safe = max(concentrations[i], 1e-10)
                    r_forward *= C_safe ** self.orders_f[j, i]

            # Reverse rate: r_r = k_r * Π(C_i^beta_i)
            r_reverse = k_r_T[j]
            for i in range(len(concentrations)):
                if self.orders_r[j, i] != 0:
                    C_safe = max(concentrations[i], 1e-10)
                    r_reverse *= C_safe ** self.orders_r[j, i]

            # Net rate
            reaction_rates[j] = r_forward - r_reverse

        # Apply stoichiometry: net_rate_i = sum_j(nu_ij * r_j)
        net_rates = self.stoich.T @ reaction_rates

        return cast(npt.NDArray[Any], net_rates)

    def validate_parameters(self) -> bool:
        """Validate kinetic parameters.

        Returns:
            True if parameters are valid.
        """
        # Check k_f > 0
        if not np.all(self.k_f > 0):
            logger.error("k_f must be positive")
            return False

        # Check k_r > 0
        if not np.all(self.k_r > 0):
            logger.error("k_r must be positive")
            return False

        # Check K_eq > 0 (if present)
        if self.K_eq is not None and not np.all(self.K_eq > 0):
            logger.error("K_eq must be positive")
            return False

        # Check A_f > 0 (if present)
        if self.A_f is not None and not np.all(self.A_f > 0):
            logger.error("A_f must be positive")
            return False

        # Check E_a_f >= 0 (if present)
        if self.E_a_f is not None and not np.all(self.E_a_f >= 0):
            logger.error("E_a_f must be non-negative")
            return False

        # Check A_r > 0 (if present)
        if self.A_r is not None and not np.all(self.A_r > 0):
            logger.error("A_r must be positive")
            return False

        # Check E_a_r >= 0 (if present)
        if self.E_a_r is not None and not np.all(self.E_a_r >= 0):
            logger.error("E_a_r must be non-negative")
            return False

        return True

    def get_equilibrium_constant(self, temperature: float) -> npt.NDArray[Any]:
        """Get equilibrium constants at given temperature.

        Args:
            temperature: Temperature in Kelvin.

        Returns:
            K_eq values, shape (num_reactions,).
        """
        from reactor_twin.utils.constants import R_GAS

        if (
            self.A_f is not None
            and self.E_a_f is not None
            and self.A_r is not None
            and self.E_a_r is not None
        ):
            k_f_T = self.A_f * np.exp(-self.E_a_f / (R_GAS * temperature))
            k_r_T = self.A_r * np.exp(-self.E_a_r / (R_GAS * temperature))
            return cast(npt.NDArray[Any], k_f_T / k_r_T)
        elif self.K_eq is not None:
            return self.K_eq
        else:
            return cast(npt.NDArray[Any], self.k_f / self.k_r)

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> ReversibleKinetics:
        """Deserialize reversible kinetics from configuration.

        Args:
            config: Configuration dictionary.

        Returns:
            ReversibleKinetics instance.
        """
        return cls(
            name=config["name"],
            num_reactions=config["num_reactions"],
            params=config["params"],
        )


__all__ = ["ReversibleKinetics"]
