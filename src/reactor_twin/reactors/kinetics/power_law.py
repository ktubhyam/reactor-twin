"""Power law kinetics for empirical rate expressions."""

from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from reactor_twin.reactors.kinetics.base import AbstractKinetics
from reactor_twin.utils.registry import KINETICS_REGISTRY

logger = logging.getLogger(__name__)


@KINETICS_REGISTRY.register("power_law")
class PowerLawKinetics(AbstractKinetics):
    """Power law kinetics: r = k * Π(C_i^alpha_i).

    General empirical rate expression where reaction rate is proportional
    to concentrations raised to arbitrary powers (not necessarily stoichiometric).

    Rate law: r_j = k_j * Π(C_i^alpha_ij)

    Temperature dependence can be included via k(T) = A * exp(-E_a/(R*T)).

    Parameters:
        k: Rate constants, shape (num_reactions,).
        orders: Reaction orders (exponents), shape (num_reactions, num_species).
        stoich: Stoichiometric coefficients, shape (num_reactions, num_species).
        A: Pre-exponential factors (optional), shape (num_reactions,).
        E_a: Activation energies (optional), shape (num_reactions,).
    """

    def __init__(self, name: str, num_reactions: int, params: dict[str, Any]):
        """Initialize power law kinetics.

        Args:
            name: Kinetics identifier.
            num_reactions: Number of reactions.
            params: Dictionary with keys 'k', 'orders', 'stoich',
                   optionally 'A', 'E_a'.
        """
        super().__init__(name, num_reactions, params)

        # Validate parameters
        required = ["k", "orders", "stoich"]
        for key in required:
            if key not in params:
                raise ValueError(f"Missing required parameter: {key}")

        self.k = np.array(params["k"])
        self.orders = np.array(params["orders"])
        self.stoich = np.array(params["stoich"])

        # Temperature dependence (optional)
        self.A = params.get("A")
        self.E_a = params.get("E_a")
        if self.A is not None:
            self.A = np.array(self.A)
        if self.E_a is not None:
            self.E_a = np.array(self.E_a)

        # Validate shapes
        assert self.k.shape == (num_reactions,), "k shape mismatch"
        assert self.orders.shape[0] == num_reactions, "orders shape mismatch"
        assert self.stoich.shape[0] == num_reactions, "stoich shape mismatch"

    def compute_rates(
        self,
        concentrations: npt.NDArray[Any],
        temperature: float,
    ) -> npt.NDArray[Any]:
        """Compute reaction rates using power law.

        Args:
            concentrations: Species concentrations, shape (num_species,).
            temperature: Temperature in Kelvin.

        Returns:
            Net production rates for each species, shape (num_species,).
        """
        # Rate constants (with temperature dependence if provided)
        if self.A is not None and self.E_a is not None:
            # k(T) = A * exp(-E_a/(R*T))
            from reactor_twin.utils.constants import R_GAS

            k_T = self.A * np.exp(-self.E_a / (R_GAS * temperature))
        else:
            k_T = self.k

        # Compute reaction rates: r_j = k_j * Π(C_i^alpha_ij)
        reaction_rates = k_T.copy()

        for j in range(self.num_reactions):
            for i in range(len(concentrations)):
                if self.orders[j, i] != 0:
                    # Avoid 0^0 and negative concentrations
                    C_safe = max(concentrations[i], 1e-10)
                    reaction_rates[j] *= C_safe ** self.orders[j, i]

        # Apply stoichiometry: net_rate_i = sum_j(nu_ij * r_j)
        net_rates = self.stoich.T @ reaction_rates

        return cast(npt.NDArray[Any], net_rates)

    def validate_parameters(self) -> bool:
        """Validate kinetic parameters.

        Returns:
            True if parameters are valid.
        """
        # Check k > 0
        if not np.all(self.k > 0):
            logger.error("k must be positive")
            return False

        # Check A > 0 (if present)
        if self.A is not None and not np.all(self.A > 0):
            logger.error("A must be positive")
            return False

        # Check E_a >= 0 (if present)
        if self.E_a is not None and not np.all(self.E_a >= 0):
            logger.error("E_a must be non-negative")
            return False

        return True

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> PowerLawKinetics:
        """Deserialize power law kinetics from configuration.

        Args:
            config: Configuration dictionary.

        Returns:
            PowerLawKinetics instance.
        """
        return cls(
            name=config["name"],
            num_reactions=config["num_reactions"],
            params=config["params"],
        )


__all__ = ["PowerLawKinetics"]
