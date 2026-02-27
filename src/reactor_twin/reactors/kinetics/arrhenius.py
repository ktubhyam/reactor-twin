"""Arrhenius kinetics implementation."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from reactor_twin.reactors.kinetics.base import AbstractKinetics
from reactor_twin.utils.registry import KINETICS_REGISTRY

logger = logging.getLogger(__name__)

# Gas constant (J/(mol*K))
R_GAS = 8.314


@KINETICS_REGISTRY.register("arrhenius")
class ArrheniusKinetics(AbstractKinetics):
    """Simple Arrhenius kinetics: r = k0 * exp(-Ea/(R*T)) * prod(C_i^order_i).

    Supports elementary and non-elementary reactions with arbitrary reaction orders.

    Parameters:
        k0: Pre-exponential factor, shape (num_reactions,).
        Ea: Activation energy (J/mol), shape (num_reactions,).
        orders: Reaction orders for each species, shape (num_reactions, num_species).
        stoich: Stoichiometric coefficients, shape (num_reactions, num_species).
    """

    def __init__(self, name: str, num_reactions: int, params: dict[str, Any]):
        """Initialize Arrhenius kinetics.

        Args:
            name: Kinetics identifier.
            num_reactions: Number of reactions.
            params: Dictionary with keys 'k0', 'Ea', 'orders', 'stoich'.
        """
        super().__init__(name, num_reactions, params)

        # Validate parameters
        required = ["k0", "Ea", "stoich"]
        for key in required:
            if key not in params:
                raise ValueError(f"Missing required parameter: {key}")

        self.k0 = np.array(params["k0"])
        self.Ea = np.array(params["Ea"])
        self.stoich = np.array(params["stoich"])  # (reactions, species)

        # Reaction orders default to stoichiometric coefficients for reactants
        if "orders" in params:
            self.orders = np.array(params["orders"])
        else:
            # For elementary reactions, order = stoichiometry for reactants
            self.orders = -np.minimum(self.stoich, 0)

        # Validate shapes
        assert self.k0.shape == (num_reactions,), "k0 shape mismatch"
        assert self.Ea.shape == (num_reactions,), "Ea shape mismatch"
        assert self.stoich.shape[0] == num_reactions, "stoich shape mismatch"
        assert self.orders.shape == self.stoich.shape, "orders shape mismatch"

    def compute_rates(
        self,
        concentrations: np.ndarray,
        temperature: float,
    ) -> np.ndarray:
        """Compute reaction rates using Arrhenius law.

        Args:
            concentrations: Species concentrations, shape (num_species,).
            temperature: Temperature in Kelvin.

        Returns:
            Net production rates for each species, shape (num_species,).
        """
        # Compute rate constants: k = k0 * exp(-Ea/(R*T))
        k = self.k0 * np.exp(-self.Ea / (R_GAS * temperature))

        # Compute reaction rates: r_j = k_j * prod(C_i^order_ij)
        reaction_rates = k.copy()
        for j in range(self.num_reactions):
            for i in range(len(concentrations)):
                if self.orders[j, i] > 0:
                    reaction_rates[j] *= concentrations[i] ** self.orders[j, i]

        # Apply stoichiometry: net_rate_i = sum_j(nu_ij * r_j)
        net_rates = self.stoich.T @ reaction_rates  # (species,)

        return net_rates

    def validate_parameters(self) -> bool:
        """Validate kinetic parameters.

        Returns:
            True if parameters are valid.
        """
        # Check k0 > 0
        if not np.all(self.k0 > 0):
            logger.error("k0 must be positive")
            return False

        # Check Ea >= 0
        if not np.all(self.Ea >= 0):
            logger.error("Ea must be non-negative")
            return False

        return True

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> ArrheniusKinetics:
        """Deserialize Arrhenius kinetics from configuration.

        Args:
            config: Configuration dictionary.

        Returns:
            ArrheniusKinetics instance.
        """
        return cls(
            name=config["name"],
            num_reactions=config["num_reactions"],
            params=config["params"],
        )


__all__ = ["ArrheniusKinetics"]
