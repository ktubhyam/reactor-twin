"""Langmuir-Hinshelwood kinetics for heterogeneous catalysis."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from reactor_twin.reactors.kinetics.base import AbstractKinetics
from reactor_twin.utils.registry import KINETICS_REGISTRY

logger = logging.getLogger(__name__)


@KINETICS_REGISTRY.register("langmuir_hinshelwood")
class LangmuirHinshelwoodKinetics(AbstractKinetics):
    """Langmuir-Hinshelwood kinetics for surface-catalyzed reactions.

    For heterogeneous catalytic reactions where multiple species compete
    for active sites on a catalyst surface.

    General rate law (single reaction):
        r = (k * Π(C_i^alpha_i)) / (1 + Σ(K_j * C_j))^beta

    where:
        - k: Rate constant
        - K_j: Adsorption equilibrium constants
        - alpha_i: Reaction orders in numerator
        - beta: Denominator exponent (typically 1 or 2)

    Common forms:
    1. Single reactant A → products:
       r = (k * K_A * C_A) / (1 + K_A * C_A)

    2. Two reactants A + B → products:
       r = (k * K_A * K_B * C_A * C_B) / (1 + K_A * C_A + K_B * C_B)²

    3. With product inhibition:
       r = (k * K_A * C_A) / (1 + K_A * C_A + K_P * C_P)

    Temperature dependence:
        k(T) = A * exp(-E_a / (R*T))
        K_j(T) = K_j0 * exp(ΔH_ads,j / (R*T))

    Parameters:
        k: Rate constants, shape (num_reactions,).
        K_ads: Adsorption constants, shape (num_reactions, num_species).
        orders_num: Numerator reaction orders, shape (num_reactions, num_species).
        orders_den: Denominator exponent, shape (num_reactions,).
        stoich: Stoichiometric coefficients, shape (num_reactions, num_species).
        A: Pre-exponential factors (optional), shape (num_reactions,).
        E_a: Activation energies (optional), shape (num_reactions,).
    """

    def __init__(self, name: str, num_reactions: int, params: dict[str, Any]):
        """Initialize Langmuir-Hinshelwood kinetics.

        Args:
            name: Kinetics identifier.
            num_reactions: Number of reactions.
            params: Dictionary with keys 'k', 'K_ads', 'orders_num', 'orders_den',
                   'stoich', optionally 'A', 'E_a'.
        """
        super().__init__(name, num_reactions, params)

        # Validate parameters
        required = ["k", "K_ads", "orders_num", "orders_den", "stoich"]
        for key in required:
            if key not in params:
                raise ValueError(f"Missing required parameter: {key}")

        self.k = np.array(params["k"])
        self.K_ads = np.array(params["K_ads"])
        self.orders_num = np.array(params["orders_num"])
        self.orders_den = np.array(params["orders_den"])
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
        assert self.K_ads.shape[0] == num_reactions, "K_ads shape mismatch"
        assert self.orders_num.shape[0] == num_reactions, "orders_num shape mismatch"
        assert self.orders_den.shape == (num_reactions,), "orders_den shape mismatch"
        assert self.stoich.shape[0] == num_reactions, "stoich shape mismatch"

    def compute_rates(
        self,
        concentrations: np.ndarray,
        temperature: float,
    ) -> np.ndarray:
        """Compute reaction rates using Langmuir-Hinshelwood law.

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

        # Compute reaction rates: r_j = (k_j * Π(C_i^alpha_i)) / (1 + Σ(K_j * C_j))^beta
        reaction_rates = np.zeros(self.num_reactions)

        for j in range(self.num_reactions):
            # Numerator: k * Π(C_i^alpha_i)
            numerator = k_T[j]
            for i in range(len(concentrations)):
                if self.orders_num[j, i] != 0:
                    C_safe = max(concentrations[i], 1e-10)
                    numerator *= C_safe ** self.orders_num[j, i]

            # Denominator: (1 + Σ(K_j * C_j))^beta
            adsorption_term = 1.0
            for i in range(len(concentrations)):
                if self.K_ads[j, i] > 0:
                    C_safe = max(concentrations[i], 1e-10)
                    adsorption_term += self.K_ads[j, i] * C_safe

            denominator = adsorption_term ** self.orders_den[j]

            # Reaction rate
            reaction_rates[j] = numerator / denominator

        # Apply stoichiometry: net_rate_i = sum_j(nu_ij * r_j)
        net_rates = self.stoich.T @ reaction_rates

        return net_rates

    def validate_parameters(self) -> bool:
        """Validate kinetic parameters.

        Returns:
            True if parameters are valid.
        """
        # Check k > 0
        if not np.all(self.k > 0):
            logger.error("k must be positive")
            return False

        # Check K_ads >= 0
        if not np.all(self.K_ads >= 0):
            logger.error("K_ads must be non-negative")
            return False

        # Check orders_den > 0
        if not np.all(self.orders_den > 0):
            logger.error("orders_den must be positive")
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
    def from_dict(cls, config: dict[str, Any]) -> LangmuirHinshelwoodKinetics:
        """Deserialize Langmuir-Hinshelwood kinetics from configuration.

        Args:
            config: Configuration dictionary.

        Returns:
            LangmuirHinshelwoodKinetics instance.
        """
        return cls(
            name=config["name"],
            num_reactions=config["num_reactions"],
            params=config["params"],
        )


__all__ = ["LangmuirHinshelwoodKinetics"]
