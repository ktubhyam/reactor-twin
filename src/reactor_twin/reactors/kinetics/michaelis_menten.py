"""Michaelis-Menten enzyme kinetics."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from reactor_twin.reactors.kinetics.base import AbstractKinetics
from reactor_twin.utils.registry import KINETICS_REGISTRY

logger = logging.getLogger(__name__)


@KINETICS_REGISTRY.register("michaelis_menten")
class MichaelisMentenKinetics(AbstractKinetics):
    """Michaelis-Menten enzyme kinetics.

    For enzyme-catalyzed reactions: E + S <-> ES -> E + P

    Rate law: r = (V_max * [S]) / (K_m + [S])

    where:
    - V_max: Maximum reaction rate
    - K_m: Michaelis constant (substrate concentration at half V_max)
    - [S]: Substrate concentration

    For competitive inhibition:
        r = (V_max * [S]) / (K_m * (1 + [I]/K_i) + [S])

    Parameters:
        V_max: Maximum reaction rates, shape (num_reactions,).
        K_m: Michaelis constants, shape (num_reactions,).
        K_i: Inhibition constants (optional), shape (num_reactions,).
        substrate_indices: Index of substrate for each reaction.
        inhibitor_indices: Index of inhibitor for each reaction (optional).
        stoich: Stoichiometric coefficients, shape (num_reactions, num_species).
    """

    def __init__(self, name: str, num_reactions: int, params: dict[str, Any]):
        """Initialize Michaelis-Menten kinetics.

        Args:
            name: Kinetics identifier.
            num_reactions: Number of reactions.
            params: Dictionary with keys 'V_max', 'K_m', 'substrate_indices',
                   'stoich', optionally 'K_i', 'inhibitor_indices'.
        """
        super().__init__(name, num_reactions, params)

        # Validate parameters
        required = ["V_max", "K_m", "substrate_indices", "stoich"]
        for key in required:
            if key not in params:
                raise ValueError(f"Missing required parameter: {key}")

        self.V_max = np.array(params["V_max"])
        self.K_m = np.array(params["K_m"])
        self.substrate_indices = np.array(params["substrate_indices"], dtype=int)
        self.stoich = np.array(params["stoich"])

        # Competitive inhibition (optional)
        self.K_i = np.array(params.get("K_i", np.zeros(num_reactions)))
        self.inhibitor_indices = params.get("inhibitor_indices")
        if self.inhibitor_indices is not None:
            self.inhibitor_indices = np.array(self.inhibitor_indices, dtype=int)

        # Validate shapes
        assert self.V_max.shape == (num_reactions,), "V_max shape mismatch"
        assert self.K_m.shape == (num_reactions,), "K_m shape mismatch"
        assert self.substrate_indices.shape == (num_reactions,), "substrate_indices shape mismatch"
        assert self.stoich.shape[0] == num_reactions, "stoich shape mismatch"

    def compute_rates(
        self,
        concentrations: np.ndarray,
        temperature: float,
    ) -> np.ndarray:
        """Compute reaction rates using Michaelis-Menten law.

        Args:
            concentrations: Species concentrations, shape (num_species,).
            temperature: Temperature (not used for basic MM kinetics).

        Returns:
            Net production rates for each species, shape (num_species,).
        """
        # Compute reaction rates: r_j = (V_max * [S]) / (K_m + [S])
        reaction_rates = np.zeros(self.num_reactions)

        for j in range(self.num_reactions):
            # Get substrate concentration
            S_idx = self.substrate_indices[j]
            S = concentrations[S_idx]

            # Michaelis-Menten rate
            rate = (self.V_max[j] * S) / (self.K_m[j] + S)

            # Apply competitive inhibition if present
            if self.inhibitor_indices is not None and self.K_i[j] > 0:
                I_idx = self.inhibitor_indices[j]
                I = concentrations[I_idx]
                inhibition_factor = 1 + I / self.K_i[j]
                rate = rate / inhibition_factor

            reaction_rates[j] = rate

        # Apply stoichiometry: net_rate_i = sum_j(nu_ij * r_j)
        net_rates = self.stoich.T @ reaction_rates  # (num_species,)

        return net_rates

    def validate_parameters(self) -> bool:
        """Validate kinetic parameters.

        Returns:
            True if parameters are valid.
        """
        # Check V_max > 0
        if not np.all(self.V_max > 0):
            logger.error("V_max must be positive")
            return False

        # Check K_m > 0
        if not np.all(self.K_m > 0):
            logger.error("K_m must be positive")
            return False

        # Check K_i > 0 (if present)
        if self.inhibitor_indices is not None:
            if not np.all(self.K_i > 0):
                logger.error("K_i must be positive")
                return False

        return True

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> MichaelisMentenKinetics:
        """Deserialize Michaelis-Menten kinetics from configuration.

        Args:
            config: Configuration dictionary.

        Returns:
            MichaelisMentenKinetics instance.
        """
        return cls(
            name=config["name"],
            num_reactions=config["num_reactions"],
            params=config["params"],
        )


__all__ = ["MichaelisMentenKinetics"]
