"""Monod growth kinetics for bioreactor systems."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from reactor_twin.reactors.kinetics.base import AbstractKinetics
from reactor_twin.utils.registry import KINETICS_REGISTRY

logger = logging.getLogger(__name__)


@KINETICS_REGISTRY.register("monod")
class MonodKinetics(AbstractKinetics):
    """Monod growth kinetics for microbial/enzymatic systems.

    Models substrate-limited growth with optional product formation:

        mu = mu_max * S / (K_s + S)

    Species convention: [Substrate, Biomass, Product, ...]

    Reactions:
        1. Growth:    S + X -> 2X   (rate = mu * X)
        2. Product:   X -> X + P    (rate = q_p * X)

    Net rates:
        dS/dt = -mu * X / Y_xs + dilution
        dX/dt = +mu * X + dilution
        dP/dt = +q_p * X + dilution

    Attributes:
        mu_max: Maximum specific growth rate (1/time).
        K_s: Half-saturation constant (concentration units).
        Y_xs: Yield coefficient (g biomass / g substrate).
        q_p: Specific product formation rate (g product / (g biomass * time)).
        substrate_idx: Index of substrate in concentration vector.
        biomass_idx: Index of biomass in concentration vector.
        product_idx: Index of product in concentration vector (or None).
    """

    def __init__(
        self,
        name: str,
        num_species: int = 3,
        params: dict[str, Any] | None = None,
        mu_max: float = 0.5,
        K_s: float = 0.1,
        Y_xs: float = 0.5,
        q_p: float = 0.0,
        substrate_idx: int = 0,
        biomass_idx: int = 1,
        product_idx: int | None = 2,
    ):
        """Initialize Monod kinetics.

        Args:
            name: Kinetics identifier.
            num_species: Number of chemical species.
            params: Optional parameter dict (overrides keyword args).
            mu_max: Maximum specific growth rate.
            K_s: Half-saturation constant.
            Y_xs: Yield coefficient (biomass per substrate).
            q_p: Specific product formation rate.
            substrate_idx: Index of substrate species.
            biomass_idx: Index of biomass species.
            product_idx: Index of product species (None if no product).
        """
        params = params or {}
        self.mu_max = params.get("mu_max", mu_max)
        self.K_s = params.get("K_s", K_s)
        self.Y_xs = params.get("Y_xs", Y_xs)
        self.q_p = params.get("q_p", q_p)
        self.substrate_idx = params.get("substrate_idx", substrate_idx)
        self.biomass_idx = params.get("biomass_idx", biomass_idx)
        self.product_idx = params.get("product_idx", product_idx)

        num_reactions = 2 if self.product_idx is not None else 1
        self.num_species = num_species
        super().__init__(name, num_reactions, params or {})

        logger.debug(
            f"Initialized MonodKinetics: mu_max={self.mu_max}, "
            f"K_s={self.K_s}, Y_xs={self.Y_xs}, q_p={self.q_p}"
        )

    def compute_rates(
        self,
        concentrations: np.ndarray,
        temperature: float,
    ) -> np.ndarray:
        """Compute net production rates using Monod kinetics.

        Args:
            concentrations: Species concentrations, shape (num_species,).
            temperature: Temperature (K). Not used in basic Monod.

        Returns:
            Net production rates, shape (num_species,).
        """
        S = max(concentrations[self.substrate_idx], 0.0)
        X = max(concentrations[self.biomass_idx], 0.0)

        # Monod growth rate
        mu = self.mu_max * S / (self.K_s + S)

        # Net rates for each species
        rates = np.zeros(self.num_species)

        # Substrate consumption: dS/dt = -mu * X / Y_xs
        rates[self.substrate_idx] = -mu * X / self.Y_xs

        # Biomass growth: dX/dt = +mu * X
        rates[self.biomass_idx] = mu * X

        # Product formation: dP/dt = q_p * X
        if self.product_idx is not None and self.q_p > 0:
            rates[self.product_idx] = self.q_p * X

        return rates

    def get_specific_growth_rate(
        self,
        substrate_conc: float,
    ) -> float:
        """Compute specific growth rate mu(S).

        Args:
            substrate_conc: Substrate concentration.

        Returns:
            Specific growth rate.
        """
        S = max(substrate_conc, 0.0)
        return self.mu_max * S / (self.K_s + S)

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> MonodKinetics:
        """Deserialize from configuration."""
        return cls(
            name=config["name"],
            num_species=config.get("num_species", 3),
            params=config.get("params", {}),
        )


__all__ = ["MonodKinetics"]
