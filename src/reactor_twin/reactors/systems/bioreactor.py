"""Bioreactor CSTR benchmark with Monod kinetics."""

from __future__ import annotations

import logging

from reactor_twin.reactors.cstr import CSTRReactor
from reactor_twin.reactors.kinetics.monod import MonodKinetics

logger = logging.getLogger(__name__)


def create_bioreactor_cstr(
    name: str = "bioreactor_cstr",
    isothermal: bool = True,
) -> CSTRReactor:
    """Create a bioreactor CSTR with Monod growth kinetics.

    Biological system: S (substrate) + X (cells) -> P (product) + X (more cells)

    Monod kinetics:
        mu = mu_max * S / (K_s + S)
        dX/dt = mu * X - D * X
        dS/dt = -mu * X / Y_xs - D * (S - S_in)
        dP/dt = q_p * X - D * P

    where:
        - mu: Specific growth rate (1/h)
        - mu_max: Maximum growth rate (1/h)
        - K_s: Half-saturation constant (g/L)
        - Y_xs: Yield coefficient (g cells / g substrate)
        - q_p: Specific production rate (g product / (g cells * h))
        - D: Dilution rate F/V (1/h)

    Species:
        0: Substrate (S) - glucose, carbon source
        1: Biomass (X) - cell concentration
        2: Product (P) - metabolite, protein

    Reactor parameters (typical aerobic fermentation):
        - V = 1000 L (1 m3 bioreactor)
        - F = 100 L/h (dilution rate D = 0.1 1/h)
        - S_in = 20 g/L (feed glucose)
        - T = 310 K (37C, mesophilic)

    Kinetic parameters:
        - mu_max = 0.5 1/h
        - K_s = 0.1 g/L
        - Y_xs = 0.5 g/g (50% substrate to biomass)
        - q_p = 0.2 g/(g*h) (product formation rate)

    Args:
        name: Reactor identifier.
        isothermal: If True, operate at constant temperature.

    Returns:
        Configured CSTRReactor instance.
    """
    num_species = 3

    kinetics = MonodKinetics(
        name="monod_growth",
        num_species=num_species,
        mu_max=0.5,
        K_s=0.1,
        Y_xs=0.5,
        q_p=0.2,
        substrate_idx=0,
        biomass_idx=1,
        product_idx=2,
    )

    params: dict = {
        "V": 1000.0,
        "F": 100.0,
        "T": 310.0,
        "T_feed": 310.0,
        "C_feed": [20.0, 0.0, 0.0],
        "C_initial": [10.0, 0.5, 0.0],
    }

    if not isothermal:
        params.update(
            {
                "rho": 1000.0,
                "Cp": 4184.0,
                "UA": 5000.0,
                "T_coolant": 310.0,
            }
        )

    reactor = CSTRReactor(
        name=name,
        num_species=num_species,
        params=params,
        kinetics=kinetics,
        isothermal=isothermal,
    )

    logger.info(f"Created bioreactor CSTR '{name}' with proper Monod kinetics")
    return reactor


def get_bioreactor_species_names() -> list[str]:
    """Get species names for bioreactor."""
    return ["Substrate (S)", "Biomass (X)", "Product (P)"]


__all__ = ["create_bioreactor_cstr", "get_bioreactor_species_names"]
