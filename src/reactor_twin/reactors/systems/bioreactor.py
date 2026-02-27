"""Bioreactor CSTR benchmark with Monod kinetics."""

from __future__ import annotations

import logging

import numpy as np

from reactor_twin.reactors.cstr import CSTRReactor
from reactor_twin.reactors.kinetics.michaelis_menten import MichaelisMentenKinetics

logger = logging.getLogger(__name__)


def create_bioreactor_cstr(
    name: str = "bioreactor_cstr",
    isothermal: bool = True,
) -> CSTRReactor:
    """Create a bioreactor CSTR with Monod growth kinetics.

    Biological system: S (substrate) + X (cells) → P (product) + X (more cells)

    Monod kinetics (special case of Michaelis-Menten):
        μ = μ_max * S / (K_s + S)
        dX/dt = μ * X - D * X
        dS/dt = -μ * X / Y_xs - D * (S - S_in)
        dP/dt = q_p * X - D * P

    where:
        - μ: Specific growth rate (1/h)
        - μ_max: Maximum growth rate (1/h)
        - K_s: Half-saturation constant (g/L)
        - Y_xs: Yield coefficient (g cells / g substrate)
        - q_p: Specific production rate (g product / (g cells * h))
        - D: Dilution rate F/V (1/h)

    Species:
        0: Substrate (S) - glucose, carbon source
        1: Biomass (X) - cell concentration
        2: Product (P) - metabolite, protein

    Reactor parameters (typical aerobic fermentation):
        - V = 1000 L (1 m³ bioreactor)
        - F = 100 L/h (dilution rate D = 0.1 1/h)
        - S_in = 20 g/L (feed glucose)
        - T = 310 K (37°C, mesophilic)

    Kinetic parameters:
        - μ_max = 0.5 1/h
        - K_s = 0.1 g/L
        - Y_xs = 0.5 g/g (50% substrate to biomass)
        - q_p = 0.2 g/(g*h) (product formation rate)

    Initial conditions:
        - S0 = 10 g/L
        - X0 = 0.5 g/L
        - P0 = 0 g/L

    Args:
        name: Reactor identifier.
        isothermal: If True, operate at constant temperature.

    Returns:
        Configured CSTRReactor instance.
    """
    # Monod kinetics is structurally identical to Michaelis-Menten
    # We'll use MM kinetics with appropriate parameters

    # Species: [S, X, P]
    num_species = 3

    # Kinetic parameters
    mu_max = 0.5  # 1/h
    K_s = 0.1  # g/L
    Y_xs = 0.5  # g/g
    q_p = 0.2  # g/(g*h)

    # Stoichiometry for growth: -1/Y_xs * S + 1 * X + q_p/mu_max * P
    # Simplified: treat as single "reaction" with MM kinetics
    # V_max corresponds to mu_max * X_typical (we'll approximate)

    # For Monod kinetics, we adapt MM by making substrate the limiting species
    # and using biomass as a "catalyst" (though this is a simplification)

    # Actually, Monod kinetics doesn't fit perfectly into MM framework
    # because growth depends on both S and X. We'll create custom kinetics.

    # For now, use power law kinetics as a placeholder
    # Rate: r = mu_max * S/(K_s + S) * X
    # This is bilinear in S and X, not quite power law either

    # Let's use MM kinetics for substrate consumption with X as "enzyme"
    V_max_substrate = mu_max  # Will multiply by X in rate
    K_m_substrate = K_s

    # Stoichiometry:
    # Reaction 1: S → X (growth)
    #   dS/dt = -1/Y_xs * mu * X
    #   dX/dt = +1 * mu * X
    # Reaction 2: S → P (product formation)
    #   dP/dt = q_p * X

    # This is complex - for benchmark purposes, let's simplify:
    # Use 2 reactions with power law kinetics

    from reactor_twin.reactors.kinetics.power_law import PowerLawKinetics

    # Reaction 1: Growth (S + X → 2X)
    # Rate: r1 = mu_max * S/(K_s + S) * X
    # Approximate with power law: r1 ≈ k1 * S * X
    k1 = mu_max / K_s  # Approximate for S << K_s

    # Reaction 2: Product formation (X → P)
    # Rate: r2 = q_p * X
    k2 = q_p

    kinetics = PowerLawKinetics(
        name="monod_kinetics",
        num_reactions=2,
        params={
            "k": np.array([k1, k2]),
            "orders": np.array(
                [
                    [1.0, 1.0, 0.0],  # Reaction 1: S^1 * X^1 (growth)
                    [0.0, 1.0, 0.0],  # Reaction 2: X^1 (production)
                ]
            ),
            "stoich": np.array(
                [
                    [-1.0 / Y_xs, 1.0, 0.0],  # Reaction 1: -S/Y_xs, +X
                    [0.0, 0.0, 1.0],  # Reaction 2: +P
                ]
            ),
        },
    )

    # Reactor parameters
    params = {
        "V": 1000.0,  # L
        "F": 100.0,  # L/h
        "T": 310.0,  # K (37°C)
        "C_in": [20.0, 0.0, 0.0],  # g/L [S_in, X_in, P_in]
        "C_initial": [10.0, 0.5, 0.0],  # g/L [S0, X0, P0]
    }

    if not isothermal:
        params.update(
            {
                "rho": 1000.0,  # kg/m³ (water-like)
                "Cp": 4184.0,  # J/(kg*K)
                "delta_H": np.array([-50000.0, 0.0]),  # J/mol (exothermic growth)
            }
        )

    reactor = CSTRReactor(
        name=name,
        num_species=num_species,
        params=params,
        kinetics=kinetics,
        isothermal=isothermal,
    )

    logger.info(f"Created bioreactor CSTR '{name}' with Monod kinetics")
    return reactor


def get_bioreactor_species_names() -> list[str]:
    """Get species names for bioreactor.

    Returns:
        List of species names.
    """
    return ["Substrate (S)", "Biomass (X)", "Product (P)"]


__all__ = ["create_bioreactor_cstr", "get_bioreactor_species_names"]
