"""Consecutive reactions CSTR benchmark (A→B→C)."""

from __future__ import annotations

import logging

import numpy as np

from reactor_twin.reactors.cstr import CSTRReactor
from reactor_twin.reactors.kinetics.arrhenius import ArrheniusKinetics

logger = logging.getLogger(__name__)


def create_consecutive_cstr(
    name: str = "consecutive_abc_cstr",
    isothermal: bool = True,
) -> CSTRReactor:
    """Create a CSTR with consecutive reactions A→B→C.

    Reaction network:
        Reaction 1: A → B  (rate r1 = k1 * C_A)
        Reaction 2: B → C  (rate r2 = k2 * C_B)

    This is a classic selectivity problem. Want to maximize B production
    by choosing appropriate residence time τ = V/F and temperature.

    Species:
        0: A (reactant)
        1: B (desired intermediate product)
        2: C (over-reaction product)

    Reactor parameters (typical liquid-phase):
        - V = 100 L
        - F = 100 L/min (τ = 1 min)
        - C_A_in = 2.0 mol/L
        - T = 350 K

    Kinetic parameters (both first-order):
        Reaction 1 (A→B):
            - k1,0 = 1.0e3 min⁻¹
            - E_a1 = 50 kJ/mol

        Reaction 2 (B→C):
            - k2,0 = 5.0e2 min⁻¹
            - E_a2 = 60 kJ/mol

    Initial conditions:
        - C_A0 = 1.0 mol/L
        - C_B0 = 0.0 mol/L
        - C_C0 = 0.0 mol/L

    Analytical solution at steady state (CSTR):
        C_A_ss = C_A_in / (1 + k1 * τ)
        C_B_ss = (k1 * τ * C_A_in) / [(1 + k1 * τ) * (1 + k2 * τ)]
        C_C_ss = C_A_in - C_A_ss - C_B_ss

    Optimal τ for maximum B: τ_opt = 1 / sqrt(k1 * k2)

    Args:
        name: Reactor identifier.
        isothermal: If True, operate at constant temperature.

    Returns:
        Configured CSTRReactor instance.
    """
    # Species: [A, B, C]
    num_species = 3

    # Kinetic parameters
    k1_0 = 1.0e3  # min⁻¹ (A→B)
    E_a1 = 50.0e3  # J/mol
    k2_0 = 5.0e2  # min⁻¹ (B→C)
    E_a2 = 60.0e3  # J/mol

    # Stoichiometry matrix (reactions x species)
    # Reaction 1: A → B  =>  [-1, +1, 0]
    # Reaction 2: B → C  =>  [0, -1, +1]
    stoich = np.array(
        [
            [-1.0, 1.0, 0.0],  # Reaction 1
            [0.0, -1.0, 1.0],  # Reaction 2
        ]
    )

    kinetics = ArrheniusKinetics(
        name="consecutive_arrhenius",
        num_reactions=2,
        params={
            "k0": np.array([k1_0, k2_0]),
            "Ea": np.array([E_a1, E_a2]),
            "stoich": stoich,
            "orders": np.array(
                [
                    [1.0, 0.0, 0.0],  # Reaction 1: first order in A
                    [0.0, 1.0, 0.0],  # Reaction 2: first order in B
                ]
            ),
        },
    )

    # Reactor parameters
    params = {
        "V": 100.0,  # L
        "F": 100.0,  # L/min (τ = 1 min)
        "T": 350.0,  # K
        "C_in": [2.0, 0.0, 0.0],  # mol/L [A_in, B_in, C_in]
        "C_initial": [1.0, 0.0, 0.0],  # mol/L [A0, B0, C0]
    }

    if not isothermal:
        params.update(
            {
                "rho": 800.0,  # kg/m³ (organic liquid)
                "Cp": 2000.0,  # J/(kg*K)
                "delta_H": np.array([-30000.0, -40000.0]),  # J/mol (both exothermic)
            }
        )

    reactor = CSTRReactor(
        name=name,
        num_species=num_species,
        params=params,
        kinetics=kinetics,
        isothermal=isothermal,
    )

    logger.info(f"Created consecutive reactions CSTR '{name}' (A→B→C)")
    return reactor


def get_consecutive_species_names() -> list[str]:
    """Get species names for consecutive reactions.

    Returns:
        List of species names.
    """
    return ["A (reactant)", "B (intermediate)", "C (product)"]


def compute_optimal_residence_time(k1: float, k2: float) -> float:
    """Compute optimal residence time for maximum B concentration.

    For consecutive reactions A→B→C in a CSTR:
        τ_opt = 1 / sqrt(k1 * k2)

    Args:
        k1: Rate constant for A→B (min⁻¹).
        k2: Rate constant for B→C (min⁻¹).

    Returns:
        Optimal residence time (min).
    """
    return 1.0 / np.sqrt(k1 * k2)


def compute_steady_state_concentrations(
    C_A_in: float,
    k1: float,
    k2: float,
    tau: float,
) -> tuple[float, float, float]:
    """Compute steady-state concentrations for consecutive reactions.

    Args:
        C_A_in: Inlet concentration of A (mol/L).
        k1: Rate constant for A→B (min⁻¹).
        k2: Rate constant for B→C (min⁻¹).
        tau: Residence time (min).

    Returns:
        Tuple of (C_A_ss, C_B_ss, C_C_ss) in mol/L.
    """
    C_A_ss = C_A_in / (1 + k1 * tau)
    C_B_ss = (k1 * tau * C_A_in) / ((1 + k1 * tau) * (1 + k2 * tau))
    C_C_ss = C_A_in - C_A_ss - C_B_ss
    return C_A_ss, C_B_ss, C_C_ss


__all__ = [
    "create_consecutive_cstr",
    "get_consecutive_species_names",
    "compute_optimal_residence_time",
    "compute_steady_state_concentrations",
]
