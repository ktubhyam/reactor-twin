"""Parallel competing reactions CSTR benchmark (A→B, A→C)."""

from __future__ import annotations

import logging
from typing import cast

import numpy as np

from reactor_twin.reactors.cstr import CSTRReactor
from reactor_twin.reactors.kinetics.arrhenius import ArrheniusKinetics

logger = logging.getLogger(__name__)


def create_parallel_cstr(
    name: str = "parallel_abc_cstr",
    isothermal: bool = True,
) -> CSTRReactor:
    """Create a CSTR with parallel competing reactions A→B and A→C.

    Reaction network:
        Reaction 1: A → B  (rate r1 = k1 * C_A^n1)
        Reaction 2: A → C  (rate r2 = k2 * C_A^n2)

    This is a classic selectivity problem. Want to maximize B (desired)
    and minimize C (byproduct) by choosing temperature and concentration.

    Species:
        0: A (reactant)
        1: B (desired product)
        2: C (undesired byproduct)

    Reactor parameters (typical liquid-phase):
        - V = 100 L
        - F = 50 L/min (τ = 2 min)
        - C_A_in = 3.0 mol/L
        - T = 340 K

    Kinetic parameters:
        Reaction 1 (A→B, desired):
            - k1,0 = 2.0e4 min⁻¹
            - E_a1 = 55 kJ/mol
            - n1 = 1.0 (first order)

        Reaction 2 (A→C, undesired):
            - k2,0 = 1.0e5 min⁻¹
            - E_a2 = 70 kJ/mol
            - n2 = 2.0 (second order)

    Initial conditions:
        - C_A0 = 2.0 mol/L
        - C_B0 = 0.0 mol/L
        - C_C0 = 0.0 mol/L

    Selectivity analysis:
        S_B/C = r1 / r2 = (k1 * C_A^n1) / (k2 * C_A^n2)
              = (k1 / k2) * C_A^(n1 - n2)

    Since n1 < n2, selectivity increases with lower C_A (high dilution).
    Since E_a1 < E_a2, selectivity increases with lower T.

    Strategy for high B selectivity:
        - Low temperature (favor lower E_a)
        - High dilution (favor lower reaction order)
        - Short residence time (minimize A depletion)

    Args:
        name: Reactor identifier.
        isothermal: If True, operate at constant temperature.

    Returns:
        Configured CSTRReactor instance.
    """
    # Species: [A, B, C]
    num_species = 3

    # Kinetic parameters
    k1_0 = 2.0e4  # min⁻¹ (A→B)
    E_a1 = 55.0e3  # J/mol
    n1 = 1.0  # First order

    k2_0 = 1.0e5  # min⁻¹ (A→C)
    E_a2 = 70.0e3  # J/mol
    n2 = 2.0  # Second order

    # Stoichiometry matrix (reactions x species)
    # Reaction 1: A → B  =>  [-1, +1, 0]
    # Reaction 2: A → C  =>  [-1, 0, +1]
    stoich = np.array(
        [
            [-1.0, 1.0, 0.0],  # Reaction 1
            [-1.0, 0.0, 1.0],  # Reaction 2
        ]
    )

    kinetics = ArrheniusKinetics(
        name="parallel_arrhenius",
        num_reactions=2,
        params={
            "k0": np.array([k1_0, k2_0]),
            "Ea": np.array([E_a1, E_a2]),
            "stoich": stoich,
            "orders": np.array(
                [
                    [n1, 0.0, 0.0],  # Reaction 1: order n1 in A
                    [n2, 0.0, 0.0],  # Reaction 2: order n2 in A
                ]
            ),
        },
    )

    # Reactor parameters
    params = {
        "V": 100.0,  # L
        "F": 50.0,  # L/min (τ = 2 min)
        "T_feed": 340.0,  # K
        "C_feed": [3.0, 0.0, 0.0],  # mol/L [A_in, B_in, C_in]
        "C_initial": [2.0, 0.0, 0.0],  # mol/L [A0, B0, C0]
    }

    if not isothermal:
        params.update(
            {
                "rho": 800.0,  # kg/m³ (organic liquid)
                "Cp": 2000.0,  # J/(kg*K)
                "dH_rxn": np.array([-35000.0, -50000.0]),  # J/mol (both exothermic)
            }
        )

    reactor = CSTRReactor(
        name=name,
        num_species=num_species,
        params=params,
        kinetics=kinetics,
        isothermal=isothermal,
    )

    logger.info(f"Created parallel competing reactions CSTR '{name}' (A→B, A→C)")
    return reactor


def get_parallel_species_names() -> list[str]:
    """Get species names for parallel reactions.

    Returns:
        List of species names.
    """
    return ["A (reactant)", "B (desired product)", "C (byproduct)"]


def compute_selectivity(
    C_A: float,
    k1: float,
    k2: float,
    n1: float,
    n2: float,
) -> float:
    """Compute instantaneous selectivity S_B/C.

    S_B/C = r1 / r2 = (k1 * C_A^n1) / (k2 * C_A^n2)

    Args:
        C_A: Concentration of A (mol/L).
        k1: Rate constant for A→B.
        k2: Rate constant for A→C.
        n1: Reaction order for A→B.
        n2: Reaction order for A→C.

    Returns:
        Selectivity ratio (dimensionless).
    """
    if C_A <= 0:
        return 0.0
    r1 = k1 * (C_A**n1)
    r2 = k2 * (C_A**n2)
    return cast(float, r1 / (r2 + 1e-10))


def compute_yield(C_A_in: float, C_A: float, C_B: float) -> float:
    """Compute yield of B.

    Yield = moles of B produced / moles of A consumed

    Args:
        C_A_in: Inlet concentration of A (mol/L).
        C_A: Current concentration of A (mol/L).
        C_B: Current concentration of B (mol/L).

    Returns:
        Yield (dimensionless, 0-1).
    """
    A_consumed = C_A_in - C_A
    if A_consumed <= 0:
        return 0.0
    return C_B / A_consumed


__all__ = [
    "create_parallel_cstr",
    "get_parallel_species_names",
    "compute_selectivity",
    "compute_yield",
]
