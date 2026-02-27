"""Exothermic first-order A -> B reaction in CSTR.

Classic benchmark from chemical engineering textbooks. Used for testing
bifurcation analysis, stability, and control strategies.

Reference: Fogler, H.S. "Elements of Chemical Reaction Engineering"
"""

from __future__ import annotations

import logging

import numpy as np

from reactor_twin.reactors.cstr import CSTRReactor
from reactor_twin.reactors.kinetics.arrhenius import ArrheniusKinetics

logger = logging.getLogger(__name__)


def create_exothermic_cstr(
    name: str = "exothermic_ab_cstr",
    isothermal: bool = False,
) -> CSTRReactor:
    """Create CSTR with exothermic A -> B reaction.

    System:
        A -> B  (exothermic, first-order)

    Parameters from ND Pyomo Cookbook:
        V = 100 L
        F = 100 L/min
        rho = 1000 g/L
        Cp = 0.239 J/(g*K)
        dH_r = -5e4 J/mol
        E_a = 7.27e4 J/mol
        k_0 = 7.2e10 1/min
        UA = 5e4 J/(min*K)
        C_A0 = 1.0 mol/L
        T_0 = 350 K
        T_c = 300 K

    Args:
        name: Reactor identifier.
        isothermal: If True, ignore energy balance (fixed temperature).

    Returns:
        Configured CSTR reactor instance.
    """
    # Kinetics: A -> B
    kinetics = ArrheniusKinetics(
        name="A_to_B",
        num_reactions=1,
        params={
            "k0": np.array([7.2e10]),  # 1/min
            "Ea": np.array([7.27e4]),  # J/mol
            "stoich": np.array([[-1, 1]]),  # A -> B
            "orders": np.array([[1, 0]]),  # First-order in A
        },
    )

    # Reactor parameters
    reactor_params = {
        "V": 100.0,  # L
        "F": 100.0,  # L/min
        "C_feed": [1.0, 0.0],  # [C_A0, C_B0] mol/L
        "T_feed": 350.0,  # K
        "C_initial": [0.5, 0.0],  # Initial concentrations
        "T_initial": 350.0,  # Initial temperature
    }

    if not isothermal:
        # Add thermal parameters
        reactor_params.update({
            "rho": 1000.0,  # g/L
            "Cp": 0.239,  # J/(g*K)
            "UA": 5e4,  # J/(min*K)
            "T_coolant": 300.0,  # K
        })

    reactor = CSTRReactor(
        name=name,
        num_species=2,
        params=reactor_params,
        kinetics=kinetics,
        isothermal=isothermal,
    )

    logger.info(f"Created exothermic A->B CSTR: {reactor}")
    return reactor


__all__ = ["create_exothermic_cstr"]
