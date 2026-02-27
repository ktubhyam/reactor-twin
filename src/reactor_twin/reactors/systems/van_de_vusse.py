"""Van de Vusse reaction system in CSTR.

Complex reaction network with series-parallel pathways:
    A -> B -> C  (desired product B)
    2A -> D      (unwanted byproduct)

Classic benchmark for testing Neural ODE on multi-reaction systems.

Reference: Van de Vusse, J.G. (1964). Plug-flow type reactor versus tank reactor.
           Chemical Engineering Science, 19(12), 994-996.
"""

from __future__ import annotations

import logging

import numpy as np

from reactor_twin.reactors.cstr import CSTRReactor
from reactor_twin.reactors.kinetics.arrhenius import ArrheniusKinetics

logger = logging.getLogger(__name__)

from reactor_twin.utils.constants import R_GAS


def create_van_de_vusse_cstr(
    name: str = "van_de_vusse_cstr",
    isothermal: bool = True,
    temperature: float = 403.0,  # 130 C
) -> CSTRReactor:
    """Create CSTR with Van de Vusse reaction network.

    Reactions:
        1. A -> B       k1 = 50/hr  (E1 = 5e4 J/mol)
        2. B -> C       k2 = 100/hr (E2 = 7.5e4 J/mol)
        3. 2A -> D      k3 = 10 L/(mol*hr) (E3 = 6e4 J/mol)

    Parameters:
        C_A0 = 5.1 mol/L
        tau = V/F = 20 s = 1/180 hr
        T = 130 C = 403 K (isothermal)

    Stoichiometry matrix (reactions x species: A, B, C, D):
        [[-1,  1,  0,  0],   # A -> B
         [ 0, -1,  1,  0],   # B -> C
         [-2,  0,  0,  1]]   # 2A -> D

    Args:
        name: Reactor identifier.
        isothermal: If True, operate at constant temperature.
        temperature: Operating temperature (K).

    Returns:
        Configured CSTR reactor instance.
    """
    # Compute rate constants at operating temperature
    k1_0 = 50.0  # 1/hr at T_ref
    E1 = 5e4  # J/mol
    k1 = k1_0 * np.exp(-E1 / (R_GAS * temperature))

    k2_0 = 100.0  # 1/hr at T_ref
    E2 = 7.5e4  # J/mol
    k2 = k2_0 * np.exp(-E2 / (R_GAS * temperature))

    k3_0 = 10.0  # L/(mol*hr) at T_ref
    E3 = 6e4  # J/mol
    k3 = k3_0 * np.exp(-E3 / (R_GAS * temperature))

    # Kinetics
    kinetics = ArrheniusKinetics(
        name="van_de_vusse",
        num_reactions=3,
        params={
            "k0": np.array([k1_0, k2_0, k3_0]),  # Pre-exponential factors
            "Ea": np.array([E1, E2, E3]),  # Activation energies
            "stoich": np.array([
                [-1,  1,  0,  0],  # A -> B
                [ 0, -1,  1,  0],  # B -> C
                [-2,  0,  0,  1],  # 2A -> D
            ]),
            "orders": np.array([
                [1, 0, 0, 0],  # r1 = k1 * C_A
                [0, 1, 0, 0],  # r2 = k2 * C_B
                [2, 0, 0, 0],  # r3 = k3 * C_A^2
            ]),
        },
    )

    # Reactor parameters
    tau = 20.0 / 3600.0  # Residence time: 20 s = 1/180 hr
    V = 1.0  # Arbitrary volume (L), normalized
    F = V / tau  # Flow rate (L/hr)

    reactor_params = {
        "V": V,
        "F": F,
        "C_feed": [5.1, 0.0, 0.0, 0.0],  # Feed only A
        "T_feed": temperature,
        "C_initial": [1.0, 0.5, 0.0, 0.0],  # Start with some A and B
        "T_initial": temperature,
    }

    reactor = CSTRReactor(
        name=name,
        num_species=4,  # A, B, C, D
        params=reactor_params,
        kinetics=kinetics,
        isothermal=isothermal,
    )

    logger.info(f"Created Van de Vusse CSTR: {reactor}")
    logger.info(f"  Residence time: {tau * 3600:.1f} s")
    logger.info(f"  Operating temperature: {temperature:.1f} K ({temperature - 273.15:.1f} C)")
    logger.info(f"  Rate constants: k1={k1:.2e}, k2={k2:.2e}, k3={k3:.2e}")

    return reactor


__all__ = ["create_van_de_vusse_cstr"]
