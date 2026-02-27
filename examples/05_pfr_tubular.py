"""Plug Flow Reactor (PFR) with Arrhenius kinetics.

Demonstrates:
1. Creating a PFR with spatial discretization (Method of Lines)
2. Simulating to steady state
3. Showing axial concentration profiles along reactor length

Run: python examples/05_pfr_tubular.py
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from reactor_twin import ArrheniusKinetics, PlugFlowReactor


def main() -> None:
    """Run PFR tubular reactor example."""
    print("=" * 60)
    print("Example 05: Plug Flow Reactor (PFR)")
    print("=" * 60)

    # 1. Define kinetics: A -> B (first-order)
    print("\n1. Defining reaction kinetics...")
    print("   Reaction: A -> B (first-order, Arrhenius)")

    kinetics = ArrheniusKinetics(
        name="A_to_B_pfr",
        num_reactions=1,
        params={
            "k0": np.array([1e6]),  # Pre-exponential (1/s)
            "Ea": np.array([40000.0]),  # Activation energy (J/mol)
            "stoich": np.array([[-1, 1]]),  # A -> B
            "orders": np.array([[1, 0]]),  # First order in A
        },
    )

    # 2. Create PFR
    print("\n2. Creating PFR with 20 spatial cells...")
    num_cells = 20
    reactor = PlugFlowReactor(
        name="tubular_pfr",
        num_species=2,
        params={
            "L": 2.0,  # Reactor length (m)
            "u": 0.5,  # Axial velocity (m/s)
            "D": 0.01,  # Dispersion coefficient (m^2/s)
            "C_in": [1.0, 0.0],  # Inlet: pure A
            "T": 400.0,  # Temperature (K)
        },
        kinetics=kinetics,
        num_cells=num_cells,
    )
    print(f"   Reactor: {reactor}")
    print(f"   State dim: {reactor.state_dim} ({reactor.num_species} species x {num_cells} cells)")
    print(f"   Cell size: {reactor.dz:.4f} m")

    # 3. Simulate to steady state
    print("\n3. Simulating to steady state...")
    y0 = reactor.get_initial_state()
    # PFR reaches steady state relatively quickly
    t_eval = np.linspace(0, 20, 100)

    sol = solve_ivp(
        reactor.ode_rhs,
        [0, 20],
        y0,
        t_eval=t_eval,
        method="LSODA",
    )
    print(f"   Integration success: {sol.success}")

    # 4. Extract steady-state axial profiles
    print("\n4. Steady-state axial concentration profiles:")
    y_ss = sol.y[:, -1]  # Last time point
    z_positions, C_profiles = reactor.get_axial_profile(y_ss)

    C_A_profile = C_profiles[0]  # Species A
    C_B_profile = C_profiles[1]  # Species B

    print(
        f"   {'Position (m)':>12} | {'C_A (mol/L)':>12} | {'C_B (mol/L)':>12} | {'Conversion':>10}"
    )
    print("   " + "-" * 52)

    for i in range(0, num_cells, max(1, num_cells // 10)):
        conv = 1.0 - C_A_profile[i] / 1.0
        print(
            f"   {z_positions[i]:>12.3f} | {C_A_profile[i]:>12.4f} | {C_B_profile[i]:>12.4f} | {conv:>10.3f}"
        )

    # Print outlet
    outlet = reactor.get_outlet_concentrations(y_ss)
    print("\n   Outlet concentrations:")
    print(f"     C_A = {outlet[0]:.4f} mol/L")
    print(f"     C_B = {outlet[1]:.4f} mol/L")
    print(f"     Overall conversion = {1.0 - outlet[0] / 1.0:.3f}")

    # 5. Effect of dispersion coefficient
    print("\n5. Effect of dispersion coefficient on conversion:")
    dispersion_values = [0.001, 0.01, 0.05, 0.1, 0.5]

    print(f"   {'D (m^2/s)':>10} | {'C_A_out':>10} | {'Conversion':>10}")
    print("   " + "-" * 35)

    for D_val in dispersion_values:
        reactor.dispersion = D_val
        y0_d = reactor.get_initial_state()

        sol_d = solve_ivp(
            reactor.ode_rhs,
            [0, 30],
            y0_d,
            t_eval=np.linspace(0, 30, 100),
            method="LSODA",
        )

        outlet_d = reactor.get_outlet_concentrations(sol_d.y[:, -1])
        conv_d = 1.0 - outlet_d[0] / 1.0
        print(f"   {D_val:>10.3f} | {outlet_d[0]:>10.4f} | {conv_d:>10.3f}")

    # Restore
    reactor.dispersion = 0.01

    # 6. Mass balance check
    print("\n6. Mass balance check at steady state:")
    total_outlet = outlet[0] + outlet[1]
    print(f"   Inlet total:  {sum(reactor.params['C_in']):.4f} mol/L")
    print(f"   Outlet total: {total_outlet:.4f} mol/L")

    print("\n" + "=" * 60)
    print("Example 05 complete!")
    print("Key insight: PFR achieves higher conversion than CSTR for same volume")
    print("and first-order kinetics. Axial dispersion reduces conversion by")
    print("back-mixing (making the PFR more CSTR-like).")
    print("=" * 60)


if __name__ == "__main__":
    main()
