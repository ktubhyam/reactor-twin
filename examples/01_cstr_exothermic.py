"""Non-isothermal exothermic A -> B CSTR simulation.

Demonstrates:
1. Creating a non-isothermal exothermic CSTR with create_exothermic_cstr()
2. Simulating with scipy to observe temperature runaway
3. Showing the effect of coolant temperature on steady-state behavior

Run: python examples/01_cstr_exothermic.py
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from reactor_twin import create_exothermic_cstr


def main() -> None:
    """Run exothermic CSTR example."""
    print("=" * 60)
    print("Example 01: Non-isothermal Exothermic CSTR")
    print("=" * 60)

    # 1. Create non-isothermal CSTR
    print("\n1. Creating non-isothermal exothermic CSTR (A -> B)...")
    reactor = create_exothermic_cstr(isothermal=False)
    print(f"   Reactor: {reactor}")
    print(f"   State variables: {reactor.get_state_labels()}")
    print(f"   Volume: {reactor.params['V']} L, Flow: {reactor.params['F']} L/min")

    # 2. Simulate the base case
    print("\n2. Simulating base case (T_coolant = 300 K)...")
    y0 = reactor.get_initial_state()
    t_span = np.linspace(0, 5, 200)

    sol = solve_ivp(
        reactor.ode_rhs,
        [t_span[0], t_span[-1]],
        y0,
        t_eval=t_span,
        method="LSODA",
    )

    C_A = sol.y[0]
    C_B = sol.y[1]
    T = sol.y[2]

    print(f"   Initial: C_A={C_A[0]:.3f} mol/L, C_B={C_B[0]:.3f} mol/L, T={T[0]:.1f} K")
    print(f"   Final:   C_A={C_A[-1]:.3f} mol/L, C_B={C_B[-1]:.3f} mol/L, T={T[-1]:.1f} K")
    print(f"   Temperature rise: {T[-1] - T[0]:.1f} K")

    # 3. Effect of coolant temperature
    print("\n3. Sweeping coolant temperature to show thermal behavior...")
    coolant_temps = [280, 290, 300, 310, 320]

    print(f"   {'T_coolant (K)':>14} | {'C_A_ss (mol/L)':>15} | {'T_ss (K)':>10} | {'Conversion':>10}")
    print("   " + "-" * 60)

    for T_c in coolant_temps:
        # Modify coolant temperature
        reactor.params["T_coolant"] = T_c

        sol_sweep = solve_ivp(
            reactor.ode_rhs,
            [0, 10],
            y0,
            t_eval=np.linspace(0, 10, 500),
            method="LSODA",
        )

        C_A_ss = sol_sweep.y[0, -1]
        T_ss = sol_sweep.y[2, -1]
        conversion = 1.0 - C_A_ss / reactor.params["C_feed"][0]

        print(f"   {T_c:>14.0f} | {C_A_ss:>15.4f} | {T_ss:>10.1f} | {conversion:>10.3f}")

    # 4. Show temperature runaway with high coolant temp
    print("\n4. Demonstrating thermal sensitivity...")
    reactor.params["T_coolant"] = 305.0
    y0_hot = np.array([0.5, 0.0, 370.0])  # Start hotter

    sol_hot = solve_ivp(
        reactor.ode_rhs,
        [0, 2],
        y0_hot,
        t_eval=np.linspace(0, 2, 100),
        method="LSODA",
    )

    T_hot = sol_hot.y[2]
    T_max = np.max(T_hot)
    T_final = T_hot[-1]
    print(f"   Starting at T=370 K with T_coolant=305 K:")
    print(f"   Peak temperature: {T_max:.1f} K")
    print(f"   Final temperature: {T_final:.1f} K")
    print(f"   Temperature overshoot: {T_max - 370.0:.1f} K")

    print("\n" + "=" * 60)
    print("Example 01 complete!")
    print("Key insight: Exothermic CSTRs show strong sensitivity to coolant")
    print("temperature -- small changes can cause large shifts in steady state.")
    print("=" * 60)


if __name__ == "__main__":
    main()
