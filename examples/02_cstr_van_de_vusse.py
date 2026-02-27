"""Van de Vusse reaction system in CSTR.

Demonstrates:
1. Complex reaction network: A -> B -> C, 2A -> D
2. Selectivity of desired product B vs byproducts C and D
3. Effect of residence time on selectivity and conversion

Run: python examples/02_cstr_van_de_vusse.py
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from reactor_twin import create_van_de_vusse_cstr


def main() -> None:
    """Run Van de Vusse CSTR example."""
    print("=" * 60)
    print("Example 02: Van de Vusse Reaction System")
    print("=" * 60)

    # 1. Create Van de Vusse CSTR
    print("\n1. Creating Van de Vusse CSTR...")
    print("   Reactions: A -> B -> C  (series)")
    print("              2A -> D      (parallel)")
    reactor = create_van_de_vusse_cstr()
    print(f"   Reactor: {reactor}")
    print(f"   Species: A, B, C, D (4 components)")
    print(f"   State dim: {reactor.state_dim}")

    # 2. Simulate to steady state
    print("\n2. Simulating to steady state...")
    y0 = reactor.get_initial_state()
    t_span = np.linspace(0, 0.05, 300)  # Short time (fast dynamics)

    sol = solve_ivp(
        reactor.ode_rhs,
        [t_span[0], t_span[-1]],
        y0,
        t_eval=t_span,
        method="LSODA",
    )

    C_A, C_B, C_C, C_D = sol.y[0, -1], sol.y[1, -1], sol.y[2, -1], sol.y[3, -1]
    print(f"   Steady-state concentrations:")
    print(f"     C_A = {C_A:.4f} mol/L (reactant)")
    print(f"     C_B = {C_B:.4f} mol/L (desired product)")
    print(f"     C_C = {C_C:.4f} mol/L (over-oxidation)")
    print(f"     C_D = {C_D:.4f} mol/L (dimerization)")

    C_A_feed = reactor.params["C_feed"][0]
    conversion = 1.0 - C_A / C_A_feed
    selectivity_B = C_B / (C_B + C_C + C_D) if (C_B + C_C + C_D) > 0 else 0
    print(f"   Conversion of A: {conversion:.3f}")
    print(f"   Selectivity to B: {selectivity_B:.3f}")

    # 3. Effect of residence time
    print("\n3. Effect of residence time on selectivity...")
    print("   (Varying flow rate F while keeping V constant)")

    # Original: V=1, F=V/tau where tau=20/3600
    V = reactor.params["V"]

    residence_times_s = [5, 10, 20, 40, 80, 160]  # seconds

    print(f"   {'tau (s)':>8} | {'Conv_A':>8} | {'C_B':>8} | {'C_C':>8} | {'C_D':>8} | {'Sel_B':>8}")
    print("   " + "-" * 58)

    for tau_s in residence_times_s:
        tau_hr = tau_s / 3600.0
        F_new = V / tau_hr

        reactor.params["F"] = F_new

        sol_tau = solve_ivp(
            reactor.ode_rhs,
            [0, 0.1],
            y0,
            t_eval=np.linspace(0, 0.1, 500),
            method="LSODA",
        )

        ca = sol_tau.y[0, -1]
        cb = sol_tau.y[1, -1]
        cc = sol_tau.y[2, -1]
        cd = sol_tau.y[3, -1]

        conv = 1.0 - ca / C_A_feed
        sel = cb / (cb + cc + cd) if (cb + cc + cd) > 1e-10 else 0.0

        print(f"   {tau_s:>8.0f} | {conv:>8.3f} | {cb:>8.4f} | {cc:>8.4f} | {cd:>8.4f} | {sel:>8.3f}")

    print("\n" + "=" * 60)
    print("Example 02 complete!")
    print("Key insight: There is an optimal residence time that maximizes")
    print("selectivity to B. Too short = low conversion, too long = B")
    print("decomposes to C and A dimerizes to D.")
    print("=" * 60)


if __name__ == "__main__":
    main()
