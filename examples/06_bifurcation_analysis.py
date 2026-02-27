"""Bifurcation analysis of an exothermic CSTR.

Demonstrates:
1. Sweeping coolant temperature to find multiple steady states
2. Using scipy to find steady states at each coolant temperature
3. Showing hysteresis / bifurcation behavior

Run: python examples/06_bifurcation_analysis.py
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

from reactor_twin import create_exothermic_cstr


def main() -> None:
    """Run bifurcation analysis example."""
    print("=" * 60)
    print("Example 06: Bifurcation Analysis of Exothermic CSTR")
    print("=" * 60)

    # 1. Create reactor
    print("\n1. Creating non-isothermal exothermic CSTR...")
    reactor = create_exothermic_cstr(isothermal=False)
    print(f"   Reactor: {reactor}")

    # 2. Find steady states using fsolve from different initial guesses
    print("\n2. Finding steady states via fsolve...")

    def steady_state_rhs(y: np.ndarray) -> np.ndarray:
        """RHS for steady state: dy/dt = 0."""
        return reactor.ode_rhs(0, y)

    # Try multiple initial guesses to find different steady states
    T_coolant_test = 300.0
    reactor.params["T_coolant"] = T_coolant_test

    initial_guesses = [
        np.array([0.9, 0.1, 310.0]),  # Low conversion (cold)
        np.array([0.5, 0.5, 380.0]),  # Medium conversion
        np.array([0.05, 0.95, 450.0]),  # High conversion (hot)
    ]

    print(f"   T_coolant = {T_coolant_test} K")
    found_states = []
    for i, guess in enumerate(initial_guesses):
        try:
            ss, info, ier, msg = fsolve(steady_state_rhs, guess, full_output=True)
            residual = np.linalg.norm(info["fvec"])
            if ier == 1 and residual < 1e-6 and ss[0] >= 0 and ss[1] >= 0 and ss[2] > 250:
                # Check if this is a new steady state
                is_new = True
                for prev in found_states:
                    if np.linalg.norm(ss - prev) < 1.0:
                        is_new = False
                        break
                if is_new:
                    found_states.append(ss)
                    print(
                        f"   SS {len(found_states)}: C_A={ss[0]:.4f}, C_B={ss[1]:.4f}, T={ss[2]:.1f} K (residual={residual:.2e})"
                    )
        except Exception:
            pass

    if len(found_states) < 2:
        print("   (Found fewer than expected -- fsolve is sensitive to guesses)")

    # 3. Sweep coolant temperature for bifurcation diagram
    print("\n3. Building bifurcation diagram by sweeping T_coolant...")
    print("   (Running dynamic simulation to steady state at each T_coolant)")

    coolant_range = np.linspace(280, 330, 26)

    # Forward sweep: start from cold state
    print("\n   Forward sweep (cold start):")
    print(f"   {'T_coolant':>10} | {'T_ss':>10} | {'C_A_ss':>10} | {'Conv':>8}")
    print("   " + "-" * 44)

    y_current_fwd = np.array([1.0, 0.0, 300.0])  # Cold initial condition

    for T_c in coolant_range:
        reactor.params["T_coolant"] = T_c
        sol = solve_ivp(
            reactor.ode_rhs,
            [0, 20],
            y_current_fwd,
            t_eval=[20],
            method="LSODA",
        )
        ss = sol.y[:, -1]
        y_current_fwd = ss.copy()
        conv = 1.0 - ss[0] / reactor.params["C_feed"][0]
        print(f"   {T_c:>10.1f} | {ss[2]:>10.1f} | {ss[0]:>10.4f} | {conv:>8.3f}")

    # Reverse sweep: start from hot state
    print("\n   Reverse sweep (hot start):")
    print(f"   {'T_coolant':>10} | {'T_ss':>10} | {'C_A_ss':>10} | {'Conv':>8}")
    print("   " + "-" * 44)

    y_current_rev = np.array([0.01, 0.99, 450.0])  # Hot initial condition

    for T_c in reversed(coolant_range):
        reactor.params["T_coolant"] = T_c
        sol = solve_ivp(
            reactor.ode_rhs,
            [0, 20],
            y_current_rev,
            t_eval=[20],
            method="LSODA",
        )
        ss = sol.y[:, -1]
        y_current_rev = ss.copy()
        conv = 1.0 - ss[0] / reactor.params["C_feed"][0]
        print(f"   {T_c:>10.1f} | {ss[2]:>10.1f} | {ss[0]:>10.4f} | {conv:>8.3f}")

    print("\n" + "=" * 60)
    print("Example 06 complete!")
    print("Key insight: Exothermic CSTRs can exhibit multiple steady states.")
    print("Forward and reverse sweeps may settle on different branches,")
    print("demonstrating hysteresis -- a hallmark of bifurcation behavior.")
    print("=" * 60)


if __name__ == "__main__":
    main()
