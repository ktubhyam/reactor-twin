"""Batch reactor with consecutive reactions A -> B -> C.

Demonstrates:
1. Creating a BatchReactor with ArrheniusKinetics for 2 reactions
2. Finding the optimal batch time for maximum B yield
3. Concentration profiles over time

Run: python examples/03_batch_consecutive.py
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from reactor_twin import ArrheniusKinetics, BatchReactor


def main() -> None:
    """Run batch consecutive reactions example."""
    print("=" * 60)
    print("Example 03: Batch Reactor with Consecutive A -> B -> C")
    print("=" * 60)

    # 1. Define kinetics: A -> B -> C
    print("\n1. Defining consecutive reaction kinetics...")
    print("   Reaction 1: A -> B  (k1 = 0.5 /min at 350 K)")
    print("   Reaction 2: B -> C  (k2 = 0.1 /min at 350 K)")

    kinetics = ArrheniusKinetics(
        name="consecutive_ABC",
        num_reactions=2,
        params={
            "k0": np.array([0.5, 0.1]),       # Pre-exponential (1/min)
            "Ea": np.array([0.0, 0.0]),        # Zero Ea for simplicity (constant k)
            "stoich": np.array([
                [-1,  1,  0],  # A -> B
                [ 0, -1,  1],  # B -> C
            ]),
            "orders": np.array([
                [1, 0, 0],  # First order in A
                [0, 1, 0],  # First order in B
            ]),
        },
    )

    # 2. Create batch reactor
    print("\n2. Creating batch reactor...")
    reactor = BatchReactor(
        name="batch_consecutive",
        num_species=3,
        params={
            "V": 10.0,       # L
            "T": 350.0,      # K (isothermal)
            "C_initial": [1.0, 0.0, 0.0],  # Pure A initially
        },
        kinetics=kinetics,
        isothermal=True,
    )
    print(f"   Reactor: {reactor}")
    print(f"   Initial: C_A=1.0, C_B=0.0, C_C=0.0 mol/L")

    # 3. Simulate
    print("\n3. Simulating batch reaction...")
    y0 = reactor.get_initial_state()
    t_eval = np.linspace(0, 30, 300)

    sol = solve_ivp(
        reactor.ode_rhs,
        [0, 30],
        y0,
        t_eval=t_eval,
        method="LSODA",
    )

    C_A = sol.y[0]
    C_B = sol.y[1]
    C_C = sol.y[2]

    # 4. Find optimal batch time for max B
    print("\n4. Finding optimal batch time for maximum B yield...")
    idx_max_B = np.argmax(C_B)
    t_opt = t_eval[idx_max_B]
    C_B_max = C_B[idx_max_B]

    print(f"   Optimal batch time: {t_opt:.2f} min")
    print(f"   Maximum C_B: {C_B_max:.4f} mol/L")
    print(f"   At t_opt: C_A={C_A[idx_max_B]:.4f}, C_B={C_B[idx_max_B]:.4f}, C_C={C_C[idx_max_B]:.4f}")

    # Analytical solution for t_opt: t* = ln(k1/k2) / (k1 - k2)
    k1, k2 = 0.5, 0.1
    t_opt_analytical = np.log(k1 / k2) / (k1 - k2)
    print(f"   Analytical t_opt: {t_opt_analytical:.2f} min")

    # 5. Show profiles at key times
    print("\n5. Concentration profiles at key times:")
    print(f"   {'Time (min)':>10} | {'C_A':>8} | {'C_B':>8} | {'C_C':>8} | {'Yield_B':>8}")
    print("   " + "-" * 50)

    key_times = [0, 2, 4, t_opt, 10, 20, 30]
    for t_key in key_times:
        idx = np.argmin(np.abs(t_eval - t_key))
        yield_B = C_B[idx] / 1.0  # Normalized by initial C_A
        print(f"   {t_eval[idx]:>10.1f} | {C_A[idx]:>8.4f} | {C_B[idx]:>8.4f} | {C_C[idx]:>8.4f} | {yield_B:>8.3f}")

    # 6. Mass balance check
    print("\n6. Mass balance verification:")
    total = C_A[-1] + C_B[-1] + C_C[-1]
    print(f"   Sum of concentrations at t=30: {total:.6f} mol/L")
    print(f"   Initial total: {y0.sum():.6f} mol/L")
    print(f"   Error: {abs(total - y0.sum()):.2e}")

    print("\n" + "=" * 60)
    print("Example 03 complete!")
    print("Key insight: In consecutive reactions A->B->C, the intermediate B")
    print(f"reaches a maximum at t={t_opt:.1f} min. Stopping the batch early")
    print("maximizes yield of the desired product.")
    print("=" * 60)


if __name__ == "__main__":
    main()
