"""Model Predictive Control (MPC) with a NeuralODE plant model.

Demonstrates:
1. Using MPCController with a NeuralODE as the dynamics model
2. Setting a reference trajectory (setpoint tracking)
3. Running receding-horizon MPC steps
4. Showing control inputs and tracking performance

Run: python examples/13_mpc_control.py
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.integrate import solve_ivp

from reactor_twin import (
    ArrheniusKinetics,
    CSTRReactor,
    MPCController,
    NeuralODE,
)

np.random.seed(42)
torch.manual_seed(42)


def main() -> None:
    """Run MPC control example."""
    print("=" * 60)
    print("Example 13: Model Predictive Control (MPC)")
    print("=" * 60)

    state_dim = 2
    input_dim = 1

    # 1. Create a CSTR for reference data
    print("\n1. Generating CSTR reference data...")
    kinetics = ArrheniusKinetics(
        name="A_to_B",
        num_reactions=1,
        params={
            "k0": np.array([0.3]),
            "Ea": np.array([0.0]),
            "stoich": np.array([[-1, 1]]),
            "orders": np.array([[1, 0]]),
        },
    )
    reactor = CSTRReactor(
        name="cstr_mpc",
        num_species=2,
        params={
            "V": 10.0,
            "F": 1.0,
            "C_feed": [1.0, 0.0],
            "T_feed": 350.0,
            "C_initial": [0.5, 0.5],
        },
        kinetics=kinetics,
        isothermal=True,
    )

    y0 = reactor.get_initial_state()
    t_eval = np.linspace(0, 10, 50)
    sol = solve_ivp(reactor.ode_rhs, [0, 10], y0, t_eval=t_eval, method="LSODA")
    print(f"   Steady state: C_A={sol.y[0, -1]:.4f}, C_B={sol.y[1, -1]:.4f}")

    # 2. Create NeuralODE with input_dim=1 for MPC
    #    We train it using ode_func directly (Euler steps) to handle the control input.
    print("\n2. Training NeuralODE with control input...")
    model = NeuralODE(
        state_dim=state_dim,
        input_dim=input_dim,
        hidden_dim=32,
        num_layers=2,
        solver="rk4",
        adjoint=False,
    )

    # Train using Euler rollout (matching how MPC uses the model):
    # The ode_func accepts (t, z, u), so we do Euler integration ourselves
    # with u=0 to match the reactor's open-loop dynamics.
    targets = torch.tensor(sol.y.T, dtype=torch.float32)  # (50, 2)
    dt_train = t_eval[1] - t_eval[0]

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()

    for epoch in range(200):
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0)

        z = torch.tensor(y0, dtype=torch.float32).unsqueeze(0)  # (1, 2)
        u_zero = torch.zeros(1, input_dim)  # No control during training

        for k in range(min(20, len(t_eval) - 1)):
            t_k = torch.tensor(0.0)
            dzdt = model.ode_func(t_k, z, u_zero)
            z = z + dzdt * dt_train
            target_k = targets[k + 1].unsqueeze(0)
            total_loss = total_loss + torch.mean((z - target_k) ** 2)

        total_loss = total_loss / 20
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"   Epoch {epoch + 1}: loss = {total_loss.item():.6f}")

    print(f"   Model: state_dim={state_dim}, input_dim={input_dim}")

    # 3. Set up MPC
    print("\n3. Setting up MPC controller...")
    horizon = 5
    dt_mpc = 0.2

    mpc = MPCController(
        model=model,
        horizon=horizon,
        dt=dt_mpc,
        max_iter=10,
    )
    print(f"   Horizon: {horizon} steps")
    print(f"   dt: {dt_mpc}")

    # 4. Define reference and initial state
    print("\n4. Defining setpoint and running MPC...")
    y_ref = torch.tensor([0.23, 0.77], dtype=torch.float32)
    z_current = torch.tensor([0.8, 0.2], dtype=torch.float32)

    print(f"   Reference: C_A={y_ref[0]:.2f}, C_B={y_ref[1]:.2f}")
    print(f"   Start:     C_A={z_current[0]:.2f}, C_B={z_current[1]:.2f}")

    # 5. Run MPC loop
    num_mpc_steps = 20
    state_history = [z_current.numpy().copy()]
    control_history = []
    cost_history = []

    print(f"\n5. Running {num_mpc_steps} MPC steps...")
    print(f"   {'Step':>5} | {'C_A':>8} | {'C_B':>8} | {'Control':>8} | {'Cost':>10}")
    print("   " + "-" * 48)

    for step in range(num_mpc_steps):
        u_applied, info = mpc.step(z_current, y_ref)

        control_val = u_applied.item()
        cost_val = info["cost"]

        control_history.append(control_val)
        cost_history.append(cost_val)

        print(
            f"   {step:>5} | {z_current[0].item():>8.4f} | {z_current[1].item():>8.4f} | "
            f"{control_val:>8.4f} | {cost_val:>10.4f}"
        )

        # Simulate one step forward using Euler with the model
        z_batch = z_current.unsqueeze(0)
        u_batch = u_applied.unsqueeze(0)
        with torch.no_grad():
            dzdt = model.ode_func(torch.tensor(0.0), z_batch, u_batch).squeeze(0)
        z_current = z_current + dzdt * dt_mpc

        state_history.append(z_current.detach().numpy().copy())

    # 6. Results
    print("\n6. MPC results summary:")
    state_history = np.array(state_history)
    control_history = np.array(control_history)

    tracking_error = np.sqrt(np.mean((state_history[-1] - y_ref.numpy()) ** 2))
    initial_error = np.sqrt(np.mean((state_history[0] - y_ref.numpy()) ** 2))

    print(f"   Initial tracking error:  {initial_error:.4f}")
    print(f"   Final tracking error:    {tracking_error:.4f}")
    print(f"   Error reduction:         {initial_error / max(tracking_error, 1e-10):.1f}x")
    print(f"   Mean control effort:     {np.mean(np.abs(control_history)):.4f}")
    print(f"   Max control:             {np.max(np.abs(control_history)):.4f}")
    print(f"   Final state: C_A={state_history[-1, 0]:.4f}, C_B={state_history[-1, 1]:.4f}")
    print(f"   Reference:   C_A={y_ref[0]:.4f}, C_B={y_ref[1]:.4f}")

    print("\n" + "=" * 60)
    print("Example 13 complete!")
    print("Key insight: MPC with a NeuralODE plant model can drive the system")
    print("toward a desired setpoint. The receding-horizon approach provides")
    print("feedback to handle model mismatch and disturbances.")
    print("=" * 60)


if __name__ == "__main__":
    main()
