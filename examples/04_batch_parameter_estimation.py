"""Parameter estimation: learn batch reactor dynamics from noisy data.

Demonstrates:
1. Generating synthetic batch data with known kinetics
2. Adding measurement noise
3. Training a NeuralODE to learn the dynamics
4. Comparing learned vs true trajectories

Run: python examples/04_batch_parameter_estimation.py
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.integrate import solve_ivp

from reactor_twin import ArrheniusKinetics, BatchReactor, NeuralODE

np.random.seed(42)
torch.manual_seed(42)


def main() -> None:
    """Run batch parameter estimation example."""
    print("=" * 60)
    print("Example 04: Batch Reactor Parameter Estimation")
    print("=" * 60)

    # 1. Generate synthetic ground-truth data
    print("\n1. Generating ground-truth batch reactor data...")
    kinetics = ArrheniusKinetics(
        name="true_kinetics",
        num_reactions=1,
        params={
            "k0": np.array([0.3]),
            "Ea": np.array([0.0]),
            "stoich": np.array([[-1, 1]]),
            "orders": np.array([[1, 0]]),
        },
    )

    reactor = BatchReactor(
        name="batch_true",
        num_species=2,
        params={
            "V": 1.0,
            "T": 350.0,
            "C_initial": [1.0, 0.0],
        },
        kinetics=kinetics,
        isothermal=True,
    )

    y0 = reactor.get_initial_state()
    t_eval = np.linspace(0, 10, 50)

    sol = solve_ivp(
        reactor.ode_rhs,
        [0, 10],
        y0,
        t_eval=t_eval,
        method="LSODA",
    )
    true_data = sol.y.T  # (50, 2)
    print(f"   True data shape: {true_data.shape}")
    print(f"   C_A: {true_data[0, 0]:.3f} -> {true_data[-1, 0]:.3f}")
    print(f"   C_B: {true_data[0, 1]:.3f} -> {true_data[-1, 1]:.3f}")

    # 2. Add noise
    print("\n2. Adding measurement noise (sigma=0.02)...")
    noise_level = 0.02
    noisy_data = true_data + noise_level * np.random.randn(*true_data.shape)
    noisy_data = np.maximum(noisy_data, 0)  # Keep non-negative
    print(f"   Noisy data range: [{noisy_data.min():.4f}, {noisy_data.max():.4f}]")

    # 3. Prepare PyTorch tensors
    z0 = torch.tensor(noisy_data[0], dtype=torch.float32).unsqueeze(0)  # (1, 2)
    t_span = torch.tensor(t_eval, dtype=torch.float32)
    targets = torch.tensor(noisy_data, dtype=torch.float32).unsqueeze(0)  # (1, 50, 2)

    # 4. Create Neural ODE
    print("\n3. Creating Neural ODE model...")
    model = NeuralODE(
        state_dim=2,
        hidden_dim=32,
        num_layers=2,
        solver="rk4",
        adjoint=False,
    )
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {num_params:,}")

    # 5. Train
    print("\n4. Training Neural ODE on noisy data...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 200

    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        preds = model(z0, t_span)
        loss_dict = model.compute_loss(preds, targets)
        loss_dict["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"   Epoch {epoch + 1:>4d}/{num_epochs}: loss = {loss_dict['total'].item():.6f}")

    # 6. Evaluate
    print("\n5. Comparing learned vs true trajectories...")
    model.eval()
    with torch.no_grad():
        z0_true = torch.tensor(y0, dtype=torch.float32).unsqueeze(0)
        predictions = model(z0_true, t_span)

    pred_np = predictions[0].numpy()  # (50, 2)
    true_np = true_data  # (50, 2)

    # Compute errors at key times
    print(f"   {'Time':>6} | {'True C_A':>10} | {'Pred C_A':>10} | {'True C_B':>10} | {'Pred C_B':>10}")
    print("   " + "-" * 55)

    for idx in [0, 12, 24, 36, 49]:
        t = t_eval[idx]
        print(
            f"   {t:>6.1f} | {true_np[idx, 0]:>10.4f} | {pred_np[idx, 0]:>10.4f} | "
            f"{true_np[idx, 1]:>10.4f} | {pred_np[idx, 1]:>10.4f}"
        )

    # Overall error
    mse = np.mean((pred_np - true_np) ** 2)
    rmse = np.sqrt(mse)
    print(f"\n   Overall RMSE: {rmse:.6f}")
    print(f"   Noise level:  {noise_level:.6f}")
    print(f"   RMSE vs noise ratio: {rmse / noise_level:.2f}")

    print("\n" + "=" * 60)
    print("Example 04 complete!")
    print("Key insight: A NeuralODE can learn reactor dynamics from noisy data.")
    print("The learned model recovers the underlying smooth trajectory despite")
    print("measurement noise.")
    print("=" * 60)


if __name__ == "__main__":
    main()
