"""State estimation with Extended Kalman Filter (EKF).

Demonstrates:
1. Using EKFStateEstimator with a trained NeuralODE as the process model
2. Generating noisy measurements from a CSTR simulation
3. Running EKF to fuse model predictions with measurements
4. Comparing filtered states vs true states

Run: python examples/11_state_estimation_ekf.py
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.integrate import solve_ivp

from reactor_twin import (
    ArrheniusKinetics,
    CSTRReactor,
    EKFStateEstimator,
    NeuralODE,
)

np.random.seed(42)
torch.manual_seed(42)


def main() -> None:
    """Run EKF state estimation example."""
    print("=" * 60)
    print("Example 11: EKF State Estimation with NeuralODE")
    print("=" * 60)

    # 1. Create a simple CSTR
    print("\n1. Creating isothermal CSTR and generating true trajectory...")
    kinetics = ArrheniusKinetics(
        name="A_to_B",
        num_reactions=1,
        params={
            "k0": np.array([0.5]),
            "Ea": np.array([0.0]),
            "stoich": np.array([[-1, 1]]),
            "orders": np.array([[1, 0]]),
        },
    )
    reactor = CSTRReactor(
        name="cstr_ekf",
        num_species=2,
        params={
            "V": 10.0,
            "F": 1.0,
            "C_feed": [1.0, 0.0],
            "T_feed": 350.0,
            "C_initial": [0.8, 0.2],
        },
        kinetics=kinetics,
        isothermal=True,
    )

    state_dim = 2
    y0 = reactor.get_initial_state()
    dt = 0.1
    num_steps = 100
    t_eval = np.linspace(0, num_steps * dt, num_steps + 1)

    sol = solve_ivp(
        reactor.ode_rhs,
        [0, t_eval[-1]],
        y0,
        t_eval=t_eval,
        method="LSODA",
    )
    true_states = sol.y.T  # (101, 2)
    print(f"   True trajectory: {true_states.shape}")
    print(f"   C_A: {true_states[0, 0]:.3f} -> {true_states[-1, 0]:.3f}")

    # 2. Generate noisy measurements
    print("\n2. Generating noisy measurements...")
    measurement_noise = 0.05
    measurements_np = true_states[1:] + measurement_noise * np.random.randn(num_steps, state_dim)
    measurements_np = np.maximum(measurements_np, 0)
    measurements = torch.tensor(measurements_np, dtype=torch.float32)
    print(f"   Measurement noise: sigma = {measurement_noise}")
    print(f"   Measurements shape: {measurements.shape}")

    # 3. Train a NeuralODE on clean data
    print("\n3. Training NeuralODE as process model...")
    model = NeuralODE(
        state_dim=state_dim,
        input_dim=0,
        hidden_dim=32,
        num_layers=2,
        solver="rk4",
        adjoint=False,
    )

    # Train on the true data
    z0_train = torch.tensor(y0, dtype=torch.float32).unsqueeze(0)
    t_span_train = torch.tensor(t_eval[:51], dtype=torch.float32)  # First half
    targets_train = torch.tensor(true_states[:51], dtype=torch.float32).unsqueeze(0)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        preds = model(z0_train, t_span_train)
        loss_dict = model.compute_loss(preds, targets_train)
        loss_dict["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"   Epoch {epoch + 1}: loss = {loss_dict['total'].item():.6f}")

    # 4. Set up EKF
    print("\n4. Setting up Extended Kalman Filter...")
    ekf = EKFStateEstimator(
        model=model,
        state_dim=state_dim,
        Q=1e-3,  # Process noise
        R=measurement_noise**2,  # Measurement noise (match true noise)
        P0=0.1,  # Initial uncertainty
        dt=dt,
    )
    print(f"   Q (process noise): {1e-3}")
    print(f"   R (measurement noise): {measurement_noise**2}")

    # 5. Run EKF filter
    print("\n5. Running EKF filter over measurements...")
    z0_ekf = torch.tensor(y0, dtype=torch.float32)
    t_span_ekf = torch.tensor(t_eval[1:], dtype=torch.float32)

    result = ekf.filter(
        measurements=measurements,
        z0=z0_ekf,
        t_span=t_span_ekf,
    )

    filtered_states = result["states"].numpy()  # (100, 2)
    innovations = result["innovations"].numpy()  # (100, 2)

    print(f"   Filtered states shape: {filtered_states.shape}")

    # 6. Compare accuracy
    print("\n6. Comparing true vs measured vs filtered states:")
    print(
        f"   {'Time':>6} | {'True C_A':>9} | {'Meas C_A':>9} | {'Filt C_A':>9} | "
        f"{'True C_B':>9} | {'Meas C_B':>9} | {'Filt C_B':>9}"
    )
    print("   " + "-" * 68)

    for idx in [0, 20, 40, 60, 80, 99]:
        t = (idx + 1) * dt
        print(
            f"   {t:>6.1f} | {true_states[idx + 1, 0]:>9.4f} | {measurements_np[idx, 0]:>9.4f} | "
            f"{filtered_states[idx, 0]:>9.4f} | {true_states[idx + 1, 1]:>9.4f} | "
            f"{measurements_np[idx, 1]:>9.4f} | {filtered_states[idx, 1]:>9.4f}"
        )

    # 7. Error analysis
    print("\n7. Error analysis:")
    mse_meas = np.mean((measurements_np - true_states[1:]) ** 2)
    mse_filt = np.mean((filtered_states - true_states[1:]) ** 2)
    rmse_meas = np.sqrt(mse_meas)
    rmse_filt = np.sqrt(mse_filt)

    print(f"   RMSE (measurements):     {rmse_meas:.6f}")
    print(f"   RMSE (EKF filtered):     {rmse_filt:.6f}")
    print(f"   Improvement ratio:       {rmse_meas / max(rmse_filt, 1e-10):.2f}x")

    # Innovation analysis
    innov_mean = np.mean(np.abs(innovations), axis=0)
    print(f"   Mean |innovation| C_A:   {innov_mean[0]:.6f}")
    print(f"   Mean |innovation| C_B:   {innov_mean[1]:.6f}")

    print("\n" + "=" * 60)
    print("Example 11 complete!")
    print("Key insight: The EKF combines the NeuralODE's learned dynamics with")
    print("noisy measurements to produce state estimates that are more accurate")
    print("than raw measurements alone.")
    print("=" * 60)


if __name__ == "__main__":
    main()
