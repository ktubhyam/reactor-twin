"""Uncertainty quantification with ensemble of NeuralODEs.

Demonstrates:
1. Training multiple NeuralODEs with different random seeds
2. Using the ensemble spread as an uncertainty estimate
3. Comparing predictions and confidence intervals

Note: If torchsde is available, a NeuralSDE could be used directly.
This example uses the ensemble approach which works without extra deps.

Run: python examples/09_neural_sde_uncertainty.py
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.integrate import solve_ivp

from reactor_twin import ArrheniusKinetics, BatchReactor, NeuralODE

np.random.seed(42)


def main() -> None:
    """Run uncertainty quantification example."""
    print("=" * 60)
    print("Example 09: Uncertainty Quantification (Ensemble NeuralODEs)")
    print("=" * 60)

    # 1. Generate ground-truth data with noise
    print("\n1. Generating noisy batch reactor data...")
    kinetics = ArrheniusKinetics(
        name="A_to_B",
        num_reactions=1,
        params={
            "k0": np.array([0.4]),
            "Ea": np.array([0.0]),
            "stoich": np.array([[-1, 1]]),
            "orders": np.array([[1, 0]]),
        },
    )
    reactor = BatchReactor(
        name="batch",
        num_species=2,
        params={"V": 1.0, "T": 350.0, "C_initial": [1.0, 0.0]},
        kinetics=kinetics,
        isothermal=True,
    )

    y0 = reactor.get_initial_state()
    t_eval = np.linspace(0, 8, 30)

    sol = solve_ivp(reactor.ode_rhs, [0, 8], y0, t_eval=t_eval, method="LSODA")
    true_data = sol.y.T  # (30, 2)

    # Add noise
    noise_level = 0.03
    noisy_data = true_data + noise_level * np.random.randn(*true_data.shape)
    noisy_data = np.maximum(noisy_data, 0)

    z0 = torch.tensor(noisy_data[0], dtype=torch.float32).unsqueeze(0)
    t_span = torch.tensor(t_eval, dtype=torch.float32)
    targets = torch.tensor(noisy_data, dtype=torch.float32).unsqueeze(0)

    print(f"   Data points: {len(t_eval)}")
    print(f"   Noise level: {noise_level}")

    # 2. Train ensemble
    print("\n2. Training ensemble of NeuralODEs (5 members)...")
    num_ensemble = 5
    num_epochs = 150
    models = []

    for i in range(num_ensemble):
        seed = 42 + i * 7
        torch.manual_seed(seed)

        model = NeuralODE(
            state_dim=2,
            hidden_dim=32,
            num_layers=2,
            solver="rk4",
            adjoint=False,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        model.train()
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            preds = model(z0, t_span)
            loss_dict = model.compute_loss(preds, targets)
            loss_dict["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        final_loss = loss_dict["total"].item()
        models.append(model)
        print(f"   Member {i + 1}/{num_ensemble}: seed={seed}, final_loss={final_loss:.6f}")

    # 3. Generate ensemble predictions
    print("\n3. Generating ensemble predictions...")
    ensemble_preds = []

    for model in models:
        model.eval()
        with torch.no_grad():
            z0_true = torch.tensor(y0, dtype=torch.float32).unsqueeze(0)
            pred = model(z0_true, t_span)
        ensemble_preds.append(pred[0].numpy())

    ensemble_preds = np.array(ensemble_preds)  # (5, 30, 2)

    # Compute statistics
    mean_pred = ensemble_preds.mean(axis=0)   # (30, 2)
    std_pred = ensemble_preds.std(axis=0)     # (30, 2)

    # 4. Display results
    print("\n4. Ensemble predictions with uncertainty:")
    print(f"   {'Time':>6} | {'True C_A':>9} | {'Mean C_A':>9} | {'Std C_A':>8} | "
          f"{'True C_B':>9} | {'Mean C_B':>9} | {'Std C_B':>8}")
    print("   " + "-" * 70)

    for idx in range(0, len(t_eval), 5):
        t = t_eval[idx]
        print(
            f"   {t:>6.1f} | {true_data[idx, 0]:>9.4f} | {mean_pred[idx, 0]:>9.4f} | "
            f"{std_pred[idx, 0]:>8.4f} | {true_data[idx, 1]:>9.4f} | {mean_pred[idx, 1]:>9.4f} | "
            f"{std_pred[idx, 1]:>8.4f}"
        )

    # 5. Coverage analysis
    print("\n5. Uncertainty coverage analysis (2-sigma interval):")
    for dim, name in enumerate(["C_A", "C_B"]):
        lower = mean_pred[:, dim] - 2 * std_pred[:, dim]
        upper = mean_pred[:, dim] + 2 * std_pred[:, dim]
        covered = ((true_data[:, dim] >= lower) & (true_data[:, dim] <= upper)).sum()
        coverage = covered / len(t_eval) * 100
        mean_width = (2 * 2 * std_pred[:, dim]).mean()
        print(f"   {name}: coverage = {coverage:.0f}%, mean interval width = {mean_width:.4f}")

    # 6. Summary statistics
    print("\n6. Summary:")
    mse_ensemble = np.mean((mean_pred - true_data) ** 2)
    mean_uncertainty = std_pred.mean()
    print(f"   Ensemble MSE: {mse_ensemble:.6f}")
    print(f"   Mean uncertainty (std): {mean_uncertainty:.6f}")
    print(f"   Uncertainty / noise ratio: {mean_uncertainty / noise_level:.2f}")

    # Individual model MSEs
    print(f"\n   Individual model MSEs:")
    for i, preds in enumerate(ensemble_preds):
        mse_i = np.mean((preds - true_data) ** 2)
        print(f"     Model {i + 1}: MSE = {mse_i:.6f}")

    print("\n" + "=" * 60)
    print("Example 09 complete!")
    print("Key insight: Ensemble diversity provides calibrated uncertainty.")
    print("Where models agree, we have high confidence; where they disagree,")
    print("we know predictions are less reliable.")
    print("=" * 60)


if __name__ == "__main__":
    main()
