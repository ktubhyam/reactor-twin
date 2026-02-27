"""Quickstart example: Train a Neural ODE on CSTR data.

This example demonstrates the complete workflow:
1. Create a CSTR reactor with Arrhenius kinetics
2. Generate training data using scipy
3. Train a Neural ODE to learn the dynamics
4. Apply physics constraints (positivity)
5. Evaluate predictions vs ground truth

Run: python examples/00_quickstart.py
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.integrate import solve_ivp

from reactor_twin import (
    ArrheniusKinetics,
    CSTRReactor,
    NeuralODE,
    PositivityConstraint,
)

# Set random seeds
np.random.seed(42)
torch.manual_seed(42)


def main() -> None:
    """Run quickstart example."""
    print("=" * 60)
    print("ReactorTwin Quickstart Example")
    print("=" * 60)

    # 1. Define reactor
    print("\n1. Creating CSTR reactor with A -> B kinetics...")
    kinetics = ArrheniusKinetics(
        name="A_to_B",
        num_reactions=1,
        params={
            "k0": [1e10],  # 1/min
            "Ea": [50000.0],  # J/mol
            "stoich": np.array([[-1, 1]]),  # A -> B
        },
    )

    reactor = CSTRReactor(
        name="exothermic_cstr",
        num_species=2,
        params={
            "V": 100.0,  # L
            "F": 10.0,  # L/min
            "C_feed": [1.0, 0.0],  # mol/L
            "T_feed": 350.0,  # K
        },
        kinetics=kinetics,
        isothermal=True,
    )
    print(f"   Created: {reactor}")

    # 2. Generate training data
    print("\n2. Generating training data with scipy...")
    y0 = reactor.get_initial_state()
    t_span = np.linspace(0, 100, 50)

    sol = solve_ivp(
        reactor.ode_rhs,
        [t_span[0], t_span[-1]],
        y0,
        t_eval=t_span,
        method="LSODA",
    )

    # Convert to PyTorch
    z0 = torch.tensor(y0, dtype=torch.float32).unsqueeze(0)  # (1, 2)
    t_span_torch = torch.tensor(t_span, dtype=torch.float32)
    targets = torch.tensor(sol.y.T, dtype=torch.float32).unsqueeze(0)  # (1, 50, 2)

    print(f"   Data shape: {targets.shape}")
    print(f"   C_A: {sol.y[0, 0]:.3f} -> {sol.y[0, -1]:.3f} mol/L")
    print(f"   C_B: {sol.y[1, 0]:.3f} -> {sol.y[1, -1]:.3f} mol/L")

    # 3. Create Neural ODE
    print("\n3. Creating Neural ODE model...")
    model = NeuralODE(
        state_dim=2,
        hidden_dim=32,
        num_layers=2,
        solver="dopri5",
        adjoint=True,
    )
    print(f"   Model: {model}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 4. Train (single step demonstration)
    print("\n4. Training Neural ODE...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    predictions = model(z0, t_span_torch)
    losses = model.compute_loss(predictions, targets)

    print(f"   Initial loss: {losses['total']:.6f}")

    # Single training step
    losses_after = model.train_step(
        batch={"z0": z0, "t_span": t_span_torch, "targets": targets},
        optimizer=optimizer,
    )
    print(f"   After 1 step: {losses_after['total']:.6f}")

    # 5. Apply physics constraints
    print("\n5. Applying positivity constraint...")
    constraint = PositivityConstraint(mode="hard", method="softplus")

    predictions_unconstrained = model.predict(z0, t_span_torch)
    predictions_constrained, _ = constraint(predictions_unconstrained)

    print(f"   Min value (unconstrained): {predictions_unconstrained.min():.6f}")
    print(f"   Min value (constrained): {predictions_constrained.min():.6f}")

    # 6. Evaluate
    print("\n6. Final predictions:")
    pred_final = predictions_constrained[0, -1].detach().numpy()
    true_final = targets[0, -1].numpy()

    print(f"   C_A: pred={pred_final[0]:.3f}, true={true_final[0]:.3f}")
    print(f"   C_B: pred={pred_final[1]:.3f}, true={true_final[1]:.3f}")

    print("\n" + "=" * 60)
    print("Quickstart complete! ✓")
    print("Next steps:")
    print("  - See examples/ for more advanced examples")
    print("  - See notebooks/ for interactive tutorials")
    print("  - Run 'reactor-twin-dashboard' for visualization")
    print("=" * 60)


if __name__ == "__main__":
    main()
