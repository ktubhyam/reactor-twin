"""Comparing hard vs soft positivity constraints on a NeuralODE.

Demonstrates:
1. Training the same model architecture with hard vs soft positivity constraints
2. Hard constraints enforce non-negative concentrations architecturally
3. Soft constraints add a penalty but allow (small) violations

Run: python examples/07_hard_vs_soft_constraints.py
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.integrate import solve_ivp

from reactor_twin import (
    ArrheniusKinetics,
    BatchReactor,
    NeuralODE,
    PositivityConstraint,
)

np.random.seed(42)
torch.manual_seed(42)


def generate_data() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate batch reactor data for A -> B."""
    kinetics = ArrheniusKinetics(
        name="simple",
        num_reactions=1,
        params={
            "k0": np.array([0.5]),
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
    t_eval = np.linspace(0, 8, 40)

    sol = solve_ivp(reactor.ode_rhs, [0, 8], y0, t_eval=t_eval, method="LSODA")

    z0 = torch.tensor(y0, dtype=torch.float32).unsqueeze(0)
    t_span = torch.tensor(t_eval, dtype=torch.float32)
    targets = torch.tensor(sol.y.T, dtype=torch.float32).unsqueeze(0)

    return z0, t_span, targets


def train_model(
    z0: torch.Tensor,
    t_span: torch.Tensor,
    targets: torch.Tensor,
    constraint_mode: str,
    num_epochs: int = 150,
) -> tuple[NeuralODE, PositivityConstraint, list[float]]:
    """Train a NeuralODE with specified constraint mode."""
    torch.manual_seed(42)

    model = NeuralODE(
        state_dim=2,
        hidden_dim=32,
        num_layers=2,
        solver="rk4",
        adjoint=False,
    )

    constraint = PositivityConstraint(
        mode=constraint_mode,
        method="softplus" if constraint_mode == "hard" else "relu",
        weight=10.0,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []

    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        preds = model(z0, t_span)

        # Apply constraint
        if constraint_mode == "hard":
            preds_constrained, violation = constraint(preds)
            loss_dict = model.compute_loss(preds_constrained, targets)
            loss = loss_dict["total"]
        else:
            preds_constrained, violation = constraint(preds)
            loss_dict = model.compute_loss(preds, targets)
            loss = loss_dict["total"] + violation

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(loss.item())

    return model, constraint, losses


def main() -> None:
    """Run hard vs soft constraints comparison."""
    print("=" * 60)
    print("Example 07: Hard vs Soft Positivity Constraints")
    print("=" * 60)

    # 1. Generate data
    print("\n1. Generating batch reactor data (A -> B)...")
    z0, t_span, targets = generate_data()
    print(f"   Data shape: z0={z0.shape}, targets={targets.shape}")

    # 2. Train with hard constraints
    print("\n2. Training with HARD constraints (softplus projection)...")
    model_hard, constraint_hard, losses_hard = train_model(z0, t_span, targets, "hard")
    print(f"   Final loss: {losses_hard[-1]:.6f}")

    # 3. Train with soft constraints
    print("\n3. Training with SOFT constraints (penalty-based)...")
    model_soft, constraint_soft, losses_soft = train_model(z0, t_span, targets, "soft")
    print(f"   Final loss: {losses_soft[-1]:.6f}")

    # 4. Compare predictions
    print("\n4. Comparing predictions...")

    model_hard.eval()
    model_soft.eval()

    with torch.no_grad():
        preds_hard = model_hard(z0, t_span)
        preds_hard_constrained, _ = constraint_hard(preds_hard)

        preds_soft = model_soft(z0, t_span)

    hard_np = preds_hard_constrained[0].numpy()
    soft_np = preds_soft[0].numpy()
    true_np = targets[0].numpy()

    # Check for negative values
    min_hard = hard_np.min()
    min_soft = soft_np.min()
    min_hard_raw = preds_hard[0].numpy().min()

    print(f"   Hard constraint (raw)   min value: {min_hard_raw:.6f}")
    print(f"   Hard constraint (proj)  min value: {min_hard:.6f}")
    print(f"   Soft constraint         min value: {min_soft:.6f}")

    num_neg_hard = (hard_np < 0).sum()
    num_neg_soft = (soft_np < 0).sum()

    print(f"\n   Negative values (hard projected): {num_neg_hard}")
    print(f"   Negative values (soft):           {num_neg_soft}")

    # 5. Compare accuracy at key times
    print("\n5. Accuracy comparison at key time points:")
    print(
        f"   {'Time':>6} | {'True C_A':>9} | {'Hard C_A':>9} | {'Soft C_A':>9} | {'True C_B':>9} | {'Hard C_B':>9} | {'Soft C_B':>9}"
    )
    print("   " + "-" * 68)

    for idx in [0, 10, 20, 30, 39]:
        t = t_span[idx].item()
        print(
            f"   {t:>6.1f} | {true_np[idx, 0]:>9.4f} | {hard_np[idx, 0]:>9.4f} | "
            f"{soft_np[idx, 0]:>9.4f} | {true_np[idx, 1]:>9.4f} | {hard_np[idx, 1]:>9.4f} | "
            f"{soft_np[idx, 1]:>9.4f}"
        )

    # 6. Summary
    mse_hard = np.mean((hard_np - true_np) ** 2)
    mse_soft = np.mean((soft_np - true_np) ** 2)

    print("\n6. Summary:")
    print(f"   MSE (hard constraint): {mse_hard:.6f}")
    print(f"   MSE (soft constraint): {mse_soft:.6f}")
    print(f"   Hard guarantees non-negative: {min_hard >= 0}")
    print(f"   Soft guarantees non-negative: {min_soft >= 0}")

    print("\n" + "=" * 60)
    print("Example 07 complete!")
    print("Key insight: Hard constraints guarantee physical feasibility")
    print("(non-negative concentrations) by architecture, while soft")
    print("constraints only penalize violations and may still produce")
    print("small negative values.")
    print("=" * 60)


if __name__ == "__main__":
    main()
