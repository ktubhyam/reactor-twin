"""Multi-objective optimization with physics constraints.

Demonstrates:
1. Using MultiObjectiveLoss with data and physics losses
2. Training a NeuralODE with a ConstraintPipeline
3. Showing how different loss weights affect the Pareto tradeoff
4. Comparing constrained vs unconstrained training

Run: python examples/14_multi_objective_optimization.py
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.integrate import solve_ivp

from reactor_twin import (
    ArrheniusKinetics,
    BatchReactor,
    ConstraintPipeline,
    MultiObjectiveLoss,
    NeuralODE,
    PositivityConstraint,
)

np.random.seed(42)
torch.manual_seed(42)


def generate_data() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate batch reactor data."""
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
    t_eval = np.linspace(0, 8, 40)

    sol = solve_ivp(reactor.ode_rhs, [0, 8], y0, t_eval=t_eval, method="LSODA")

    z0 = torch.tensor(y0, dtype=torch.float32).unsqueeze(0)
    t_span = torch.tensor(t_eval, dtype=torch.float32)
    targets = torch.tensor(sol.y.T, dtype=torch.float32).unsqueeze(0)

    return z0, t_span, targets


def train_with_weights(
    z0: torch.Tensor,
    t_span: torch.Tensor,
    targets: torch.Tensor,
    data_weight: float,
    constraint_weight: float,
    reg_weight: float,
    num_epochs: int = 150,
) -> dict:
    """Train a model with specific loss weights."""
    torch.manual_seed(42)

    model = NeuralODE(
        state_dim=2,
        hidden_dim=32,
        num_layers=2,
        solver="rk4",
        adjoint=False,
    )

    # Set up constraints
    positivity = PositivityConstraint(
        mode="soft",
        weight=constraint_weight,
    )

    loss_fn = MultiObjectiveLoss(
        weights={
            "data": data_weight,
            "positivity": constraint_weight,
            "regularization": reg_weight,
        },
        constraints=[positivity],
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    loss_history = {"total": [], "data": [], "constraint": [], "regularization": []}

    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        preds = model(z0, t_span)

        losses = loss_fn(preds, targets, model=model)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        loss_history["total"].append(losses["total"].item())
        loss_history["data"].append(losses["data"].item())
        constraint_val = sum(
            v.item() for k, v in losses.items()
            if k not in ("total", "data", "physics", "regularization")
        )
        loss_history["constraint"].append(constraint_val)
        reg_val = losses.get("regularization", torch.tensor(0.0)).item()
        loss_history["regularization"].append(reg_val)

    # Evaluate
    model.eval()
    with torch.no_grad():
        preds = model(z0, t_span)

    pred_np = preds[0].numpy()
    true_np = targets[0].numpy()
    mse = np.mean((pred_np - true_np) ** 2)
    min_val = pred_np.min()
    num_negative = (pred_np < 0).sum()

    return {
        "model": model,
        "loss_history": loss_history,
        "mse": mse,
        "min_val": min_val,
        "num_negative": num_negative,
        "pred_np": pred_np,
    }


def main() -> None:
    """Run multi-objective optimization example."""
    print("=" * 60)
    print("Example 14: Multi-Objective Optimization")
    print("=" * 60)

    # 1. Generate data
    print("\n1. Generating batch reactor data (A -> B)...")
    z0, t_span, targets = generate_data()
    print(f"   Data shape: {targets.shape}")

    # 2. Train with different weight configurations
    print("\n2. Training with different loss weight configurations...")

    configs = [
        {"name": "Data only",           "data": 1.0, "constraint": 0.0,  "reg": 0.0},
        {"name": "Data + Constraint",   "data": 1.0, "constraint": 10.0, "reg": 0.0},
        {"name": "Data + Reg",          "data": 1.0, "constraint": 0.0,  "reg": 0.001},
        {"name": "Balanced",            "data": 1.0, "constraint": 5.0,  "reg": 0.001},
        {"name": "Strong constraint",   "data": 1.0, "constraint": 50.0, "reg": 0.001},
    ]

    results = []
    for cfg in configs:
        print(f"\n   Training: {cfg['name']}...")
        print(f"     Weights: data={cfg['data']}, constraint={cfg['constraint']}, reg={cfg['reg']}")

        result = train_with_weights(
            z0, t_span, targets,
            data_weight=cfg["data"],
            constraint_weight=cfg["constraint"],
            reg_weight=cfg["reg"],
        )
        result["name"] = cfg["name"]
        results.append(result)

        print(f"     Final loss: {result['loss_history']['total'][-1]:.6f}")
        print(f"     Data MSE: {result['mse']:.6f}")
        print(f"     Min value: {result['min_val']:.6f}")
        print(f"     Negative values: {result['num_negative']}")

    # 3. Pareto comparison
    print("\n3. Pareto tradeoff comparison:")
    print(f"   {'Config':<20} | {'Data MSE':>10} | {'Min Value':>10} | {'Negatives':>10}")
    print("   " + "-" * 56)

    for r in results:
        print(
            f"   {r['name']:<20} | {r['mse']:>10.6f} | {r['min_val']:>10.6f} | "
            f"{r['num_negative']:>10d}"
        )

    # 4. ConstraintPipeline demonstration
    print("\n4. Using ConstraintPipeline for post-hoc constraint enforcement...")
    pipeline = ConstraintPipeline([
        PositivityConstraint(mode="hard", method="softplus"),
    ])

    # Apply to unconstrained model
    unconstrained_result = results[0]  # "Data only"
    with torch.no_grad():
        preds_unconstrained = unconstrained_result["model"](z0, t_span)
        preds_constrained, violations = pipeline(preds_unconstrained)

    min_before = preds_unconstrained[0].numpy().min()
    min_after = preds_constrained[0].numpy().min()
    print(f"   Before pipeline: min value = {min_before:.6f}")
    print(f"   After pipeline:  min value = {min_after:.6f}")
    print(f"   Violations: {violations}")

    # 5. Detailed comparison at key times
    print("\n5. Predictions comparison (Balanced vs Data-only):")
    true_np = targets[0].numpy()
    balanced = results[3]["pred_np"]
    data_only = results[0]["pred_np"]

    print(f"   {'Time':>6} | {'True C_A':>9} | {'Bal C_A':>9} | {'DO C_A':>9} | "
          f"{'True C_B':>9} | {'Bal C_B':>9} | {'DO C_B':>9}")
    print("   " + "-" * 65)

    for idx in [0, 10, 20, 30, 39]:
        t = t_span[idx].item()
        print(
            f"   {t:>6.1f} | {true_np[idx, 0]:>9.4f} | {balanced[idx, 0]:>9.4f} | "
            f"{data_only[idx, 0]:>9.4f} | {true_np[idx, 1]:>9.4f} | {balanced[idx, 1]:>9.4f} | "
            f"{data_only[idx, 1]:>9.4f}"
        )

    # 6. Summary
    print("\n6. Summary of multi-objective tradeoffs:")
    print("   - Pure data fitting gives best MSE but may violate constraints")
    print("   - Adding positivity constraint prevents negative concentrations")
    print("   - Regularization improves generalization at small MSE cost")
    print("   - ConstraintPipeline can enforce constraints post-hoc")

    print("\n" + "=" * 60)
    print("Example 14 complete!")
    print("Key insight: Multi-objective loss balances data fidelity with")
    print("physics constraints. The Pareto frontier shows that stronger")
    print("constraints increase physical plausibility at modest accuracy cost.")
    print("=" * 60)


if __name__ == "__main__":
    main()
