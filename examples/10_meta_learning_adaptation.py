"""Meta-learning for cross-reactor transfer with Reptile.

Demonstrates:
1. Using ReptileMetaLearner for cross-reactor transfer learning
2. Creating tasks from different reactor configurations
3. Meta-training, then fine-tuning on a new reactor
4. Showing few-shot adaptation performance

Run: python examples/10_meta_learning_adaptation.py
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.integrate import solve_ivp

from reactor_twin import (
    ArrheniusKinetics,
    CSTRReactor,
    NeuralODE,
    ReactorDataGenerator,
    ReptileMetaLearner,
)

np.random.seed(42)
torch.manual_seed(42)


def create_reactor_with_rate(k: float) -> CSTRReactor:
    """Create a simple isothermal CSTR with a given rate constant."""
    kinetics = ArrheniusKinetics(
        name="A_to_B",
        num_reactions=1,
        params={
            "k0": np.array([k]),
            "Ea": np.array([0.0]),  # Constant rate
            "stoich": np.array([[-1, 1]]),
            "orders": np.array([[1, 0]]),
        },
    )
    return CSTRReactor(
        name=f"cstr_k{k}",
        num_species=2,
        params={
            "V": 10.0,
            "F": 1.0,
            "C_feed": [1.0, 0.0],
            "T_feed": 350.0,
            "C_initial": [0.5, 0.0],
            "T_initial": 350.0,
        },
        kinetics=kinetics,
        isothermal=True,
    )


def evaluate_model(
    model: NeuralODE,
    reactor: CSTRReactor,
    t_eval: np.ndarray,
) -> float:
    """Evaluate model on a reactor, return MSE."""
    y0 = reactor.get_initial_state()
    sol = solve_ivp(
        reactor.ode_rhs, [t_eval[0], t_eval[-1]], y0,
        t_eval=t_eval, method="LSODA",
    )
    true_traj = sol.y.T  # (T, 2)

    z0 = torch.tensor(y0, dtype=torch.float32).unsqueeze(0)
    t_span = torch.tensor(t_eval, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        preds = model(z0, t_span)
    pred_np = preds[0].numpy()

    return float(np.mean((pred_np - true_traj) ** 2))


def main() -> None:
    """Run meta-learning adaptation example."""
    print("=" * 60)
    print("Example 10: Meta-Learning for Cross-Reactor Transfer")
    print("=" * 60)

    # 1. Create reactor tasks with varying rate constants
    print("\n1. Creating reactor tasks for meta-learning...")
    train_rates = [0.1, 0.3, 0.5, 0.8, 1.2]
    test_rate = 0.6  # Novel reactor for few-shot adaptation

    train_reactors = [create_reactor_with_rate(k) for k in train_rates]
    test_reactor = create_reactor_with_rate(test_rate)

    print(f"   Training tasks: k = {train_rates}")
    print(f"   Test task:      k = {test_rate}")

    # 2. Create data generators
    print("\n2. Setting up data generators...")
    train_generators = [ReactorDataGenerator(r) for r in train_reactors]
    test_generator = ReactorDataGenerator(test_reactor)

    t_eval = np.linspace(0, 5, 30)

    # 3. Create model and meta-learner
    print("\n3. Creating NeuralODE and ReptileMetaLearner...")
    model = NeuralODE(
        state_dim=2,
        hidden_dim=32,
        num_layers=2,
        solver="rk4",
        adjoint=False,
    )
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {num_params:,}")

    meta_learner = ReptileMetaLearner(
        model=model,
        meta_lr=0.1,
        inner_lr=0.01,
        inner_steps=3,
    )

    # 4. Evaluate before meta-training
    print("\n4. Evaluating BEFORE meta-training...")
    mse_before = evaluate_model(model, test_reactor, t_eval)
    print(f"   MSE on test reactor (k={test_rate}): {mse_before:.6f}")

    # 5. Meta-train
    print("\n5. Meta-training with Reptile (10 steps)...")
    displacements = meta_learner.meta_train(
        task_generators=train_generators,
        num_steps=10,
        t_span=(0.0, 5.0),
        t_eval=t_eval,
        tasks_per_step=3,
        batch_size=4,
        log_interval=5,
    )
    print(f"   Final displacement: {displacements[-1]:.6f}")

    # 6. Evaluate after meta-training (before fine-tuning)
    print("\n6. Evaluating AFTER meta-training (before fine-tuning)...")
    mse_after_meta = evaluate_model(model, test_reactor, t_eval)
    print(f"   MSE on test reactor: {mse_after_meta:.6f}")

    # 7. Fine-tune on test reactor (few-shot)
    print("\n7. Fine-tuning on test reactor (10 steps, few-shot)...")
    fine_tune_losses = meta_learner.fine_tune(
        task_generator=test_generator,
        t_span=(0.0, 5.0),
        t_eval=t_eval,
        num_steps=10,
        batch_size=4,
    )
    print(f"   Fine-tune losses: {fine_tune_losses[0]:.6f} -> {fine_tune_losses[-1]:.6f}")

    # 8. Evaluate after fine-tuning
    print("\n8. Evaluating AFTER fine-tuning...")
    mse_after_ft = evaluate_model(model, test_reactor, t_eval)
    print(f"   MSE on test reactor: {mse_after_ft:.6f}")

    # 9. Compare against training from scratch
    print("\n9. Comparison: training from scratch (same budget)...")
    torch.manual_seed(123)
    model_scratch = NeuralODE(
        state_dim=2, hidden_dim=32, num_layers=2, solver="rk4", adjoint=False,
    )
    optimizer = torch.optim.Adam(model_scratch.parameters(), lr=0.01)

    model_scratch.train()
    for step in range(10):
        batch = test_generator.generate_batch(4, (0, 5), t_eval)
        model_scratch.train_step(batch, optimizer)

    mse_scratch = evaluate_model(model_scratch, test_reactor, t_eval)
    print(f"   MSE (from scratch, 10 steps): {mse_scratch:.6f}")

    # 10. Summary
    print("\n10. Summary:")
    print(f"   {'Method':<35} | {'MSE':>10}")
    print("   " + "-" * 50)
    print(f"   {'Random init (no training)':<35} | {mse_before:>10.6f}")
    print(f"   {'After meta-training (no fine-tune)':<35} | {mse_after_meta:>10.6f}")
    print(f"   {'Meta-trained + 10-step fine-tune':<35} | {mse_after_ft:>10.6f}")
    print(f"   {'From scratch (10 steps)':<35} | {mse_scratch:>10.6f}")

    print("\n" + "=" * 60)
    print("Example 10 complete!")
    print("Key insight: Meta-learning with Reptile provides a good initialization")
    print("that adapts quickly to new reactor configurations with only a few")
    print("gradient steps, outperforming training from scratch.")
    print("=" * 60)


if __name__ == "__main__":
    main()
