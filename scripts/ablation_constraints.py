"""Ablation study: No constraints vs Soft constraints vs Hard constraints.

Experiment design:
  - Reactor: Exothermic CSTR (A -> B), 3 state dimensions (C_A, C_B, T)
  - Model: NeuralODE, identical architecture (3-32-32-3, tanh, ~2K params)
  - Seeds: 3 random seeds for variance estimates
  - Epochs: 300 per run

Metrics:
  - Prediction MSE on 20% hold-out trajectories
  - Mass balance violation: |sum(dC/dt) - 0| (species must be conserved)
  - Positivity violation: max(0, -min(C)) (concentrations must be >= 0)
  - Convergence epoch: first epoch where val loss < threshold

Output: prints a LaTeX-ready table + CSV to results/ablation_constraints.csv

Run: python3 scripts/ablation_constraints.py
"""

from __future__ import annotations

import csv
import logging
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.integrate import solve_ivp

from reactor_twin import (
    ArrheniusKinetics,
    CSTRReactor,
    MassBalanceConstraint,
    NeuralODE,
    PositivityConstraint,
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

SEEDS = [42, 43, 44]
N_EPOCHS = 300
LR = 1e-3
HIDDEN_DIM = 32
NUM_LAYERS = 2
T_END = 10.0
N_TRAIN_TRAJ = 8
N_VAL_TRAJ = 4
N_TIMES = 50
CONVERGENCE_THRESHOLD = 0.05
N_SPECIES = 2  # C_A, C_B — temperature (index 2) excluded from constraints

RESULTS_DIR = Path(__file__).parent.parent / "results"


def _make_reactor() -> CSTRReactor:
    kinetics = ArrheniusKinetics(
        name="ab",
        num_reactions=1,
        params={
            "k0": np.array([1.0]),
            "Ea": np.array([5000.0]),
            "stoich": np.array([[-1, 1]]),
            "orders": np.array([[1, 0]]),
            "delta_H": np.array([-50000.0]),
        },
    )
    return CSTRReactor(
        name="exothermic",
        num_species=2,
        params={
            "V": 1.0,
            "F": 0.5,
            "T_feed": 350.0,
            "C_feed": [2.0, 0.0],
            "T_initial": 350.0,
            "C_initial": [1.0, 0.0],
            "rho": 1000.0,
            "Cp": 4.18,
            "UA": 500.0,
            "T_coolant": 300.0,
        },
        kinetics=kinetics,
        isothermal=False,
    )


def _generate_dataset(
    n_traj: int,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    reactor = _make_reactor()
    t_eval = np.linspace(0, T_END, N_TIMES)

    z0_list: list[np.ndarray] = []
    traj_list: list[np.ndarray] = []

    for _ in range(n_traj):
        y0 = reactor.get_initial_state().copy()
        # Perturb initial conditions
        y0[:2] *= (0.8 + 0.4 * rng.random(2))
        y0[2] += rng.uniform(-20, 20)

        sol = solve_ivp(reactor.ode_rhs, [0, T_END], y0, t_eval=t_eval, method="RK45")
        if not sol.success:
            continue
        z0_list.append(y0)
        traj_list.append(sol.y.T)

    z0_t = torch.tensor(np.array(z0_list), dtype=torch.float32)
    traj_t = torch.tensor(np.array(traj_list), dtype=torch.float32)
    t_span = torch.tensor(t_eval, dtype=torch.float32)
    return z0_t, t_span, traj_t


def _mass_balance_violation(preds: torch.Tensor) -> float:
    """Mean absolute change in total species concentration across time."""
    # C_A + C_B should be constant (A -> B reaction, total species conserved)
    total = preds[..., :2].sum(dim=-1)  # (batch, time)
    drift = (total - total[:, 0:1]).abs().mean().item()
    return drift


def _positivity_violation(preds: torch.Tensor) -> float:
    """Max negative concentration (0 if always non-negative)."""
    concs = preds[..., :2]
    return float(max(0.0, -concs.min().item()))


def _train(
    z0: torch.Tensor,
    t_span: torch.Tensor,
    targets: torch.Tensor,
    z0_val: torch.Tensor,
    targets_val: torch.Tensor,
    constraint_mode: str,
    seed: int,
) -> dict[str, Any]:
    """Train NeuralODE under specified constraint mode.

    Args:
        z0: Training initial conditions, shape (batch, state_dim).
        t_span: Time points, shape (T,).
        targets: Training targets, shape (batch, T, state_dim).
        z0_val: Validation initial conditions.
        targets_val: Validation targets.
        constraint_mode: 'none', 'soft', or 'hard'.
        seed: Random seed.

    Returns:
        Dict with metrics: mse_val, mass_violation, pos_violation, conv_epoch, train_time_s.
    """
    torch.manual_seed(seed)

    state_dim = z0.shape[1]
    model = NeuralODE(
        state_dim=state_dim,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        solver="rk4",
        adjoint=False,
    )

    mass_constraint: MassBalanceConstraint | None = None
    pos_constraint: PositivityConstraint | None = None

    if constraint_mode in ("soft", "hard"):
        mass_constraint = MassBalanceConstraint(mode=constraint_mode, weight=1.0)
        pos_constraint = PositivityConstraint(mode=constraint_mode, weight=1.0)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    conv_epoch = N_EPOCHS
    t0 = time.perf_counter()

    model.train()
    for epoch in range(N_EPOCHS):
        optimizer.zero_grad()

        # Reset cached initial_mass each epoch (batch identity changes between train/val)
        if mass_constraint is not None:
            mass_constraint.reset()

        preds = model(z0, t_span)

        # Data loss
        loss = torch.mean((preds - targets) ** 2)

        # Constraint losses — operate on concentration dims only (not temperature)
        if constraint_mode == "soft" and mass_constraint and pos_constraint:
            _, mass_viol = mass_constraint(preds[..., :N_SPECIES])
            _, pos_viol = pos_constraint(preds[..., :N_SPECIES])
            loss = loss + mass_viol + pos_viol
        elif constraint_mode == "hard" and mass_constraint and pos_constraint:
            # Mass balance first, positivity (ReLU) last. Positivity is the
            # final projection so it is exactly guaranteed on every forward pass.
            # The subsequent ReLU introduces a small residual mass imbalance,
            # which is consistent (low variance) but non-zero.
            concs, mass_viol = mass_constraint(preds[..., :N_SPECIES])
            concs, pos_viol = pos_constraint(concs)
            preds = torch.cat([concs, preds[..., N_SPECIES:]], dim=-1)
            loss = torch.mean((preds - targets) ** 2)

        loss.backward()  # type: ignore[no-untyped-call]
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Track convergence (no constraint applied to val for fairness)
        if epoch < conv_epoch:
            with torch.no_grad():
                val_preds = model(z0_val, t_span)
                val_loss = torch.mean((val_preds - targets_val) ** 2).item()
                if val_loss < CONVERGENCE_THRESHOLD:
                    conv_epoch = epoch + 1

    train_time = time.perf_counter() - t0

    # Evaluate on val set
    model.eval()
    with torch.no_grad():
        if mass_constraint is not None:
            mass_constraint.reset()
        val_preds = model(z0_val, t_span)

        if constraint_mode == "hard" and mass_constraint and pos_constraint:
            concs, _ = mass_constraint(val_preds[..., :N_SPECIES])
            concs, _ = pos_constraint(concs)
            val_preds = torch.cat([concs, val_preds[..., N_SPECIES:]], dim=-1)

    mse_val = torch.mean((val_preds - targets_val) ** 2).item()
    mass_viol = _mass_balance_violation(val_preds)
    pos_viol = _positivity_violation(val_preds)

    return {
        "mse_val": mse_val,
        "mass_violation": mass_viol,
        "pos_violation": pos_viol,
        "conv_epoch": conv_epoch,
        "train_time_s": train_time,
    }


def _mean_std(values: list[float]) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    print("Generating dataset...")
    z0_train, t_span, traj_train = _generate_dataset(N_TRAIN_TRAJ, rng)
    z0_val, _, traj_val = _generate_dataset(N_VAL_TRAJ, rng)

    modes = ["none", "soft", "hard"]
    all_results: dict[str, list[dict[str, Any]]] = {m: [] for m in modes}

    for mode in modes:
        print(f"\nRunning mode={mode!r} ...")
        for seed in SEEDS:
            r = _train(z0_train, t_span, traj_train, z0_val, traj_val, mode, seed)
            all_results[mode].append(r)
            print(
                f"  seed={seed}  MSE={r['mse_val']:.4f}  "
                f"MassViol={r['mass_violation']:.4f}  "
                f"PosViol={r['pos_violation']:.4f}  "
                f"ConvEpoch={r['conv_epoch']}"
            )

    # Build summary table
    print("\n\n" + "=" * 80)
    print("ABLATION RESULTS — Hard vs Soft vs No Constraints")
    print("=" * 80)

    header = f"{'Mode':<10} {'MSE':>12} {'Mass Viol':>14} {'Pos Viol':>12} {'Conv Epoch':>12}"
    print(header)
    print("-" * 62)

    csv_rows = []
    for mode in modes:
        mses = [r["mse_val"] for r in all_results[mode]]
        mvs = [r["mass_violation"] for r in all_results[mode]]
        pvs = [r["pos_violation"] for r in all_results[mode]]
        ces = [float(r["conv_epoch"]) for r in all_results[mode]]

        mse_m, mse_s = _mean_std(mses)
        mv_m, mv_s = _mean_std(mvs)
        pv_m, pv_s = _mean_std(pvs)
        ce_m, ce_s = _mean_std(ces)

        print(
            f"{mode:<10} "
            f"{mse_m:.4f}±{mse_s:.4f}  "
            f"{mv_m:.4f}±{mv_s:.4f}  "
            f"{pv_m:.4f}±{pv_s:.4f}  "
            f"{ce_m:.0f}±{ce_s:.0f}"
        )
        csv_rows.append({
            "mode": mode,
            "mse_mean": mse_m, "mse_std": mse_s,
            "mass_violation_mean": mv_m, "mass_violation_std": mv_s,
            "pos_violation_mean": pv_m, "pos_violation_std": pv_s,
            "conv_epoch_mean": ce_m, "conv_epoch_std": ce_s,
        })

    print("\nLaTeX table row (for paper):")
    for row in csv_rows:
        mode_label = {"none": "No constraints", "soft": "Soft (penalty)", "hard": "Hard (projection)"}[row["mode"]]
        print(
            f"  {mode_label} & "
            f"${row['mse_mean']:.4f} \\pm {row['mse_std']:.4f}$ & "
            f"${row['mass_violation_mean']:.4f} \\pm {row['mass_violation_std']:.4f}$ & "
            f"${row['pos_violation_mean']:.4f} \\pm {row['pos_violation_std']:.4f}$ & "
            f"${row['conv_epoch_mean']:.0f} \\pm {row['conv_epoch_std']:.0f}$ \\\\"
        )

    # Write CSV
    csv_path = RESULTS_DIR / "ablation_constraints.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nCSV saved to {csv_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
