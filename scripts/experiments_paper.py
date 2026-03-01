"""Paper experiments: hard vs soft physics constraints for neural ODEs.

Three benchmark systems:
  1. Exothermic A->B CSTR  (state=[C_A, C_B, T],     dim=3, constraint=positivity)
  2. Van de Vusse CSTR      (state=[C_A,C_B,C_C,C_D], dim=4, constraint=positivity)
  3. A->B->C Batch reactor  (state=[C_A, C_B, C_C],   dim=3, constraint=mass_balance)

Three conditions (applied uniformly):
  - none:  No constraints (baseline)
  - soft:  Soft penalty in training loss (lambda=1.0)
  - hard:  Exact architectural projection at inference

Three random seeds (42, 43, 44) for variance estimates.

Metrics per system:
  - mse_short:     MSE over first 30% of rollout horizon
  - mse_long:      MSE over last 50% of rollout horizon (long-horizon stress test)
  - physics_viol:  Positivity violation rate (% negative concs) for CSTRs;
                   mass drift |ΔΣC| for batch reactor

Additional experiments:
  - Lambda sweep:    soft constraint weight in {0.01, 0.1, 1.0, 10.0, 100.0}
                     (exothermic CSTR, 3 seeds each)
  - Speedup benchmark: scipy first-principles vs NeuralODE inference
  - Per-timestep MSE: saved for Figure 1 (long-rollout curve)

Output:
  - results/paper_results.json  (full numerical results + per-timestep MSE)
  - LaTeX table rows printed to stdout

Run: python3 scripts/experiments_paper.py
"""

from __future__ import annotations

import csv
import json
import logging
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.integrate import solve_ivp

from reactor_twin import (
    BatchReactor,
    MassBalanceConstraint,
    NeuralODE,
    PositivityConstraint,
    PowerLawKinetics,
    create_exothermic_cstr,
    create_van_de_vusse_cstr,
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
SEEDS = [42, 43, 44]
N_EPOCHS = 100
LR = 1e-3
HIDDEN_DIM = 32
NUM_LAYERS = 2
N_TRAIN_TRAJ = 16
N_VAL_TRAJ = 8
N_TIMES = 50
LAMBDA_WEIGHT = 1.0
LAMBDA_SWEEP_VALUES = [0.01, 0.1, 1.0, 10.0, 100.0]
N_SPEEDUP_TRAJ = 100

SHORT_END = int(0.3 * N_TIMES)   # 15 — first 30 %
LONG_START = int(0.5 * N_TIMES)  # 25 — last 50 %

CONDITIONS = ["none", "soft", "hard"]
RESULTS_DIR = Path(__file__).parent.parent / "results"


# ---------------------------------------------------------------------------
# Reactor factories
# ---------------------------------------------------------------------------

def _make_batch_abc() -> BatchReactor:
    """A->B->C batch reactor. state=[C_A, C_B, C_C], all species, no temperature."""
    kinetics = PowerLawKinetics(
        name="abc_series",
        num_reactions=2,
        params={
            "k": np.array([1.0, 0.5]),       # min^{-1}
            "orders": np.array([[1, 0, 0], [0, 1, 0]]),
            "stoich": np.array([[-1, 1, 0], [0, -1, 1]]),
        },
    )
    return BatchReactor(
        name="batch_abc",
        num_species=3,
        params={
            "V": 1.0,
            "T": 350.0,
            "C_initial": [1.0, 0.0, 0.0],
        },
        kinetics=kinetics,
        isothermal=True,
        constant_volume=True,
    )


# Per-system metadata.
#   t_end:            integration end time (units match each reactor's kinetics)
#   n_species:        number of species dims (may be < state_dim for CSTR with T)
#   stoich:           stoichiometric matrix (reactions x n_species); used for hard MB
#   constraint_type:  "positivity" or "mass_balance"
#   pos_indices:      PositivityConstraint indices (None = all dims)

_VDV_STOICH = np.array([
    [-1.0,  1.0, 0.0, 0.0],   # A -> B
    [ 0.0, -1.0, 1.0, 0.0],   # B -> C
    [-2.0,  0.0, 0.0, 1.0],   # 2A -> D
])

_BATCH_STOICH = np.array([
    [-1.0, 1.0, 0.0],   # A -> B
    [ 0.0, -1.0, 1.0],  # B -> C
])

_EXO_STOICH = np.array([[-1.0, 1.0]])  # A -> B (2 species)

SYSTEMS: dict[str, dict[str, Any]] = {
    "exothermic_cstr": {
        "name": "Exothermic CSTR",
        "factory": create_exothermic_cstr,
        "t_end": 10.0,         # minutes (10τ, τ=1 min)
        "n_species": 2,        # C_A, C_B (T is index 2)
        "stoich": _EXO_STOICH,
        "constraint_type": "positivity",
        "pos_indices": [0, 1], # Constrain C_A, C_B only — not T
    },
    "vdv_cstr": {
        "name": "Van de Vusse CSTR",
        "factory": create_van_de_vusse_cstr,
        "t_end": 0.05,         # hours (≈9τ, τ=1/180 hr)
        "n_species": 4,        # all species
        "stoich": _VDV_STOICH,
        "constraint_type": "positivity",
        "pos_indices": None,   # all 4 concentration dims
    },
    "batch_abc": {
        "name": "Batch A→B→C",
        "factory": _make_batch_abc,
        "t_end": 8.0,          # minutes
        "n_species": 3,        # all species (no T)
        "stoich": _BATCH_STOICH,
        "constraint_type": "mass_balance",
        "pos_indices": None,
    },
}


# ---------------------------------------------------------------------------
# Initial-condition perturbation
# ---------------------------------------------------------------------------

def _perturb(
    y0: np.ndarray,
    sys_key: str,
    rng: np.random.Generator,
) -> np.ndarray:
    y = y0.copy()
    if sys_key == "exothermic_cstr":
        y[0] *= 0.8 + 0.4 * rng.random()   # C_A ∈ [0.8, 1.2] × default
        y[1] = max(y[1] * (0.8 + 0.4 * rng.random()), 0.0)
        y[2] += rng.uniform(-15.0, 15.0)    # T ± 15 K
    elif sys_key == "vdv_cstr":
        y[0] *= 0.5 + 1.0 * rng.random()   # C_A broad range
        y[1] *= 0.5 + 1.0 * rng.random()   # C_B; C_C=C_D=0
    elif sys_key == "batch_abc":
        y[0] *= 0.5 + 1.0 * rng.random()   # C_A ∈ [0.5, 1.5] mol/L
    return y


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def _generate_dataset(
    sys_key: str,
    n_traj: int,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate trajectories via scipy integration.

    Returns:
        z0:      (n_traj, state_dim) float32
        t_span:  (N_TIMES,) float32
        targets: (n_traj, N_TIMES, state_dim) float32
    """
    cfg = SYSTEMS[sys_key]
    reactor = cfg["factory"]()
    t_eval = np.linspace(0.0, cfg["t_end"], N_TIMES)
    y0_default = reactor.get_initial_state()

    z0_list: list[np.ndarray] = []
    traj_list: list[np.ndarray] = []

    for _ in range(n_traj):
        y0 = _perturb(y0_default, sys_key, rng)
        sol = solve_ivp(
            reactor.ode_rhs,
            [0.0, cfg["t_end"]],
            y0,
            t_eval=t_eval,
            method="LSODA",
            rtol=1e-7,
            atol=1e-9,
        )
        if not sol.success:
            # Retry with default IC
            sol = solve_ivp(
                reactor.ode_rhs,
                [0.0, cfg["t_end"]],
                y0_default.copy(),
                t_eval=t_eval,
                method="LSODA",
                rtol=1e-7,
                atol=1e-9,
            )
            if not sol.success:
                continue
        z0_list.append(sol.y[:, 0])   # initial state from solved trajectory
        traj_list.append(sol.y.T)     # (N_TIMES, state_dim)

    if not z0_list:
        raise RuntimeError(f"All trajectories failed for system {sys_key!r}")

    z0 = torch.tensor(np.array(z0_list), dtype=torch.float32)
    t_span = torch.tensor(t_eval, dtype=torch.float32)
    targets = torch.tensor(np.array(traj_list), dtype=torch.float32)
    return z0, t_span, targets


# ---------------------------------------------------------------------------
# Constraint builders
# ---------------------------------------------------------------------------

def _build_constraint(
    sys_key: str,
    condition: str,
) -> MassBalanceConstraint | PositivityConstraint | None:
    """Return the appropriate constraint for (system, condition), or None."""
    if condition == "none":
        return None
    cfg = SYSTEMS[sys_key]
    mode = "soft" if condition == "soft" else "hard"
    if cfg["constraint_type"] == "positivity":
        return PositivityConstraint(
            mode=mode,
            weight=LAMBDA_WEIGHT,
            indices=cfg["pos_indices"],
            method="softplus",
        )
    else:  # mass_balance
        stoich_tensor = (
            torch.tensor(cfg["stoich"], dtype=torch.float32)
            if mode == "hard" else None
        )
        return MassBalanceConstraint(
            mode=mode,
            weight=LAMBDA_WEIGHT,
            stoich_matrix=stoich_tensor,
            check_total_mass=True,
        )


# ---------------------------------------------------------------------------
# Physics-violation metrics
# ---------------------------------------------------------------------------

def _pos_viol_pct(preds: torch.Tensor, indices: list[int] | None, n_species: int) -> float:
    """Fraction of concentration entries that are strictly negative (%)."""
    if indices is not None:
        concs = preds[..., indices]
    else:
        concs = preds[..., :n_species]
    return 100.0 * (concs < 0.0).float().mean().item()


def _mass_drift(preds: torch.Tensor, n_species: int) -> float:
    """Mean absolute total-mole drift from t=0 (valid for closed systems)."""
    total = preds[..., :n_species].sum(dim=-1)   # (batch, time)
    ref = total[:, 0:1]                           # (batch, 1)
    return (total - ref).abs().mean().item()


def _compute_physics_viol(
    preds: torch.Tensor,
    sys_key: str,
) -> float:
    cfg = SYSTEMS[sys_key]
    if cfg["constraint_type"] == "positivity":
        return _pos_viol_pct(preds, cfg["pos_indices"], cfg["n_species"])
    else:
        return _mass_drift(preds, cfg["n_species"])


def _mse_slice(preds: torch.Tensor, targets: torch.Tensor, start: int, end: int) -> float:
    return torch.mean((preds[:, start:end, :] - targets[:, start:end, :]) ** 2).item()


def _per_timestep_mse(preds: torch.Tensor, targets: torch.Tensor) -> list[float]:
    """MSE at each time step, averaged over batch and state dims."""
    diff2 = (preds - targets) ** 2  # (batch, time, state_dim)
    return diff2.mean(dim=[0, 2]).tolist()


# ---------------------------------------------------------------------------
# Apply constraint projection to trajectory
# ---------------------------------------------------------------------------

def _apply_hard_constraint(
    preds: torch.Tensor,
    constraint: MassBalanceConstraint | PositivityConstraint | None,
    sys_key: str,
) -> torch.Tensor:
    """Apply hard projection to a (batch, time, state_dim) trajectory."""
    if constraint is None:
        return preds
    if isinstance(constraint, MassBalanceConstraint):
        constraint.reset()
    z_proj, _ = constraint(preds)
    return z_proj


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _train_one(
    sys_key: str,
    z0_train: torch.Tensor,
    t_span: torch.Tensor,
    targets_train: torch.Tensor,
    z0_val: torch.Tensor,
    targets_val: torch.Tensor,
    condition: str,
    seed: int,
    lambda_weight: float = LAMBDA_WEIGHT,
) -> dict[str, Any]:
    """Train a NeuralODE under one (system, condition, seed).

    Returns dict with mse_short, mse_long, physics_viol, train_time_s,
    and per_timestep_mse (list of N_TIMES floats).
    """
    torch.manual_seed(seed)
    state_dim = z0_train.shape[1]

    model = NeuralODE(
        state_dim=state_dim,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        solver="rk4",
        adjoint=False,
    )

    constraint = _build_constraint(sys_key, condition)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    cfg = SYSTEMS[sys_key]
    n_species = cfg["n_species"]
    pos_indices = cfg["pos_indices"]

    t0 = time.perf_counter()
    model.train()

    for _ in range(N_EPOCHS):
        optimizer.zero_grad()
        preds = model(z0_train, t_span)  # (batch, time, state_dim)

        if condition == "none" or constraint is None:
            loss = torch.mean((preds - targets_train) ** 2)

        elif condition == "soft":
            loss = torch.mean((preds - targets_train) ** 2)
            if cfg["constraint_type"] == "positivity":
                # Apply only to species dims
                if pos_indices is not None:
                    concs = preds[..., pos_indices]
                else:
                    concs = preds[..., :n_species]
                _, viol = constraint(concs)
            else:  # mass_balance — species only, no temperature
                constraint.reset()  # type: ignore[union-attr]
                _, viol = constraint(preds[..., :n_species])
            loss = loss + lambda_weight * viol

        else:  # hard
            preds_proj = _apply_hard_constraint(preds, constraint, sys_key)
            loss = torch.mean((preds_proj - targets_train) ** 2)

        loss.backward()  # type: ignore[no-untyped-call]
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    train_time = time.perf_counter() - t0

    # Evaluation
    model.eval()
    with torch.no_grad():
        val_preds = model(z0_val, t_span)
        if condition == "hard":
            val_preds = _apply_hard_constraint(val_preds, constraint, sys_key)

    mse_short = _mse_slice(val_preds, targets_val, 0, SHORT_END)
    mse_long = _mse_slice(val_preds, targets_val, LONG_START, N_TIMES)
    phys_viol = _compute_physics_viol(val_preds, sys_key)
    ts_mse = _per_timestep_mse(val_preds, targets_val)

    return {
        "mse_short": mse_short,
        "mse_long": mse_long,
        "physics_viol": phys_viol,
        "train_time_s": train_time,
        "per_timestep_mse": ts_mse,
    }


# ---------------------------------------------------------------------------
# Main ablation
# ---------------------------------------------------------------------------

def run_ablation(rng_seed: int = 0) -> dict[str, Any]:
    """Run 3 systems × 3 conditions × 3 seeds."""
    rng = np.random.default_rng(rng_seed)
    all_results: dict[str, dict[str, list[dict[str, Any]]]] = {}

    for sys_key, cfg in SYSTEMS.items():
        print(f"\n{'='*60}")
        print(f"System: {cfg['name']}")
        print(f"{'='*60}")

        # Generate data once per system
        z0_train, t_span, traj_train = _generate_dataset(sys_key, N_TRAIN_TRAJ, rng)
        z0_val, _, traj_val = _generate_dataset(sys_key, N_VAL_TRAJ, rng)
        print(f"  Generated {z0_train.shape[0]} train / {z0_val.shape[0]} val trajectories")

        sys_results: dict[str, list[dict[str, Any]]] = {c: [] for c in CONDITIONS}

        for cond in CONDITIONS:
            print(f"\n  Condition: {cond!r}")
            for seed in SEEDS:
                r = _train_one(
                    sys_key, z0_train, t_span, traj_train,
                    z0_val, traj_val, cond, seed,
                )
                sys_results[cond].append(r)
                viol_label = "mass_drift" if cfg["constraint_type"] == "mass_balance" else "pos_viol%"
                print(
                    f"    seed={seed}  mse_short={r['mse_short']:.5f}  "
                    f"mse_long={r['mse_long']:.5f}  "
                    f"{viol_label}={r['physics_viol']:.4f}  "
                    f"t={r['train_time_s']:.1f}s"
                )

        all_results[sys_key] = sys_results

    return all_results


# ---------------------------------------------------------------------------
# Lambda sweep
# ---------------------------------------------------------------------------

def run_lambda_sweep(rng_seed: int = 1) -> dict[str, Any]:
    """Sweep soft constraint weight for exothermic CSTR (3 seeds each)."""
    print(f"\n{'='*60}")
    print("Lambda sweep — Exothermic CSTR, soft positivity")
    print(f"{'='*60}")

    rng = np.random.default_rng(rng_seed)
    z0_train, t_span, traj_train = _generate_dataset("exothermic_cstr", N_TRAIN_TRAJ, rng)
    z0_val, _, traj_val = _generate_dataset("exothermic_cstr", N_VAL_TRAJ, rng)

    sweep_results: dict[float, list[dict[str, Any]]] = {}

    for lam in LAMBDA_SWEEP_VALUES:
        print(f"\n  lambda={lam}")
        runs: list[dict[str, Any]] = []
        for seed in SEEDS:
            r = _train_one(
                "exothermic_cstr",
                z0_train, t_span, traj_train,
                z0_val, traj_val,
                condition="soft",
                seed=seed,
                lambda_weight=lam,
            )
            runs.append(r)
            print(
                f"    seed={seed}  mse_long={r['mse_long']:.5f}  "
                f"pos_viol%={r['physics_viol']:.4f}"
            )
        sweep_results[lam] = runs

    return sweep_results


# ---------------------------------------------------------------------------
# Speedup benchmark
# ---------------------------------------------------------------------------

def run_speedup_benchmark(rng_seed: int = 2) -> dict[str, Any]:
    """Scipy solve_ivp vs NeuralODE inference timing per system."""
    print(f"\n{'='*60}")
    print(f"Speedup benchmark ({N_SPEEDUP_TRAJ} trajectories per system)")
    print(f"{'='*60}")

    rng = np.random.default_rng(rng_seed)
    speedup_results: dict[str, dict[str, float]] = {}

    for sys_key, cfg in SYSTEMS.items():
        print(f"\n  System: {cfg['name']}")
        reactor = cfg["factory"]()
        t_eval = np.linspace(0.0, cfg["t_end"], N_TIMES)
        y0_default = reactor.get_initial_state()
        state_dim = len(y0_default)

        # --- scipy timing ---
        ics = np.stack([
            _perturb(y0_default, sys_key, rng) for _ in range(N_SPEEDUP_TRAJ)
        ])
        t_scipy_start = time.perf_counter()
        for i in range(N_SPEEDUP_TRAJ):
            solve_ivp(
                reactor.ode_rhs,
                [0.0, cfg["t_end"]],
                ics[i],
                t_eval=t_eval,
                method="LSODA",
                rtol=1e-7,
                atol=1e-9,
            )
        scipy_time = time.perf_counter() - t_scipy_start

        # --- NeuralODE timing (untrained, architecture-level throughput) ---
        torch.manual_seed(42)
        model = NeuralODE(
            state_dim=state_dim,
            hidden_dim=HIDDEN_DIM,
            num_layers=NUM_LAYERS,
            solver="rk4",
            adjoint=False,
        )
        model.eval()
        z0_bench = torch.tensor(ics, dtype=torch.float32)
        t_span_bench = torch.tensor(t_eval, dtype=torch.float32)

        # Warm-up
        with torch.no_grad():
            _ = model(z0_bench[:4], t_span_bench)

        t_torch_start = time.perf_counter()
        with torch.no_grad():
            _ = model(z0_bench, t_span_bench)
        torch_time = time.perf_counter() - t_torch_start

        speedup = scipy_time / max(torch_time, 1e-9)
        print(f"    scipy={scipy_time:.2f}s  pytorch={torch_time:.4f}s  speedup={speedup:.0f}x")
        speedup_results[sys_key] = {
            "scipy_s": scipy_time,
            "pytorch_s": torch_time,
            "speedup": speedup,
            "n_traj": N_SPEEDUP_TRAJ,
        }

    return speedup_results


# ---------------------------------------------------------------------------
# Mean/std helper
# ---------------------------------------------------------------------------

def _ms(values: list[float]) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


# ---------------------------------------------------------------------------
# Print LaTeX tables
# ---------------------------------------------------------------------------

def _print_ablation_table(all_results: dict[str, Any]) -> None:
    print("\n\n" + "=" * 80)
    print("TABLE 1 — Main Ablation (hard vs soft vs no constraints)")
    print("System | Condition | MSE_short ± std | MSE_long ± std | Physics Viol ± std")
    print("=" * 80)

    for sys_key, sys_res in all_results.items():
        cfg = SYSTEMS[sys_key]
        viol_name = "Mass drift" if cfg["constraint_type"] == "mass_balance" else "Pos viol %"
        for cond in CONDITIONS:
            runs = sys_res[cond]
            ms_m, ms_s = _ms([r["mse_short"] for r in runs])
            ml_m, ml_s = _ms([r["mse_long"] for r in runs])
            pv_m, pv_s = _ms([r["physics_viol"] for r in runs])
            cond_label = {
                "none": "No constraint",
                "soft": "Soft (λ=1)",
                "hard": "Hard (projection)",
            }[cond]
            print(
                f"  {cfg['name']:<24} | {cond_label:<18} | "
                f"{ms_m:.5f}±{ms_s:.5f} | "
                f"{ml_m:.5f}±{ml_s:.5f} | "
                f"{pv_m:.4f}±{pv_s:.4f}"
            )
        print()

    print("\nLaTeX rows (paste into Table 1):")
    for sys_key, sys_res in all_results.items():
        cfg = SYSTEMS[sys_key]
        for cond in CONDITIONS:
            runs = sys_res[cond]
            ms_m, ms_s = _ms([r["mse_short"] for r in runs])
            ml_m, ml_s = _ms([r["mse_long"] for r in runs])
            pv_m, pv_s = _ms([r["physics_viol"] for r in runs])
            cond_label = {
                "none": "No constraint",
                "soft": r"Soft ($\lambda=1$)",
                "hard": r"\textbf{Hard (projection)}",
            }[cond]
            viol_str = (
                r"\textbf{0.00}" if (cond == "hard" and pv_m < 1e-4)
                else f"${pv_m:.4f} \\pm {pv_s:.4f}$"
            )
            print(
                f"  {cfg['name']} & {cond_label} & "
                f"${ms_m:.5f} \\pm {ms_s:.5f}$ & "
                f"${ml_m:.5f} \\pm {ml_s:.5f}$ & "
                f"{viol_str} \\\\"
            )
        print("  \\midrule")


def _print_lambda_table(sweep: dict[str, Any]) -> None:
    print("\n\n" + "=" * 60)
    print("LAMBDA SWEEP — Exothermic CSTR, soft positivity")
    print("lambda | MSE_long | Pos Viol %")
    print("-" * 60)
    for lam_str, runs in sweep.items():
        lam = float(lam_str)
        ml_m, ml_s = _ms([r["mse_long"] for r in runs])
        pv_m, pv_s = _ms([r["physics_viol"] for r in runs])
        print(f"  {lam:8.2f} | {ml_m:.5f}±{ml_s:.5f} | {pv_m:.4f}±{pv_s:.4f}")

    print("\nLaTeX rows:")
    for lam_str, runs in sweep.items():
        lam = float(lam_str)
        ml_m, ml_s = _ms([r["mse_long"] for r in runs])
        pv_m, pv_s = _ms([r["physics_viol"] for r in runs])
        print(
            f"  ${lam}$ & ${ml_m:.5f} \\pm {ml_s:.5f}$ & "
            f"${pv_m:.4f} \\pm {pv_s:.4f}$ \\\\"
        )


def _print_speedup_table(speedup: dict[str, Any]) -> None:
    print("\n\n" + "=" * 60)
    print("TABLE 2 — Speedup Benchmark")
    print("System | scipy (s) | PyTorch (s) | Speedup")
    print("-" * 60)
    for sys_key, r in speedup.items():
        cfg = SYSTEMS[sys_key]
        print(
            f"  {cfg['name']:<24} | "
            f"{r['scipy_s']:7.2f} | {r['pytorch_s']:9.4f} | {r['speedup']:.0f}x"
        )
    print("\nLaTeX rows:")
    for sys_key, r in speedup.items():
        cfg = SYSTEMS[sys_key]
        print(
            f"  {cfg['name']} & {r['scipy_s']:.2f} s & "
            f"{r['pytorch_s']*1000:.1f} ms & "
            f"{r['speedup']:.0f}$\\times$ \\\\"
        )


def _print_paper_summary(all_results: dict[str, Any]) -> None:
    """Print the specific numbers needed to fill [X] placeholders in paper."""
    print("\n\n" + "=" * 80)
    print("PAPER FILL-IN SUMMARY")
    print("=" * 80)

    # Hard constraint physics violations
    print("\n[Hard constraint exact violation = 0 verification]")
    for sys_key, sys_res in all_results.items():
        cfg = SYSTEMS[sys_key]
        runs = sys_res["hard"]
        pv_m, pv_s = _ms([r["physics_viol"] for r in runs])
        print(f"  {cfg['name']}: physics_viol(hard) = {pv_m:.2e} ± {pv_s:.2e}")

    # MSE improvement: hard vs none
    print("\n[MSE improvement: hard vs none (long horizon)]")
    for sys_key, sys_res in all_results.items():
        cfg = SYSTEMS[sys_key]
        ml_none = _ms([r["mse_long"] for r in sys_res["none"]])[0]
        ml_hard = _ms([r["mse_long"] for r in sys_res["hard"]])[0]
        improvement = 100.0 * (ml_none - ml_hard) / (ml_none + 1e-12)
        print(f"  {cfg['name']}: {improvement:.1f}% MSE reduction (none→hard)")

    # Soft vs hard physics violation comparison
    print("\n[Soft vs hard physics violation]")
    for sys_key, sys_res in all_results.items():
        cfg = SYSTEMS[sys_key]
        pv_soft = _ms([r["physics_viol"] for r in sys_res["soft"]])[0]
        pv_hard = _ms([r["physics_viol"] for r in sys_res["hard"]])[0]
        print(f"  {cfg['name']}: soft={pv_soft:.4f}  hard={pv_hard:.2e}")


# ---------------------------------------------------------------------------
# Save results to JSON
# ---------------------------------------------------------------------------

def _save_json(
    all_results: dict[str, Any],
    sweep: dict[str, Any],
    speedup: dict[str, Any],
) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Convert float keys for JSON serialization
    sweep_serializable = {str(k): v for k, v in sweep.items()}

    output = {
        "metadata": {
            "seeds": SEEDS,
            "n_epochs": N_EPOCHS,
            "hidden_dim": HIDDEN_DIM,
            "num_layers": NUM_LAYERS,
            "n_train_traj": N_TRAIN_TRAJ,
            "n_val_traj": N_VAL_TRAJ,
            "n_times": N_TIMES,
            "lambda_weight": LAMBDA_WEIGHT,
            "short_end_idx": SHORT_END,
            "long_start_idx": LONG_START,
        },
        "ablation": all_results,
        "lambda_sweep": sweep_serializable,
        "speedup": speedup,
    }

    out_path = RESULTS_DIR / "paper_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Also save a CSV summary
    csv_path = RESULTS_DIR / "paper_ablation_summary.csv"
    rows = []
    for sys_key, sys_res in all_results.items():
        cfg = SYSTEMS[sys_key]
        for cond in CONDITIONS:
            runs = sys_res[cond]
            ms_m, ms_s = _ms([r["mse_short"] for r in runs])
            ml_m, ml_s = _ms([r["mse_long"] for r in runs])
            pv_m, pv_s = _ms([r["physics_viol"] for r in runs])
            rows.append({
                "system": sys_key,
                "system_name": cfg["name"],
                "condition": cond,
                "constraint_type": cfg["constraint_type"],
                "mse_short_mean": ms_m, "mse_short_std": ms_s,
                "mse_long_mean": ml_m, "mse_long_std": ml_s,
                "physics_viol_mean": pv_m, "physics_viol_std": pv_s,
            })
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV summary saved to {csv_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("ReactorTwin Paper Experiments")
    print(f"Seeds: {SEEDS}  Epochs: {N_EPOCHS}  "
          f"Hidden: {HIDDEN_DIM}  Layers: {NUM_LAYERS}")
    print(f"Train traj: {N_TRAIN_TRAJ}  Val traj: {N_VAL_TRAJ}  "
          f"Time points: {N_TIMES}")

    t_total_start = time.perf_counter()

    # 1. Main ablation
    all_results = run_ablation(rng_seed=0)

    # 2. Lambda sweep
    sweep = run_lambda_sweep(rng_seed=1)

    # 3. Speedup benchmark
    speedup = run_speedup_benchmark(rng_seed=2)

    # Print tables
    _print_ablation_table(all_results)
    _print_lambda_table(sweep)
    _print_speedup_table(speedup)
    _print_paper_summary(all_results)

    # Save
    _save_json(all_results, sweep, speedup)

    total_time = time.perf_counter() - t_total_start
    print(f"\nTotal wall time: {total_time / 60:.1f} min")


if __name__ == "__main__":
    main()
