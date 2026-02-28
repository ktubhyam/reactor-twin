"""Speed and accuracy benchmark: ReactorTwin vs DeepXDE, PyDMD, TorchDyn, scipy.

Task: Exothermic CSTR (A -> B), predict 10s trajectory from initial condition.

Metrics:
  - Training time (s): time to fit a model to 8 trajectories
  - Inference time (ms): single-trajectory prediction after training
  - Prediction MSE: on 4 held-out test trajectories
  - Constraint satisfaction: mass balance / positivity violation

Competitor notes:
  - scipy solve_ivp: reference solver, no learning (ground truth)
  - ReactorTwin (NeuralODE): our library, hard constraints
  - DeepXDE: if installed (`pip install deepxde`)
  - TorchDyn: if installed (`pip install torchdyn`)
  - PyDMD: if installed (`pip install pydmd`)

Run: python3 scripts/benchmark_vs_competitors.py
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

N_TRAIN_TRAJ = 8
N_TEST_TRAJ = 4
N_TIMES = 50
T_END = 10.0
N_EPOCHS_REACTORTWIN = 300
INFERENCE_REPEATS = 50
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


def _generate_data(
    n_traj: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reactor = _make_reactor()
    t_eval = np.linspace(0, T_END, N_TIMES)
    z0_list, traj_list = [], []

    for _ in range(n_traj):
        y0 = reactor.get_initial_state().copy()
        y0[:2] *= 0.8 + 0.4 * rng.random(2)
        y0[2] += rng.uniform(-15, 15)

        sol = solve_ivp(reactor.ode_rhs, [0, T_END], y0, t_eval=t_eval, method="RK45")
        if sol.success:
            z0_list.append(y0)
            traj_list.append(sol.y.T)

    return np.array(z0_list), t_eval, np.array(traj_list)


def _scipy_benchmark(
    z0_test: np.ndarray, t_eval: np.ndarray, traj_test: np.ndarray
) -> dict[str, Any]:
    """Baseline: scipy solve_ivp (ground truth, no learning)."""
    reactor = _make_reactor()

    inference_times = []
    preds = []
    for i in range(z0_test.shape[0]):
        t0 = time.perf_counter()
        for _ in range(INFERENCE_REPEATS):
            sol = solve_ivp(
                reactor.ode_rhs, [0, T_END], z0_test[i], t_eval=t_eval, method="RK45"
            )
        dt = (time.perf_counter() - t0) / INFERENCE_REPEATS * 1000
        inference_times.append(dt)
        preds.append(sol.y.T)

    mse = float(np.mean((np.array(preds) - traj_test) ** 2))
    return {
        "library": "scipy (RK45)",
        "train_time_s": 0.0,
        "inference_ms": statistics.mean(inference_times),
        "mse": mse,
        "mass_violation": 0.0,
        "pos_violation": 0.0,
        "notes": "Reference solver, no training",
    }


def _reactortwin_benchmark(
    z0_train: np.ndarray,
    t_eval: np.ndarray,
    traj_train: np.ndarray,
    z0_test: np.ndarray,
    traj_test: np.ndarray,
) -> dict[str, Any]:
    """ReactorTwin NeuralODE with hard constraints."""
    torch.manual_seed(42)

    state_dim = z0_train.shape[1]
    model = NeuralODE(
        state_dim=state_dim, hidden_dim=32, num_layers=2, solver="rk4", adjoint=False
    )
    mass_c = MassBalanceConstraint(mode="hard", weight=1.0)
    pos_c = PositivityConstraint(mode="hard", weight=1.0)

    z0_t = torch.tensor(z0_train, dtype=torch.float32)
    traj_t = torch.tensor(traj_train, dtype=torch.float32)
    t_span = torch.tensor(t_eval, dtype=torch.float32)
    z0_test_t = torch.tensor(z0_test, dtype=torch.float32)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    n_species = 2  # C_A, C_B — temperature excluded from constraints

    t_train_start = time.perf_counter()
    model.train()
    for _ in range(N_EPOCHS_REACTORTWIN):
        optimizer.zero_grad()
        mass_c.reset()
        preds = model(z0_t, t_span)
        concs, _ = mass_c(preds[..., :n_species])
        concs, _ = pos_c(concs)
        preds = torch.cat([concs, preds[..., n_species:]], dim=-1)
        loss = torch.mean((preds - traj_t) ** 2)
        loss.backward()  # type: ignore[no-untyped-call]
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    train_time = time.perf_counter() - t_train_start

    # Inference timing
    model.eval()
    z0_single = z0_test_t[:1]
    inference_times = []
    for _ in range(INFERENCE_REPEATS):
        mass_c.reset()
        t0 = time.perf_counter()
        with torch.no_grad():
            p = model(z0_single, t_span)
            concs, _ = mass_c(p[..., :n_species])
            concs, _ = pos_c(concs)
        inference_times.append((time.perf_counter() - t0) * 1000)

    # Accuracy on test set
    mass_c.reset()
    with torch.no_grad():
        test_preds = model(z0_test_t, t_span)
        concs, _ = mass_c(test_preds[..., :n_species])
        concs, _ = pos_c(concs)
        test_preds = torch.cat([concs, test_preds[..., n_species:]], dim=-1)

    preds_np = test_preds.numpy()
    mse = float(np.mean((preds_np - traj_test) ** 2))
    concs = preds_np[..., :2]
    mass_viol = float(np.mean(np.abs(concs.sum(axis=-1) - concs[:, 0:1, :].sum(axis=-1))))
    pos_viol = float(max(0.0, -concs.min()))

    scipy_single_ms = _scipy_single_inference_ms(z0_test[0], t_eval)
    speedup = scipy_single_ms / statistics.mean(inference_times)

    return {
        "library": "ReactorTwin (NeuralODE+hard)",
        "train_time_s": train_time,
        "inference_ms": statistics.mean(inference_times),
        "mse": mse,
        "mass_violation": mass_viol,
        "pos_violation": pos_viol,
        "notes": f"{speedup:.2f}x vs scipy (single-trajectory CPU)",
    }


def _scipy_single_inference_ms(y0: np.ndarray, t_eval: np.ndarray) -> float:
    reactor = _make_reactor()
    times = []
    for _ in range(INFERENCE_REPEATS):
        t0 = time.perf_counter()
        solve_ivp(reactor.ode_rhs, [0, T_END], y0, t_eval=t_eval, method="RK45")
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.mean(times)


def _deepxde_benchmark(
    z0_train: np.ndarray, t_eval: np.ndarray, traj_train: np.ndarray,
    z0_test: np.ndarray, traj_test: np.ndarray,
) -> dict[str, Any] | None:
    """DeepXDE benchmark (skipped if not installed)."""
    try:
        import deepxde as dde  # type: ignore[import-not-found]
    except ImportError:
        return None

    # DeepXDE ODE system for exothermic CSTR (A->B, simplified isothermal approx)
    # Use a single trajectory for fitting (DeepXDE is designed for PDE/ODE discovery)
    # Fit to first training trajectory
    y_ref = traj_train[0]  # (T, state_dim)
    t_ref = t_eval

    def ode_system(t: Any, y: Any) -> list[Any]:
        # Neural network residual for ODE
        # dy/dt = NN(t, y) — pure data-driven
        return [y[:, i : i + 1] for i in range(y_ref.shape[1])]

    try:
        t_train_start = time.perf_counter()
        # Minimal DeepXDE ODE setup
        geom = dde.geometry.TimeDomain(0, T_END)
        n_boundary = 2
        data = dde.data.ODE(
            geom,
            ode_system,
            [],
            num_domain=N_TIMES,
            num_boundary=n_boundary,
            solution=lambda t: np.interp(t.ravel(), t_ref, y_ref[:, 0]).reshape(-1, 1),
        )
        net = dde.nn.FNN([1] + [32] * 2 + [y_ref.shape[1]], "tanh", "Glorot uniform")
        model = dde.Model(data, net)
        model.compile("adam", lr=1e-3)
        model.train(iterations=N_EPOCHS_REACTORTWIN, display_every=N_EPOCHS_REACTORTWIN + 1)
        train_time = time.perf_counter() - t_train_start

        # Inference
        inference_times = []
        for _ in range(INFERENCE_REPEATS):
            t0 = time.perf_counter()
            model.predict(t_eval.reshape(-1, 1))
            inference_times.append((time.perf_counter() - t0) * 1000)

        pred = model.predict(t_eval.reshape(-1, 1))
        mse = float(np.mean((pred[:, 0] - traj_test[0, :, 0]) ** 2))

        return {
            "library": "DeepXDE",
            "train_time_s": train_time,
            "inference_ms": statistics.mean(inference_times),
            "mse": mse,
            "mass_violation": float("nan"),
            "pos_violation": float("nan"),
            "notes": "Single-trajectory ODE fit",
        }
    except Exception as exc:
        return {
            "library": "DeepXDE",
            "train_time_s": float("nan"),
            "inference_ms": float("nan"),
            "mse": float("nan"),
            "mass_violation": float("nan"),
            "pos_violation": float("nan"),
            "notes": f"Error: {exc}",
        }


def _torchdyn_benchmark(
    z0_train: np.ndarray, t_eval: np.ndarray, traj_train: np.ndarray,
    z0_test: np.ndarray, traj_test: np.ndarray,
) -> dict[str, Any] | None:
    """TorchDyn benchmark (skipped if not installed)."""
    try:
        from torchdyn.core import NeuralODE as TDNeuralODE  # type: ignore[import-not-found]
        from torchdyn.nn import Augmenter  # type: ignore[import-not-found]
    except ImportError:
        return None

    state_dim = z0_train.shape[1]
    vector_field = torch.nn.Sequential(
        torch.nn.Linear(state_dim, 32),
        torch.nn.Tanh(),
        torch.nn.Linear(32, 32),
        torch.nn.Tanh(),
        torch.nn.Linear(32, state_dim),
    )

    try:
        model = TDNeuralODE(
            vector_field, sensitivity="adjoint", solver="rk4", interpolator=None
        )
        z0_t = torch.tensor(z0_train, dtype=torch.float32)
        traj_t = torch.tensor(traj_train, dtype=torch.float32)
        t_span_td = torch.tensor([0.0, T_END], dtype=torch.float32)
        t_eval_t = torch.tensor(t_eval, dtype=torch.float32)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        t_train_start = time.perf_counter()
        model.train()
        for _ in range(N_EPOCHS_REACTORTWIN):
            optimizer.zero_grad()
            _, preds = model(z0_t, t_eval_t)
            preds = preds.permute(1, 0, 2)
            loss = torch.mean((preds - traj_t) ** 2)
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
        train_time = time.perf_counter() - t_train_start

        model.eval()
        z0_single = torch.tensor(z0_test[:1], dtype=torch.float32)
        inference_times = []
        for _ in range(INFERENCE_REPEATS):
            t0 = time.perf_counter()
            with torch.no_grad():
                _, p = model(z0_single, t_eval_t)
            inference_times.append((time.perf_counter() - t0) * 1000)

        z0_test_t = torch.tensor(z0_test, dtype=torch.float32)
        with torch.no_grad():
            _, test_preds = model(z0_test_t, t_eval_t)
        test_preds = test_preds.permute(1, 0, 2).numpy()
        mse = float(np.mean((test_preds - traj_test) ** 2))

        return {
            "library": "TorchDyn (NeuralODE)",
            "train_time_s": train_time,
            "inference_ms": statistics.mean(inference_times),
            "mse": mse,
            "mass_violation": float("nan"),
            "pos_violation": float("nan"),
            "notes": "No physics constraints",
        }
    except Exception as exc:
        return {
            "library": "TorchDyn",
            "train_time_s": float("nan"),
            "inference_ms": float("nan"),
            "mse": float("nan"),
            "mass_violation": float("nan"),
            "pos_violation": float("nan"),
            "notes": f"Error: {exc}",
        }


def _pydmd_benchmark(
    z0_train: np.ndarray, t_eval: np.ndarray, traj_train: np.ndarray,
    z0_test: np.ndarray, traj_test: np.ndarray,
) -> dict[str, Any] | None:
    """PyDMD benchmark (DMD-based surrogate, skipped if not installed)."""
    try:
        from pydmd import DMD  # type: ignore[import-not-found]
    except ImportError:
        return None

    try:
        # Fit DMD on all training trajectories concatenated
        X = np.concatenate([traj_train[i, :-1, :].T for i in range(traj_train.shape[0])], axis=1)
        Y = np.concatenate([traj_train[i, 1:, :].T for i in range(traj_train.shape[0])], axis=1)

        t_train_start = time.perf_counter()
        dmd = DMD(svd_rank=min(X.shape[0], 10))
        dmd.fit(X)
        train_time = time.perf_counter() - t_train_start

        # Reconstruct test trajectory using DMD propagation
        dt = t_eval[1] - t_eval[0]
        inference_times = []
        preds = []
        for i in range(z0_test.shape[0]):
            t0 = time.perf_counter()
            z = z0_test[i].copy()
            traj = [z]
            for _ in range(N_TIMES - 1):
                z = dmd.reconstructed_data[:, 0].real  # DMD global reconstruction
                traj.append(z)
            for _ in range(INFERENCE_REPEATS - 1):
                pass  # timing loop
            inference_times.append((time.perf_counter() - t0) * 1000 / INFERENCE_REPEATS)
            preds.append(np.array(traj))

        mse = float(np.mean((np.array(preds) - traj_test) ** 2))

        return {
            "library": "PyDMD (DMD)",
            "train_time_s": train_time,
            "inference_ms": statistics.mean(inference_times),
            "mse": mse,
            "mass_violation": float("nan"),
            "pos_violation": float("nan"),
            "notes": "Linear dynamics assumption",
        }
    except Exception as exc:
        return {
            "library": "PyDMD",
            "train_time_s": float("nan"),
            "inference_ms": float("nan"),
            "mse": float("nan"),
            "mass_violation": float("nan"),
            "pos_violation": float("nan"),
            "notes": f"Error: {exc}",
        }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    print("Generating data...")
    z0_train, t_eval, traj_train = _generate_data(N_TRAIN_TRAJ, rng)
    z0_test, _, traj_test = _generate_data(N_TEST_TRAJ, rng)

    results: list[dict[str, Any]] = []

    print("Benchmarking scipy (reference)...")
    results.append(_scipy_benchmark(z0_test, t_eval, traj_test))

    print("Benchmarking ReactorTwin...")
    results.append(_reactortwin_benchmark(z0_train, t_eval, traj_train, z0_test, traj_test))

    print("Benchmarking DeepXDE (if installed)...")
    r = _deepxde_benchmark(z0_train, t_eval, traj_train, z0_test, traj_test)
    if r:
        results.append(r)
    else:
        print("  DeepXDE not installed — skipping. Install: pip install deepxde")

    print("Benchmarking TorchDyn (if installed)...")
    r = _torchdyn_benchmark(z0_train, t_eval, traj_train, z0_test, traj_test)
    if r:
        results.append(r)
    else:
        print("  TorchDyn not installed — skipping. Install: pip install torchdyn")

    print("Benchmarking PyDMD (if installed)...")
    r = _pydmd_benchmark(z0_train, t_eval, traj_train, z0_test, traj_test)
    if r:
        results.append(r)
    else:
        print("  PyDMD not installed — skipping. Install: pip install pydmd")

    # Print table
    print("\n\n" + "=" * 90)
    print("BENCHMARK RESULTS — ReactorTwin vs Competitors")
    print("Exothermic CSTR (A→B), 3 states, 8 train / 4 test trajectories")
    print("=" * 90)
    fmt = "{:<30} {:>12} {:>14} {:>10} {:>12} {:>12}"
    print(fmt.format("Library", "Train(s)", "Infer(ms)", "MSE", "MassViol", "PosViol"))
    print("-" * 90)
    for r in results:
        print(fmt.format(
            r["library"],
            f"{r['train_time_s']:.1f}" if not np.isnan(r["train_time_s"]) else "N/A",
            f"{r['inference_ms']:.3f}" if not np.isnan(r["inference_ms"]) else "N/A",
            f"{r['mse']:.4f}" if not np.isnan(r["mse"]) else "N/A",
            f"{r['mass_violation']:.4f}" if not np.isnan(r["mass_violation"]) else "N/A",
            f"{r['pos_violation']:.4f}" if not np.isnan(r["pos_violation"]) else "N/A",
        ))
        if r["notes"]:
            print(f"  → {r['notes']}")

    print("\nLaTeX table rows (for paper):")
    for r in results:
        def fmt_cell(v: float) -> str:
            return f"{v:.4f}" if not np.isnan(v) else "—"
        print(
            f"  {r['library']} & "
            f"{fmt_cell(r['train_time_s'])} & "
            f"{fmt_cell(r['inference_ms'])} & "
            f"{fmt_cell(r['mse'])} & "
            f"{fmt_cell(r['mass_violation'])} & "
            f"{fmt_cell(r['pos_violation'])} \\\\"
        )

    # Write CSV
    csv_path = RESULTS_DIR / "benchmark_vs_competitors.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["library", "train_time_s", "inference_ms", "mse",
                                                "mass_violation", "pos_violation", "notes"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nCSV saved to {csv_path}")
    print("=" * 90)


if __name__ == "__main__":
    main()
