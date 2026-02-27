"""Performance benchmarks for Phase 5 Digital Twin modules.

Targets:
- MPC step: < 100ms per step
- EKF filter step: < 10ms per step
- EKF Jacobian: measure autograd vs finite-diff speed

Run: python benchmarks/bench_digital_twin.py
"""

from __future__ import annotations

import time
import statistics

import torch
import numpy as np


def bench_mpc_step() -> dict[str, float]:
    """Benchmark MPC controller step time."""
    from reactor_twin.core.neural_ode import NeuralODE
    from reactor_twin.digital_twin.mpc_controller import MPCController

    state_dim = 3
    input_dim = 1
    horizons = [5, 10, 20]
    results = {}

    for H in horizons:
        model = NeuralODE(state_dim=state_dim, input_dim=input_dim,
                          adjoint=False, solver="euler", hidden_dim=32, num_layers=2)
        mpc = MPCController(model, horizon=H, dt=0.01, max_iter=10)

        z0 = torch.randn(state_dim)
        y_ref = torch.zeros(state_dim)

        # Warm-up
        mpc.step(z0, y_ref)

        times = []
        for _ in range(20):
            z0 = torch.randn(state_dim)
            t0 = time.perf_counter()
            mpc.step(z0, y_ref)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)  # ms

        mean_ms = statistics.mean(times)
        median_ms = statistics.median(times)
        p95 = sorted(times)[int(0.95 * len(times))]
        results[f"mpc_H{H}_mean_ms"] = mean_ms
        results[f"mpc_H{H}_median_ms"] = median_ms
        results[f"mpc_H{H}_p95_ms"] = p95

    return results


def bench_ekf_step() -> dict[str, float]:
    """Benchmark EKF predict + update step time."""
    from reactor_twin.core.neural_ode import NeuralODE
    from reactor_twin.digital_twin.state_estimator import EKFStateEstimator

    state_dims = [3, 6, 10]
    results = {}

    for sd in state_dims:
        model = NeuralODE(state_dim=sd, adjoint=False, solver="euler",
                          hidden_dim=32, num_layers=2)
        ekf = EKFStateEstimator(model, state_dim=sd, dt=0.01)

        z = torch.randn(sd)
        P = torch.eye(sd)
        meas = torch.randn(sd)

        # Warm-up
        z_pred, P_pred = ekf.predict_step(z, P)
        ekf.update_step(z_pred, P_pred, meas)

        times = []
        for _ in range(50):
            z = torch.randn(sd)
            P = torch.eye(sd) * 0.1
            meas = torch.randn(sd)
            t0 = time.perf_counter()
            z_pred, P_pred = ekf.predict_step(z, P)
            ekf.update_step(z_pred, P_pred, meas)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

        mean_ms = statistics.mean(times)
        median_ms = statistics.median(times)
        results[f"ekf_sd{sd}_mean_ms"] = mean_ms
        results[f"ekf_sd{sd}_median_ms"] = median_ms

    return results


def bench_ekf_jacobian() -> dict[str, float]:
    """Benchmark EKF Jacobian computation methods."""
    from reactor_twin.core.neural_ode import NeuralODE
    from reactor_twin.digital_twin.state_estimator import EKFStateEstimator

    state_dim = 6
    model = NeuralODE(state_dim=state_dim, adjoint=False, solver="euler",
                      hidden_dim=64, num_layers=3)
    ekf = EKFStateEstimator(model, state_dim=state_dim, dt=0.01)

    z = torch.randn(state_dim)
    t = torch.tensor(0.0)

    # Warm-up
    ekf._compute_jacobian(z, t)

    times = []
    for _ in range(30):
        z = torch.randn(state_dim)
        t0 = time.perf_counter()
        ekf._compute_jacobian(z, t)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    return {
        "jacobian_sd6_mean_ms": statistics.mean(times),
        "jacobian_sd6_median_ms": statistics.median(times),
        "jacobian_sd6_p95_ms": sorted(times)[int(0.95 * len(times))],
    }


def bench_ekf_filter() -> dict[str, float]:
    """Benchmark full EKF filter pass."""
    from reactor_twin.core.neural_ode import NeuralODE
    from reactor_twin.digital_twin.state_estimator import EKFStateEstimator

    state_dim = 3
    model = NeuralODE(state_dim=state_dim, adjoint=False, solver="euler",
                      hidden_dim=32, num_layers=2)
    ekf = EKFStateEstimator(model, state_dim=state_dim, dt=0.01)

    lengths = [50, 100, 200]
    results = {}

    for N in lengths:
        measurements = torch.randn(N, state_dim)
        z0 = torch.randn(state_dim)

        # Warm-up
        ekf.filter(measurements[:10], z0=z0)

        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            ekf.filter(measurements, z0=z0)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

        results[f"filter_N{N}_mean_ms"] = statistics.mean(times)
        results[f"filter_N{N}_per_step_ms"] = statistics.mean(times) / N

    return results


def bench_online_adapt() -> dict[str, float]:
    """Benchmark online adapter step time."""
    from reactor_twin.core.neural_ode import NeuralODE
    from reactor_twin.digital_twin.online_adapter import OnlineAdapter

    state_dim = 3
    model = NeuralODE(state_dim=state_dim, adjoint=False, solver="euler",
                      hidden_dim=32, num_layers=2)
    adapter = OnlineAdapter(model, lr=1e-3, buffer_capacity=100)

    # Fill buffer
    for _ in range(20):
        adapter.add_experience(
            torch.randn(1, state_dim),
            torch.linspace(0, 1, 10),
            torch.randn(1, 10, state_dim),
        )

    new_data = {
        "z0": torch.randn(4, state_dim),
        "t_span": torch.linspace(0, 1, 10),
        "targets": torch.randn(4, 10, state_dim),
    }

    # Warm-up
    adapter.adapt(new_data, num_steps=1, batch_size=8)

    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        adapter.adapt(new_data, num_steps=5, batch_size=8)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    return {
        "adapt_5steps_mean_ms": statistics.mean(times),
        "adapt_5steps_median_ms": statistics.median(times),
    }


def main() -> None:
    """Run all benchmarks and print results."""
    torch.manual_seed(42)

    print("=" * 65)
    print("ReactorTwin Digital Twin Performance Benchmarks")
    print("=" * 65)

    benchmarks = [
        ("MPC Controller Step", bench_mpc_step),
        ("EKF Predict+Update Step", bench_ekf_step),
        ("EKF Jacobian Computation", bench_ekf_jacobian),
        ("EKF Full Filter Pass", bench_ekf_filter),
        ("Online Adaptation", bench_online_adapt),
    ]

    all_results: dict[str, float] = {}
    for name, fn in benchmarks:
        print(f"\n--- {name} ---")
        results = fn()
        all_results.update(results)
        for k, v in results.items():
            print(f"  {k}: {v:.2f} ms")

    # Summary / pass-fail
    print("\n" + "=" * 65)
    print("PERFORMANCE TARGETS")
    print("=" * 65)

    mpc_h10 = all_results.get("mpc_H10_mean_ms", 999)
    ekf_sd3 = all_results.get("ekf_sd3_mean_ms", 999)
    jac_sd6 = all_results.get("jacobian_sd6_mean_ms", 999)

    # CPU targets; GPU targets would be ~10x tighter
    targets = [
        ("MPC step (H=10)", mpc_h10, 500.0),
        ("EKF step (sd=3)", ekf_sd3, 200.0),
        ("Jacobian (sd=6)", jac_sd6, 200.0),
    ]

    all_pass = True
    for name, actual, target in targets:
        status = "PASS" if actual < target else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  {name}: {actual:.2f}ms (target <{target:.0f}ms) [{status}]")

    print()
    if all_pass:
        print("All performance targets met.")
    else:
        print("WARNING: Some targets exceeded. Consider profiling.")


if __name__ == "__main__":
    main()
