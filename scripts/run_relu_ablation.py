"""Run VDV CSTR hard_relu ablation and merge into paper_results.json.

Usage: uv run --no-sync python3 scripts/run_relu_ablation.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# Ensure repo root is on path so experiments_paper imports correctly
sys.path.insert(0, str(Path(__file__).parent))

from experiments_paper import (
    LAMBDA_WEIGHT,
    N_TRAIN_TRAJ,
    N_VAL_TRAJ,
    RESULTS_DIR,
    SEEDS,
    SYSTEMS,
    _generate_dataset,
    _train_one,
)

CONDITION = "hard_relu"
SYS_KEY = "vdv_cstr"


def main() -> None:
    rng = np.random.default_rng(0)
    cfg = SYSTEMS[SYS_KEY]

    print(f"System: {cfg['name']}")
    print(f"Condition: {CONDITION!r}")
    print(f"Seeds: {SEEDS}")
    print()

    z0_train, t_span, traj_train = _generate_dataset(SYS_KEY, N_TRAIN_TRAJ, rng)
    z0_val, _, traj_val = _generate_dataset(SYS_KEY, N_VAL_TRAJ, rng)
    print(f"Data: {z0_train.shape[0]} train / {z0_val.shape[0]} val trajectories\n")

    run_results: list[dict] = []
    for seed in SEEDS:
        r = _train_one(
            SYS_KEY, z0_train, t_span, traj_train,
            z0_val, traj_val, CONDITION, seed,
            lambda_weight=LAMBDA_WEIGHT,
        )
        run_results.append(r)
        print(
            f"  seed={seed}  nmse_long={r['nmse_long']:.5f}  "
            f"pos_viol%={r['physics_viol']:.4f}  "
            f"drift={r['pre_proj_drift']:.4f}  "
            f"t={r['train_time_s']:.1f}s"
        )

    # Summary
    nmse_vals = [r["nmse_long"] for r in run_results]
    viol_vals = [r["physics_viol"] for r in run_results]
    print(f"\nNMSE:  {float(np.mean(nmse_vals)):.3f} ± {float(np.std(nmse_vals)):.3f}")
    print(f"Viol%: {float(np.mean(viol_vals)):.2f} ± {float(np.std(viol_vals)):.2f}")

    # Merge into paper_results.json
    results_path = RESULTS_DIR / "paper_results.json"
    with results_path.open() as f:
        data = json.load(f)

    data["ablation"][SYS_KEY][CONDITION] = run_results

    with results_path.open("w") as f:
        json.dump(data, f, indent=2)

    print(f"\nSaved {CONDITION} results to {results_path}")

    # Print LaTeX row for Table 1
    nmse_mean = float(np.mean(nmse_vals))
    nmse_std = float(np.std(nmse_vals))
    viol_mean = float(np.mean(viol_vals))
    viol_std = float(np.std(viol_vals))
    print(
        f"\nLaTeX row:\n"
        f"  Hard (ReLU) & {nmse_mean:.3f}$\\pm${nmse_std:.3f} "
        f"& {viol_mean:.2f}$\\pm${viol_std:.2f} \\\\"
    )


if __name__ == "__main__":
    main()
