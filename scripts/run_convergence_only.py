"""Run convergence curve only and merge into existing paper_results.json.

Loads the existing results, runs run_convergence_curve(), adds the data,
saves the updated JSON, then calls generate_figures.py.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

RESULTS_PATH = Path(__file__).parent.parent / "results" / "paper_results.json"

# Load the experiments module
spec = importlib.util.spec_from_file_location(
    "exp", Path(__file__).parent / "experiments_paper.py"
)
m = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
spec.loader.exec_module(m)  # type: ignore[union-attr]

print("Running convergence curve (500 epochs × 4 conditions × 3 seeds on exo CSTR)...")
convergence = m.run_convergence_curve(rng_seed=3)

# Load existing results
with open(RESULTS_PATH) as f:
    data = json.load(f)

# Merge
data["convergence"] = convergence

# Save
with open(RESULTS_PATH, "w") as f:
    json.dump(data, f, indent=2)
print(f"Updated {RESULTS_PATH}")

# Regenerate figures
subprocess.run(
    ["uv", "run", "--no-sync", "python3", "scripts/generate_figures.py"],
    cwd=Path(__file__).parent.parent,
    check=True,
)
print("Done.")
