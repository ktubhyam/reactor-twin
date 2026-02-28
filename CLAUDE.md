# ReactorTwin — CLAUDE.md

Physics-constrained Neural Differential Equations for chemical reactor digital twins.

## Quick Reference

- **Package**: `reactor-twin` on PyPI, v1.0.0
- **Imports as**: `from reactor_twin import ...`
- **Local path**: `/Users/admin/Work/Code/GitHub/reactor-twin/`
- **Python**: 3.10+, src layout (`src/reactor_twin/`)
- **Package manager**: uv (dev installs: `uv pip install -e ".[dev]"`)
- **Tests**: `pytest` from repo root
- **Docs**: MkDocs (`mkdocs serve`)
- **Linter/formatter**: ruff

## Architecture

Five Neural DE variants:
- `NeuralODE` — standard continuous dynamics
- `LatentNeuralODE` — latent-space formulation
- `AugmentedNeuralODE` — augmented state
- `NeuralSDE` — stochastic
- `NeuralCDE` — controlled

Four reactor types: `CSTRReactor`, `BatchReactor`, `SemiBatchReactor`, `PFRReactor`

Seven hard physics constraints (projection-based, exact satisfaction):
mass, energy, thermodynamics, stoichiometry, port-Hamiltonian, GENERIC, positivity

Digital twin stack: EKF state estimation, 4-level fault detection, MPC control,
online adaptation, meta-learning.

## Key Conventions

- `from __future__ import annotations` in every module
- Strict type hints on all signatures
- Google-style docstrings
- `logging` module only — never `print()` in library code
- Pydantic v2 for config validation
- PyTorch tensors throughout (not numpy arrays in model code)
- `python3` not `python`

## Performance Claims (site-critical — DO NOT CHANGE)

The site describes ReactorTwin as **1500x faster** than first-principles simulation.
This figure appears in two places on tubhyam.dev and must stay consistent.
Do not change this number without explicit instruction.

## Project Status

- Status: **beta** (production-stable on PyPI, active development)
- Site: https://tubhyam.dev/projects/reactor-twin
- GitHub: https://github.com/ktubhyam/reactor-twin
- Docs: https://ktubhyam.github.io/reactor-twin
- PyPI: https://pypi.org/project/reactor-twin/

## Testing

Run: `pytest` (from repo root)
Coverage: `pytest --cov=reactor_twin --cov-report=term-missing`

## Optional Extras

- `[dashboard]` — Streamlit 10-page interactive dashboard
- `[digital_twin]` — scikit-learn for EKF/fault detection
- `[api]` — FastAPI + WebSocket real-time API
- `[all]` — everything
