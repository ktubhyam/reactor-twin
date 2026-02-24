# ReactorTwin

Surrogate digital twin for industrial chemical reactors using neural ODEs and physics-informed neural networks (PINNs).

## What Is This?

ReactorTwin creates fast, accurate surrogate models of chemical reactors that run 1000x faster than traditional CFD simulations while preserving physical constraints. It enables real-time monitoring, optimization, and what-if analysis for industrial chemical processes.

## Features

- **Neural ODE reactor models** — Learn reactor dynamics from process data
- **Physics-informed training** — Enforce mass balance, energy balance, and thermodynamic constraints
- **Real-time digital twin dashboard** — Live visualization of reactor state
- **What-if scenario analysis** — Predict outcomes of operating condition changes
- **Anomaly detection** — Flag deviations from expected behavior
- **Multi-reactor support** — CSTR, PFR, batch, and semi-batch configurations

## Architecture

```
Process Data (T, P, C, F)
    → Feature Engineering (residence time, Da, Pe)
    → Neural ODE Encoder (physics-constrained)
    → Latent State z(t)
    → Decoder: concentration, temperature, conversion profiles
    → Dashboard: real-time monitoring + optimization
```

## Tech Stack

- **ML Framework:** PyTorch + torchdiffeq (Neural ODEs)
- **Physics:** Cantera (thermodynamics), custom PDE solvers
- **Dashboard:** Next.js + D3.js for real-time visualization
- **Deployment:** Docker + FastAPI for model serving

## Target Applications

- Continuous stirred-tank reactors (CSTR) in pharmaceutical manufacturing
- Tubular reactors in petrochemical processing
- Batch fermentation reactors in biotech
- Fluidized bed reactors in catalytic cracking

## Getting Started

```bash
pip install -e .
python train.py --reactor cstr --data examples/cstr_experiment.csv
python serve.py --port 8000
```

## License

MIT
