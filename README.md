# ReactorTwin

Surrogate digital twin for industrial chemical reactors using neural ODEs and physics-informed neural networks (PINNs). Creates fast, accurate reactor models that run 1000x faster than CFD while enforcing mass balance, energy balance, and thermodynamic constraints.

**[Project Page](https://tubhyam.dev/projects/reactor-twin)**

---

## The Problem

High-fidelity CFD simulations of chemical reactors take minutes to hours per condition. Engineers need real-time feedback for monitoring, optimization, and what-if analysis — but can't wait for full simulations. Classical reduced-order models sacrifice too much physics.

## Our Approach

ReactorTwin learns reactor dynamics from process data using neural ODEs (`dz/dt = f_theta(z, t, u)`) with physics-informed constraints that guarantee conservation laws hold. The result: a surrogate model that's 1000x faster than CFD with < 2% RMSE on concentration and < 1 K error on temperature.

## Features

- **Neural ODE reactor models** — learn continuous-time dynamics from process data via torchdiffeq
- **Physics-informed training** — enforce mass balance, energy balance, and thermodynamic consistency (Gibbs free energy bounds)
- **Real-time dashboard** — live visualization of reactor state, concentration profiles, temperature evolution
- **What-if scenario analysis** — predict outcomes of operating condition changes in milliseconds
- **Anomaly detection** — flag deviations from expected behavior using latent-space monitoring
- **Multi-reactor support** — CSTR, PFR, batch, and semi-batch configurations

## Architecture

```
Process Data (T, P, C, F)
  -> Feature Engineering (residence time, Da, Pe)
  -> Neural ODE Encoder (physics-constrained)
  -> Latent State z(t)
  -> Decoder: concentration, temperature, conversion profiles
  -> Dashboard: real-time monitoring + optimization
```

## Performance Targets

| Metric | CFD | ReactorTwin |
|--------|-----|-------------|
| CSTR steady-state | 45 s | 0.03 s (1500x) |
| Parameter sweep (100 conditions) | 4.5 hours | 3 s |
| Concentration RMSE | — | < 2% |
| Temperature error | — | < 1 K |

## Tech Stack

- **ML Framework:** PyTorch + torchdiffeq (neural ODEs)
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

## Author

**Tubhyam Karthikeyan** — [tubhyam.dev](https://tubhyam.dev) | [GitHub](https://github.com/ktubhyam)

## License

MIT
