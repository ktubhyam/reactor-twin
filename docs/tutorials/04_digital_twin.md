# Tutorial 4: Digital Twin Pipeline

The digital twin layer provides online monitoring, fault detection, and control capabilities.

!!! tip "Interactive Notebook"
    This tutorial is available as a Jupyter notebook: [`notebooks/04_digital_twin.ipynb`](https://github.com/ktubhyam/reactor-twin/blob/main/notebooks/04_digital_twin.ipynb)

## Architecture

```
Sensor Data -> EKF -> Fault Detector -> MPC Controller
                |         |                    |
                v         v                    v
           State Est.  Alarms          Control Actions
```

## EKF State Estimation

The Extended Kalman Filter fuses Neural ODE predictions with noisy measurements:

```python
from reactor_twin import EKFStateEstimator

ekf = EKFStateEstimator(
    model=trained_model,
    state_dim=2,
    Q=1e-3,           # Process noise covariance
    R=0.05**2,        # Measurement noise covariance
    P0=0.1,           # Initial state uncertainty
    dt=0.1,
)

result = ekf.filter(measurements=measurements, z0=z0, t_span=t_span)
filtered_states = result["states"]      # Optimal state estimates
innovations = result["innovations"]      # Prediction errors
covariances = result["covariances"]      # Uncertainty estimates
```

## Fault Detection

Multi-level fault detection with SPC charts:

```python
from reactor_twin.digital_twin import SPCChart

spc = SPCChart(
    num_vars=2,
    ewma_lambda=0.2,   # EWMA smoothing factor
    ewma_L=3.0,        # EWMA control limit (sigma)
    cusum_k=0.5,       # CUSUM allowance
    cusum_h=5.0,       # CUSUM decision interval
)

# Learn normal operation statistics
spc.set_baseline(normal_data)

# Monitor in real-time
result = spc.update(new_observation)
if result["ewma_alarm"].any():
    print("EWMA alarm triggered!")
if result["cusum_alarm"].any():
    print("CUSUM alarm triggered!")
```

## Model Predictive Control

Optimal control using Neural ODE as the plant model:

```python
from reactor_twin import MPCController

mpc = MPCController(
    model=model_with_control_input,
    horizon=5,         # Prediction horizon
    dt=0.2,            # Time step
    max_iter=10,       # LBFGS iterations
)

# Single MPC step
u_applied, info = mpc.step(z_current, y_ref)
# info: {"cost", "trajectory", "controls"}
```

## Online Adaptation

Continual learning with Elastic Weight Consolidation:

```python
from reactor_twin import OnlineAdapter

adapter = OnlineAdapter(
    model=model,
    buffer_size=100,
    ewc_lambda=100.0,  # EWC regularization strength
    lr=1e-4,
)

adapter.add_experience(z0_new, t_new, targets_new)
losses = adapter.adapt(num_steps=20)
adapter.consolidate()  # Snapshot for next EWC round
```

## Meta-Learning

Cross-reactor transfer with Reptile:

```python
from reactor_twin import ReptileMetaLearner

meta = ReptileMetaLearner(model=model, inner_lr=1e-3, meta_lr=1e-3)
meta.meta_train(task_generators, num_steps=100)
meta.fine_tune(new_reactor_generator, num_steps=10)  # Few-shot adaptation
```

## Next Steps

- [Tutorial 5: Custom Extensions](05_custom_extensions.md)
- [API Reference: Digital Twin](../api/digital-twin.md)
