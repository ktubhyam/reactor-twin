# Digital Twin

The digital twin module provides online monitoring and control capabilities on top of trained Neural ODE models.

## Components

| Component | Class | Purpose |
|-----------|-------|---------|
| EKF State Estimator | `EKFStateEstimator` | Fuse model predictions with noisy measurements |
| Fault Detector | `FaultDetector`, `SPCChart` | Multi-level anomaly detection (SPC, residual, isolation, classification) |
| MPC Controller | `MPCController` | Optimal control using Neural ODE as plant model |
| Online Adapter | `OnlineAdapter` | Continual learning with Elastic Weight Consolidation |
| Meta-Learner | `ReptileMetaLearner` | Cross-reactor transfer learning |

## Architecture

```
Sensor Data -> EKF -> Fault Detector -> MPC Controller
                |         |                    |
                v         v                    v
           State Est.  Alarms          Control Actions
                |
                v
           Online Adapter -> Meta-Learner
```

## EKF State Estimation

The Extended Kalman Filter uses autograd Jacobians (`torch.func.jacrev`) to linearize the Neural ODE dynamics at each timestep:

```python
from reactor_twin import EKFStateEstimator

ekf = EKFStateEstimator(model=trained_model, state_dim=2, Q=1e-3, R=0.05**2, P0=0.1, dt=0.1)
result = ekf.filter(measurements=measurements, z0=z0, t_span=t_span)
```

## Fault Detection

Four levels of fault detection:

1. **L1 (SPC)**: EWMA and CUSUM control charts on raw sensor data
2. **L2 (Residual)**: One-step-ahead prediction error monitoring
3. **L3 (Isolation)**: Per-variable Mahalanobis contribution analysis
4. **L4 (Classification)**: SVM/Random Forest on residual features

## MPC Control

Receding-horizon optimization using LBFGS with warm-starting:

```python
from reactor_twin import MPCController

mpc = MPCController(model=model, horizon=5, dt=0.2, max_iter=10)
u_applied, info = mpc.step(z_current, y_ref)
```

## Online Adaptation

Elastic Weight Consolidation prevents catastrophic forgetting when adapting to new operating conditions:

```python
from reactor_twin import OnlineAdapter

adapter = OnlineAdapter(model=model, buffer_size=100, ewc_lambda=100.0, lr=1e-4)
adapter.add_experience(z0_new, t_new, targets_new)
losses = adapter.adapt(num_steps=20)
adapter.consolidate()
```

## Meta-Learning

Reptile meta-learning enables few-shot adaptation to new reactor types:

```python
from reactor_twin import ReptileMetaLearner

meta = ReptileMetaLearner(model=model, inner_lr=1e-3, meta_lr=1e-3)
meta.meta_train(task_generators, num_steps=100)
meta.fine_tune(new_reactor_generator, num_steps=10)
```

See [Tutorial 4: Digital Twin Pipeline](../tutorials/04_digital_twin.md) for a complete walkthrough.
