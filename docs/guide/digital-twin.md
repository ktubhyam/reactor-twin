# Digital Twin

The digital twin module provides online monitoring and control capabilities.

## Components

1. **EKF State Estimator**: Extended Kalman Filter for state estimation
2. **Fault Detector**: Multi-level fault detection (SPC, residual, isolation, classification)
3. **MPC Controller**: Model Predictive Control
4. **Online Adapter**: Continual learning with EWC
5. **Meta-Learner**: Reptile meta-learning for fast adaptation

## Architecture

```
Sensor Data -> EKF -> Fault Detector -> MPC Controller
                |         |                    |
                v         v                    v
           State Est.  Alarms          Control Actions
```
