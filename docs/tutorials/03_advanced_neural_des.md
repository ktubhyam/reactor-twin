# Tutorial 3: Advanced Neural Differential Equations

ReactorTwin provides 5 Neural DE variants for learning reactor dynamics.

!!! tip "Interactive Notebook"
    This tutorial is available as a Jupyter notebook: [`notebooks/03_advanced_neural_des.ipynb`](https://github.com/ktubhyam/reactor-twin/blob/main/notebooks/03_advanced_neural_des.ipynb)

## Neural DE Variants

| Variant | Class | Use Case |
|---------|-------|----------|
| Neural ODE | `NeuralODE` | Standard dynamics learning |
| Augmented Neural ODE | `AugmentedNeuralODE` | Complex dynamics requiring trajectory crossing |
| Latent Neural ODE | `LatentNeuralODE` | High-dimensional observations, encoder-decoder |
| Neural SDE | `NeuralSDE` | Stochastic dynamics, uncertainty quantification |
| Neural CDE | `NeuralCDE` | Irregular time series, controlled inputs |

## Standard Neural ODE

The baseline model: $\frac{dy}{dt} = f_\theta(t, y)$

```python
from reactor_twin import NeuralODE

model = NeuralODE(state_dim=2, hidden_dim=32, num_layers=2, solver="rk4", adjoint=False)
preds = model(z0, t_span)  # (batch, time, state_dim)
```

## Augmented Neural ODE

Extends the state space with extra dimensions to increase expressivity:

```python
from reactor_twin.core import AugmentedNeuralODE

model = AugmentedNeuralODE(
    state_dim=2, augment_dim=4,  # 4 extra dimensions
    hidden_dim=32, num_layers=2, solver="rk4", adjoint=False,
)
```

## Latent Neural ODE

Encoder-decoder architecture with VAE-style latent space:

```python
from reactor_twin.core import LatentNeuralODE

model = LatentNeuralODE(
    state_dim=2, latent_dim=4,
    encoder_hidden_dim=32, decoder_hidden_dim=32,
    encoder_type="mlp",
    hidden_dim=32, num_layers=2, solver="rk4", adjoint=False,
)

# Training includes KL divergence loss
loss_dict = model.compute_loss(preds, targets)
# loss_dict: {"total", "reconstruction", "kl"}
```

## Neural SDE

Stochastic dynamics with built-in uncertainty: $dy = f_\theta dt + g_\phi dW$

```python
from reactor_twin.core import NeuralSDE

model = NeuralSDE(
    state_dim=2, hidden_dim=32, num_layers=2,
    noise_type="diagonal", sde_type="ito",
)
# Multiple forward passes give different samples
```

## Neural CDE

For irregular time series with control signals:

```python
from reactor_twin.core import NeuralCDE

model = NeuralCDE(state_dim=2, hidden_dim=32, num_layers=2, solver="rk4")
```

## Next Steps

- [Tutorial 4: Digital Twin Pipeline](04_digital_twin.md)
- [Tutorial 5: Custom Extensions](05_custom_extensions.md)
