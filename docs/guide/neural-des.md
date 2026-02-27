# Neural Differential Equations

ReactorTwin supports five Neural DE variants for learning reactor dynamics.

| Variant | Key | Description |
|---------|-----|-------------|
| Neural ODE | `neural_ode` | Standard Neural ODE |
| Augmented Neural ODE | `augmented_neural_ode` | Extended state space |
| Latent Neural ODE | `latent_neural_ode` | Encoder-decoder architecture |
| Neural SDE | `neural_sde` | Stochastic dynamics |
| Neural CDE | `neural_cde` | Controlled differential equations |

## Training Loop

```python
from reactor_twin import NeuralODE
from reactor_twin.training import Trainer, ReactorDataGenerator

model = NeuralODE(state_dim=2, hidden_dim=64, num_layers=3)
trainer = Trainer(model=model, device="cpu")
# trainer.train(...)
```
