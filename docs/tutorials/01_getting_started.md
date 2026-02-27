# Tutorial 1: Getting Started

This tutorial covers the complete workflow from creating a reactor to training a Neural ODE.

!!! tip "Interactive Notebook"
    This tutorial is available as a Jupyter notebook: [`notebooks/01_getting_started.ipynb`](https://github.com/ktubhyam/reactor-twin/blob/main/notebooks/01_getting_started.ipynb)

## What You'll Learn

1. Creating a CSTR reactor with Arrhenius kinetics
2. Simulating reactor dynamics with scipy
3. Training a Neural ODE to learn the dynamics
4. Applying positivity constraints
5. Using the built-in Trainer

## Creating a Reactor

```python
from reactor_twin import ArrheniusKinetics, CSTRReactor

kinetics = ArrheniusKinetics(
    name="A_to_B",
    num_reactions=1,
    params={
        "k0": [1e10],
        "Ea": [50000.0],
        "stoich": [[-1, 1]],
    },
)

reactor = CSTRReactor(
    name="my_cstr",
    num_species=2,
    params={"V": 100, "F": 10, "C_feed": [1.0, 0.0], "T_feed": 350},
    kinetics=kinetics,
    isothermal=True,
)
```

## Generating Training Data

```python
import numpy as np
from scipy.integrate import solve_ivp

y0 = reactor.get_initial_state()
t_span = np.linspace(0, 10, 100)
sol = solve_ivp(reactor.ode_rhs, [0, 10], y0, t_eval=t_span, method="LSODA")
```

## Training a Neural ODE

```python
import torch
from reactor_twin import NeuralODE

model = NeuralODE(state_dim=2, hidden_dim=64, num_layers=3, solver="rk4", adjoint=False)

z0 = torch.tensor(y0, dtype=torch.float32).unsqueeze(0)
t_tensor = torch.tensor(t_span, dtype=torch.float32)
targets = torch.tensor(sol.y.T, dtype=torch.float32).unsqueeze(0)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
model.train()
for epoch in range(300):
    optimizer.zero_grad()
    preds = model(z0, t_tensor)
    loss = model.compute_loss(preds, targets)["total"]
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
```

## Applying Constraints

```python
from reactor_twin import PositivityConstraint

constraint = PositivityConstraint(mode="hard", method="softplus")
constrained_preds, violation = constraint(preds)
# Guaranteed: constrained_preds >= 0
```

## Using the Trainer

```python
from reactor_twin import Trainer, ReactorDataGenerator

data_gen = ReactorDataGenerator(reactor=reactor)
z0_batch, t_batch, targets_batch = data_gen.generate(
    num_trajectories=5, t_span=(0, 10), num_points=50, noise_std=0.01,
)

trainer = Trainer(model=model, device="cpu")
history = trainer.train(train_data=(z0_batch, t_batch, targets_batch), num_epochs=50, lr=5e-4)
```

## Next Steps

- [Tutorial 2: Physics Constraints](02_physics_constraints.md) - Explore all 7 constraint types
- [Tutorial 3: Advanced Neural DEs](03_advanced_neural_des.md) - Latent, Augmented, SDE, CDE variants
