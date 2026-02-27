# Quickstart

This guide walks through the complete workflow: create a reactor, generate data, train a Neural ODE, and apply physics constraints.

See `examples/00_quickstart.py` for the full runnable script.

## 1. Define a Reactor

```python
from reactor_twin import CSTRReactor, ArrheniusKinetics
import numpy as np

kinetics = ArrheniusKinetics(
    name="A_to_B",
    num_reactions=1,
    params={
        "k0": [1e10],
        "Ea": [50000.0],
        "stoich": np.array([[-1, 1]]),
    },
)

reactor = CSTRReactor(
    name="exothermic_cstr",
    num_species=2,
    params={"V": 100.0, "F": 10.0, "C_feed": [1.0, 0.0], "T_feed": 350.0},
    kinetics=kinetics,
    isothermal=True,
)
```

## 2. Generate Training Data

```python
from scipy.integrate import solve_ivp

y0 = reactor.get_initial_state()
t_span = np.linspace(0, 10, 50)
sol = solve_ivp(reactor.ode_rhs, [0, 10], y0, t_eval=t_span, method="LSODA")
```

## 3. Train a Neural ODE

```python
import torch
from reactor_twin import NeuralODE

model = NeuralODE(state_dim=2, hidden_dim=32, num_layers=2)
z0 = torch.tensor(y0, dtype=torch.float32).unsqueeze(0)
t_torch = torch.tensor(t_span, dtype=torch.float32)
targets = torch.tensor(sol.y.T, dtype=torch.float32).unsqueeze(0)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
for epoch in range(100):
    preds = model(z0, t_torch)
    loss = model.compute_loss(preds, targets)["total"]
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## 4. Apply Physics Constraints

```python
from reactor_twin import PositivityConstraint

constraint = PositivityConstraint(mode="hard", method="softplus")
predictions = model.predict(z0, t_torch)
constrained, _ = constraint(predictions)
print(f"Min value: {constrained.min():.4f}")  # Always >= 0
```
