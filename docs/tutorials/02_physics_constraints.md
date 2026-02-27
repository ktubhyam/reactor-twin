# Tutorial 2: Physics Constraints

ReactorTwin enforces physical laws during Neural DE predictions with 7 constraint types.

!!! tip "Interactive Notebook"
    This tutorial is available as a Jupyter notebook: [`notebooks/02_physics_constraints.ipynb`](https://github.com/ktubhyam/reactor-twin/blob/main/notebooks/02_physics_constraints.ipynb)

## Constraint Modes

All constraints support two modes:

| Mode | Behavior | Use Case |
|------|----------|----------|
| **Hard** | Projects predictions onto the constraint manifold | Strict enforcement, physically valid outputs |
| **Soft** | Adds a penalty term to the loss function | Regularization during training |

## Available Constraints

| Constraint | Key | What it Enforces |
|-----------|-----|------------------|
| Positivity | `positivity` | Concentrations >= 0 |
| Mass Balance | `mass_balance` | Conservation of total mass |
| Energy Balance | `energy_balance` | Conservation of energy |
| Stoichiometry | `stoichiometry` | Stoichiometric consistency |
| Port-Hamiltonian | `port_hamiltonian` | Hamiltonian structure preservation |
| GENERIC | `generic` | GENERIC thermodynamic framework |
| Thermodynamic | `thermodynamic` | Gibbs/entropy constraints |

## Hard vs Soft Example

```python
from reactor_twin import PositivityConstraint

# Hard: projects negative values to zero
hard = PositivityConstraint(mode="hard", method="softplus")
corrected, _ = hard(predictions)  # All values >= 0

# Soft: computes penalty for negative values
soft = PositivityConstraint(mode="soft", method="relu", weight=10.0)
_, penalty = soft(predictions)  # Add penalty to loss
```

## Constraint Pipeline

Combine multiple constraints:

```python
from reactor_twin import ConstraintPipeline, PositivityConstraint, MassBalanceConstraint

pipeline = ConstraintPipeline([
    PositivityConstraint(mode="hard", method="softplus"),
    MassBalanceConstraint(mode="soft", stoichiometry=stoich, total_mass=1.0),
])

corrected, total_penalty = pipeline(predictions)
```

## Next Steps

- [Tutorial 3: Advanced Neural DEs](03_advanced_neural_des.md)
- [Tutorial 4: Digital Twin Pipeline](04_digital_twin.md)
