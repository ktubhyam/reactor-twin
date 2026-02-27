# Physics Constraints

Physics constraints enforce physical laws during Neural DE predictions.

All constraints support two modes:
- **Hard mode**: Projects predictions onto the constraint manifold
- **Soft mode**: Adds a penalty term to the loss function

| Constraint | Key | What it Enforces |
|-----------|-----|------------------|
| Positivity | `positivity` | Concentrations >= 0 |
| Mass Balance | `mass_balance` | Conservation of total mass |
| Energy Balance | `energy_balance` | Conservation of energy |
| Stoichiometry | `stoichiometry` | Stoichiometric consistency |
| Port-Hamiltonian | `port_hamiltonian` | Hamiltonian structure |
| GENERIC | `generic` | GENERIC thermodynamic framework |
| Thermodynamic | `thermodynamic` | Gibbs/entropy constraints |

## Example

```python
from reactor_twin import PositivityConstraint

constraint = PositivityConstraint(mode="hard", method="softplus")
constrained_preds, violation = constraint(predictions)
```
