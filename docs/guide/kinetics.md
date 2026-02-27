# Kinetics Models

ReactorTwin provides six kinetics models registered via the plugin system.

| Model | Key | Use Case |
|-------|-----|----------|
| Arrhenius | `arrhenius` | Temperature-dependent reactions |
| Michaelis-Menten | `michaelis_menten` | Enzyme kinetics |
| Power Law | `power_law` | General rate expressions |
| Langmuir-Hinshelwood | `langmuir_hinshelwood` | Surface catalysis |
| Reversible | `reversible` | Equilibrium-limited reactions |
| Monod | `monod` | Microbial growth |

## Using the Registry

```python
from reactor_twin.utils import KINETICS_REGISTRY

# List available models
print(KINETICS_REGISTRY.list_keys())

# Get a class by key
KineticsClass = KINETICS_REGISTRY.get("arrhenius")
```
