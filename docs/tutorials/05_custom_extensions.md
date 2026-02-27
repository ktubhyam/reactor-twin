# Tutorial 5: Custom Extensions

ReactorTwin's registry system makes it easy to add custom components.

!!! tip "Interactive Notebook"
    This tutorial is available as a Jupyter notebook: [`notebooks/05_custom_extensions.ipynb`](https://github.com/ktubhyam/reactor-twin/blob/main/notebooks/05_custom_extensions.ipynb)

## The Registry System

ReactorTwin uses 5 registries for component discovery:

| Registry | Contents |
|----------|----------|
| `KINETICS_REGISTRY` | Kinetics models (arrhenius, michaelis_menten, ...) |
| `REACTOR_REGISTRY` | Reactor types (cstr, batch, semi_batch, pfr) |
| `CONSTRAINT_REGISTRY` | Physics constraints (positivity, mass_balance, ...) |
| `NEURAL_DE_REGISTRY` | Neural DE variants (neural_ode, augmented_neural_ode, ...) |
| `DIGITAL_TWIN_REGISTRY` | Digital twin components (ekf, fault_detector, ...) |

## Exploring Registries

```python
from reactor_twin.utils import KINETICS_REGISTRY

# List available models
print(KINETICS_REGISTRY.list_keys())
# ['arrhenius', 'michaelis_menten', 'power_law', ...]

# Get a class by key
KineticsClass = KINETICS_REGISTRY.get("arrhenius")
```

## Creating Custom Kinetics

Subclass `AbstractKinetics` and implement `compute_rates()`:

```python
from reactor_twin.reactors.kinetics.base import AbstractKinetics

class MyKinetics(AbstractKinetics):
    def __init__(self, name, params):
        super().__init__(name=name, num_reactions=1, params=params)
        self.k = params["k"]

    def compute_rates(self, concentrations, temperature):
        rate = self.k * concentrations[0]
        return np.array([-rate, rate])

# Register it
KINETICS_REGISTRY.register("my_kinetics", MyKinetics)
```

## Creating Custom Registries

```python
from reactor_twin.utils import Registry

MY_REGISTRY = Registry("my_components")
MY_REGISTRY.register("key", MyClass)
cls = MY_REGISTRY.get("key")
```

## Next Steps

- [API Reference](../api/core.md) for full class documentation
- [Examples](https://github.com/ktubhyam/reactor-twin/tree/main/examples) for runnable scripts
