# Reactor Models

ReactorTwin includes four reactor types that cover the most common chemical engineering configurations.

## CSTR (Continuous Stirred-Tank Reactor)

The workhorse of chemical engineering. Assumes perfect mixing.

```python
from reactor_twin import CSTRReactor, ArrheniusKinetics

reactor = CSTRReactor(
    name="my_cstr",
    num_species=2,
    params={"V": 100, "F": 10, "C_feed": [1.0, 0.0], "T_feed": 350},
    kinetics=kinetics,
    isothermal=True,
)
```

**Key parameters:** `V` (volume), `F` (flow rate), `C_feed` (feed concentrations), `T_feed` (feed temperature).

## Batch Reactor

No inlet or outlet flow. Concentrations change only due to reactions.

## Semi-Batch Reactor

Feed enters but no outlet. Volume increases over time.

## Plug Flow Reactor (PFR)

Tubular reactor with axial flow. Discretized using Method of Lines.
