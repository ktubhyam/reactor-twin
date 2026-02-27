# Reactor Models

ReactorTwin includes four reactor types that cover the most common chemical engineering configurations.

## CSTR (Continuous Stirred-Tank Reactor)

The workhorse of chemical engineering. Assumes perfect mixing — all state variables are uniform throughout the reactor volume.

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

For non-isothermal operation, set `isothermal=False` and provide heat transfer parameters:

- `UA` — heat transfer coefficient times area (J/min/K)
- `T_coolant` — coolant temperature (K)
- `dH_rxn` — list of heats of reaction per reaction (J/mol)

## Batch Reactor

No inlet or outlet flow. Concentrations change only due to reactions.

```python
from reactor_twin import BatchReactor

reactor = BatchReactor(
    name="batch",
    num_species=2,
    params={"V": 1.0, "T": 350.0, "C_initial": [1.0, 0.0]},
    kinetics=kinetics,
    isothermal=True,
)
```

**Key parameters:** `V` (volume), `T` (temperature), `C_initial` (initial concentrations).

## Semi-Batch Reactor

Feed enters but no outlet. Volume increases over time.

```python
from reactor_twin import SemiBatchReactor

reactor = SemiBatchReactor(
    name="semi_batch",
    num_species=2,
    params={
        "V0": 50.0, "F_in": 5.0,
        "C_feed": [2.0, 0.0], "T_feed": 350.0,
        "C_initial": [0.5, 0.0],
    },
    kinetics=kinetics,
    isothermal=True,
)
```

**Key parameters:** `V0` (initial volume), `F_in` (feed flow rate), `C_feed`, `T_feed`, `C_initial`.

## Plug Flow Reactor (PFR)

Tubular reactor with axial flow. Discretized using Method of Lines into `N` spatial cells.

```python
from reactor_twin import PlugFlowReactor

reactor = PlugFlowReactor(
    name="pfr",
    num_species=2,
    params={
        "L": 10.0, "D": 0.1, "u": 1.0,
        "C_inlet": [1.0, 0.0], "N": 20,
    },
    kinetics=kinetics,
    isothermal=True,
)
```

**Key parameters:** `L` (length), `D` (diameter), `u` (velocity), `C_inlet` (inlet concentrations), `N` (grid cells).

## Benchmark Systems

ReactorTwin provides 5 pre-built reactor systems for testing and benchmarking:

| System | Function | Description |
|--------|----------|-------------|
| Exothermic A→B | `create_exothermic_cstr()` | Non-isothermal with thermal runaway |
| Van de Vusse | `create_van_de_vusse_cstr()` | Series-parallel reactions |
| Bioreactor | `create_bioreactor_cstr()` | Monod kinetics microbial growth |
| Consecutive A→B→C | `create_consecutive_cstr()` | Sequential reactions |
| Parallel A→B, A→C | `create_parallel_cstr()` | Competing reactions |
