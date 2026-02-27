# ReactorTwin Project Structure

## Status: Architecture Complete, Ready for Implementation

**Date:** 2026-02-27
**Version:** 0.1.0 (Architecture Phase)

---

## Overview

ReactorTwin is a physics-constrained Neural Differential Equation framework for chemical reactor digital twins. The complete architecture has been defined with:

- **Foundational abstractions** (base classes, interfaces)
- **Registry-based plugin system** (extensible without modifying library code)
- **Concrete reference implementations** (Neural ODE, CSTR, Arrhenius kinetics, positivity constraint)
- **Full project configuration** (pyproject.toml, tooling, CI/CD ready)

---

## Directory Structure

```
reactor-twin/
├── README.md
├── LICENSE (MIT)
├── IMPLEMENTATION_PLAN.md         # 2,560-line technical design document
├── PROJECT_STRUCTURE.md           # This file
├── pyproject.toml                 # ✅ Complete with all dependencies
├── src/
│   └── reactor_twin/              # Main package
│       ├── __init__.py            # ✅ Exports core API
│       ├── core/                  # Neural DE Engine
│       │   ├── __init__.py        # ✅ Module exports
│       │   ├── base.py            # ✅ AbstractNeuralDE (complete)
│       │   ├── ode_func.py        # ✅ AbstractODEFunc, MLPODEFunc, HybridODEFunc (complete)
│       │   ├── neural_ode.py      # ✅ NeuralODE (complete with torchdiffeq)
│       │   ├── latent_neural_ode.py      # ⏳ TODO
│       │   ├── augmented_neural_ode.py   # ⏳ TODO
│       │   ├── neural_sde.py             # ⏳ TODO
│       │   ├── neural_cde.py             # ⏳ TODO
│       │   ├── hybrid_ode.py             # ⏳ TODO (use HybridODEFunc)
│       │   ├── solvers.py                # ⏳ TODO
│       │   ├── adjoint.py                # ⏳ TODO
│       │   └── stiffness.py              # ⏳ TODO
│       ├── physics/               # Physics Enforcement
│       │   ├── __init__.py        # ✅ Module exports
│       │   ├── constraints.py     # ✅ AbstractConstraint, ConstraintPipeline (complete)
│       │   ├── positivity.py      # ✅ PositivityConstraint (complete)
│       │   ├── mass_balance.py           # ⏳ TODO
│       │   ├── energy_balance.py         # ⏳ TODO
│       │   ├── thermodynamics.py         # ⏳ TODO
│       │   ├── stoichiometry.py          # ⏳ TODO
│       │   ├── port_hamiltonian.py       # ⏳ TODO
│       │   └── generic.py                # ⏳ TODO
│       ├── reactors/              # Reactor Library
│       │   ├── __init__.py        # ✅ Module exports
│       │   ├── base.py            # ✅ AbstractReactor (complete)
│       │   ├── cstr.py            # ✅ CSTRReactor (complete)
│       │   ├── batch.py                  # ⏳ TODO
│       │   ├── semi_batch.py             # ⏳ TODO
│       │   ├── pfr.py                    # ⏳ TODO
│       │   ├── multi_phase.py            # ⏳ TODO
│       │   ├── population_balance.py     # ⏳ TODO
│       │   └── kinetics/          # Reaction kinetics plugins
│       │       ├── __init__.py    # ✅ Module exports
│       │       ├── base.py        # ✅ AbstractKinetics (complete)
│       │       ├── arrhenius.py   # ✅ ArrheniusKinetics (complete)
│       │       ├── langmuir_hinshelwood.py  # ⏳ TODO
│       │       ├── michaelis_menten.py      # ⏳ TODO
│       │       ├── power_law.py             # ⏳ TODO
│       │       └── reversible.py            # ⏳ TODO
│       ├── training/              # Training Engine
│       │   ├── __init__.py               # ⏳ TODO
│       │   ├── trainer.py                # ⏳ TODO
│       │   ├── losses.py                 # ⏳ TODO
│       │   ├── data_generator.py         # ⏳ TODO
│       │   ├── schedulers.py             # ⏳ TODO
│       │   ├── curriculum.py             # ⏳ TODO
│       │   ├── meta_learning.py          # ⏳ TODO
│       │   └── stiff_training.py         # ⏳ TODO
│       ├── digital_twin/          # Digital Twin Layer
│       │   ├── __init__.py               # ⏳ TODO
│       │   ├── state_estimator.py        # ⏳ TODO (EKF + Neural ODE)
│       │   ├── fault_detector.py         # ⏳ TODO
│       │   ├── anomaly_diagnosis.py      # ⏳ TODO
│       │   ├── scenario_simulator.py     # ⏳ TODO
│       │   ├── optimal_control.py        # ⏳ TODO (MPC)
│       │   ├── online_adapter.py         # ⏳ TODO
│       │   └── transfer_learning.py      # ⏳ TODO
│       ├── dashboard/             # Visualization & Dashboard
│       │   ├── __init__.py               # ⏳ TODO
│       │   ├── app.py                    # ⏳ TODO (Streamlit main)
│       │   └── pages/            # Dashboard pages (10 pages)
│       │       ├── reactor_sim.py        # ⏳ TODO
│       │       ├── phase_portrait.py     # ⏳ TODO
│       │       ├── bifurcation.py        # ⏳ TODO
│       │       ├── rtd_analysis.py       # ⏳ TODO
│       │       ├── parameter_sweep.py    # ⏳ TODO
│       │       ├── sensitivity.py        # ⏳ TODO
│       │       ├── pareto.py             # ⏳ TODO
│       │       ├── fault_monitor.py      # ⏳ TODO
│       │       ├── comparison.py         # ⏳ TODO
│       │       └── model_explorer.py     # ⏳ TODO
│       ├── api/                   # API & Deployment
│       │   ├── __init__.py               # ⏳ TODO
│       │   ├── server.py                 # ⏳ TODO (FastAPI)
│       │   ├── schemas.py                # ⏳ TODO (Pydantic)
│       │   └── websocket.py              # ⏳ TODO
│       └── utils/                 # Utilities
│           ├── __init__.py        # ✅ Module exports
│           ├── registry.py        # ✅ Registry system (complete)
│           ├── visualization.py          # ⏳ TODO
│           ├── metrics.py                # ⏳ TODO
│           ├── config.py                 # ⏳ TODO
│           └── numerical.py              # ⏳ TODO
├── tests/                         # Test suite
│   ├── conftest.py                       # ⏳ TODO (shared fixtures)
│   ├── test_core/                        # ⏳ TODO (8 test files)
│   ├── test_physics/                     # ⏳ TODO (5 test files)
│   ├── test_reactors/                    # ⏳ TODO (4 test files)
│   ├── test_training/                    # ⏳ TODO (4 test files)
│   ├── test_digital_twin/                # ⏳ TODO (3 test files)
│   ├── test_integration/                 # ⏳ TODO (3 test files)
│   └── benchmarks/                       # ⏳ TODO (3 benchmark scripts)
├── examples/                      # Example scripts
│   ├── 01_cstr_exothermic.py             # ⏳ TODO
│   ├── 02_cstr_van_de_vusse.py           # ⏳ TODO
│   ├── 03_batch_consecutive.py           # ⏳ TODO
│   ├── 04_batch_parameter_estimation.py  # ⏳ TODO
│   ├── 05_pfr_tubular.py                 # ⏳ TODO
│   ├── 06_bifurcation_analysis.py        # ⏳ TODO
│   ├── 07_hard_vs_soft_constraints.py    # ⏳ TODO
│   ├── 08_latent_neural_ode.py           # ⏳ TODO
│   ├── 09_neural_sde_uncertainty.py      # ⏳ TODO
│   ├── 10_meta_learning_adaptation.py    # ⏳ TODO
│   ├── 11_state_estimation_ekf.py        # ⏳ TODO
│   ├── 12_fault_detection.py             # ⏳ TODO
│   ├── 13_mpc_control.py                 # ⏳ TODO
│   ├── 14_multi_objective_optimization.py # ⏳ TODO
│   └── 15_population_balance.py          # ⏳ TODO
├── notebooks/                     # Tutorial notebooks
│   ├── tutorial_01_basics.ipynb          # ⏳ TODO
│   ├── tutorial_02_physics_constraints.ipynb  # ⏳ TODO
│   ├── tutorial_03_custom_reactions.ipynb     # ⏳ TODO
│   ├── tutorial_04_digital_twin.ipynb         # ⏳ TODO
│   ├── tutorial_05_advanced_neural_des.ipynb  # ⏳ TODO
│   └── paper_figures.ipynb               # ⏳ TODO
├── configs/                       # YAML configuration files
│   ├── cstr_exothermic.yaml              # ⏳ TODO
│   ├── cstr_van_de_vusse.yaml            # ⏳ TODO
│   ├── batch_consecutive.yaml            # ⏳ TODO
│   ├── pfr_tubular.yaml                  # ⏳ TODO
│   └── meta_learning.yaml                # ⏳ TODO
├── Dockerfile                            # ⏳ TODO
├── docker-compose.yml                    # ⏳ TODO
└── .github/
    └── workflows/
        ├── ci.yml                        # ⏳ TODO
        └── publish.yml                   # ⏳ TODO
```

---

## Architecture Components

### 1. Core Neural DE Engine (`core/`)

**Status:** Foundational architecture complete.

**What's Done:**
- ✅ `AbstractNeuralDE` base class with:
  - `forward(z0, t_span, controls)` - ODE integration
  - `compute_loss(predictions, targets)` - Multi-objective loss
  - `train_step(batch, optimizer)` - Training loop
  - `save()` / `load()` - Checkpointing
- ✅ `AbstractODEFunc` base class for ODE right-hand-side
- ✅ `MLPODEFunc` - Standard MLP with softplus activation
- ✅ `HybridODEFunc` - Physics + neural correction
- ✅ `NeuralODE` - Complete implementation with torchdiffeq adjoint method

**What's Next:**
- Latent Neural ODE (encoder/decoder architecture)
- Augmented Neural ODE (extra dimensions)
- Neural SDE (stochastic dynamics)
- Neural CDE (irregular time series)
- Port-Hamiltonian ODE function
- Stiffness detection and adaptive handling

---

### 2. Physics Constraints (`physics/`)

**Status:** Framework complete.

**What's Done:**
- ✅ `AbstractConstraint` base class with:
  - `project(z)` - Hard constraint projection
  - `compute_violation(z)` - Soft constraint penalty
  - Hard/soft mode switching
- ✅ `ConstraintPipeline` - Compose multiple constraints
- ✅ `PositivityConstraint` - Non-negativity enforcement (softplus/ReLU/square)

**What's Next:**
- Mass balance constraint (stoichiometric projection)
- Energy balance constraint (computed from rates)
- Thermodynamic constraint (Gibbs monotonicity, equilibrium)
- Stoichiometric constraint (predict rates not species)
- Port-Hamiltonian constraint (structure-preserving)
- GENERIC constraint (reversible-irreversible coupling)

---

### 3. Reactor Library (`reactors/`)

**Status:** Reference implementation complete.

**What's Done:**
- ✅ `AbstractReactor` base class with:
  - `ode_rhs(t, y, u)` - ODE for scipy integration
  - `get_initial_state()` - Initial conditions
  - `get_state_labels()` - Variable names
  - `validate_state(y)` - Physical constraints
  - `to_dict()` / `from_dict()` - Serialization
- ✅ `CSTRReactor` - Continuous stirred-tank reactor
  - Mass balances for all species
  - Energy balance (non-isothermal mode)
  - Plug-in kinetics support

**What's Next:**
- Batch reactor (time-varying volume)
- Semi-batch reactor (continuous feed + batch)
- PFR (plug flow with Method of Lines)
- Multi-phase reactor (gas-liquid with mass transfer)
- Population balance reactor (crystallization)

---

### 4. Reaction Kinetics (`reactors/kinetics/`)

**Status:** Reference implementation complete.

**What's Done:**
- ✅ `AbstractKinetics` base class with:
  - `compute_rates(C, T)` - Reaction rate calculation
  - `validate_parameters()` - Physical checks
  - `to_dict()` / `from_dict()` - Serialization
- ✅ `ArrheniusKinetics` - Temperature-dependent rates
  - Support for elementary and non-elementary reactions
  - Arbitrary reaction orders
  - Stoichiometry matrix

**What's Next:**
- Langmuir-Hinshelwood kinetics (heterogeneous catalysis)
- Michaelis-Menten kinetics (enzyme reactions)
- Power law kinetics (empirical)
- Reversible kinetics (equilibrium-limited)

---

### 5. Registry System (`utils/registry.py`)

**Status:** Complete.

**What's Done:**
- ✅ `Registry` class - Plugin registration and lookup
- ✅ Global registries for:
  - `REACTOR_REGISTRY` - Reactor types
  - `KINETICS_REGISTRY` - Kinetics models
  - `CONSTRAINT_REGISTRY` - Physics constraints
  - `NEURAL_DE_REGISTRY` - Neural DE variants
  - `ODE_FUNC_REGISTRY` - ODE functions
  - `SOLVER_REGISTRY` - ODE solvers

**Usage Example:**
```python
from reactor_twin import REACTOR_REGISTRY, CSTRReactor

# Automatically registered via decorator
assert "cstr" in REACTOR_REGISTRY

# Retrieve and instantiate
reactor_cls = REACTOR_REGISTRY.get("cstr")
reactor = reactor_cls(name="my_cstr", num_species=2, params={...})

# Register custom reactor
@REACTOR_REGISTRY.register("my_custom_reactor")
class MyReactor(AbstractReactor):
    ...
```

---

## Configuration Files

### pyproject.toml ✅

**Complete configuration including:**
- Core dependencies: torch, torchdiffeq, numpy, scipy
- Optional dependency groups:
  - `[dashboard]` - Streamlit
  - `[api]` - FastAPI, uvicorn
  - `[sde]` - torchsde
  - `[cde]` - torchcde
  - `[fast]` - torchode (faster solver)
  - `[thermo]` - cantera, coolprop
  - `[analysis]` - SALib (sensitivity analysis)
  - `[tracking]` - wandb
  - `[deploy]` - ONNX export
  - `[dev]` - pytest, ruff, mypy, hypothesis
- CLI entry points:
  - `reactor-twin-dashboard` → Streamlit app
  - `reactor-twin-api` → FastAPI server
- Tool configuration:
  - Ruff (linter + formatter)
  - Mypy (strict type checking)
  - Pytest (with coverage)

### Installation

```bash
# Development install with all dependencies
pip install -e ".[all]"

# Minimal install (core only)
pip install -e .

# With specific features
pip install -e ".[dashboard,api,dev]"
```

---

## What Works Right Now

You can already:

1. **Define custom reactors:**
```python
from reactor_twin import AbstractReactor, REACTOR_REGISTRY

@REACTOR_REGISTRY.register("my_reactor")
class MyCustomReactor(AbstractReactor):
    def ode_rhs(self, t, y, u):
        # Your reactor equations
        return dydt
```

2. **Use the CSTR reactor:**
```python
from reactor_twin import CSTRReactor
from reactor_twin.reactors.kinetics import ArrheniusKinetics

# Define kinetics
kinetics = ArrheniusKinetics(
    name="A_to_B",
    num_reactions=1,
    params={
        "k0": [1e10],
        "Ea": [50000],
        "stoich": [[-1, 1]],  # A -> B
    }
)

# Create CSTR
reactor = CSTRReactor(
    name="exothermic_cstr",
    num_species=2,
    params={
        "V": 100.0,
        "F": 10.0,
        "C_feed": [1.0, 0.0],
        "T_feed": 350.0,
    },
    kinetics=kinetics,
    isothermal=True,
)

# Integrate with scipy
from scipy.integrate import solve_ivp
y0 = reactor.get_initial_state()
sol = solve_ivp(reactor.ode_rhs, [0, 100], y0, dense_output=True)
```

3. **Train a Neural ODE:**
```python
from reactor_twin import NeuralODE
import torch

# Create Neural ODE
model = NeuralODE(
    state_dim=2,
    hidden_dim=64,
    num_layers=3,
    solver="dopri5",
    adjoint=True,
)

# Forward pass
z0 = torch.randn(32, 2)  # batch of initial states
t_span = torch.linspace(0, 10, 50)
z_trajectory = model(z0, t_span)  # (batch, time, state_dim)

# Train
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
losses = model.train_step(
    batch={
        "z0": z0,
        "t_span": t_span,
        "targets": z_trajectory,  # Ground truth from reactor
    },
    optimizer=optimizer,
)
```

4. **Apply physics constraints:**
```python
from reactor_twin import PositivityConstraint, ConstraintPipeline

# Create constraint pipeline
pipeline = ConstraintPipeline([
    PositivityConstraint(mode="hard", method="softplus"),
])

# Apply to predictions
z_constrained, violations = pipeline(z_trajectory)
```

---

## Implementation Phases

See `IMPLEMENTATION_PLAN.md` for the full 5-phase, 10-week implementation schedule.

**Phase 1** (Weeks 1-2): Core Neural ODE + 1 CSTR benchmark
**Phase 2** (Weeks 3-4): Physics constraints + 4 more CSTRs
**Phase 3** (Weeks 5-6): Advanced DEs (Latent/Augmented/SDE)
**Phase 4** (Weeks 7-8): PFR, multi-phase, PBE
**Phase 5** (Weeks 9-10): Digital twin features + dashboard

---

## Next Steps for Implementation

1. **Create test infrastructure** (`tests/conftest.py` with fixtures)
2. **Implement remaining Neural DE variants** (Latent, Augmented, SDE, CDE)
3. **Implement remaining physics constraints** (mass, energy, thermodynamics)
4. **Implement remaining reactor types** (Batch, PFR, multi-phase, PBE)
5. **Build training engine** (Trainer, multi-objective losses, curriculum)
6. **Build digital twin components** (EKF, fault detection, MPC)
7. **Build dashboard** (10-page Streamlit app)
8. **Write 15 example scripts** (demonstrate all features)
9. **Write 5 tutorial notebooks** (onboarding for users)
10. **Set up CI/CD** (pytest, ruff, mypy, auto-publish)

---

## Architecture Principles

**Followed throughout the codebase:**

1. **Inheritance hierarchy** - Abstract base classes define interfaces
2. **Registry-based plugins** - Extensible without modifying library code
3. **Hard/soft constraints** - Physics enforcement via projection or penalty
4. **Type safety** - Strict type hints on all function signatures
5. **Logging** - `logger = logging.getLogger(__name__)` in each module
6. **Serialization** - `to_dict()` / `from_dict()` for all components
7. **Docstrings** - Google-style with array shapes
8. **Python 3.10+** - Modern syntax (`X | Y` unions, match statements)
9. **Import order** - stdlib → third-party → relative

---

## Summary

✅ **Architecture:** Complete and reviewed
✅ **Foundational code:** Base classes, registry, reference implementations
✅ **Configuration:** pyproject.toml, tooling, dependencies
⏳ **Implementation:** Ready to proceed with Phase 1

**File count:**
- Created: 18 core architecture files
- Remaining: ~102 implementation files

The project is now in a state where implementation can proceed systematically,
following the established patterns and interfaces.
