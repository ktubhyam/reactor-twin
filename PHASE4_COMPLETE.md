# Phase 4: Additional Reactors - COMPLETE ✅

**Completion Date:** 2026-02-27

## Overview

Phase 4 expands ReactorTwin's reactor library and kinetics models, adding 3 new reactor types, 4 new kinetics models, and 3 benchmark systems for validation.

---

## New Reactor Types (3)

### 1. BatchReactor (`reactors/batch.py`)

**Description:** Closed system with no inflow/outflow. Supports time-varying volume for gas-phase reactions.

**Features:**
- Constant or variable volume modes
- Isothermal or non-isothermal operation
- Volume change: `dV/dt = dn_total * V / n_total` (ideal gas)
- Energy balance with heat of reaction and external heating

**State Variables:**
- `[C_1, ..., C_n]` (isothermal, constant volume)
- `[C_1, ..., C_n, V]` (variable volume)
- `[C_1, ..., C_n, V, T]` (non-isothermal, variable volume)

**Use Cases:**
- Pharmaceutical batch synthesis
- Polymerization reactors
- Gas-phase batch reactions with pressure/temperature effects

---

### 2. SemiBatchReactor (`reactors/semi_batch.py`)

**Description:** Continuous feed with no outflow. Volume increases over time.

**Features:**
- Continuous inflow, no outflow (batch-like)
- Volume balance: `dV/dt = F_in`
- Mass balance includes dilution: `dC/dt = rates + (F_in/V) * (C_in - C)`
- Dynamic feed rate and temperature control

**State Variables:**
- `[C_1, ..., C_n, V]` (isothermal)
- `[C_1, ..., C_n, V, T]` (non-isothermal)

**Use Cases:**
- Specialty chemicals with controlled addition
- Fed-batch bioreactors
- Runaway reaction mitigation via slow feed

---

### 3. PlugFlowReactor (PFR) (`reactors/pfr.py`)

**Description:** Tubular reactor with axial flow. Uses Method of Lines (MOL) to discretize spatial dimension.

**Governing PDE:**
```
∂C_i/∂t = -u * ∂C_i/∂z + D * ∂²C_i/∂z² + Σ(nu_ij * r_j)
```

**Features:**
- Method of Lines discretization (N cells)
- Advection + dispersion + reaction terms
- Boundary conditions: Dirichlet (inlet), Neumann (outlet)
- Axial concentration profiles via `get_axial_profile()`

**State Variables:**
- `[C_1,0, ..., C_1,N-1, C_2,0, ..., C_M,N-1]`
- Total dimension: `num_species * num_cells`

**Parameters:**
- `L`: Reactor length (m)
- `u`: Axial velocity (m/s)
- `D`: Dispersion coefficient (m²/s)
- `num_cells`: Spatial discretization (default 50)

**Use Cases:**
- Industrial tubular reactors
- Catalytic converters
- Fixed-bed reactors

---

## New Kinetics Models (4)

### 1. MichaelisMentenKinetics (`kinetics/michaelis_menten.py`)

**Description:** Enzyme-catalyzed reactions with saturation kinetics.

**Rate Law:**
```
r = (V_max * [S]) / (K_m + [S])
```

**Features:**
- Competitive inhibition: `r = (V_max * [S]) / (K_m * (1 + [I]/K_i) + [S])`
- Substrate and inhibitor indices
- Parameters: `V_max`, `K_m`, `K_i` (optional)

**Use Cases:**
- Enzyme reactions (proteases, kinases)
- Biocatalysis
- Metabolic pathways

---

### 2. PowerLawKinetics (`kinetics/power_law.py`)

**Description:** Empirical rate expressions with arbitrary reaction orders.

**Rate Law:**
```
r_j = k_j * Π(C_i^alpha_ij)
```

**Features:**
- Non-stoichiometric orders (alpha_ij)
- Temperature dependence: `k(T) = A * exp(-E_a/(R*T))`
- Parameters: `k`, `orders`, `stoich`, `A` (optional), `E_a` (optional)

**Use Cases:**
- Empirical fitting of experimental data
- Complex reaction networks
- Non-elementary reactions

---

### 3. LangmuirHinshelwoodKinetics (`kinetics/langmuir_hinshelwood.py`)

**Description:** Heterogeneous catalysis with surface adsorption competition.

**Rate Law:**
```
r = (k * Π(C_i^alpha_i)) / (1 + Σ(K_j * C_j))^beta
```

**Features:**
- Adsorption equilibrium constants `K_ads`
- Numerator orders `orders_num`, denominator exponent `orders_den`
- Temperature dependence for both `k` and `K_ads`

**Common Forms:**
- Single reactant: `r = (k * K_A * C_A) / (1 + K_A * C_A)`
- Two reactants: `r = (k * K_A * K_B * C_A * C_B) / (1 + K_A * C_A + K_B * C_B)²`

**Use Cases:**
- Catalytic converters
- Heterogeneous catalysis (Pt, Pd, Ni)
- Surface reactions on zeolites

---

### 4. ReversibleKinetics (`kinetics/reversible.py`)

**Description:** Equilibrium-limited reactions with forward and reverse rates.

**Rate Law:**
```
r = k_f * Π(C_reactants^alpha) - k_r * Π(C_products^beta)
```

**Features:**
- Specify `k_r` directly or via `K_eq = k_f / k_r`
- van 't Hoff temperature dependence for `K_eq`
- Independent `A_f`, `E_a_f`, `A_r`, `E_a_r`

**Use Cases:**
- Esterification/hydrolysis
- Isomerization reactions
- Gas-phase equilibria

---

## New Benchmark Systems (3)

### 1. Bioreactor (`systems/bioreactor.py`)

**Description:** Aerobic fermentation with Monod growth kinetics.

**Species:**
- S: Substrate (glucose)
- X: Biomass (cells)
- P: Product (metabolite)

**Reactions:**
- Growth: `S + X → 2X` (rate: `μ_max * S/(K_s + S) * X`)
- Production: `X → P` (rate: `q_p * X`)

**Parameters:**
- V = 1000 L, F = 100 L/h
- μ_max = 0.5 1/h, K_s = 0.1 g/L
- Y_xs = 0.5 g/g, q_p = 0.2 g/(g*h)
- S_in = 20 g/L, T = 310 K (37°C)

**Use Cases:**
- E. coli fermentation
- Yeast ethanol production
- Recombinant protein expression

---

### 2. Consecutive Reactions (`systems/consecutive.py`)

**Description:** Selectivity problem for intermediate product (A→B→C).

**Reactions:**
- Reaction 1: A → B (k1 = 1.0e3 min⁻¹, E_a1 = 50 kJ/mol)
- Reaction 2: B → C (k2 = 5.0e2 min⁻¹, E_a2 = 60 kJ/mol)

**Optimal Residence Time:**
```
τ_opt = 1 / sqrt(k1 * k2)
```

**Steady-State Concentrations:**
```
C_A_ss = C_A_in / (1 + k1 * τ)
C_B_ss = (k1 * τ * C_A_in) / [(1 + k1 * τ) * (1 + k2 * τ)]
C_C_ss = C_A_in - C_A_ss - C_B_ss
```

**Parameters:**
- V = 100 L, F = 100 L/min (τ = 1 min)
- C_A_in = 2.0 mol/L, T = 350 K

**Use Cases:**
- Intermediate synthesis (e.g., B is desired)
- Residence time optimization
- Temperature sensitivity studies

---

### 3. Parallel Competing (`systems/parallel.py`)

**Description:** Selectivity problem with desired and undesired products (A→B, A→C).

**Reactions:**
- Reaction 1: A → B (desired, k1 = 2.0e4 min⁻¹, E_a1 = 55 kJ/mol, n1 = 1.0)
- Reaction 2: A → C (byproduct, k2 = 1.0e5 min⁻¹, E_a2 = 70 kJ/mol, n2 = 2.0)

**Selectivity:**
```
S_B/C = r1 / r2 = (k1 / k2) * C_A^(n1 - n2)
```

**Strategy for High B Selectivity:**
- Low temperature (favor lower E_a)
- High dilution (favor lower order)
- Short residence time

**Parameters:**
- V = 100 L, F = 50 L/min (τ = 2 min)
- C_A_in = 3.0 mol/L, T = 340 K

**Use Cases:**
- Petrochemical selectivity (aromatics vs. olefins)
- Fine chemical synthesis
- Minimizing byproduct formation

---

## Summary Statistics

**Total Modules Added:** 10
- Reactors: 3
- Kinetics: 4
- Benchmark Systems: 3

**Total Project Modules:** 50+
- Phase 1: 18 modules
- Phase 2: 13 modules
- Phase 3: 4 modules
- Phase 4: 10 modules
- Supporting: 5+ (docs, workflows)

**Test Coverage:** TBD (Phase 6)

---

## Next Steps (Phase 5)

Phase 5 focuses on digital twin capabilities:

1. **State Estimation**
   - Extended Kalman Filter (EKF)
   - Neural ODE + EKF fusion
   - Covariance propagation

2. **Fault Detection**
   - Statistical process control
   - Residual-based detection
   - Fault classification

3. **Model Predictive Control (MPC)**
   - Neural ODE as plant model
   - Gradient-based optimization
   - Real-time capable (< 100ms)

4. **Online Adaptation**
   - Replay buffer
   - Continual learning
   - Elastic Weight Consolidation

5. **Meta-Learning**
   - Reptile for cross-reactor transfer
   - Few-shot adaptation

6. **Streamlit Dashboard**
   - 10 pages: simulator, phase portraits, bifurcation, RTD, parameter sweeps, sensitivity, Pareto, fault monitoring, validation, latent space

**ETA:** Phase 5 completion by 2026-03-08

---

## Installation

```bash
pip install -e ".[dev]"
```

---

## Example Usage

### Batch Reactor

```python
from reactor_twin.reactors.batch import BatchReactor
from reactor_twin.reactors.kinetics.arrhenius import ArrheniusKinetics

kinetics = ArrheniusKinetics(...)
reactor = BatchReactor(
    name="batch",
    num_species=2,
    params={"V": 10.0, "T": 350.0, "C_initial": [1.0, 0.0]},
    kinetics=kinetics,
    constant_volume=True,
    isothermal=True,
)
```

### PFR with Method of Lines

```python
from reactor_twin.reactors.pfr import PlugFlowReactor

reactor = PlugFlowReactor(
    name="pfr",
    num_species=2,
    params={"L": 5.0, "u": 0.1, "D": 0.001, "C_in": [1.0, 0.0], "T": 350.0},
    kinetics=kinetics,
    num_cells=100,  # Spatial discretization
)

# Get axial profiles
z_positions, C_profiles = reactor.get_axial_profile(y)
```

### Michaelis-Menten Kinetics

```python
from reactor_twin.reactors.kinetics.michaelis_menten import MichaelisMentenKinetics

kinetics = MichaelisMentenKinetics(
    name="enzyme",
    num_reactions=1,
    params={
        "V_max": [0.5],
        "K_m": [0.1],
        "substrate_indices": [0],
        "stoich": [[-1, 1]],
    },
)
```

### Consecutive Reactions Benchmark

```python
from reactor_twin.reactors.systems.consecutive import (
    create_consecutive_cstr,
    compute_optimal_residence_time,
    compute_steady_state_concentrations,
)

reactor = create_consecutive_cstr()

# Compute optimal residence time
k1, k2 = 1000.0, 500.0
tau_opt = compute_optimal_residence_time(k1, k2)

# Analytical steady state
C_A_ss, C_B_ss, C_C_ss = compute_steady_state_concentrations(
    C_A_in=2.0, k1=k1, k2=k2, tau=tau_opt
)
```

---

## Contributors

- Tubhyam Karthikeyan (takarthikeyan25@gmail.com)
- Claude Sonnet 4.5 (noreply@anthropic.com)

---

## License

MIT License - see LICENSE file for details.
