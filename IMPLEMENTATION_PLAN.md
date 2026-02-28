# ReactorTwin: Comprehensive Implementation Plan

## Physics-Constrained Neural Differential Equations for Chemical Reactor Digital Twins

**Author:** Tubhyam Karthikeyan (ICT Mumbai, Computational Chemistry & ML)
**Date:** 2026-02-27
**Version:** 2.0 -- Deep Research Edition

---

## Table of Contents

1. [Vision & Scope](#1-vision--scope)
2. [Core Architecture](#2-core-architecture)
3. [Component Deep Dive](#3-component-deep-dive)
4. [Reactor Library](#4-reactor-library)
5. [Physics Enforcement](#5-physics-enforcement)
6. [Digital Twin Features](#6-digital-twin-features)
7. [Visualization & Dashboard](#7-visualization--dashboard)
8. [Testing & Validation Strategy](#8-testing--validation-strategy)
9. [Phased Implementation Plan](#9-phased-implementation-plan)
10. [Tech Stack & Dependencies](#10-tech-stack--dependencies)
11. [Benchmarks & Success Metrics](#11-benchmarks--success-metrics)
12. [References](#12-references)

---

## 1. Vision & Scope

### 1.1 What ReactorTwin Is

ReactorTwin is a Python library that creates fast, physics-guaranteed surrogate models of chemical reactors using Neural Differential Equations. It is not just a Neural ODE wrapper -- it is a complete digital twin framework that combines:

- **Multiple Neural DE families** (Neural ODEs, Latent Neural ODEs, Augmented Neural ODEs, Neural SDEs, Neural CDEs) unified under a single API
- **Hard physics constraints** enforced architecturally (not just via loss penalties) -- mass conservation, energy conservation, thermodynamic consistency, positivity of concentrations, stoichiometric constraints
- **A reactor library** spanning CSTRs, batch reactors, PFRs, semi-batch reactors, and multi-phase reactors with plug-in reaction kinetics (Arrhenius, Langmuir-Hinshelwood, Michaelis-Menten)
- **Real-time digital twin capabilities** -- state estimation via Extended Kalman Filter fused with Neural ODE predictions, fault detection, anomaly diagnosis, predictive maintenance, what-if scenario simulation, and Model Predictive Control
- **An interactive visualization dashboard** with phase portraits, bifurcation diagrams, RTD curves, sensitivity heatmaps, Pareto fronts, and 3D reactor cross-sections
- **Production-grade software engineering** -- typed Python, plugin architecture for extensibility, property-based testing, CI/CD, Docker deployment, FastAPI serving

### 1.2 Why It Matters

**The field is active but fragmented.** Papers exist (GRxnODE, ChemNODE, Stiff-PINN, Phy-ChemNODE, SPIN-ODE, FMEnets), but each solves one piece. No single open-source project ties Neural DEs + hard physics constraints + multiple reactor types + digital twin capabilities + interactive deployment together.

**Industry demand is real.** Shell demonstrated 10^6-10^8x speedup using PINNs for reactor surrogate modeling via NVIDIA PhysicsNeMo. Digital twins for chemical processes are a multi-billion-dollar industrial research area. Aspen Plus and gPROMS are proprietary and expensive; machine learning surrogates are the clear future direction, with recent work showing 93.8% reduction in simulation time while maintaining high predictive accuracy.

**Academic novelty is achievable.** The foundation model paper (Wang et al., 2024/2025) used Reptile meta-learning with TensorFlow and discrete MLPs. ReactorTwin uses continuous-time Neural DEs with hard conservation guarantees -- a genuine architectural improvement. The combination of Port-Hamiltonian structure with Neural ODEs for open reactor systems has not been demonstrated in the chemical engineering literature.

**Portfolio impact is exceptional.** This project sits at the intersection of scientific ML, chemical engineering, and software engineering, demonstrating depth in all three. The bifurcation diagrams, phase portraits, and real-time dashboard create immediate visual impact.

### 1.3 Scope Boundaries

**In scope:**
- Lumped-parameter reactor models (CSTRs, batch, semi-batch)
- 1D distributed-parameter models (PFR via method of lines)
- Single-phase and simple two-phase (gas-liquid) reactors
- Homogeneous and heterogeneous catalysis kinetics
- Population balance equations for crystallization (simplified 1D)
- Real-time state estimation and control
- Interactive web dashboard

**Out of scope (future work):**
- Full 3D CFD coupling
- Multi-component thermodynamic property prediction (use Cantera/CoolProp)
- Quantum chemistry integration
- Hardware-in-the-loop control

---

## 2. Core Architecture

### 2.1 System Architecture Overview

```
+-------------------------------------------------------------------+
|                        ReactorTwin Library                         |
+-------------------------------------------------------------------+
|                                                                    |
|  +------------------+  +------------------+  +------------------+  |
|  |   Neural DE      |  |   Reactor        |  |   Physics        |  |
|  |   Engine         |  |   Library        |  |   Enforcement    |  |
|  +------------------+  +------------------+  +------------------+  |
|  | NeuralODE        |  | CSTR             |  | MassBalance      |  |
|  | LatentNeuralODE  |  | BatchReactor     |  | EnergyBalance    |  |
|  | AugmentedNODE    |  | PFR              |  | Thermodynamic    |  |
|  | NeuralSDE        |  | SemiBatch        |  | Positivity       |  |
|  | NeuralCDE        |  | MultiPhase       |  | Stoichiometric   |  |
|  | HybridODE        |  | PopulationBal.   |  | PortHamiltonian  |  |
|  +--------+---------+  +--------+---------+  +--------+---------+  |
|           |                      |                      |          |
|  +--------v----------------------v----------------------v--------+ |
|  |                    Training Engine                             | |
|  +---------------------------------------------------------------+ |
|  | DataGenerator | Trainer | LossScheduler | CurriculumManager   | |
|  | AdjointBackprop | StiffnessHandler | MetaLearner              | |
|  +---------------------------+-----------------------------------+ |
|                              |                                     |
|  +---------------------------v-----------------------------------+ |
|  |                    Digital Twin Layer                          | |
|  +---------------------------------------------------------------+ |
|  | StateEstimator (EKF+NODE) | FaultDetector | AnomalyDiagnosis  | |
|  | ScenarioSimulator | OptimalController (MPC) | OnlineAdapter   | |
|  +---------------------------+-----------------------------------+ |
|                              |                                     |
|  +---------------------------v-----------------------------------+ |
|  |              Visualization & Deployment                        | |
|  +---------------------------------------------------------------+ |
|  | StreamlitDashboard | FastAPI Server | ONNX Export | Docker     | |
|  +---------------------------------------------------------------+ |
+-------------------------------------------------------------------+
```

### 2.2 Directory Structure

```
reactor-twin/
|-- README.md
|-- LICENSE (MIT)
|-- IMPLEMENTATION_PLAN.md
|-- pyproject.toml
|-- src/
|   |-- reactor_twin/
|   |   |-- __init__.py
|   |   |-- core/                          # Neural DE Engine
|   |   |   |-- __init__.py
|   |   |   |-- base.py                    # AbstractNeuralDE base class
|   |   |   |-- neural_ode.py              # Standard Neural ODE
|   |   |   |-- latent_neural_ode.py       # Encoder -> Latent ODE -> Decoder
|   |   |   |-- augmented_neural_ode.py    # Extra dims for expressivity
|   |   |   |-- neural_sde.py              # Stochastic DE for uncertainty
|   |   |   |-- neural_cde.py              # Controlled DE for irregular data
|   |   |   |-- hybrid_ode.py              # Known physics + learned correction
|   |   |   |-- ode_func.py                # ODE right-hand-side networks
|   |   |   |-- solvers.py                 # Solver backend abstraction
|   |   |   |-- adjoint.py                 # Adjoint sensitivity utilities
|   |   |   |-- stiffness.py               # Stiffness detection & handling
|   |   |-- physics/                        # Physics Enforcement
|   |   |   |-- __init__.py
|   |   |   |-- constraints.py             # Hard constraint projections
|   |   |   |-- mass_balance.py            # Mass/mole conservation
|   |   |   |-- energy_balance.py          # Energy conservation
|   |   |   |-- thermodynamics.py          # Gibbs, entropy, equilibrium
|   |   |   |-- positivity.py              # Non-negative concentrations
|   |   |   |-- stoichiometry.py           # Stoichiometric consistency
|   |   |   |-- port_hamiltonian.py        # Port-Hamiltonian structure
|   |   |   |-- generic.py                 # GENERIC framework
|   |   |-- reactors/                       # Reactor Library
|   |   |   |-- __init__.py
|   |   |   |-- base.py                    # AbstractReactor interface
|   |   |   |-- cstr.py                    # CSTR equations & configs
|   |   |   |-- batch.py                   # Batch reactor
|   |   |   |-- semi_batch.py              # Semi-batch reactor
|   |   |   |-- pfr.py                     # Plug flow reactor (MOL)
|   |   |   |-- multi_phase.py             # Gas-liquid reactor
|   |   |   |-- population_balance.py      # PBE for crystallization
|   |   |   |-- kinetics/                  # Reaction kinetics plugins
|   |   |   |   |-- __init__.py
|   |   |   |   |-- base.py               # AbstractKinetics
|   |   |   |   |-- arrhenius.py           # Arrhenius rate law
|   |   |   |   |-- langmuir_hinshelwood.py # Heterogeneous catalysis
|   |   |   |   |-- michaelis_menten.py    # Enzyme kinetics
|   |   |   |   |-- power_law.py           # Power law kinetics
|   |   |   |   |-- reversible.py          # Equilibrium-limited reactions
|   |   |   |-- systems/                   # Pre-built reaction systems
|   |   |   |   |-- __init__.py
|   |   |   |   |-- exothermic_ab.py       # A -> B exothermic
|   |   |   |   |-- van_de_vusse.py        # A->B->C, 2A->D
|   |   |   |   |-- consecutive_abc.py     # A -> B -> C
|   |   |   |   |-- parallel_competing.py  # A -> B, A -> C
|   |   |   |   |-- reversible_ab.py       # A <-> B
|   |   |   |   |-- series_parallel.py     # Complex network
|   |   |   |   |-- bioreactor.py          # Monod/enzyme kinetics
|   |   |-- training/                       # Training Engine
|   |   |   |-- __init__.py
|   |   |   |-- trainer.py                 # Main training loop
|   |   |   |-- losses.py                  # Multi-objective loss functions
|   |   |   |-- data_generator.py          # ODE-based data generation
|   |   |   |-- schedulers.py              # Loss weight scheduling
|   |   |   |-- curriculum.py              # Curriculum learning manager
|   |   |   |-- meta_learning.py           # MAML/Reptile for adaptation
|   |   |   |-- stiff_training.py          # Stiffness-aware training
|   |   |-- digital_twin/                   # Digital Twin Layer
|   |   |   |-- __init__.py
|   |   |   |-- state_estimator.py         # EKF + Neural ODE fusion
|   |   |   |-- fault_detector.py          # Fault detection & isolation
|   |   |   |-- anomaly_diagnosis.py       # VAE-based anomaly scoring
|   |   |   |-- scenario_simulator.py      # What-if analysis engine
|   |   |   |-- optimal_control.py         # MPC with Neural ODE plant
|   |   |   |-- online_adapter.py          # Continual learning from data
|   |   |   |-- transfer_learning.py       # Cross-reactor adaptation
|   |   |-- dashboard/                      # Visualization & Dashboard
|   |   |   |-- __init__.py
|   |   |   |-- app.py                     # Streamlit main app
|   |   |   |-- pages/
|   |   |   |   |-- reactor_sim.py         # Real-time simulation
|   |   |   |   |-- phase_portrait.py      # Phase space visualization
|   |   |   |   |-- bifurcation.py         # Bifurcation diagrams
|   |   |   |   |-- rtd_analysis.py        # Residence time distribution
|   |   |   |   |-- parameter_sweep.py     # Parameter sweep heatmaps
|   |   |   |   |-- sensitivity.py         # Sensitivity analysis
|   |   |   |   |-- pareto.py              # Multi-objective optimization
|   |   |   |   |-- fault_monitor.py       # Real-time fault monitoring
|   |   |   |   |-- comparison.py          # Neural DE vs ground truth
|   |   |   |   |-- model_explorer.py      # Latent space visualization
|   |   |-- api/                            # API & Deployment
|   |   |   |-- __init__.py
|   |   |   |-- server.py                  # FastAPI model serving
|   |   |   |-- schemas.py                 # Pydantic request/response
|   |   |   |-- websocket.py               # Real-time streaming
|   |   |-- utils/
|   |   |   |-- __init__.py
|   |   |   |-- visualization.py           # Plotting utilities
|   |   |   |-- metrics.py                 # RMSE, conservation error, etc.
|   |   |   |-- config.py                  # Configuration management
|   |   |   |-- registry.py               # Plugin registry system
|   |   |   |-- numerical.py              # Numerical utilities
|-- tests/
|   |-- conftest.py                        # Shared fixtures
|   |-- test_core/
|   |   |-- test_neural_ode.py
|   |   |-- test_latent_ode.py
|   |   |-- test_augmented_ode.py
|   |   |-- test_neural_sde.py
|   |   |-- test_hybrid_ode.py
|   |   |-- test_solvers.py
|   |-- test_physics/
|   |   |-- test_mass_balance.py
|   |   |-- test_energy_balance.py
|   |   |-- test_thermodynamics.py
|   |   |-- test_positivity.py
|   |   |-- test_port_hamiltonian.py
|   |-- test_reactors/
|   |   |-- test_cstr.py
|   |   |-- test_batch.py
|   |   |-- test_pfr.py
|   |   |-- test_kinetics.py
|   |-- test_training/
|   |   |-- test_trainer.py
|   |   |-- test_data_generator.py
|   |   |-- test_losses.py
|   |   |-- test_meta_learning.py
|   |-- test_digital_twin/
|   |   |-- test_state_estimator.py
|   |   |-- test_fault_detector.py
|   |   |-- test_optimal_control.py
|   |-- test_integration/
|   |   |-- test_end_to_end_cstr.py
|   |   |-- test_end_to_end_batch.py
|   |   |-- test_conservation_laws.py     # Property-based tests
|   |-- benchmarks/
|   |   |-- bench_solver_speed.py
|   |   |-- bench_training_convergence.py
|   |   |-- bench_inference_throughput.py
|-- examples/
|   |-- 01_cstr_exothermic.py
|   |-- 02_cstr_van_de_vusse.py
|   |-- 03_batch_consecutive.py
|   |-- 04_batch_parameter_estimation.py
|   |-- 05_pfr_tubular.py
|   |-- 06_bifurcation_analysis.py
|   |-- 07_hard_vs_soft_constraints.py
|   |-- 08_latent_neural_ode.py
|   |-- 09_neural_sde_uncertainty.py
|   |-- 10_meta_learning_adaptation.py
|   |-- 11_state_estimation_ekf.py
|   |-- 12_fault_detection.py
|   |-- 13_mpc_control.py
|   |-- 14_multi_objective_optimization.py
|   |-- 15_population_balance.py
|-- notebooks/
|   |-- tutorial_01_basics.ipynb
|   |-- tutorial_02_physics_constraints.ipynb
|   |-- tutorial_03_custom_reactions.ipynb
|   |-- tutorial_04_digital_twin.ipynb
|   |-- tutorial_05_advanced_neural_des.ipynb
|   |-- paper_figures.ipynb                # Reproducible figures for paper
|-- configs/
|   |-- cstr_exothermic.yaml
|   |-- cstr_van_de_vusse.yaml
|   |-- batch_consecutive.yaml
|   |-- pfr_tubular.yaml
|   |-- meta_learning.yaml
|-- Dockerfile
|-- docker-compose.yml
|-- .github/
|   |-- workflows/
|   |   |-- ci.yml
|   |   |-- publish.yml
```

### 2.3 Class Hierarchy

```
AbstractNeuralDE (core/base.py)
  |-- NeuralODE (core/neural_ode.py)
  |-- LatentNeuralODE (core/latent_neural_ode.py)
  |-- AugmentedNeuralODE (core/augmented_neural_ode.py)
  |-- NeuralSDE (core/neural_sde.py)
  |-- NeuralCDE (core/neural_cde.py)
  |-- HybridODE (core/hybrid_ode.py)

AbstractODEFunc (core/ode_func.py)
  |-- MLPODEFunc          -- Standard MLP right-hand side
  |-- ResNetODEFunc        -- Residual connections
  |-- HybridODEFunc        -- Physics + neural correction
  |-- PortHamiltonianFunc  -- Structure-preserving

AbstractReactor (reactors/base.py)
  |-- CSTRReactor (reactors/cstr.py)
  |-- BatchReactor (reactors/batch.py)
  |-- SemiBatchReactor (reactors/semi_batch.py)
  |-- PFRReactor (reactors/pfr.py)
  |-- MultiPhaseReactor (reactors/multi_phase.py)
  |-- PopulationBalanceReactor (reactors/population_balance.py)

AbstractKinetics (reactors/kinetics/base.py)
  |-- ArrheniusKinetics
  |-- LangmuirHinshelwoodKinetics
  |-- MichaelisMentenKinetics
  |-- PowerLawKinetics
  |-- ReversibleKinetics

AbstractConstraint (physics/constraints.py)
  |-- MassBalanceConstraint
  |-- EnergyBalanceConstraint
  |-- ThermodynamicConstraint
  |-- PositivityConstraint
  |-- StoichiometricConstraint
  |-- PortHamiltonianConstraint

AbstractDigitalTwin (digital_twin/)
  |-- StateEstimator
  |-- FaultDetector
  |-- AnomalyDiagnosis
  |-- ScenarioSimulator
  |-- OptimalController
  |-- OnlineAdapter
```

### 2.4 Data Flow

```
                   Operating Conditions u(t)
                          |
                          v
+---------------------------------------------------+
|                  Data Generation                    |
|  scipy.solve_ivp(reactor.ode_rhs, tspan, y0)      |
|  -> trajectories: (N_traj, N_time, N_state)        |
+-------------------------+-------------------------+
                          |
                          v
+---------------------------------------------------+
|                  Training Pipeline                  |
|                                                     |
|  1. Encode (Latent ODE only):                      |
|     z_0 = encoder(observations)                    |
|                                                     |
|  2. Integrate:                                      |
|     z(t) = odeint(f_theta, z_0, t_span)            |
|     where f_theta = physics(z) + neural_net(z,t,u) |
|                                                     |
|  3. Apply Hard Constraints:                         |
|     z_constrained = project(z, constraint_manifold) |
|                                                     |
|  4. Decode (Latent ODE only):                      |
|     x_pred = decoder(z_constrained)                |
|                                                     |
|  5. Compute Loss:                                   |
|     L = w_d*L_data + w_p*L_physics + w_c*L_constr  |
|       + w_t*L_thermo + w_s*L_stability              |
|                                                     |
|  6. Backpropagate:                                  |
|     adjoint method (O(1) memory) or                 |
|     discretize-then-optimize (higher accuracy)      |
+-------------------------+-------------------------+
                          |
                          v
+---------------------------------------------------+
|              Digital Twin Runtime                    |
|                                                     |
|  Trained Neural DE model serves as:                 |
|  - Fast simulator (1000x+ speedup)                 |
|  - Plant model in MPC controller                   |
|  - State estimator (fused with EKF)                |
|  - Anomaly detection baseline                      |
|  - What-if scenario engine                         |
+---------------------------------------------------+
```

### 2.5 Plugin Architecture

ReactorTwin uses a registry-based plugin system inspired by DeepXDE's loosely-coupled design. Users can register custom reactors, kinetics, constraints, and Neural DE variants without modifying library source code.

```python
from reactor_twin.utils.registry import Registry

# Global registries
REACTOR_REGISTRY = Registry("reactors")
KINETICS_REGISTRY = Registry("kinetics")
CONSTRAINT_REGISTRY = Registry("constraints")
NEURAL_DE_REGISTRY = Registry("neural_des")
SOLVER_REGISTRY = Registry("solvers")

# Register a custom reactor
@REACTOR_REGISTRY.register("my_reactor")
class MyCustomReactor(AbstractReactor):
    ...

# Register custom kinetics
@KINETICS_REGISTRY.register("custom_langmuir")
class CustomLangmuirKinetics(AbstractKinetics):
    ...

# Use in configuration
config = {
    "reactor": "my_reactor",
    "kinetics": "custom_langmuir",
    "neural_de": "latent_neural_ode",
    "constraints": ["mass_balance", "positivity"],
}
twin = ReactorTwin.from_config(config)
```

---

## 3. Component Deep Dive

### 3.1 Neural ODE Engine

#### 3.1.1 Standard Neural ODE

The foundational model (Chen et al., 2018). Parameterizes the derivative `dz/dt = f_theta(z, t, u)` where `f_theta` is a neural network, and solves the IVP with a black-box ODE integrator. The adjoint method enables O(1) memory backpropagation.

**Implementation approach:**

```python
class NeuralODE(AbstractNeuralDE):
    """Standard Neural ODE for reactor dynamics."""

    def __init__(
        self,
        ode_func: AbstractODEFunc,
        solver: str = "dopri5",
        atol: float = 1e-6,
        rtol: float = 1e-3,
        adjoint: bool = True,
        solver_options: dict | None = None,
    ):
        super().__init__()
        self.ode_func = ode_func
        self.solver = solver
        self.atol = atol
        self.rtol = rtol
        self._integrate = odeint_adjoint if adjoint else odeint

    def forward(
        self,
        z0: torch.Tensor,        # (batch, state_dim)
        t_span: torch.Tensor,    # (num_timesteps,)
        u: torch.Tensor | None,  # (batch, control_dim) operating conditions
    ) -> torch.Tensor:            # (num_timesteps, batch, state_dim)
        # Wrap ODE func to include control inputs
        func = lambda t, z: self.ode_func(t, z, u)
        trajectory = self._integrate(
            func, z0, t_span,
            method=self.solver,
            atol=self.atol,
            rtol=self.rtol,
        )
        return trajectory
```

**Key design decisions:**
- **Softplus activation** (not ReLU): Smooth derivatives are critical for ODE solvers. ReLU's discontinuous gradient at zero causes adaptive step-size solvers to waste function evaluations trying to resolve the kink. Softplus provides C-infinity smoothness. We also support Swish/SiLU and ELU as alternatives.
- **Residual connections in the ODE function**: Help gradient flow and reduce the number of function evaluations (NFEs) needed by the solver.
- **Control inputs as conditioning**: Operating conditions `u` are concatenated at every layer (FiLM-style conditioning), not just the input layer. This allows the network to modulate its behavior at every depth.

#### 3.1.2 Latent Neural ODE

Introduced by Rubanova et al. (2019). For high-dimensional or partially-observed systems, we first encode observations into a latent space, evolve the latent state with a Neural ODE, then decode back to observation space. This is critical for:
- Systems with many species (reduce 50+ dimensions to 5-10 latent dims)
- Partially observed reactors (only temperature measured, not all concentrations)
- Irregular time series from real sensors

**Architecture:**

```
Observations x(t_1), ..., x(t_N)
        |
        v
   ODE-RNN Encoder
   (processes observations in reverse time)
        |
        v
   z_0 ~ q(z_0 | x)   [variational posterior]
        |
        v
   Neural ODE:  dz/dt = f_theta(z, t)
        |
        v
   z(t_1), ..., z(t_M)  [latent trajectory]
        |
        v
   Decoder: x_hat = g_phi(z)
        |
        v
   Reconstructed observations
```

**Why this matters for reactors:**
- Phy-ChemNODE (2025) demonstrated that autoencoders + Neural ODE can handle stiff hydrocarbon kinetics by reducing dimensionality before integration, turning a 50-species stiff system into a 5-dimensional non-stiff latent system.
- The variational formulation naturally provides uncertainty estimates (the posterior variance of z_0).

**Implementation approach:**

```python
class LatentNeuralODE(AbstractNeuralDE):
    """Latent Neural ODE with recognition network."""

    def __init__(
        self,
        obs_dim: int,
        latent_dim: int,
        hidden_dim: int = 128,
        encoder_type: str = "ode_rnn",  # or "gru", "transformer"
        ode_func: AbstractODEFunc | None = None,
        variational: bool = True,
    ):
        super().__init__()
        self.encoder = ODERNNEncoder(obs_dim, latent_dim, hidden_dim)
        self.ode_func = ode_func or MLPODEFunc(latent_dim, hidden_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, obs_dim),
        )
        self.variational = variational
        if variational:
            self.mean_head = nn.Linear(hidden_dim, latent_dim)
            self.logvar_head = nn.Linear(hidden_dim, latent_dim)
```

#### 3.1.3 Augmented Neural ODE

Dupont et al. (2019) showed that standard Neural ODEs preserve input topology -- they cannot learn functions that require the trajectories to cross in state space. Augmented Neural ODEs append extra dimensions `a` to the state, evolving `d[z, a]/dt = f_theta([z, a], t)` where `a` is initialized to zero. The extra dimensions provide "room" for trajectories to avoid crossing.

**Why this matters for reactors:**
- Complex reaction networks can produce dynamics where trajectories in concentration space appear to cross (due to different operating conditions).
- Augmenting with 2-4 extra dimensions empirically reduces NFEs by 30-50% and improves accuracy.
- ANODE is essentially free -- just add extra output dimensions to the ODE function.

**Implementation:** Wraps a standard NeuralODE but pads the initial state with zeros and truncates the output.

#### 3.1.4 Neural SDE (Stochastic Differential Equations)

For uncertainty quantification, we model stochastic dynamics:

```
dz = f_theta(z, t) dt + g_phi(z, t) dW
```

where `f_theta` is the drift (deterministic part), `g_phi` is the diffusion (noise), and `dW` is a Wiener process. This naturally captures:
- **Aleatoric uncertainty**: Inherent randomness in reactor operation (feed fluctuations, catalyst deactivation)
- **Epistemic uncertainty**: Model uncertainty due to limited training data

**Key algorithms:**
- Euler-Maruyama integration for forward pass
- Stochastic adjoint method for backpropagation
- The diffusion function `g_phi` can be diagonal (each state has independent noise), low-rank (correlated noise), or scalar (same noise scale for all states)

**Implementation uses** `torchsde` library with a custom wrapper:

```python
class NeuralSDE(AbstractNeuralDE):
    """Neural SDE for uncertainty-aware reactor modeling."""

    def __init__(
        self,
        drift_func: AbstractODEFunc,
        diffusion_func: nn.Module,
        noise_type: str = "diagonal",  # "diagonal", "scalar", "general"
        sde_type: str = "ito",         # "ito" or "stratonovich"
    ):
        ...

    def forward(self, z0, t_span, u=None, n_samples: int = 100):
        """Returns (n_samples, n_timesteps, batch, state_dim) with uncertainty."""
        ...
```

#### 3.1.5 Neural CDE (Controlled Differential Equations)

Kidger et al. (2020). For irregularly-sampled reactor data (sensor measurements at non-uniform intervals), Neural CDEs are the natural formulation:

```
dz/dt = f_theta(z) * dX/dt
```

where `X(t)` is a continuous path constructed from the irregular observations via natural cubic spline interpolation. The control path `X` drives the evolution, making Neural CDEs robust to missing data and irregular sampling.

**Why this matters for reactors:**
- Real sensor data is often irregularly sampled (different sensors at different rates)
- Lab measurements may be sparse (hourly samples for some species, continuous for temperature)
- Log-NCDEs (Walker et al., 2024) further improve efficiency using the log-ODE method from rough path theory

**Implementation uses** `torchcde` library:

```python
class NeuralCDE(AbstractNeuralDE):
    """Neural CDE for irregular time series from reactor sensors."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        interpolation: str = "cubic",  # "cubic", "linear", "rectilinear"
    ):
        ...
```

#### 3.1.6 Hybrid Physics + Neural ODE

The key architectural innovation of ReactorTwin. Rather than purely black-box or purely physics loss, we decompose the dynamics:

```
dz/dt = f_known(z, t, u, params) + f_neural(z, t, u)
```

- `f_known`: Analytically computed from known reactor equations (flow terms, known kinetics, heat transfer)
- `f_neural`: Learned correction for unknown/uncertain terms (unknown kinetics, fouling, catalyst deactivation, unmodeled mixing effects)

**This hybrid approach offers:**
- Much better data efficiency (the network only learns the residual, not the entire dynamics)
- Physical interpretability (known terms are explicit)
- Better extrapolation (known physics dominates outside training distribution)
- Faster convergence (network starts from zero correction)

```python
class HybridODEFunc(AbstractODEFunc):
    """Hybrid physics + neural correction ODE function."""

    def __init__(
        self,
        reactor: AbstractReactor,
        neural_correction: nn.Module,
        correction_scale: float = 0.1,  # Limit neural correction magnitude
    ):
        super().__init__()
        self.reactor = reactor
        self.neural_correction = neural_correction
        self.correction_scale = correction_scale

    def forward(self, t: torch.Tensor, z: torch.Tensor, u: torch.Tensor):
        f_physics = self.reactor.known_physics(z, t, u)
        f_neural = self.neural_correction(t, z, u)
        # Scale correction to prevent it from dominating physics
        f_neural = self.correction_scale * torch.tanh(f_neural)
        return f_physics + f_neural
```

### 3.2 ODE Function Networks

The ODE right-hand side `f_theta(z, t, u)` can be implemented with several architectures:

#### 3.2.1 MLPODEFunc

Standard MLP with skip connections. Input: `[z, t, u]` concatenated. Output: `dz/dt`.

```python
class MLPODEFunc(AbstractODEFunc):
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 4,
        control_dim: int = 0,
        activation: str = "softplus",
        use_time: bool = True,
        layer_norm: bool = True,
    ):
        ...
        # Build layers with skip connections every 2 layers
        input_dim = state_dim + (1 if use_time else 0) + control_dim
        layers = []
        for i in range(num_layers):
            in_features = input_dim if i == 0 else hidden_dim
            if i > 0 and i % 2 == 0:
                in_features += input_dim  # Skip connection
            layers.append(nn.Linear(in_features, hidden_dim))
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(get_activation(activation))
        self.layers = nn.ModuleList(layers)
        self.output = nn.Linear(hidden_dim, state_dim)
```

#### 3.2.2 ResNetODEFunc

Deep residual ODE function for complex dynamics. Each block computes `z + delta_z` with bottleneck architecture.

#### 3.2.3 PortHamiltonianODEFunc

Structure-preserving ODE function based on Port-Hamiltonian formulation. This is a key differentiator for ReactorTwin. Chemical reactors are open thermodynamic systems with energy exchange at their ports (feed/product streams, heat exchanger). The Port-Hamiltonian formulation naturally captures this:

```
dz/dt = (J(z) - R(z)) * grad_H(z) + B(z) * u(t)
```

where:
- `H(z)` is the Hamiltonian (total stored energy) -- learned by a neural network
- `J(z)` is the skew-symmetric interconnection matrix (energy-conserving coupling)
- `R(z)` is the positive semi-definite dissipation matrix (entropy production)
- `B(z)` is the input matrix (port interactions)
- `u(t)` is the external input (feed conditions, coolant)

**Why this is powerful for reactors:**
- Energy conservation is guaranteed by construction (J is skew-symmetric)
- Dissipation is guaranteed non-negative (R is PSD) -- entropy always increases
- The second law of thermodynamics is built into the architecture
- Port-metriplectic extensions (Hernandez et al., 2023) further incorporate thermodynamic consistency

```python
class PortHamiltonianODEFunc(AbstractODEFunc):
    """Port-Hamiltonian ODE for thermodynamically consistent reactor modeling."""

    def __init__(self, state_dim: int, hidden_dim: int, port_dim: int):
        super().__init__()
        # Learn the Hamiltonian as a scalar function
        self.hamiltonian_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, 1),
        )
        # Learn J as skew-symmetric: J = A - A^T
        self.J_net = nn.Linear(state_dim, state_dim * state_dim)
        # Learn R as PSD: R = L @ L^T
        self.L_net = nn.Linear(state_dim, state_dim * state_dim)
        # Input matrix
        self.B_net = nn.Linear(state_dim, state_dim * port_dim)

    def forward(self, t, z, u):
        # Compute Hamiltonian gradient
        z_req = z.detach().requires_grad_(True)
        H = self.hamiltonian_net(z_req)
        grad_H = torch.autograd.grad(H.sum(), z_req, create_graph=True)[0]

        # Skew-symmetric J
        A = self.J_net(z).reshape(-1, z.shape[-1], z.shape[-1])
        J = A - A.transpose(-2, -1)

        # PSD R = L @ L^T
        L = self.L_net(z).reshape(-1, z.shape[-1], z.shape[-1])
        R = L @ L.transpose(-2, -1)

        # Port input
        B = self.B_net(z).reshape(-1, z.shape[-1], u.shape[-1])

        # Port-Hamiltonian dynamics
        dz_dt = torch.bmm(J - R, grad_H.unsqueeze(-1)).squeeze(-1) + torch.bmm(B, u.unsqueeze(-1)).squeeze(-1)
        return dz_dt
```

### 3.3 Solver Backend

ReactorTwin abstracts the ODE solver backend, supporting multiple libraries:

| Backend | Solvers | Strengths | When to Use |
|---------|---------|-----------|-------------|
| **torchdiffeq** | dopri5, bosh3, adams, euler, midpoint, rk4, implicit_adams | Most mature, adjoint support, well-tested | Default for most problems |
| **torchode** | dopri5, tsit5 | 4x faster batched solving, JIT-compatible, per-sample step sizes | Large batch training |
| **torchsde** | euler, milstein, srk | SDE integration | Neural SDE |
| **torchcde** | - | CDE with path interpolation | Irregular time series |

**Solver selection heuristic:**
1. Non-stiff systems: `dopri5` (explicit Runge-Kutta 4/5 with adaptive stepping)
2. Moderately stiff: `implicit_adams` or `tsit5` (Tsitouras 5th order)
3. Very stiff: `scipy_solver` wrapper with BDF or Radau (implicit methods)
4. Stiff Neural ODE training: Semi-implicit methods following the approach of Caldana & Hesthaven (2025), using time reparameterization to reduce stiffness before using explicit solvers

```python
class SolverFactory:
    """Factory for creating ODE solvers with appropriate settings."""

    @staticmethod
    def create(
        backend: str = "torchdiffeq",
        method: str = "dopri5",
        atol: float = 1e-6,
        rtol: float = 1e-3,
        adjoint: bool = True,
        stiffness_detection: bool = True,
    ) -> Solver:
        if backend == "torchdiffeq":
            return TorchdiffeqSolver(method, atol, rtol, adjoint)
        elif backend == "torchode":
            return TorchodeSolver(method, atol, rtol)
        elif backend == "torchsde":
            return TorchsdeSolver(method, atol, rtol)
        ...
```

### 3.4 Stiffness Handling

Chemical kinetics often involve timescales spanning 10+ orders of magnitude. ReactorTwin employs a multi-strategy approach:

**Strategy 1: Time Reparameterization**
Learn an adaptive time map `tau(t)` that stretches time in regions of fast dynamics and compresses it where dynamics are slow. The map is data-driven, induced by the adaptive time-stepping of an implicit solver on reference solutions (Caldana & Hesthaven, 2025). This produces a non-stiff system solvable with fast explicit schemes.

```python
class TimeReparameterization(nn.Module):
    """Learned time map to reduce stiffness."""

    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.time_map = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),  # Ensure monotonically increasing
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Map physical time to computational time."""
        return torch.cumsum(self.time_map(t.unsqueeze(-1)).squeeze(-1), dim=0)
```

**Strategy 2: QSSA Preprocessing**
For systems with known fast variables, apply Quasi-Steady-State Approximation before training. Set `dC_fast/dt = 0` and solve algebraically for fast species, reducing the dimension and stiffness of the ODE system. This follows the approach validated in Stiff-PINN (Ji & Deng, 2021).

**Strategy 3: Implicit Neural ODE Solvers**
Use single-step implicit methods (backward Euler, implicit midpoint, SDIRK) for the Neural ODE integration. Recent work (2024) demonstrated that implicit Neural ODE methods can learn stiff dynamics, enabling the model to take larger steps than classical explicit methods.

**Strategy 4: Multi-Scale Architecture**
Separate fast and slow dynamics into different neural networks, each integrated at appropriate timescales. Inspired by Multiscale PINNs for Stiff Chemical Kinetics (2023).

### 3.5 Adjoint Sensitivity Analysis

ReactorTwin supports two gradient computation strategies with different tradeoffs:

| Strategy | Memory | Speed | Accuracy | Use Case |
|----------|--------|-------|----------|----------|
| **Adjoint (optimize-then-discretize)** | O(1) -- constant | Slower (solves backward ODE) | Can have numerical errors for stiff systems | Large models, long time horizons |
| **Direct backprop (discretize-then-optimize)** | O(N_steps) -- stores all states | Faster | Exact gradients | Small models, debugging, stiff systems |

**When to use which:**
- Adjoint for production training with long trajectories (>1000 timesteps) and large models (>100K parameters)
- Direct backprop for debugging, short trajectories, and stiff systems where adjoint accuracy degrades
- Symplectic adjoint method (Zhuang et al., 2021) as a middle ground: exact gradients with O(1) memory using symplectic integrators

```python
class AdjointConfig:
    """Configuration for adjoint sensitivity computation."""

    method: str = "adjoint"  # "adjoint", "direct", "symplectic"
    adjoint_atol: float = 1e-6
    adjoint_rtol: float = 1e-3
    # Checkpointing for direct backprop
    checkpoint_segments: int | None = None  # Gradient checkpointing
```

---

## 4. Reactor Library

### 4.1 CSTR (Continuous Stirred-Tank Reactor)

**Governing equations (general form):**

```
dC_i/dt = (F/V)(C_i0 - C_i) + sum_j(nu_ij * r_j(C, T))

dT/dt = (F/V)(T_0 - T) + sum_j((-dH_j) * r_j(C, T)) / (rho * Cp)
        - UA/(V * rho * Cp) * (T - T_c)

dT_c/dt = (F_c/V_c)(T_ci - T_c) + UA/(V_c * rho_c * Cp_c) * (T - T_c)
```

where:
- `C_i`: concentration of species i [mol/L]
- `F`: volumetric flow rate [L/min]
- `V`: reactor volume [L]
- `C_i0`: feed concentration [mol/L]
- `nu_ij`: stoichiometric coefficient of species i in reaction j
- `r_j`: rate of reaction j [mol/(L*min)]
- `T`: reactor temperature [K]
- `T_0`: feed temperature [K]
- `dH_j`: heat of reaction j [J/mol]
- `rho`: density [g/L]
- `Cp`: heat capacity [J/(g*K)]
- `UA`: heat transfer coefficient [J/(min*K)]
- `T_c`: coolant temperature [K]

**Benchmark systems:**

1. **Exothermic irreversible A -> B** (Fogler, Bequette benchmark)
   - 3 state variables: [C_A, T, T_c]
   - Exhibits multiple steady states (S-curve bifurcation)
   - Ignition/extinction behavior
   - Parameters from MATLAB MPC toolbox

2. **Van de Vusse reaction** (A -> B -> C, 2A -> D)
   - 4 state variables: [C_A, C_B, C_C, C_D]
   - Non-minimum phase behavior (inverse response in C_B)
   - Classic nonlinear control benchmark
   - Selectivity optimization challenge

3. **Exothermic reversible A <-> B**
   - Temperature-dependent equilibrium constant
   - Tests thermodynamic consistency
   - K_eq = k_f/k_r = exp(-dG/RT)

4. **Series-parallel A -> B -> C, A -> D**
   - Different activation energies for competing pathways
   - Temperature programming to maximize desired product B
   - Tests selective constraint enforcement

5. **Bioreactor (Monod kinetics)**
   - dX/dt = (mu - D) * X, where mu = mu_max * S / (K_s + S)
   - dS/dt = D * (S_f - S) - mu * X / Y_xs
   - Exhibits washout and multiple steady states

### 4.2 Batch Reactor

**Governing equations:**

```
dC_i/dt = sum_j(nu_ij * r_j(C, T))

dT/dt = sum_j((-dH_j) * r_j(C, T)) / (rho * Cp)
        - UA/(V * rho * Cp) * (T - T_c)
```

No flow terms. Same kinetics as CSTR. Key differences:
- Time-varying profiles (no steady state)
- Optimal stopping time problems
- Ideal for kinetic parameter estimation (inverse problems)
- Simpler conservation laws (closed system: total mass constant)

**Benchmark systems:**

1. **Consecutive A -> B -> C**: Maximize intermediate B at optimal batch time
2. **Parallel competing A -> B, A -> C**: Different E_a values, temperature programming
3. **Esterification with equilibrium**: Fischer esterification, equilibrium-limited

### 4.3 Semi-Batch Reactor

**Governing equations:**

```
dC_i/dt = (F_in/V)(C_i_feed - C_i) + sum_j(nu_ij * r_j(C, T))

dV/dt = F_in   (volume changes as feed is added)
```

Combines batch and CSTR features. Feed addition rate F_in(t) is a time-varying control input. Critical for:
- Exothermic reactions where reagent is added slowly for safety
- Polymerization (monomer feed rate controls molecular weight distribution)
- Crystallization (anti-solvent addition rate controls crystal size)

### 4.4 PFR (Plug Flow Reactor)

**Governing equations (1D):**

```
dC_i/dz = -r_i(C, T) / u

dT/dz = sum_j((-dH_j) * r_j) / (u * rho * Cp) - 4U/(d * u * rho * Cp) * (T - T_w)
```

where `z` is the axial position and `u` is the linear velocity.

**Discretization via Method of Lines (MOL):**
- Discretize spatial domain into N cells: z_0, z_1, ..., z_N
- Approximate spatial derivatives with finite differences
- Transform PDE into system of N*n_species coupled ODEs
- Apply Neural ODE to the resulting large ODE system

**For non-ideal PFR (axial dispersion):**

```
dC_i/dt = D_ax * d^2C_i/dz^2 - u * dC_i/dz + r_i(C, T)
```

FMEnets (2025) demonstrated physics-informed neural networks for non-ideal PFR design with Navier-Stokes coupling.

### 4.5 Multi-Phase Reactor

**Gas-liquid reactor model:**

```
Liquid phase:
dC_i^L/dt = (F_L/V_L)(C_i0^L - C_i^L) + k_L*a*(C_i^* - C_i^L) + sum_j(nu_ij * r_j)

Gas phase:
dC_i^G/dt = (F_G/V_G)(C_i0^G - C_i^G) - k_L*a*(C_i^* - C_i^L) * (V_L/V_G)
```

where `k_L*a` is the volumetric mass transfer coefficient and `C_i^*` is the equilibrium interfacial concentration (Henry's law: `C_i^* = P_i / H_i`).

**Application:** Gas-liquid hydrogenation, oxidation, fermentation with oxygen transfer.

### 4.6 Population Balance Reactor

**For crystallization/polymerization, the state includes a particle size distribution:**

```
dn/dt + d(G*n)/dL = B(L) - D(L)

where:
n(L, t) = particle size distribution [#/m^4]
G = growth rate [m/s]
B = birth rate (nucleation, breakage) [#/(m^4*s)]
D = death rate (agglomeration) [#/(m^4*s)]
L = particle size [m]
```

**Discretization:** Method of moments (transform PBE into finite set of ODEs for moments mu_k) or fixed-pivot method for full distribution.

**Recent work** (2025) demonstrated physics-informed neural networks for population balance models, capturing the evolution of particle size distributions in response to nucleation, growth, aggregation, and breakage.

### 4.7 Reaction Kinetics Plugins

Each kinetics plugin implements the `AbstractKinetics` interface:

```python
class AbstractKinetics(ABC):
    """Interface for reaction kinetics models."""

    @abstractmethod
    def rate(self, C: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """Compute reaction rate(s) given concentrations and temperature."""
        ...

    @abstractmethod
    def rate_jacobian(self, C: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """Analytical Jacobian dr/dC for stiffness analysis."""
        ...

    @abstractmethod
    def equilibrium_constant(self, T: torch.Tensor) -> torch.Tensor | None:
        """Return K_eq(T) for reversible reactions, None for irreversible."""
        ...
```

**Arrhenius kinetics:**
```
r = k_0 * exp(-E_a / (R*T)) * prod(C_i^alpha_i)
```

**Langmuir-Hinshelwood kinetics (heterogeneous catalysis):**
```
r = k * K_A*C_A * K_B*C_B / (1 + K_A*C_A + K_B*C_B)^2
```

**Michaelis-Menten kinetics (enzyme/bioreactor):**
```
r = V_max * C_S / (K_m + C_S)
```

With optional inhibition: competitive, uncompetitive, noncompetitive, substrate inhibition.

**Reversible reactions:**
```
r_net = k_f * prod(C_reactants^alpha) - k_r * prod(C_products^beta)

Thermodynamic consistency: K_eq = k_f/k_r = exp(-dG_rxn / (R*T))
```

---

## 5. Physics Enforcement

### 5.1 Philosophy: Hard vs Soft Constraints

ReactorTwin provides both hard and soft constraint enforcement with a clear recommendation hierarchy:

| Constraint Type | Enforcement | Guarantee | Computational Cost |
|----------------|-------------|-----------|-------------------|
| **Hard architectural** | Built into network structure | Exact (machine precision) | Moderate (projection step) |
| **Hard projection** | Post-process network output | Exact after projection | Low (single linear solve) |
| **Soft loss penalty** | Added to training loss | Approximate (~10^-2 to 10^-4) | Low |
| **Soft augmented Lagrangian** | Adaptive penalty + multiplier | Better than soft (~10^-4 to 10^-6) | Moderate |

**Recommendation:** Use hard constraints for fundamental conservation laws (mass, energy) and soft constraints for weaker physical knowledge (approximate thermodynamic relations, empirical correlations).

### 5.2 Mass Balance Constraint

**Physical law:** In a closed system, total mass is conserved. In an open system (reactor with flow), the change in total mass equals mass in minus mass out.

**For CSTR with constant density:**
```
sum_i(M_i * dC_i/dt) = (F/V) * sum_i(M_i * (C_i0 - C_i)) + sum_i(M_i * sum_j(nu_ij * r_j))
```

The second term must satisfy: `sum_i(M_i * nu_ij) = 0` for each reaction j (atom balance).

**Hard constraint implementation:**

```python
class MassBalanceConstraint(AbstractConstraint):
    """Projects Neural ODE output onto mass-conserving manifold."""

    def __init__(
        self,
        molecular_weights: torch.Tensor,  # (n_species,)
        stoichiometry: torch.Tensor,       # (n_species, n_reactions)
    ):
        self.M = molecular_weights
        self.nu = stoichiometry
        # Verify atom balance: M^T @ nu = 0 for each reaction
        residual = self.M @ self.nu
        assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-10)

    def project(self, dC_dt: torch.Tensor, flow_terms: torch.Tensor) -> torch.Tensor:
        """Project concentration derivatives onto mass-conserving manifold.

        The neural network outputs unconstrained dC_dt.
        We decompose: dC_dt = flow_contribution + reaction_contribution
        The reaction contribution must lie in the column space of nu (stoichiometry).
        Project the neural output onto span(nu) to enforce this.
        """
        reaction_rates = dC_dt - flow_terms  # Remove known flow terms
        # Project onto column space of stoichiometry matrix
        # reaction_contribution = nu @ (nu^+ @ reaction_rates)
        # where nu^+ is the pseudoinverse
        nu_pinv = torch.linalg.pinv(self.nu)
        projected_rates = self.nu @ (nu_pinv @ reaction_rates.unsqueeze(-1)).squeeze(-1)
        return flow_terms + projected_rates
```

**Soft constraint (augmented Lagrangian):**

```python
def mass_balance_loss(dC_dt, flow_terms, nu, M):
    """Augmented Lagrangian for mass balance violation."""
    reaction_rates = dC_dt - flow_terms
    # Violation: reaction rates must be in column space of nu
    null_space_component = reaction_rates - nu @ torch.linalg.lstsq(nu, reaction_rates).solution
    violation = (M * null_space_component).sum(dim=-1)  # Weighted by molecular weight
    return violation.pow(2).mean()
```

### 5.3 Energy Balance Constraint

**Physical law:** First law of thermodynamics applied to the reactor:

```
rho * Cp * V * dT/dt = F * rho * Cp * (T_in - T) + V * sum_j((-dH_j) * r_j) - UA * (T - T_c)
```

**Hard constraint:** Given the neural network's predicted reaction rates r_j, the temperature derivative is fully determined by the energy balance. We do NOT let the neural network independently predict dT/dt -- instead, we compute it from the energy balance equation using the network's predicted rates.

```python
class EnergyBalanceConstraint(AbstractConstraint):
    """Enforces energy conservation by computing dT/dt from rates."""

    def apply(self, state, reaction_rates, params):
        T = state[..., self.T_idx]
        T_c = state[..., self.Tc_idx] if self.has_coolant else params.T_c

        # dT/dt is COMPUTED from energy balance, not predicted
        heat_generation = (params.dH_rxn * reaction_rates).sum(dim=-1)
        flow_cooling = (params.F / params.V) * (params.T_in - T)
        jacket_cooling = params.UA / (params.V * params.rho * params.Cp) * (T - T_c)

        dT_dt = flow_cooling + heat_generation / (params.rho * params.Cp) - jacket_cooling
        return dT_dt
```

### 5.4 Thermodynamic Consistency

**Requirements:**
1. **Gibbs free energy must decrease** (for irreversible reactions): `dG/dt <= 0`
2. **Reaction rates approach zero at equilibrium**: When `G_products = G_reactants`, net rate = 0
3. **Arrhenius temperature dependence**: Activation energy must be positive
4. **Detailed balance**: For reversible reactions, `K_eq = k_f/k_r = exp(-dG/RT)`

**Implementation using ThermoLearn approach (2025):**

```python
class ThermodynamicConstraint(AbstractConstraint):
    """Enforces thermodynamic consistency of reaction rates."""

    def gibbs_constraint(self, reaction_rates, concentrations, T, thermo_params):
        """Ensure dG/dt <= 0 (Gibbs free energy decreases)."""
        # Chemical potentials
        mu_i = thermo_params.mu_0 + R * T * torch.log(concentrations / thermo_params.C_ref)
        # Gibbs free energy change per reaction
        dG_rxn = (self.stoich_matrix.T @ mu_i.unsqueeze(-1)).squeeze(-1)
        # Reaction rates and dG must have opposite signs
        # r_j * dG_j <= 0 for each reaction j
        violation = torch.relu(reaction_rates * dG_rxn)
        return violation.sum(dim=-1).mean()

    def equilibrium_consistency(self, k_forward, k_reverse, T, dG_standard):
        """Enforce K_eq = k_f/k_r = exp(-dG/RT)."""
        K_eq_predicted = k_forward / (k_reverse + 1e-10)
        K_eq_thermodynamic = torch.exp(-dG_standard / (R * T))
        return F.mse_loss(
            torch.log(K_eq_predicted + 1e-10),
            torch.log(K_eq_thermodynamic + 1e-10)
        )
```

### 5.5 Positivity Constraints

**Physical requirement:** Concentrations cannot be negative. Temperature (in Kelvin) cannot be negative.

**Strategy 1: Output transformation (hard)**
```python
def enforce_positivity_exp(z_raw):
    """Map unconstrained output to positive values via softplus."""
    return F.softplus(z_raw)  # smooth approximation to max(0, x)
```

**Strategy 2: Logarithmic state transformation (hard)**
Train the Neural ODE in log-concentration space: `y_i = log(C_i)`. The ODE becomes:
```
dy_i/dt = (1/C_i) * dC_i/dt
```
Concentrations are always positive by construction: `C_i = exp(y_i)`.

**Strategy 3: Projection (hard)**
After each integration step, project: `C_i = max(C_i, epsilon)` where epsilon is a small positive number (e.g., 1e-15).

**Strategy 4: Barrier loss (soft)**
```python
def positivity_barrier(C, epsilon=1e-8):
    """Log-barrier penalty for near-zero concentrations."""
    return -torch.log(C + epsilon).mean()
```

**Recommendation:** Use log-concentration transformation (Strategy 2) as default. It provides exact positivity with minimal overhead and is differentiable everywhere.

### 5.6 Stoichiometric Constraints

**Physical requirement:** Reaction rates must respect stoichiometry. If the stoichiometric matrix is `nu` (n_species x n_reactions), then the species production rates must lie in the column space of `nu`:

```
dC/dt|_rxn = nu @ r
```

where `r` is the vector of reaction rates (n_reactions,).

**Hard constraint:** Rather than predicting `dC/dt` directly, the neural network predicts the reaction rate vector `r` (lower-dimensional), and we compute species rates as `nu @ r`. This automatically enforces stoichiometric consistency.

```python
class StoichiometricODEFunc(AbstractODEFunc):
    """ODE function that predicts reaction rates, not species rates."""

    def __init__(self, state_dim, n_reactions, stoich_matrix, neural_net):
        self.nu = stoich_matrix  # (n_species, n_reactions)
        self.rate_net = neural_net  # Outputs (batch, n_reactions)

    def forward(self, t, z, u):
        # Neural network predicts reaction rates (lower dim)
        rates = F.softplus(self.rate_net(t, z, u))  # Positive rates
        # Species production rates from stoichiometry
        dC_dt_rxn = (self.nu @ rates.unsqueeze(-1)).squeeze(-1)
        # Add flow terms
        dC_dt = flow_terms(z, u) + dC_dt_rxn
        return dC_dt
```

### 5.7 Port-Hamiltonian Structure

For the most rigorous thermodynamic enforcement, ReactorTwin provides a Port-Hamiltonian formulation following recent work on port-metriplectic neural networks (Hernandez et al., 2023) and stable Port-Hamiltonian neural networks (2025).

**The GENERIC framework** (General Equation for Non-Equilibrium Reversible-Irreversible Coupling) extends Port-Hamiltonian systems:

```
dz/dt = L(z) * dE/dz + M(z) * dS/dz
```

where:
- `E(z)` is total energy (conserved)
- `S(z)` is entropy (non-decreasing)
- `L(z)` is Poisson matrix (skew-symmetric, reversible coupling)
- `M(z)` is friction matrix (positive semi-definite, irreversible dissipation)

**Degeneracy conditions:** `L * dS/dz = 0` and `M * dE/dz = 0` ensure energy conservation and entropy production are decoupled.

```python
class GENERICODEFunc(AbstractODEFunc):
    """GENERIC framework for thermodynamically consistent reactor modeling."""

    def __init__(self, state_dim, hidden_dim):
        # Learn energy and entropy as scalar functions
        self.energy_net = InputConvexNN(state_dim, hidden_dim)  # Convex for stability
        self.entropy_net = ConcaveNN(state_dim, hidden_dim)     # Concave (2nd law)

        # Learn L (Poisson, skew-symmetric) and M (friction, PSD)
        self.L_net = SkewSymmetricNet(state_dim, hidden_dim)
        self.M_net = PSDNet(state_dim, hidden_dim)

    def forward(self, t, z, u=None):
        # Gradients of energy and entropy
        dE_dz = torch.autograd.grad(self.energy_net(z).sum(), z, create_graph=True)[0]
        dS_dz = torch.autograd.grad(self.entropy_net(z).sum(), z, create_graph=True)[0]

        L = self.L_net(z)  # Skew-symmetric
        M = self.M_net(z)  # PSD

        # GENERIC dynamics
        dz_dt = torch.bmm(L, dE_dz.unsqueeze(-1)).squeeze(-1) + \
                torch.bmm(M, dS_dz.unsqueeze(-1)).squeeze(-1)

        return dz_dt
```

### 5.8 Constraint Composition

Multiple constraints can be composed via a `ConstraintPipeline`:

```python
pipeline = ConstraintPipeline([
    StoichiometricConstraint(nu_matrix),     # Stoichiometry first
    MassBalanceConstraint(mol_weights, nu),   # Mass conservation
    EnergyBalanceConstraint(thermo_params),   # Energy conservation
    PositivityConstraint(method="log_transform"),  # Non-negative concentrations
    ThermodynamicConstraint(gibbs_params),    # Thermodynamic consistency
])

# Apply all constraints to Neural ODE output
dz_dt_constrained = pipeline(dz_dt_raw, state, params)
```

---

## 6. Digital Twin Features

### 6.1 Real-Time State Estimation (EKF + Neural ODE)

**Problem:** Sensors provide noisy, incomplete measurements. We need to estimate the full reactor state (all species concentrations, temperatures) from limited sensor data.

**Approach:** Extended Kalman Filter where the state prediction step uses the trained Neural ODE as the process model.

```
Prediction step:
  x_hat(k+1|k) = NeuralODE.integrate(x_hat(k|k), dt)
  F_k = d(NeuralODE)/dx |_{x=x_hat(k|k)}    [Jacobian via autodiff]
  P(k+1|k) = F_k @ P(k|k) @ F_k^T + Q       [State covariance prediction]

Update step:
  y_tilde = z_measured - H @ x_hat(k+1|k)     [Innovation]
  S = H @ P(k+1|k) @ H^T + R                  [Innovation covariance]
  K = P(k+1|k) @ H^T @ S^{-1}                 [Kalman gain]
  x_hat(k+1|k+1) = x_hat(k+1|k) + K @ y_tilde
  P(k+1|k+1) = (I - K @ H) @ P(k+1|k)
```

**Key advantage:** The Neural ODE Jacobian `F_k` is computed exactly via automatic differentiation -- no need for finite difference approximations. This makes the EKF more accurate than traditional approaches where the Jacobian is approximated.

```python
class NeuralODEStateEstimator:
    """EKF-based state estimator using Neural ODE as process model."""

    def __init__(
        self,
        neural_ode: AbstractNeuralDE,
        observation_matrix: torch.Tensor,  # H: maps state to measurements
        process_noise: torch.Tensor,       # Q: process noise covariance
        measurement_noise: torch.Tensor,   # R: measurement noise covariance
    ):
        self.model = neural_ode
        self.H = observation_matrix
        self.Q = process_noise
        self.R = measurement_noise

    def predict(self, x_hat, P, dt):
        """Prediction step using Neural ODE."""
        x_pred = self.model.integrate(x_hat, torch.tensor([0, dt]))[-1]

        # Compute Jacobian via autograd
        x_hat_req = x_hat.detach().requires_grad_(True)
        x_next = self.model.integrate(x_hat_req, torch.tensor([0, dt]))[-1]
        F = torch.autograd.functional.jacobian(
            lambda x: self.model.integrate(x, torch.tensor([0, dt]))[-1], x_hat
        )

        P_pred = F @ P @ F.T + self.Q
        return x_pred, P_pred

    def update(self, x_pred, P_pred, z_measured):
        """Update step with new measurement."""
        innovation = z_measured - self.H @ x_pred
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ torch.linalg.inv(S)
        x_updated = x_pred + K @ innovation
        P_updated = (torch.eye(len(x_pred)) - K @ self.H) @ P_pred
        return x_updated, P_updated
```

### 6.2 Fault Detection and Anomaly Diagnosis

**Architecture:** A Variational Autoencoder (VAE) trained on normal reactor operation data. Anomalies are detected when the reconstruction error or the latent space likelihood exceeds a threshold.

**Multi-level fault detection:**

1. **Level 1 -- Statistical monitoring:** Track reconstruction error `||x - x_hat||` over a sliding window. Alert when it exceeds `mu + 3*sigma`.

2. **Level 2 -- Residual analysis:** Compare Neural ODE prediction to actual measurements. Persistent residual indicates model mismatch (potential fault).

3. **Level 3 -- Fault isolation:** Use feature attribution (integrated gradients) to identify which state variables contribute most to the anomaly, localizing the fault.

4. **Level 4 -- Fault classification:** If labeled fault data is available, train a classifier on the latent residual vector to identify fault type (sensor failure, catalyst deactivation, fouling, feed disturbance).

```python
class FaultDetector:
    """Multi-level fault detection for reactor digital twin."""

    def __init__(
        self,
        neural_ode: AbstractNeuralDE,
        vae: VariationalAutoEncoder,
        nominal_stats: NominalStatistics,  # mu, sigma from normal operation
        thresholds: FaultThresholds,
    ):
        ...

    def detect(self, measurements: torch.Tensor, timestamp: float) -> FaultReport:
        """Run all detection levels on new measurement."""
        # Level 1: Statistical monitoring
        reconstruction = self.vae.reconstruct(measurements)
        recon_error = (measurements - reconstruction).norm()
        l1_alert = recon_error > self.thresholds.reconstruction

        # Level 2: Neural ODE residual
        predicted = self.neural_ode.predict(self.current_state, dt)
        residual = measurements - self.H @ predicted
        l2_alert = residual.norm() > self.thresholds.residual

        # Level 3: Fault isolation via integrated gradients
        if l1_alert or l2_alert:
            attributions = integrated_gradients(self.vae.encoder, measurements)
            fault_location = attributions.argmax()

        return FaultReport(
            timestamp=timestamp,
            level_1_alert=l1_alert,
            level_2_alert=l2_alert,
            residual=residual,
            fault_location=fault_location if (l1_alert or l2_alert) else None,
            confidence=self._compute_confidence(recon_error, residual),
        )
```

### 6.3 What-If Scenario Simulation

The trained Neural ODE enables instant evaluation of alternative operating scenarios:

```python
class ScenarioSimulator:
    """What-if analysis engine using trained Neural ODE."""

    def run_scenario(
        self,
        base_state: torch.Tensor,
        modified_params: dict,
        time_horizon: float,
    ) -> ScenarioResult:
        """Simulate reactor with modified operating conditions."""
        ...

    def parameter_sweep(
        self,
        param_name: str,
        param_range: torch.Tensor,
        objective: str,  # "conversion", "selectivity", "yield", "temperature"
    ) -> SweepResult:
        """Sweep a parameter and compute objective across range."""
        ...

    def find_steady_states(
        self,
        param_name: str,
        param_range: torch.Tensor,
    ) -> list[SteadyState]:
        """Find all steady states by continuation method."""
        # Uses Newton-Raphson on f_theta(z, u) = 0
        # with numerical continuation in the parameter
        ...
```

### 6.4 Model Predictive Control (MPC)

Use the Neural ODE as the plant model in an MPC controller. The Neural ODE's differentiability enables gradient-based optimization of the control trajectory.

**Formulation:**

```
min_{u(0), ..., u(N-1)} sum_{k=0}^{N} ||x(k) - x_ref||^2_Q + sum_{k=0}^{N-1} ||u(k)||^2_R

subject to:
  x(k+1) = NeuralODE(x(k), u(k), dt)   [dynamics]
  x_min <= x(k) <= x_max                [state constraints]
  u_min <= u(k) <= u_max                [input constraints]
  C_i(k) >= 0                           [positivity]
```

**Optimization:** Since the Neural ODE is differentiable, we can use gradient-based methods (L-BFGS, Adam) to solve the MPC problem, computing gradients of the cost with respect to the control sequence via the adjoint method. This is much faster than finite-difference-based NLP solvers for this specific problem class.

```python
class NeuralODEMPC:
    """Model Predictive Controller using Neural ODE as plant model."""

    def __init__(
        self,
        neural_ode: AbstractNeuralDE,
        horizon: int = 20,
        dt: float = 0.1,
        Q: torch.Tensor = None,  # State cost
        R: torch.Tensor = None,  # Control cost
        state_bounds: tuple | None = None,
        control_bounds: tuple | None = None,
    ):
        ...

    def solve(
        self,
        x_current: torch.Tensor,
        x_reference: torch.Tensor,
        u_init: torch.Tensor | None = None,
    ) -> ControlTrajectory:
        """Solve MPC problem for optimal control sequence."""
        u_sequence = u_init or torch.zeros(self.horizon, self.control_dim)
        u_sequence.requires_grad_(True)

        optimizer = torch.optim.LBFGS([u_sequence], lr=0.1)

        def closure():
            optimizer.zero_grad()
            x = x_current.clone()
            cost = torch.tensor(0.0)
            for k in range(self.horizon):
                u_k = torch.clamp(u_sequence[k], self.u_min, self.u_max)
                x = self.neural_ode.step(x, u_k, self.dt)
                cost += (x - x_reference) @ self.Q @ (x - x_reference)
                cost += u_k @ self.R @ u_k
            cost.backward()
            return cost

        optimizer.step(closure)
        return ControlTrajectory(u_sequence.detach(), ...)
```

### 6.5 Online Learning / Continuous Adaptation

As new measurement data arrives, the Neural ODE model can be updated incrementally without full retraining:

**Strategy 1: Fine-tuning with replay buffer**
Keep a buffer of recent measurements. Periodically fine-tune the model on a mix of buffer data + original training data (to prevent catastrophic forgetting).

**Strategy 2: MAML/Reptile meta-learned initialization**
Pre-train the model using meta-learning (MAML or Reptile) across many operating conditions. The resulting initialization enables rapid adaptation (5-10 gradient steps) to new conditions.

**Strategy 3: Online Bayesian updating**
Maintain a posterior over model parameters using recursive Bayesian estimation. Each new measurement tightens the posterior.

```python
class OnlineAdapter:
    """Continuous model adaptation from streaming sensor data."""

    def __init__(
        self,
        neural_ode: AbstractNeuralDE,
        replay_buffer_size: int = 10000,
        adaptation_lr: float = 1e-4,
        adaptation_steps: int = 5,
        forgetting_prevention: str = "replay",  # "replay", "ewc", "si"
    ):
        ...

    def adapt(self, new_observations: torch.Tensor, new_times: torch.Tensor):
        """Update model with new measurement data."""
        self.replay_buffer.add(new_observations, new_times)
        batch = self.replay_buffer.sample(batch_size=32)

        for _ in range(self.adaptation_steps):
            pred = self.neural_ode(batch.z0, batch.times, batch.u)
            loss = F.mse_loss(pred, batch.observations)
            if self.forgetting_prevention == "ewc":
                loss += self.ewc_penalty()
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
```

### 6.6 Transfer Learning Between Reactors

Following the foundation model approach (Wang et al., 2024/2025), ReactorTwin supports transfer learning across reactor types:

**Phase 1: Pre-train** a base Neural ODE on diverse reactor data (multiple CSTR conditions, batch, PFR).

**Phase 2: Fine-tune** on the target reactor with a small amount of data. The physics constraints accelerate adaptation since the model only needs to learn the kinetics, not the conservation laws.

**Meta-learning approach:**

```python
class MetaLearner:
    """MAML/Reptile meta-learning for cross-reactor adaptation."""

    def __init__(
        self,
        neural_ode: AbstractNeuralDE,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        inner_steps: int = 5,
        algorithm: str = "reptile",  # "maml" or "reptile"
    ):
        ...

    def meta_train(self, task_distribution: TaskDistribution, n_iterations: int):
        """Train initialization that enables fast adaptation."""
        for _ in range(n_iterations):
            task = task_distribution.sample()  # Random reactor + conditions
            # Clone model parameters
            adapted_params = clone_params(self.neural_ode)

            # Inner loop: adapt to this task
            for _ in range(self.inner_steps):
                pred = forward_with_params(adapted_params, task.support_data)
                loss = task_loss(pred, task.support_labels)
                adapted_params = sgd_step(adapted_params, loss, self.inner_lr)

            if self.algorithm == "reptile":
                # Reptile: move original params toward adapted params
                for p, p_adapted in zip(self.neural_ode.parameters(), adapted_params):
                    p.data += self.outer_lr * (p_adapted - p.data)
            elif self.algorithm == "maml":
                # MAML: compute loss on query set with adapted params
                query_pred = forward_with_params(adapted_params, task.query_data)
                meta_loss = task_loss(query_pred, task.query_labels)
                meta_loss.backward()  # Second-order gradients
                self.outer_optimizer.step()
```

---

## 7. Visualization & Dashboard

### 7.1 Dashboard Architecture

The ReactorTwin dashboard is built with Streamlit and Plotly for maximum interactivity with minimal frontend code. It consists of 10 pages organized by function:

```
Dashboard
|-- Reactor Simulator (real-time simulation with sliders)
|-- Phase Portrait (2D/3D state space visualization)
|-- Bifurcation Diagram (parameter continuation)
|-- RTD Analysis (residence time distribution)
|-- Parameter Sweep (heatmap over parameter space)
|-- Sensitivity Analysis (Sobol indices, tornado plots)
|-- Pareto Optimization (multi-objective trade-offs)
|-- Fault Monitor (real-time anomaly detection)
|-- Model vs Ground Truth (validation comparison)
|-- Model Explorer (latent space, architecture, training curves)
```

### 7.2 Page Specifications

#### Page 1: Reactor Simulator

**Purpose:** Interactive real-time simulation with parameter sliders.

**Layout:**
```
+------------------------------------------+
| Sidebar:                                  |
|   Reactor Type: [CSTR/Batch/PFR]         |
|   Reaction System: [Exothermic A->B/...]  |
|   Feed Concentration: [slider 0-5 mol/L]  |
|   Feed Temperature: [slider 280-400 K]    |
|   Flow Rate: [slider 50-200 L/min]        |
|   Coolant Temperature: [slider 280-350 K] |
|   Initial Conditions: [expandable]        |
|   Simulation Time: [slider 0-100 min]     |
|   [Run Simulation] button                 |
+------------------------------------------+
| Main Panel:                               |
|   +-------------------+------------------+|
|   | Concentration     | Temperature      ||
|   | vs Time           | vs Time          ||
|   | (Plotly line)     | (Plotly line)     ||
|   +-------------------+------------------+|
|   +-------------------+------------------+|
|   | Conversion        | Key Metrics      ||
|   | vs Time           | (cards)          ||
|   | (Plotly line)     |  - Final conv.   ||
|   |                   |  - Steady state T||
|   |                   |  - Inference time||
|   +-------------------+------------------+|
+------------------------------------------+
```

**Key interaction:** Sliders update the simulation in near-real-time (< 100ms for Neural ODE inference vs seconds for scipy). This viscerally demonstrates the speedup advantage.

#### Page 2: Phase Portrait

**Purpose:** Visualize reactor dynamics in state space.

**Features:**
- 2D phase portrait: C_A vs T with trajectory arrows (streamplot)
- 3D phase portrait: C_A vs C_B vs T (Plotly 3D scatter)
- Steady state markers (stable = filled circle, unstable = open circle, saddle = cross)
- Multiple trajectories from different initial conditions (color-coded)
- Nullclines overlay (dC_A/dt = 0 and dT/dt = 0 curves)
- Vector field visualization using the Neural ODE's learned dynamics

**Implementation:**

```python
def plot_phase_portrait(neural_ode, x_range, y_range, params, n_grid=20):
    """Generate phase portrait from Neural ODE predictions."""
    # Create meshgrid
    x = torch.linspace(*x_range, n_grid)
    y = torch.linspace(*y_range, n_grid)
    X, Y = torch.meshgrid(x, y, indexing='ij')

    # Evaluate dz/dt at each grid point
    states = torch.stack([X.flatten(), Y.flatten()], dim=-1)
    with torch.no_grad():
        derivatives = neural_ode.ode_func(0, states, params)
    dX = derivatives[:, 0].reshape(n_grid, n_grid)
    dY = derivatives[:, 1].reshape(n_grid, n_grid)

    # Create streamplot with Plotly
    fig = go.Figure()
    # Add quiver arrows for vector field
    # Add trajectories from multiple ICs
    for ic in initial_conditions:
        traj = neural_ode(ic, t_span, params)
        fig.add_trace(go.Scatter(x=traj[:, 0], y=traj[:, 1], ...))
    return fig
```

#### Page 3: Bifurcation Diagram

**Purpose:** Map the steady-state structure as a function of a bifurcation parameter.

**Algorithm:**
1. For each value of the bifurcation parameter (e.g., coolant temperature T_c):
   a. Start from multiple initial conditions
   b. Integrate Neural ODE for long time to find steady states
   c. Check stability via eigenvalues of Jacobian (computed via autograd)
2. Plot steady-state values vs parameter
3. Identify bifurcation points (where eigenvalues cross zero)

**Visual output:** Classic S-curve for exothermic CSTR showing ignition/extinction hysteresis. Stable branches in solid lines, unstable in dashed. This is one of the most visually compelling outputs of the project.

#### Page 4: RTD Analysis

**Purpose:** Characterize mixing and flow patterns via residence time distribution.

**Features:**
- E(t) and F(t) curves for different reactor types
- Comparison of ideal (exponential for CSTR, delta for PFR) vs learned (from Neural ODE)
- Tanks-in-series model fitting
- Dead volume and bypass detection
- Mean residence time and variance computation

**For PFR with axial dispersion:**
```
E(t) = (1 / sqrt(4*pi*D_ax*t/u^2*L)) * exp(-(1 - t/tau)^2 * u*L / (4*D_ax*t/tau))
```

#### Page 5: Parameter Sweep

**Purpose:** Generate heatmaps of reactor performance across a 2D parameter grid.

**Workflow:**
1. User selects two parameters (e.g., T_c and F) and an objective (conversion, selectivity, yield)
2. Generate N x N grid of operating conditions
3. Run Neural ODE for all conditions in parallel (batch inference)
4. Display heatmap with optimal region highlighted
5. Show time comparison: "Neural ODE: 2.3s for 10,000 conditions. scipy: estimated 4.2 hours."

**This page is the single most impressive demo** -- it shows the practical value of the Neural ODE surrogate.

#### Page 6: Sensitivity Analysis

**Purpose:** Identify which parameters most influence reactor performance.

**Methods:**
- **Local sensitivity:** Jacobian-based (autograd), displayed as tornado/waterfall chart
- **Global sensitivity:** Sobol indices via Monte Carlo sampling, displayed as bar chart + pie chart
- **Morris method:** Elementary effects screening for many parameters

```python
def sobol_sensitivity(neural_ode, param_ranges, objective_fn, n_samples=10000):
    """Compute Sobol sensitivity indices using Saltelli sampling."""
    problem = {
        'num_vars': len(param_ranges),
        'names': list(param_ranges.keys()),
        'bounds': list(param_ranges.values()),
    }
    param_values = saltelli.sample(problem, n_samples)
    Y = torch.zeros(len(param_values))
    for i, params in enumerate(param_values):
        Y[i] = objective_fn(neural_ode, params)
    Si = sobol.analyze(problem, Y.numpy())
    return Si
```

#### Page 7: Pareto Optimization

**Purpose:** Multi-objective optimization (e.g., maximize conversion while minimizing energy cost).

**Approach:**
1. Define 2-3 objectives (conversion, selectivity, energy consumption, temperature safety margin)
2. Use NSGA-II or weighted scalarization to find Pareto front
3. Neural ODE enables thousands of evaluations in seconds
4. Display Pareto front as scatter plot with interactive point selection
5. Selecting a Pareto point shows the corresponding operating conditions and trajectory

#### Page 8: Fault Monitor

**Purpose:** Real-time monitoring page simulating sensor data stream.

**Features:**
- Live-updating line charts of all measured variables
- Traffic-light indicators: Green (normal), Yellow (warning), Red (fault)
- Anomaly score time series
- Fault isolation attribution chart (which variable is anomalous)
- Historical fault log

#### Page 9: Model vs Ground Truth

**Purpose:** Validation page comparing Neural ODE predictions to scipy ground truth.

**Features:**
- Side-by-side trajectory plots
- Residual error plots
- Conservation law violation over time
- Performance metrics table (RMSE, MAE, max error, conservation error)
- Inference speed comparison bar chart
- Training loss curves

#### Page 10: Model Explorer

**Purpose:** Inspect the trained model internals.

**Features:**
- Latent space visualization (t-SNE/UMAP of latent trajectories)
- Training convergence curves (all loss components)
- Model architecture diagram (auto-generated from nn.Module)
- NFE (number of function evaluations) distribution
- Gradient norm statistics
- Ablation study results (with/without physics constraints)

### 7.3 Visualization Utilities

Reusable plotting functions in `utils/visualization.py`:

```python
def plot_trajectory_comparison(
    t: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray,
    state_names: list[str], title: str = "Neural ODE vs Ground Truth",
) -> go.Figure: ...

def plot_bifurcation_diagram(
    param_values: np.ndarray, steady_states: list[np.ndarray],
    stability: list[bool], param_name: str, state_name: str,
) -> go.Figure: ...

def plot_parameter_sweep_heatmap(
    param1_values: np.ndarray, param2_values: np.ndarray,
    objective_values: np.ndarray, param1_name: str,
    param2_name: str, objective_name: str,
) -> go.Figure: ...

def plot_rtd_curve(
    t: np.ndarray, E_t: np.ndarray, reactor_type: str,
    model_fit: np.ndarray | None = None,
) -> go.Figure: ...

def plot_pareto_front(
    objectives: np.ndarray, objective_names: list[str],
    selected_point: int | None = None,
) -> go.Figure: ...
```

---

## 8. Testing & Validation Strategy

### 8.1 Testing Philosophy

ReactorTwin uses a three-tier testing strategy designed for scientific computing:

1. **Unit tests**: Standard pytest tests for individual functions and classes
2. **Property-based tests**: Using Hypothesis to verify physical invariants
3. **Integration tests**: End-to-end training and inference pipelines

### 8.2 Property-Based Tests (Conservation Law Verification)

The most important tests in the suite. These use the Hypothesis library to generate random reactor states and verify that physical invariants hold:

```python
from hypothesis import given, strategies as st
import hypothesis.extra.numpy as hnp

@given(
    concentrations=hnp.arrays(
        dtype=np.float64, shape=(4,),
        elements=st.floats(min_value=1e-6, max_value=10.0),
    ),
    temperature=st.floats(min_value=250.0, max_value=600.0),
)
def test_mass_conservation_hard_constraint(concentrations, temperature):
    """Hard mass balance constraint must produce zero violation."""
    state = torch.tensor([*concentrations, temperature])
    dz_dt = neural_ode.ode_func(0, state, params)
    dz_constrained = mass_constraint.project(dz_dt, flow_terms)
    total_mass_rate = (mol_weights * dz_constrained[:4]).sum()
    expected_rate = flow_in_mass - flow_out_mass
    assert abs(total_mass_rate - expected_rate) < 1e-10

@given(
    state=hnp.arrays(dtype=np.float64, shape=(5,), elements=st.floats(1e-6, 100.0))
)
def test_positivity_preserved(state):
    """Concentrations must remain non-negative after integration."""
    z0 = torch.tensor(state, dtype=torch.float64)
    trajectory = neural_ode(z0, t_span, params)
    assert (trajectory[:, :4] >= 0).all(), "Negative concentration detected"

@given(
    state=hnp.arrays(dtype=np.float64, shape=(5,), elements=st.floats(1e-6, 100.0))
)
def test_entropy_production_non_negative(state):
    """Second law: entropy production must be >= 0."""
    z = torch.tensor(state, dtype=torch.float64)
    dz_dt = neural_ode.ode_func(0, z, params)
    entropy_production = compute_entropy_production(z, dz_dt, thermo_params)
    assert entropy_production >= -1e-10, "Negative entropy production"

def test_stoichiometric_consistency():
    """Reaction rates must lie in column space of stoichiometry matrix."""
    for _ in range(100):
        state = torch.randn(5).abs()
        dz_dt = neural_ode.ode_func(0, state, params)
        reaction_part = dz_dt[:4] - flow_terms(state, params)
        null_component = reaction_part - nu @ torch.linalg.lstsq(nu, reaction_part).solution
        assert null_component.norm() < 1e-8
```

### 8.3 Unit Test Categories

| Category | What We Test | Example |
|----------|-------------|---------|
| **Shape tests** | Network outputs correct dimensions | `assert output.shape == (batch, state_dim)` |
| **Gradient tests** | Gradients flow without NaN/Inf | `assert not torch.isnan(param.grad).any()` |
| **Symmetry tests** | Skew-symmetric J in Port-Hamiltonian | `assert torch.allclose(J, -J.T, atol=1e-6)` |
| **PSD tests** | Dissipation matrix is PSD | `assert (torch.linalg.eigvalsh(R) >= -1e-6).all()` |
| **Solver tests** | ODE solver matches known analytical solution | `assert rmse(numerical, analytical) < 1e-4` |
| **Data gen tests** | Generated data satisfies conservation laws | Property-based |
| **Kinetics tests** | Rate law produces correct values for known inputs | Table-driven tests |
| **Loss tests** | Loss decreases during training | `assert loss_final < loss_initial` |
| **Constraint tests** | Hard constraints produce zero violation | Property-based |
| **Config tests** | YAML configs load correctly | Parametrized pytest |

### 8.4 Integration Tests

```python
class TestEndToEndCSTR:
    """Full training + inference pipeline for CSTR."""

    def test_training_convergence(self):
        """Model reaches < 5% RMSE within 1000 epochs on exothermic CSTR."""
        reactor = CSTRReactor(system="exothermic_ab")
        data = reactor.generate_data(n_trajectories=100, n_timesteps=100)
        model = HybridODE(reactor=reactor, hidden_dim=64)
        trainer = Trainer(model, data, max_epochs=1000)
        result = trainer.train()
        assert result.final_rmse < 0.05

    def test_generalization(self):
        """Model generalizes to unseen operating conditions."""
        test_data = reactor.generate_data(n_trajectories=20, conditions="unseen")
        rmse = evaluate(model, test_data)
        assert rmse < 0.10

    def test_speedup(self):
        """Neural ODE inference is 100x+ faster than scipy."""
        scipy_time = benchmark_scipy(reactor, n_conditions=100)
        node_time = benchmark_neural_ode(model, n_conditions=100)
        assert scipy_time / node_time > 100

    def test_conservation_over_long_rollout(self):
        """Conservation laws hold over 10x training time horizon."""
        long_traj = model(z0, t_span_10x, params)
        mass_error = compute_mass_balance_error(long_traj, params)
        assert mass_error.max() < 1e-5
```

### 8.5 Benchmarking Framework

```python
class BenchmarkSuite:
    """Standardized benchmarks for reproducible evaluation."""

    benchmarks = [
        ("CSTR-ExoAB-RMSE", CSTRReactor, "exothermic_ab", "rmse", 0.05),
        ("CSTR-VdV-RMSE", CSTRReactor, "van_de_vusse", "rmse", 0.05),
        ("CSTR-ExoAB-MassBal", CSTRReactor, "exothermic_ab", "mass_error", 1e-5),
        ("CSTR-ExoAB-Speedup", CSTRReactor, "exothermic_ab", "speedup", 100),
        ("Batch-ABC-RMSE", BatchReactor, "consecutive_abc", "rmse", 0.05),
        ("Batch-ParamEst", BatchReactor, "consecutive_abc", "param_error", 0.10),
        ("PFR-1D-RMSE", PFRReactor, "exothermic_ab", "rmse", 0.08),
    ]

    def run_all(self) -> BenchmarkReport:
        """Run all benchmarks and generate report."""
        ...
```

---

## 9. Phased Implementation Plan

### Phase 1: Foundation (Weeks 1-2) -- Core Engine

**Goal:** Working Neural ODE that learns CSTR dynamics with physics constraints.

#### Week 1: Infrastructure + Data Generation

| Day | Task | Deliverable |
|-----|------|------------|
| 1 | Project setup: pyproject.toml, src layout, ruff, mypy, pytest, CI | Working dev environment |
| 2 | Implement `AbstractReactor`, `CSTRReactor`, `ArrheniusKinetics` | Reactor base classes |
| 3 | Implement `DataGenerator`: solve CSTRs with scipy, generate 500+ trajectories | Training datasets |
| 4 | Implement `MLPODEFunc`, `AbstractNeuralDE`, `NeuralODE` | Forward pass works |
| 5 | Implement `SolverFactory` with torchdiffeq backend | ODE integration works |
| 6-7 | Tests for all above. Verify data satisfies conservation laws. | >90% coverage for Phase 1 code |

#### Week 2: Training + Physics Constraints

| Day | Task | Deliverable |
|-----|------|------------|
| 8 | Implement `Trainer` with adjoint backprop, AdamW, cosine annealing | Training loop |
| 9 | Implement `losses.py`: data loss + physics residual loss | Multi-objective loss |
| 10 | Implement `MassBalanceConstraint` (hard projection) | Mass conservation |
| 11 | Implement `EnergyBalanceConstraint`, `PositivityConstraint` | Energy conservation, C >= 0 |
| 12 | Train on exothermic CSTR. Target: < 5% RMSE, mass error < 1e-5 | First working model |
| 13 | Implement `HybridODEFunc`: physics + neural correction | Hybrid architecture |
| 14 | Write `examples/01_cstr_exothermic.py`, comparison plots | First demo |

**Phase 1 Deliverables:**
- Neural ODE learns CSTR dynamics with < 5% RMSE
- Hard mass conservation (error < 1e-5)
- Hard energy conservation
- Positivity guarantee
- Hybrid physics + neural architecture
- 100x+ inference speedup over scipy
- 1 example script with comparison plots

### Phase 2: Depth (Weeks 3-4) -- Multiple Neural DEs, Reactors, Constraints

**Goal:** Multiple Neural DE variants, Van de Vusse + batch reactor, thermodynamic constraints, bifurcation analysis.

#### Week 3: Advanced Neural DEs + More Reactors

| Day | Task | Deliverable |
|-----|------|------------|
| 15 | Implement `LatentNeuralODE` with ODE-RNN encoder | Latent ODE works |
| 16 | Implement `AugmentedNeuralODE` | ANODE variant |
| 17 | Implement Van de Vusse reaction system, train CSTR on it | 2nd benchmark |
| 18 | Implement `BatchReactor`, consecutive A->B->C system | Batch reactor works |
| 19 | Implement `StoichiometricODEFunc` (predict rates, not species) | Stoichiometric constraint |
| 20 | Implement parameter estimation (inverse problem) for batch reactor | Inverse problem demo |
| 21 | Tests + examples for all above | Examples 02-04 |

#### Week 4: Thermodynamics + Bifurcation

| Day | Task | Deliverable |
|-----|------|------------|
| 22 | Implement `ThermodynamicConstraint`: Gibbs, equilibrium consistency | Thermodynamic enforcement |
| 23 | Implement `PortHamiltonianODEFunc` | Port-Hamiltonian structure |
| 24 | Implement steady-state finder + numerical continuation | Bifurcation capability |
| 25 | Generate S-curve bifurcation diagram for exothermic CSTR | Visual demo |
| 26 | Implement `LossScheduler` with curriculum learning | Improved training |
| 27 | Implement `StiffnessHandler` with time reparameterization | Stiffness handling |
| 28 | Comprehensive testing, examples 05-07 | Phase 2 complete |

**Phase 2 Deliverables:**
- Latent Neural ODE + Augmented Neural ODE variants
- Van de Vusse and batch reactor benchmarks
- Stoichiometric, thermodynamic, Port-Hamiltonian constraints
- Bifurcation diagram for CSTR (S-curve)
- Parameter estimation from noisy data
- Stiffness handling via time reparameterization
- 7 example scripts

### Phase 3: Digital Twin (Weeks 5-6) -- State Estimation, Fault Detection, Control

**Goal:** Full digital twin capabilities.

#### Week 5: State Estimation + Fault Detection

| Day | Task | Deliverable |
|-----|------|------------|
| 29 | Implement `NeuralODEStateEstimator` (EKF + Neural ODE) | State estimation |
| 30 | Implement `FaultDetector` with VAE and multi-level detection | Fault detection |
| 31 | Implement `AnomalyDiagnosis` with integrated gradients | Fault isolation |
| 32 | Implement `ScenarioSimulator` for what-if analysis | Scenario engine |
| 33 | Implement `NeuralSDE` for uncertainty quantification | UQ capability |
| 34-35 | Tests and examples 08-12 | Digital twin demos |

#### Week 6: Control + Adaptation

| Day | Task | Deliverable |
|-----|------|------------|
| 36 | Implement `NeuralODEMPC` with gradient-based optimization | MPC controller |
| 37 | Implement `OnlineAdapter` with replay buffer | Online learning |
| 38 | Implement `MetaLearner` (Reptile) for cross-reactor transfer | Meta-learning |
| 39 | Implement `TransferLearning` module | Transfer capability |
| 40 | Integration testing: full digital twin pipeline | End-to-end works |
| 41-42 | Examples 13-14, extensive testing | Phase 3 complete |

**Phase 3 Deliverables:**
- EKF + Neural ODE state estimator
- Multi-level fault detection and isolation
- Neural SDE uncertainty quantification
- What-if scenario simulation
- MPC controller with Neural ODE plant model
- Online adaptation from streaming data
- Reptile meta-learning for cross-reactor transfer

### Phase 4: Dashboard & Deployment (Weeks 7-8)

**Goal:** Polished interactive dashboard, API server, Docker deployment.

#### Week 7: Dashboard

| Day | Task | Deliverable |
|-----|------|------------|
| 43 | Implement Streamlit app skeleton with navigation | App structure |
| 44 | Page 1: Reactor Simulator with real-time sliders | Interactive simulation |
| 45 | Pages 2-3: Phase Portrait + Bifurcation Diagram | Dynamics visualization |
| 46 | Pages 4-5: RTD Analysis + Parameter Sweep | Performance analysis |
| 47 | Pages 6-7: Sensitivity Analysis + Pareto Optimization | Optimization tools |
| 48 | Pages 8-10: Fault Monitor + Comparison + Model Explorer | Monitoring & validation |
| 49 | Dashboard polish, caching, session state | Production-ready UI |

#### Week 8: API + Deployment + Documentation

| Day | Task | Deliverable |
|-----|------|------------|
| 50 | FastAPI server with all endpoints | REST API |
| 51 | WebSocket endpoint for real-time streaming | Real-time capability |
| 52 | Dockerfile + docker-compose | Container deployment |
| 53 | Tutorial notebooks (5 notebooks) | Educational content |
| 54 | Paper figures notebook (reproducible) | Publication-ready figures |
| 55 | README with GIFs, badges, performance tables | Project presentation |
| 56 | Final testing, benchmarks, cleanup | Release-ready |

**Phase 4 Deliverables:**
- 10-page interactive Streamlit dashboard
- FastAPI REST API with WebSocket streaming
- Docker deployment
- 5 tutorial Jupyter notebooks
- Paper-ready figures notebook
- Comprehensive README with visuals

### Phase 5: Extensions (Weeks 9-10, if time permits)

**Priority-ordered stretch goals:**

1. **PFR with Method of Lines**
2. **Neural CDE for irregular sensor data**
3. **GENERIC framework**
4. **Population Balance Equations**
5. **Multi-phase reactor**
6. **3D reactor cross-section visualization**
7. **ONNX export**
8. **Streamlit Cloud deployment**

---

## 10. Tech Stack & Dependencies

### 10.1 Core Dependencies

| Package | Version | Purpose | Why This One |
|---------|---------|---------|-------------|
| `torch` | >= 2.1 | Deep learning framework | Neural ODE ecosystem, autograd for Jacobians |
| `torchdiffeq` | >= 0.2.3 | Neural ODE solver (primary) | Canonical implementation, adjoint method |
| `torchode` | >= 0.2.0 | Neural ODE solver (fast) | 4x faster batched solving, JIT-compatible |
| `torchsde` | >= 0.2.5 | Neural SDE solver | Stochastic adjoint, Euler-Maruyama |
| `torchcde` | >= 0.2.5 | Neural CDE solver | Irregular time series, path interpolation |
| `numpy` | >= 1.24 | Numerical computing | Array operations |
| `scipy` | >= 1.11 | Scientific computing | solve_ivp for data generation |

### 10.2 Visualization & Dashboard

| Package | Version | Purpose |
|---------|---------|---------|
| `streamlit` | >= 1.28 | Dashboard framework |
| `plotly` | >= 5.18 | Interactive plots with WebGL |
| `matplotlib` | >= 3.8 | Paper-quality static plots |

### 10.3 API & Deployment

| Package | Version | Purpose |
|---------|---------|---------|
| `fastapi` | >= 0.104 | REST API |
| `uvicorn` | >= 0.24 | ASGI server |
| `pydantic` | >= 2.5 | Data validation |
| `websockets` | >= 12.0 | Real-time streaming |

### 10.4 Development

| Package | Version | Purpose |
|---------|---------|---------|
| `pytest` | >= 7.4 | Testing |
| `pytest-cov` | >= 4.1 | Coverage |
| `hypothesis` | >= 6.88 | Property-based testing |
| `ruff` | >= 0.1.6 | Linting + formatting |
| `mypy` | >= 1.7 | Type checking |
| `pre-commit` | >= 3.6 | Git hooks |

### 10.5 Optional Dependencies

| Package | Purpose | Install Group |
|---------|---------|--------------|
| `cantera` | Detailed thermodynamics | `[thermo]` |
| `coolprop` | Fluid properties | `[thermo]` |
| `SALib` | Sobol sensitivity analysis | `[analysis]` |
| `optuna` | Hyperparameter optimization | `[tuning]` |
| `wandb` | Experiment tracking | `[tracking]` |
| `onnx` + `onnxruntime` | Model export | `[deploy]` |

### 10.6 pyproject.toml Configuration

```toml
[project]
name = "reactor-twin"
version = "0.1.0"
description = "Physics-constrained Neural DEs for chemical reactor digital twins"
authors = [{name = "Tubhyam Karthikeyan", email = "takarthikeyan25@gmail.com"}]
requires-python = ">=3.10"
license = {text = "MIT"}
keywords = [
    "neural-ode", "chemical-reactor", "digital-twin",
    "physics-informed", "surrogate-model",
]

dependencies = [
    "torch>=2.1",
    "torchdiffeq>=0.2.3",
    "numpy>=1.24",
    "scipy>=1.11",
    "plotly>=5.18",
    "matplotlib>=3.8",
    "pydantic>=2.5",
]

[project.optional-dependencies]
dashboard = ["streamlit>=1.28"]
api = ["fastapi>=0.104", "uvicorn>=0.24", "websockets>=12.0"]
sde = ["torchsde>=0.2.5"]
cde = ["torchcde>=0.2.5"]
fast = ["torchode>=0.2.0"]
thermo = ["cantera", "coolprop"]
analysis = ["SALib>=1.4"]
tracking = ["wandb"]
deploy = ["onnx", "onnxruntime"]
dev = [
    "pytest>=7.4", "pytest-cov>=4.1", "hypothesis>=6.88",
    "ruff>=0.1.6", "mypy>=1.7", "pre-commit>=3.6",
]
all = ["reactor-twin[dashboard,api,sde,cde,fast,analysis,dev]"]

[project.scripts]
reactor-twin-dashboard = "reactor_twin.dashboard.app:main"
reactor-twin-api = "reactor_twin.api.server:main"

[tool.ruff]
target-version = "py310"
line-length = 100

[tool.mypy]
python_version = "3.10"
strict = true

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "--cov=reactor_twin --cov-report=term-missing"
```

---

## 11. Benchmarks & Success Metrics

### 11.1 Accuracy Benchmarks

| Benchmark | Metric | Target | Stretch | Comparison |
|-----------|--------|--------|---------|------------|
| CSTR exothermic A->B | RMSE (concentration) | < 5% | < 1% | ChemNODE: ~3% |
| CSTR exothermic A->B | RMSE (temperature) | < 3 K | < 0.5 K | Foundation model: ~5 K |
| CSTR Van de Vusse | RMSE (all species) | < 5% | < 2% | GRxnODE: ~4% |
| Batch consecutive | RMSE | < 5% | < 2% | -- |
| Batch parameter est. | Parameter error | < 10% | < 5% | -- |
| PFR 1D | RMSE | < 8% | < 3% | FMEnets: ~2% |
| Long rollout (10x) | RMSE degradation | < 2x | < 1.5x | Standard NODE: 5-10x |

### 11.2 Physics Compliance

| Constraint | Hard Mode | Soft Mode | Unconstrained |
|-----------|-----------|-----------|---------------|
| Mass balance error | < 10^-8 | < 10^-3 | ~10^-1 |
| Energy balance error | < 10^-6 | < 10^-3 | ~10^-1 |
| Positivity violations | 0 | < 0.1% | ~5% |
| Stoichiometric error | < 10^-8 | < 10^-3 | ~10^-2 |
| Gibbs monotonicity | 100% | > 99% | ~80% |

### 11.3 Performance Benchmarks

| Operation | Time | Speedup vs scipy |
|-----------|------|-----------------|
| Single trajectory (200 steps) | < 5 ms | 100x+ |
| Parameter sweep (10,000 conditions) | < 5 s | 1,000x+ |
| Bifurcation diagram (1,000 points) | < 2 s | 500x+ |
| MPC optimization (20-step horizon) | < 100 ms | Real-time capable |
| State estimation update | < 10 ms | Real-time capable |
| Training (CSTR, 1000 epochs) | < 30 min | -- |

### 11.4 Software Quality

| Metric | Target | Stretch |
|--------|--------|---------|
| Test coverage | > 80% | > 95% |
| Type coverage (mypy) | > 90% | 100% |
| Ruff violations | 0 | 0 |
| Example scripts | 15 | 20 |
| Tutorial notebooks | 5 | 8 |

### 11.5 Ablation Studies (for paper)

| Ablation | What We Remove | Expected Impact |
|----------|---------------|-----------------|
| No physics constraints | Remove all constraint losses | +50-200% RMSE, mass violations |
| Soft vs hard constraints | Replace hard with soft | 10-100x worse conservation |
| No hybrid architecture | Pure black-box Neural ODE | +20-50% RMSE, worse extrapolation |
| No curriculum learning | Fixed loss weights | Training instability |
| No time reparameterization | Standard time for stiff | Training failure or 10x more epochs |
| Standard vs Latent ODE | For high-dim systems | Latent: 50% fewer NFEs |
| Augmented vs Standard | Remove extra dimensions | +20% NFEs |
| Port-Hamiltonian vs MLP | Unstructured ODE func | 10-100x worse energy conservation |
| With vs without meta-learning | Train from scratch | Meta: 10x fewer fine-tune steps |

### 11.6 Comparison with Existing Tools

| Feature | ReactorTwin | Foundation Model (Wang) | ChemNODE | DeepXDE | Aspen Plus |
|---------|------------|------------------------|----------|---------|------------|
| Continuous-time | Yes (Neural ODE) | No (discrete MLP) | Yes | No | Yes |
| Hard constraints | Yes | No | No | No | Yes (built-in) |
| Reactor types | 6+ | 3 | 0 (combustion) | General PDE | 50+ |
| Uncertainty (SDE) | Yes | No | No | No | No |
| Dashboard | Yes (10 pages) | No | No | No | Yes (GUI) |
| State estimation | Yes (EKF+NODE) | No | No | No | Yes |
| Fault detection | Yes | No | No | No | Limited |
| MPC control | Yes | No | No | No | Yes |
| Meta-learning | Yes (Reptile) | Yes (Reptile) | No | No | No |
| Open source | Yes (MIT) | Yes (MIT) | Yes | Yes (Apache) | No |
| Inference speed | ~1000x | ~100x | ~100x | ~10x | Baseline |

---

## 12. References

### Foundational Neural DE Papers

1. Chen, R.T.Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D. (2018). [Neural Ordinary Differential Equations](https://arxiv.org/abs/1806.07366). NeurIPS 2018.
2. Rubanova, Y., Chen, R.T.Q., & Duvenaud, D. (2019). [Latent ODEs for Irregularly-Sampled Time Series](https://arxiv.org/abs/1907.03907). NeurIPS 2019.
3. Dupont, E., Doucet, A., & Teh, Y.W. (2019). [Augmented Neural ODEs](https://arxiv.org/abs/1904.01681). NeurIPS 2019.
4. Kidger, P. et al. (2020). [Neural Controlled Differential Equations for Irregular Time Series](https://arxiv.org/abs/2005.08926). NeurIPS 2020 Spotlight.
5. Li, X. et al. (2020). [Scalable Gradients for Stochastic Differential Equations](https://arxiv.org/abs/2001.01328). AISTATS 2020.
6. Poli, M. et al. (2020). [Hypersolvers: Toward Fast Continuous-Depth Models](https://arxiv.org/abs/2007.09601). NeurIPS 2020.
7. Walker, B. et al. (2024). [Log Neural Controlled Differential Equations](https://arxiv.org/abs/2402.18512). ICML 2024.

### Chemical Kinetics + Neural ODEs

8. Owoyele, O. & Pal, P. (2021). [ChemNODE](https://www.sciencedirect.com/science/article/pii/S2666546821000677). Energy and AI.
9. Qian, W. et al. (2022). [GRxnODE](https://www.sciencedirect.com/science/article/abs/pii/S138589472204966X). Chemical Engineering Journal.
10. (2023). [Physics-Enhanced Neural ODE for Industrial Reactions](https://pubs.acs.org/doi/10.1021/acs.iecr.3c01471). I&EC Research.
11. (2025). [Phy-ChemNODE](https://www.frontiersin.org/journals/thermal-engineering/articles/10.3389/fther.2025.1594443/full). Frontiers in Thermal Engineering.
12. (2025). [SPIN-ODE](https://arxiv.org/html/2505.05625v1). arXiv.
13. (2025). [PC-NODE for Stiff Chemical Kinetics](https://www.tandfonline.com/doi/full/10.1080/13647830.2025.2478266). Combustion Theory and Modelling.

### Stiffness in Neural ODEs

14. Ji, W. & Deng, S. (2021). [Stiff-PINN](https://pubs.acs.org/doi/abs/10.1021/acs.jpca.1c05102). J. Phys. Chem. A.
15. Caldana & Hesthaven (2025). [Neural ODEs for Stiff Systems](https://onlinelibrary.wiley.com/doi/10.1002/nme.70060). Int. J. Numer. Methods Eng.
16. (2024). [Training Stiff Neural ODEs with Implicit Methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC11646139/).
17. (2023). [Multiscale PINNs for Stiff Chemical Kinetics](https://pubs.acs.org/doi/abs/10.1021/acs.jpca.2c06513). J. Phys. Chem. A.

### Conservation Constraints

18. (2024). [Mass and Energy Constrained Neural Networks](https://pubs.acs.org/doi/10.1021/acs.iecr.4c01429). I&EC Research.
19. (2025). [Mass, Energy, Thermodynamics Constrained NNs](https://www.sciencedirect.com/science/article/abs/pii/S000925092500329X). Computers & Chemical Engineering.
20. (2024). [Hard Linear Equality Constraints in PINNs](https://arxiv.org/abs/2402.07251).
21. (2025). [Hard Constraints for Conservation Laws](https://www.nature.com/articles/s41598-025-34263-1). Scientific Reports.
22. (2024). [HANNA: Hard-Constraint NN](https://pmc.ncbi.nlm.nih.gov/articles/PMC11575590/). Chemical Science.

### Hamiltonian, Lagrangian, Thermodynamic Structure

23. Greydanus, S. et al. (2019). Hamiltonian Neural Networks. NeurIPS 2019.
24. Cranmer, M. et al. (2020). [Lagrangian Neural Networks](https://arxiv.org/abs/2003.04630).
25. Hernandez, Q. et al. (2023). [Port-Metriplectic Neural Networks](https://link.springer.com/article/10.1007/s00466-023-02296-w). Computational Mechanics.
26. (2025). [Stable Port-Hamiltonian Neural Networks](https://arxiv.org/html/2502.02480).
27. (2024). [Hamilton-Dirac Neural Networks](https://arxiv.org/abs/2401.15485).
28. (2022). [Thermodynamics-Informed Graph Neural Networks](https://arxiv.org/abs/2203.01874).
29. (2025). [ThermoLearn](https://link.springer.com/article/10.1186/s13321-025-01033-0). J. Cheminformatics.
30. (2024). [Thermodynamics-Consistent GNNs](https://pubs.rsc.org/en/content/articlehtml/2024/sc/d4sc04554h). Chemical Science.

### Reactor Digital Twins

31. Wang, Z. et al. (2024/2025). [Foundation Model for Chemical Reactors](https://arxiv.org/abs/2405.11752). Chem. Eng. Research and Design.
32. (2025). [Cloud-Enabled Digital Twin Platform](https://www.sciencedirect.com/science/article/pii/S0098135425004910). Computers & Chemical Engineering.
33. Mane et al. (2024). [Digital Twin in Chemical Industry Review](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/dgt2.12019).
34. (2025). [Fault Diagnosis in Chemical Reactors](https://pubs.acs.org/doi/10.1021/acs.iecr.4c04042). I&EC Research.

### MPC with Neural ODEs

35. Luo, J. et al. (2023). [MPC with Neural ODE Models](https://www.researchgate.net/publication/372835364).
36. (2024). [ML-Based MPC with Input Convex NNs](https://pubs.acs.org/doi/10.1021/acs.iecr.4c02257). I&EC Research.
37. (2025). [Tutorial Review of ML-Based MPC](https://www.degruyterbrill.com/document/doi/10.1515/revce-2024-0055/html). Reviews in Chemical Engineering.

### Non-Ideal Reactors and PFR

38. (2025). [FMEnets for Non-Ideal PFR Design](https://arxiv.org/abs/2505.20300).
39. (2025). [RTD Effects on CSTR Dynamics](https://pubs.acs.org/doi/10.1021/acs.iecr.5c00604). I&EC Research.

### Population Balance Equations

40. (2025). [PINN for Population Balance Model](https://advancesincontinuousanddiscretemodels.springeropen.com/articles/10.1186/s13662-025-03876-1).
41. (2025). [2D PBM Optimization for Crystallization](https://aiche.onlinelibrary.wiley.com/doi/10.1002/aic.70094). AIChE Journal.

### Adjoint Methods

42. Kidger, P. et al. (2021). [Faster ODE Adjoints via Seminorms](https://arxiv.org/abs/2009.09457). ICML 2021.
43. Zhuang, J. et al. (2021). [Symplectic Adjoint Method](https://arxiv.org/abs/2102.09750). NeurIPS 2021.
44. (2024). [Adjoint Method for Neural ODE](https://arxiv.org/html/2402.15141v1).
45. (2025). [Guide to Neural ODEs](https://www.sciencedirect.com/science/article/pii/S2950550X25000263).

### Uncertainty Quantification

46. (2024). [Stable Neural SDEs](https://proceedings.iclr.cc/paper_files/paper/2024/file/a61023ce36d21010f1423304f8ec49af-Paper-Conference.pdf). ICLR 2024.
47. (2024). [Bayesian Neural CDEs](https://proceedings.iclr.cc/paper_files/paper/2024/file/e897b4d0914f213896f3e5c25732b1ed-Paper-Conference.pdf). ICLR 2024.
48. (2025). [UQ in Universal Differential Equations](https://pmc.ncbi.nlm.nih.gov/articles/PMC12005350/).

### Software Libraries

49. [torchdiffeq](https://github.com/rtqichen/torchdiffeq) -- Differentiable ODE solvers.
50. [torchode](https://torchode.readthedocs.io/) -- Parallel ODE solver.
51. [torchsde](https://github.com/google-research/torchsde) -- SDE solvers.
52. [torchcde](https://github.com/patrick-kidger/torchcde) -- CDE solvers.
53. [DeepXDE](https://github.com/lululxvi/deepxde) -- Scientific ML library.
54. [NeuralPDE.jl](https://github.com/SciML/NeuralPDE.jl) -- Julia PINN solvers.
55. [Cantera](https://cantera.org/) -- Chemical kinetics and thermodynamics.

### CSTR Benchmarks

56. [Exothermic CSTR -- ND Pyomo Cookbook](https://jckantor.github.io/ND-Pyomo-Cookbook/notebooks/05.02-Exothermic-CSTR.html)
57. [CSTR Model -- MATLAB MPC Toolbox](https://www.mathworks.com/help/mpc/gs/cstr-model.html)
58. [Simulation of Exothermic CSTR -- CBE 30338](https://jckantor.github.io/CBE30338/07.04-Simulation-of-an-Exothermic-CSTR.html)

### Surrogate Modeling

59. (2025). [Surrogate Hybridization for Simulation Optimization](https://pubs.acs.org/doi/10.1021/acs.iecr.4c03303). I&EC Research.
60. (2025). [SMR Reactor ML Surrogate Modeling](https://arxiv.org/abs/2507.07641).

---

## Appendix A: CSTR Benchmark Parameters

### Exothermic First-Order Reaction (A -> B)

```
V    = 100 L           # reactor volume
rho  = 1000 g/L        # density
Cp   = 0.239 J/(g*K)   # heat capacity
dH_r = -5e4 J/mol      # heat of reaction
E_a  = 7.27e4 J/mol    # activation energy
k_0  = 7.2e10 1/min    # pre-exponential factor
UA   = 5e4 J/(min*K)   # heat transfer coefficient
F    = 100 L/min       # volumetric flow rate
C_A0 = 1.0 mol/L       # feed concentration
T_0  = 350 K           # feed temperature
T_c  = 300 K           # coolant temperature
```

### Van de Vusse Reaction System

```
A -> B    (k1 = 50/hr,  E1 = 5e4 J/mol)
B -> C    (k2 = 100/hr, E2 = 7.5e4 J/mol)
2A -> D   (k3 = 10 L/(mol*hr), E3 = 6e4 J/mol)

Feed: C_A0 = 5.1 mol/L, T = 130 C (isothermal)
Residence time: tau = 20 s
```

### Bioreactor (Monod Kinetics)

```
mu_max = 0.5 1/hr      # max specific growth rate
K_s    = 0.2 g/L       # substrate saturation constant
Y_xs   = 0.5 g/g       # biomass yield coefficient
S_f    = 10.0 g/L      # feed substrate concentration
D      = 0.1-0.5 1/hr  # dilution rate (control variable)
```

---

## Appendix B: Publishability Roadmap

### Paper Title Options

1. "ReactorTwin: Physics-Constrained Neural Differential Equations with Hard Conservation Guarantees for Chemical Reactor Digital Twins"
2. "Port-Hamiltonian Neural ODEs for Thermodynamically Consistent Chemical Reactor Surrogate Modeling"
3. "From Soft to Hard: Architectural Physics Constraints in Neural ODE Surrogate Models for Chemical Reactors"

### Novel Contributions

1. Hard mass/energy conservation via architectural projection
2. Port-Hamiltonian Neural ODE for open reactor systems (first application)
3. Unified framework across reactor types with plug-in kinetics
4. Systematic ablation: hard vs soft constraints (10-100x better conservation)
5. Neural ODE-based gradient MPC for reactor control

### Target Venues

| Venue | IF | Fit | Difficulty |
|-------|-----|-----|-----------|
| *Computers & Chemical Engineering* | 4.3 | Excellent | Medium |
| *Chemical Engineering Research and Design* | 3.7 | Good | Medium |
| *Digital Chemical Engineering* | New | Excellent | Easier |
| *Chemical Engineering Science* | 4.7 | Good | Medium-Hard |
| *AIChE Journal* | 3.5 | Good | Medium |
| NeurIPS ML4PS Workshop | Top ML | Scientific ML | Competitive |

### What Reviewers Will Want

1. Ablation studies showing each component's contribution
2. Comparison with Foundation Model, ChemNODE, DeepXDE
3. Scalability analysis (species count, stiffness ratio)
4. Real-world relevance (comparison with Aspen Plus)
5. Reproducibility (open-source, fixed seeds, documented hyperparams)
6. Error analysis and limitations discussion

---

*This plan was prepared with extensive literature research across Neural ODEs, physics-informed machine learning, chemical reactor engineering, digital twin technology, and thermodynamic structure-preserving methods. All referenced works are from 2019-2025, with emphasis on 2024-2025 state-of-the-art.*
