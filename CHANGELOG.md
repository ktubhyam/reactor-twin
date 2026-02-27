# Changelog

All notable changes to ReactorTwin will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-02-28

### Added (Phase 6 -- Polish & Release)
- 5 tutorial Jupyter notebooks (getting started, physics constraints, advanced Neural DEs, digital twin pipeline, custom extensions)
- Complete MkDocs API documentation with mkdocstrings for all public classes
- Tutorial documentation pages linked to notebooks
- Tutorials section added to MkDocs navigation
- ruff lint + format clean across all 81 source files
- Per-file ruff ignores for examples (PLR0915, B007)
- GitHub CI/CD: pre-commit config, issue/PR templates, PyPI release workflow
- PEP 561 py.typed marker for type checker support

### Added (Phase 5 -- Digital Twin)
- EKFStateEstimator with Neural ODE fusion and autograd Jacobian computation
  - Joseph form covariance update for numerical stability
  - Partial observability via configurable observation indices
  - Second-order matrix exponential approximation for covariance propagation
  - Full filter pass over measurement sequences
- FaultDetector with 4-level detection:
  - L1: SPC (EWMA/CUSUM) control charts on raw sensor data
  - L2: Residual-based detection via one-step-ahead Neural ODE predictions
  - L3: Per-variable fault isolation using Mahalanobis decomposition
  - L4: ML classification (SVM / Random Forest) on residual features
  - Unified orchestrator with alarm severity levels (NORMAL/WARNING/ALARM/CRITICAL)
- MPCController with gradient-based optimization (LBFGS), constraint handling, and warm-starting
  - Quadratic stage + terminal cost (MPCObjective)
  - Hard box constraints on controls, soft quadratic penalties on outputs (ControlConstraints)
  - Differentiable Euler rollout through Neural ODE dynamics
  - Receding horizon via `step()` method
- OnlineAdapter with replay buffer and Elastic Weight Consolidation for continual learning
  - FIFO ReplayBuffer with configurable capacity and mini-batch sampling
  - ElasticWeightConsolidation with diagonal Fisher estimation
  - Mixed replay/new-data gradient steps with EWC penalty
- ReptileMetaLearner for cross-reactor transfer and few-shot adaptation
  - First-order Reptile meta-learning (no second-order gradients)
  - Task subsampling from reactor data generator pool
  - `fine_tune()` for few-shot adaptation to new reactor configurations
- Streamlit Dashboard with 10 interactive pages:
  - Reactor Simulator, Phase Portraits, Bifurcation Diagrams
  - RTD Analysis, Parameter Sweeps, Sensitivity Analysis
  - Pareto Optimization, Fault Monitoring, Model Validation, Latent Explorer
- DIGITAL_TWIN_REGISTRY added to utility registries
- All digital twin components exported from top-level `reactor_twin` namespace
- FastAPI server (`api/server.py`) for real-time reactor simulation

### Added (Phase 4 -- Additional Reactors and Kinetics)
- BatchReactor with time-varying volume for gas-phase reactions
  - Heat of reaction in energy balance for non-isothermal operation
- SemiBatchReactor with continuous feed and no outflow
  - Heat of reaction in energy balance for non-isothermal operation
- PlugFlowReactor (PFR) with Method of Lines discretization
- MichaelisMentenKinetics for enzyme-catalyzed reactions with competitive inhibition
- PowerLawKinetics for empirical rate expressions with temperature dependence
- LangmuirHinshelwoodKinetics for heterogeneous catalysis on surfaces
- ReversibleKinetics for equilibrium-limited reactions with forward/reverse rates
- MonodKinetics for microbial growth modeling
- `compute_reaction_rates()` method on AbstractKinetics base class
- Heat of reaction (`delta_H_rxn`) implementation in CSTR, Batch, and Semi-Batch reactors
- Bioreactor CSTR benchmark with Monod growth kinetics (substrate, biomass, product)
- Consecutive reactions CSTR benchmark (A->B->C) with selectivity analysis
- Parallel competing reactions CSTR benchmark (A->B, A->C) with yield optimization

### Added (Phase 3 -- Advanced Neural DEs)
- Latent Neural ODE with encoder-decoder architecture for high-dimensional systems
- Augmented Neural ODE with extra dimensions for expressivity
- Neural SDE for stochastic dynamics and uncertainty quantification
- Neural CDE for irregular time series with control path interpolation
- Support for torchsde and torchcde (optional dependencies)

### Added (Phase 2 -- Physics Constraints + Training)
- Mass balance constraint with stoichiometric projection
- Energy balance constraint for thermochemical systems
- Thermodynamic constraint (entropy monotonicity, Gibbs energy)
- Stoichiometric constraint (predict rates not species)
- Port-Hamiltonian constraint with learnable structure matrices
- GENERIC constraint (reversible-irreversible coupling)
- Exothermic A->B CSTR benchmark system
- Van de Vusse reaction network CSTR benchmark
- Complete training infrastructure (Trainer, MultiObjectiveLoss, ReactorDataGenerator)
- Multi-objective loss with constraint penalties
- Automatic data generation from reactor models

### Added (Phase 1 -- Foundation)
- Core Neural ODE with adjoint method
- Abstract base classes for all components
- Registry system for plugins
- CSTR reactor implementation
- Arrhenius kinetics
- Positivity constraint (hard/soft modes)
- MLPODEFunc and HybridODEFunc
- Complete project structure and documentation

### Fixed
- PFR inlet boundary: changed `np.tile` to `np.repeat` for correct spatial replication
- MonodKinetics `__init__` signature to match AbstractKinetics interface
- Consecutive reactions CSTR: corrected parameter names for consistency
- Parallel competing reactions CSTR: corrected parameter names for consistency

### Testing
- 700+ tests across 8 test files (test_core, test_physics, test_training, test_reactors, test_kinetics, test_digital_twin, test_systems, test_utils)
- Parametrized tests for physics constraints, reactor types, and kinetics models
- API server integration tests
- Dashboard smoke tests
- Digital twin component tests (EKF, SPC, FaultDetector, MPC, OnlineAdapter)

### Performance
- Performance benchmarks in `benchmarks/bench_digital_twin.py`
  - MPC step latency (horizons 5/10/20)
  - EKF predict+update step latency (state dims 3/6/10)
  - EKF Jacobian computation (autograd)
  - EKF full filter pass (50/100/200 steps)
  - Online adaptation (5 gradient steps)
- CPU targets: MPC < 500ms, EKF < 200ms, Jacobian < 200ms

### CI/CD
- Benchmark job added to CI workflow
- Updated GitHub Actions workflow for full test suite

### Examples
- 14 example scripts added (population balance example deferred to v0.2.0)
- `examples/00_quickstart.py` updated for Phase 5 components

## [0.0.1] - 2026-02-27

Initial architecture setup. Foundation for physics-constrained Neural DEs.

### Added
- Project structure with src layout
- Abstract base classes (AbstractNeuralDE, AbstractReactor, AbstractKinetics, AbstractConstraint)
- Plugin registry system
- Reference implementations (NeuralODE, CSTRReactor, ArrheniusKinetics, PositivityConstraint)
- Full project configuration (pyproject.toml, CI/CD, documentation)
- MIT License
- README with quickstart and examples
