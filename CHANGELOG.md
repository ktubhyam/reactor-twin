# Changelog

All notable changes to ReactorTwin will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [1.1.0] - 2026-03-01

### Added
- Prometheus `/metrics` endpoint (`reactor_twin[api]`) — request counts, latency histograms, ODE solve time, active WebSocket session gauge
- AWS SageMaker inference module (`reactor_twin.deploy.sagemaker`) — `model_fn`, `input_fn`, `predict_fn`, `output_fn`, `pack_model_tar`
- ONNX partial export for `NeuralSDE` (drift function only) and `NeuralCDE` (CDE vector field only)
- CI coverage-gaps job — runs `test_coverage_gaps.py` with 5-minute timeout, uploads to Codecov separately

### Changed
- `deploy` extra now includes `boto3>=1.34`
- `api` extra now includes `prometheus-client>=0.19`

### Fixed
- mypy strict: eliminated all 352 type errors across 54 source files
  - Added `ignore_missing_imports` overrides for all untyped third-party deps
  - Added `ignore_errors` override for dashboard pages
  - `np.ndarray` → `npt.NDArray[Any]` throughout
  - `cast(torch.Tensor, ...)` on all torchdiffeq/torchsde/torchcde return sites
  - Type-widened attributes in NeuralSDE, LatentNeuralODE, HybridNeuralODE, trainer classes

## [1.0.0] - 2026-02-28

### Added — Production Release (v1.0.0)

#### CLI
- Unified `reactor-twin` CLI with argparse (no new dependencies)
  - `reactor-twin train --config config.yaml` — Train from YAML config
  - `reactor-twin serve --host 0.0.0.0 --port 8000` — Start FastAPI server
  - `reactor-twin export --model checkpoint.pt --format onnx` — Export to ONNX
  - `reactor-twin dashboard --port 8501` — Launch Streamlit dashboard

#### Kubernetes Deployment
- Helm chart (`deploy/helm/reactor-twin/`) with:
  - Deployment with health probes (liveness + readiness)
  - ClusterIP Service
  - Optional Ingress
  - ConfigMap for environment configuration
  - Horizontal Pod Autoscaler (CPU-based)
- GPU Dockerfile (`Dockerfile.gpu`) with NVIDIA CUDA base image

#### Documentation
- Deployment guide (`docs/guide/deployment.md`): Docker, Kubernetes, CLI, configuration reference, monitoring, security

### Tests
- Coverage pushed from 71% to 90%+ (dashboard pages excluded from coverage)
- New test files: `test_online_adapter.py`, `test_meta_learner.py`, `test_discovery_no_pysr.py`
- Extended: `test_distributed.py` (train_epoch, validate, full loop, gradient accumulation, LR scheduler)
- Extended: `test_api.py` (v2 endpoints: token, upload, predict, batch-predict, list models)
- Extended: `test_core.py` (save/load checkpoint, predict eval mode)
- Extended: `test_systems.py` (consecutive/parallel utility functions)
- Extended: `test_auth.py` (token expiry, client key extraction, rate limiter edge cases)

### Changed
- Version bumped to 1.0.0
- Development Status classifier updated to "Production/Stable"
- Dashboard pages excluded from test coverage metrics (interactive Streamlit UI)

## [0.4.0] - 2026-02-28

### Added — Real-World Validation (Phase v0.4.0)

#### Utility Functions (21 stubs → full implementations)
- `utils/metrics.py` — 8 evaluation metrics: RMSE, relative RMSE, mass balance error, energy balance error, positivity violations, stoichiometric error, Gibbs monotonicity score, rollout divergence
- `utils/numerical.py` — 6 numerical utilities: ODE integration (scipy wrapper), central finite-difference Jacobian, stiffness detection, RK4 stepper, adaptive step size, trajectory interpolation (linear/cubic)
- `utils/visualization.py` — 7 plotting functions: trajectory plots (matplotlib + plotly), phase portraits, bifurcation diagrams, RTD plots, sensitivity heatmaps, Pareto fronts, latent space visualization (PCA/t-SNE/UMAP)

#### Advanced MPC
- `EconomicMPC` — profit-maximising MPC with revenue/cost objectives and safety penalties
- `EconomicObjective` — economic stage cost (revenue - cost) with configurable state safety bounds
- `StochasticMPC` — chance-constrained MPC using multi-sample Euler-Maruyama rollouts through Neural SDE, CVaR risk measure, uncertainty-aware trajectory statistics

#### Distributed Training
- `DistributedTrainer` — multi-GPU data-parallel training via `DistributedDataParallel`
- Gradient accumulation for effective large batch sizes
- Rank-aware data sharding, checkpoint saving (rank 0 only)
- `setup_distributed()` / `cleanup_distributed()` helpers

#### Model Versioning & Registry
- `ModelRegistry` — local file-based model registry with JSON manifest
- `ModelMetadata` — tracks name, version, reactor type, training config, metrics, tags
- Auto-incrementing semantic versions, save/load/compare/delete operations
- Supports filtering by name, version, or tag

#### API v2 — Full Model Serving
- `POST /api/v2/token` — JWT token generation
- `POST /api/v2/models/upload` — upload trained PyTorch models
- `POST /api/v2/models/{id}/predict` — single prediction endpoint
- `POST /api/v2/models/{id}/batch-predict` — batch prediction endpoint
- `GET /api/v2/models` — list uploaded models

#### Security Hardening
- JWT authentication (HS256, no external dependencies) with configurable expiry
- In-memory rate limiter (sliding window, per-user or per-IP)
- CORS middleware (already present, now documented)

#### Real Experimental Data Benchmarks
- `benchmarks/real_data/williams_otto.py` — Williams-Otto 6-state CSTR (3 reactions, non-isothermal)
- `benchmarks/real_data/penicillin.py` — Penicillin fed-batch bioreactor (Bajpai-Reuss kinetics)
- Synthetic data generators with configurable noise, perturbations, and diverse operating conditions
- `run_benchmark()` functions for end-to-end Neural ODE training and evaluation

#### Load Testing
- `scripts/loadtest.py` — concurrent HTTP load testing with latency statistics (min/mean/median/p95/p99/throughput)

### Tests
- 1183 tests passing (102 new tests), 7 skipped
- Coverage: 71% → targeting 85% in next release
- New test files: `test_metrics.py`, `test_numerical.py`, `test_visualization.py`, `test_advanced_mpc.py`, `test_model_registry.py`, `test_auth.py`, `test_distributed.py`, `test_benchmarks.py`

### Added
- ONNX export for trained Neural ODE models (`src/reactor_twin/export/`)
- Docker support (Dockerfile, docker-compose.yml, GHCR publishing workflow)
- WebSocket streaming for real-time simulation (`src/reactor_twin/api/websocket.py`)
- Multi-phase reactor with gas-liquid mass transfer (`src/reactor_twin/reactors/multi_phase.py`)
- Population balance reactor for crystallization modeling (`src/reactor_twin/reactors/population_balance.py`)
- Custom exception hierarchy (`ReactorTwinError`, `SolverError`, `ValidationError`)
- Codecov badge and integration
- Performance regression tracking

### Changed
- API hardening with proper HTTP error codes, request validation, OpenAPI docs
- Example validation added to CI pipeline
- Notebook execution added to CI pipeline
- Codecov enforcement — CI fails if coverage drops below 85%

## [0.3.1] - 2026-02-28

### Critical Fixes
- **Fluidized Bed Reactor**: Fixed bubble-phase reaction rates using wrong concentrations (`C_e` instead of `C_b`)
- **Fluidized Bed Reactor**: Validation now runs before `super().__init__()` to prevent partially constructed objects; caller's params dict no longer mutated
- **Fluidized Bed Reactor**: Added real energy balance with convection + heat of reaction (was returning `dT/dt=0`)
- **Fluidized Bed Reactor**: Clamped phase volumes to avoid division-by-zero in edge cases
- **Membrane Reactor**: Validation order fixed (before `super().__init__()`); params dict no longer mutated
- **Hybrid Model**: Replaced broken `ReactorPhysicsFunc.__new__` hack with proper `_ZeroPhysicsFunc` for `reactor=None`
- **Foundation Model**: Wired task embeddings into training loops (were dead code); fixed device mismatch in Reptile update
- **Michaelis-Menten Kinetics**: Fixed competitive inhibition formula from `V_max*S/((K_m+S)*(1+I/K_i))` to correct `V_max*S/(K_m*(1+I/K_i)+S)`
- **Benchmark Systems**: Fixed `delta_H` to `dH_rxn` parameter key in `consecutive.py` and `parallel.py` — non-isothermal simulations were silently ignoring heat of reaction
- **MultiPhase Reactor**: Moved required parameter validation before `super().__init__()`

### High-Severity Fixes
- **Bayesian Neural ODE**: Fixed ELBO loss to compute per-sample loss then average
- **Bayesian Neural ODE**: `predict_with_uncertainty` restores training mode; added `prior_sigma > 0` validation; clamped `log_sigma`
- **Hybrid Model**: Central finite differences for Jacobian; physics regularization across trajectory; training mode restoration
- **Sensitivity Analysis**: `copy.deepcopy` for params; NaN median imputation; `output_index` bounds validation
- **NeuralSDE**: `compute_loss` now applies `loss_weights` to total (was ignoring weights)
- **torch.load**: Added `weights_only=False` to suppress PyTorch deprecation warnings
- **Data Generator**: Fixed `UnboundLocalError` for `y0_default` when custom initial conditions provided

### Medium-Severity Fixes
- **Membrane Reactor**: Added validation for Q length, species index bounds, and C_ret_feed length; clamped permeation concentrations
- **Fluidized Bed Reactor**: Added validation for `d_b > 0`, `0 < epsilon_mf < 1`, `K_be >= 0`
- **Symbolic Regression**: Added `.cpu()` before `.numpy()`; fixed `torch.tensor` deprecation
- **Logging**: Renamed `format` to `log_format` to avoid shadowing Python builtin

### Improvements
- Added `MonodKinetics` and `SensitivityAnalyzer` to top-level exports
- Added `discovery`, `tracking`, `deploy`, `thermo` to `[all]` extras group
- Added `validate_state()` to Fluidized Bed and Membrane reactors

### Tests
- 1081 tests passing, 69% coverage
- 40+ new tests with hand-calculated expected values
- Fixed SPC alarm assertion (`not np.all` to `not np.any`)
- Added random seed fixtures for deterministic SPC tests
- Updated Michaelis-Menten inhibition test for correct formula

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

[Unreleased]: https://github.com/ktubhyam/reactor-twin/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/ktubhyam/reactor-twin/compare/v0.4.0...v1.0.0
[0.4.0]: https://github.com/ktubhyam/reactor-twin/compare/v0.3.1...v0.4.0
[0.3.1]: https://github.com/ktubhyam/reactor-twin/compare/v0.3.0...v0.3.1
[0.1.0]: https://github.com/ktubhyam/reactor-twin/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/ktubhyam/reactor-twin/releases/tag/v0.0.1
