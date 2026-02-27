# Changelog

All notable changes to ReactorTwin will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added (Phase 4)
- BatchReactor with time-varying volume for gas-phase reactions
- SemiBatchReactor with continuous feed and no outflow
- PlugFlowReactor (PFR) with Method of Lines discretization
- MichaelisMentenKinetics for enzyme-catalyzed reactions with competitive inhibition
- PowerLawKinetics for empirical rate expressions with temperature dependence
- LangmuirHinshelwoodKinetics for heterogeneous catalysis on surfaces
- ReversibleKinetics for equilibrium-limited reactions with forward/reverse rates
- Bioreactor CSTR benchmark with Monod growth kinetics (substrate, biomass, product)
- Consecutive reactions CSTR benchmark (A→B→C) with selectivity analysis
- Parallel competing reactions CSTR benchmark (A→B, A→C) with yield optimization

### Added (Phase 3)
- Latent Neural ODE with encoder-decoder architecture for high-dimensional systems
- Augmented Neural ODE with extra dimensions for expressivity
- Neural SDE for stochastic dynamics and uncertainty quantification
- Neural CDE for irregular time series with control path interpolation
- Support for torchsde and torchcde (optional dependencies)

### Added (Phase 2)
- Mass balance constraint with stoichiometric projection
- Energy balance constraint for thermochemical systems
- Thermodynamic constraint (entropy monotonicity, Gibbs energy)
- Stoichiometric constraint (predict rates not species)
- Port-Hamiltonian constraint with learnable structure matrices
- GENERIC constraint (reversible-irreversible coupling)
- Exothermic A→B CSTR benchmark system
- Van de Vusse reaction network CSTR benchmark
- Complete training infrastructure (Trainer, MultiObjectiveLoss, ReactorDataGenerator)
- Multi-objective loss with constraint penalties
- Automatic data generation from reactor models

### Added (Phase 1)
- Core Neural ODE with adjoint method
- Abstract base classes for all components
- Registry system for plugins
- CSTR reactor implementation
- Arrhenius kinetics
- Positivity constraint (hard/soft modes)
- MLPODEFunc and HybridODEFunc
- Complete project structure and documentation

## [0.1.0] - 2026-02-27

Initial architecture setup. Foundation for physics-constrained Neural DEs.

### Added
- Project structure with src layout
- Abstract base classes (AbstractNeuralDE, AbstractReactor, AbstractKinetics, AbstractConstraint)
- Plugin registry system
- Reference implementations (NeuralODE, CSTRReactor, ArrheniusKinetics, PositivityConstraint)
- Full project configuration (pyproject.toml, CI/CD, documentation)
- MIT License
- README with quickstart and examples
