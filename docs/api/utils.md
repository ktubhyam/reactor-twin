# Utilities

Registry system and constants.

## Registry

::: reactor_twin.utils.registry.Registry

## Built-in Registries

The following registries are pre-populated with all built-in components:

- `REACTOR_REGISTRY` — Reactor types (cstr, batch, semi_batch, pfr)
- `KINETICS_REGISTRY` — Kinetics models (arrhenius, michaelis_menten, power_law, langmuir_hinshelwood, reversible, monod)
- `CONSTRAINT_REGISTRY` — Physics constraints (positivity, mass_balance, energy_balance, stoichiometry, port_hamiltonian, generic, thermodynamic)
- `NEURAL_DE_REGISTRY` — Neural DE variants (neural_ode, augmented_neural_ode, latent_neural_ode, neural_sde, neural_cde)
- `DIGITAL_TWIN_REGISTRY` — Digital twin components (ekf, fault_detector, mpc, online_adapter, meta_learner)

## Constants

::: reactor_twin.utils.constants
