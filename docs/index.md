# ReactorTwin

**Physics-constrained Neural Differential Equations for Chemical Reactor Digital Twins**

ReactorTwin provides a modular framework for building digital twins of chemical reactors using physics-informed neural differential equations.

## Key Features

- **4 Reactor Types**: CSTR, Batch, Semi-batch, Plug Flow (PFR)
- **6 Kinetics Models**: Arrhenius, Michaelis-Menten, Power Law, Langmuir-Hinshelwood, Reversible, Monod
- **5 Neural DE Variants**: Neural ODE, Latent Neural ODE, Augmented Neural ODE, Neural SDE, Neural CDE
- **7 Physics Constraints**: Positivity, Mass Balance, Energy Balance, Thermodynamics, Stoichiometry, Port-Hamiltonian, GENERIC
- **Digital Twin Features**: EKF state estimation, fault detection, MPC control, online adaptation, meta-learning
- **Interactive Dashboard**: 10-page Streamlit dashboard for visualization

## Quick Example

```python
from reactor_twin import CSTRReactor, ArrheniusKinetics, NeuralODE

# Define reactor with Arrhenius kinetics
kinetics = ArrheniusKinetics(
    name="A_to_B",
    num_reactions=1,
    params={"k0": [1e10], "Ea": [50000.0], "stoich": [[-1, 1]]},
)
reactor = CSTRReactor(
    name="cstr",
    num_species=2,
    params={"V": 100, "F": 10, "C_feed": [1.0, 0.0], "T_feed": 350},
    kinetics=kinetics,
    isothermal=True,
)

# Train a Neural ODE to learn the dynamics
model = NeuralODE(state_dim=2, hidden_dim=32, num_layers=2)
```

## Installation

```bash
pip install reactor-twin
```
