"""ReactorTwin: Physics-constrained Neural DEs for chemical reactor digital twins."""

from __future__ import annotations

# Version info
__version__ = "0.1.0"
__author__ = "Tubhyam Karthikeyan"
__email__ = "takarthikeyan25@gmail.com"

# Core imports
from reactor_twin.core import AbstractNeuralDE, NeuralODE
from reactor_twin.physics import (
    AbstractConstraint,
    ConstraintPipeline,
    EnergyBalanceConstraint,
    GENERICConstraint,
    MassBalanceConstraint,
    PortHamiltonianConstraint,
    PositivityConstraint,
    StoichiometricConstraint,
    ThermodynamicConstraint,
)
from reactor_twin.reactors import AbstractReactor, CSTRReactor
from reactor_twin.reactors.kinetics import AbstractKinetics, ArrheniusKinetics
from reactor_twin.reactors.systems import create_exothermic_cstr, create_van_de_vusse_cstr
from reactor_twin.training import MultiObjectiveLoss, ReactorDataGenerator, Trainer
from reactor_twin.utils import (
    CONSTRAINT_REGISTRY,
    KINETICS_REGISTRY,
    NEURAL_DE_REGISTRY,
    REACTOR_REGISTRY,
    Registry,
)

__all__ = [
    # Version
    "__version__",
    # Core Neural DEs
    "AbstractNeuralDE",
    "NeuralODE",
    # Reactors
    "AbstractReactor",
    "CSTRReactor",
    # Benchmark Systems
    "create_exothermic_cstr",
    "create_van_de_vusse_cstr",
    # Kinetics
    "AbstractKinetics",
    "ArrheniusKinetics",
    # Physics Constraints
    "AbstractConstraint",
    "ConstraintPipeline",
    "PositivityConstraint",
    "MassBalanceConstraint",
    "EnergyBalanceConstraint",
    "ThermodynamicConstraint",
    "StoichiometricConstraint",
    "PortHamiltonianConstraint",
    "GENERICConstraint",
    # Training
    "Trainer",
    "MultiObjectiveLoss",
    "ReactorDataGenerator",
    # Registry System
    "Registry",
    "REACTOR_REGISTRY",
    "KINETICS_REGISTRY",
    "CONSTRAINT_REGISTRY",
    "NEURAL_DE_REGISTRY",
]
