"""ReactorTwin: Physics-constrained Neural DEs for chemical reactor digital twins."""

from __future__ import annotations

# Version info
__version__ = "1.1.0"
__author__ = "Tubhyam Karthikeyan"
__email__ = "takarthikeyan25@gmail.com"

# Core imports
from reactor_twin.core import AbstractNeuralDE, NeuralODE
from reactor_twin.core.bayesian_neural_ode import (
    BayesianLinear,
    BayesianMLPODEFunc,
    BayesianNeuralODE,
)
from reactor_twin.core.hybrid_model import HybridNeuralODE, ReactorPhysicsFunc
from reactor_twin.digital_twin import (
    EKFStateEstimator,
    FaultDetector,
    MPCController,
    OnlineAdapter,
    ReptileMetaLearner,
)
from reactor_twin.exceptions import (
    ConfigurationError,
    ConstraintViolationError,
    ExportError,
    ReactorTwinError,
    RegistryError,
    SolverError,
    ValidationError,
)
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
from reactor_twin.reactors import (
    AbstractReactor,
    BatchReactor,
    CSTRReactor,
    FluidizedBedReactor,
    MembraneReactor,
    MultiPhaseReactor,
    PlugFlowReactor,
    PopulationBalanceReactor,
    SemiBatchReactor,
)
from reactor_twin.reactors.kinetics import (
    AbstractKinetics,
    ArrheniusKinetics,
    LangmuirHinshelwoodKinetics,
    MichaelisMentenKinetics,
    MonodKinetics,
    PowerLawKinetics,
    ReversibleKinetics,
)
from reactor_twin.reactors.systems import (
    create_bioreactor_cstr,
    create_consecutive_cstr,
    create_exothermic_cstr,
    create_parallel_cstr,
    create_van_de_vusse_cstr,
)
from reactor_twin.training import MultiObjectiveLoss, ReactorDataGenerator, Trainer
from reactor_twin.training.foundation import (
    FoundationNeuralODE,
    FoundationTrainer,
    ReactorTaskEncoder,
)
from reactor_twin.utils import (
    CONSTRAINT_REGISTRY,
    DIGITAL_TWIN_REGISTRY,
    KINETICS_REGISTRY,
    NEURAL_DE_REGISTRY,
    REACTOR_REGISTRY,
    Registry,
)
from reactor_twin.utils.logging import JSONFormatter, RequestTracer, setup_logging
from reactor_twin.utils.sensitivity import SensitivityAnalyzer

__all__ = [
    # Version
    "__version__",
    # Exceptions
    "ReactorTwinError",
    "SolverError",
    "ValidationError",
    "ExportError",
    "ConstraintViolationError",
    "RegistryError",
    "ConfigurationError",
    # Core Neural DEs
    "AbstractNeuralDE",
    "NeuralODE",
    "BayesianNeuralODE",
    "BayesianLinear",
    "BayesianMLPODEFunc",
    "HybridNeuralODE",
    "ReactorPhysicsFunc",
    # Reactors
    "AbstractReactor",
    "BatchReactor",
    "CSTRReactor",
    "FluidizedBedReactor",
    "MembraneReactor",
    "MultiPhaseReactor",
    "PlugFlowReactor",
    "PopulationBalanceReactor",
    "SemiBatchReactor",
    # Benchmark Systems
    "create_bioreactor_cstr",
    "create_consecutive_cstr",
    "create_exothermic_cstr",
    "create_parallel_cstr",
    "create_van_de_vusse_cstr",
    # Kinetics
    "AbstractKinetics",
    "ArrheniusKinetics",
    "LangmuirHinshelwoodKinetics",
    "MichaelisMentenKinetics",
    "MonodKinetics",
    "PowerLawKinetics",
    "ReversibleKinetics",
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
    # Foundation Model
    "FoundationNeuralODE",
    "FoundationTrainer",
    "ReactorTaskEncoder",
    # Digital Twin
    "EKFStateEstimator",
    "FaultDetector",
    "MPCController",
    "OnlineAdapter",
    "ReptileMetaLearner",
    # Registry System
    "Registry",
    "REACTOR_REGISTRY",
    "KINETICS_REGISTRY",
    "CONSTRAINT_REGISTRY",
    "NEURAL_DE_REGISTRY",
    "DIGITAL_TWIN_REGISTRY",
    # Logging
    "JSONFormatter",
    "RequestTracer",
    "setup_logging",
    # Sensitivity Analysis
    "SensitivityAnalyzer",
]
