"""Core Neural Differential Equation implementations."""

from __future__ import annotations

from reactor_twin.core.augmented_neural_ode import AugmentedNeuralODE
from reactor_twin.core.base import AbstractNeuralDE
from reactor_twin.core.bayesian_neural_ode import (
    BayesianLinear,
    BayesianMLPODEFunc,
    BayesianNeuralODE,
)
from reactor_twin.core.hybrid_model import HybridNeuralODE, ReactorPhysicsFunc
from reactor_twin.core.latent_neural_ode import Decoder, Encoder, LatentNeuralODE
from reactor_twin.core.neural_cde import CDEFunc, NeuralCDE
from reactor_twin.core.neural_ode import NeuralODE
from reactor_twin.core.neural_sde import NeuralSDE, SDEFunc
from reactor_twin.core.ode_func import (
    AbstractODEFunc,
    HybridODEFunc,
    MLPODEFunc,
    PortHamiltonianODEFunc,
    ResNetODEFunc,
)

__all__ = [
    # Base classes
    "AbstractNeuralDE",
    "AbstractODEFunc",
    # Neural DE implementations
    "NeuralODE",
    "LatentNeuralODE",
    "AugmentedNeuralODE",
    "NeuralSDE",
    "NeuralCDE",
    "BayesianNeuralODE",
    "HybridNeuralODE",
    # Bayesian components
    "BayesianLinear",
    "BayesianMLPODEFunc",
    # Hybrid components
    "ReactorPhysicsFunc",
    # Encoder/Decoder for Latent ODE
    "Encoder",
    "Decoder",
    # SDE/CDE functions
    "SDEFunc",
    "CDEFunc",
    # ODE functions
    "MLPODEFunc",
    "ResNetODEFunc",
    "HybridODEFunc",
    "PortHamiltonianODEFunc",
]
