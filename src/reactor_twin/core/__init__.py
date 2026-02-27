"""Core Neural Differential Equation implementations."""

from __future__ import annotations

from reactor_twin.core.base import AbstractNeuralDE
from reactor_twin.core.neural_ode import NeuralODE
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
    # ODE functions
    "MLPODEFunc",
    "ResNetODEFunc",
    "HybridODEFunc",
    "PortHamiltonianODEFunc",
]
