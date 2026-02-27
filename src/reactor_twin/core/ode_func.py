"""ODE right-hand-side function networks."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class AbstractODEFunc(nn.Module, ABC):
    """Abstract base class for ODE right-hand-side functions.

    The ODE function computes dz/dt = f(t, z, u) where:
    - t is time (scalar or batch)
    - z is state vector, shape (batch, state_dim)
    - u is control input, shape (batch, input_dim)
    """

    def __init__(self, state_dim: int, input_dim: int = 0):
        """Initialize ODE function.

        Args:
            state_dim: Dimension of state vector z.
            input_dim: Dimension of control input u.
        """
        super().__init__()
        self.state_dim = state_dim
        self.input_dim = input_dim

    @abstractmethod
    def forward(
        self,
        t: torch.Tensor,
        z: torch.Tensor,
        u: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute dz/dt.

        Args:
            t: Time, shape () or (batch,).
            z: State, shape (batch, state_dim).
            u: Control input, shape (batch, input_dim). Defaults to None.

        Returns:
            Time derivative dz/dt, shape (batch, state_dim).
        """
        raise NotImplementedError("Subclasses must implement forward()")


class MLPODEFunc(AbstractODEFunc):
    """Standard MLP-based ODE function."""

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        input_dim: int = 0,
        activation: str = "softplus",
    ):
        """Initialize MLP ODE function.

        Args:
            state_dim: State dimension.
            hidden_dim: Hidden layer width.
            num_layers: Number of hidden layers.
            input_dim: Control input dimension.
            activation: Activation function name.
        """
        super().__init__(state_dim, input_dim)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Build MLP
        input_size = state_dim + input_dim + 1  # z + u + t
        layers: list[nn.Module] = []

        for i in range(num_layers):
            in_features = input_size if i == 0 else hidden_dim
            out_features = state_dim if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_features, out_features))
            if i < num_layers - 1:
                if activation == "softplus":
                    layers.append(nn.Softplus())
                elif activation == "tanh":
                    layers.append(nn.Tanh())
                elif activation == "relu":
                    layers.append(nn.ReLU())
                else:
                    raise ValueError(f"Unknown activation: {activation}")

        self.net = nn.Sequential(*layers)
        logger.debug(
            f"Initialized MLPODEFunc: state_dim={state_dim}, "
            f"hidden={hidden_dim}, layers={num_layers}"
        )

    def forward(
        self,
        t: torch.Tensor,
        z: torch.Tensor,
        u: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute dz/dt.

        Args:
            t: Time, shape () or (batch,).
            z: State, shape (batch, state_dim).
            u: Control, shape (batch, input_dim).

        Returns:
            dz/dt, shape (batch, state_dim).
        """
        batch_size = z.shape[0]

        # Expand time to batch
        if t.ndim == 0:
            t = t.expand(batch_size, 1)
        else:
            t = t.reshape(batch_size, 1)

        # Concatenate inputs
        inputs = [z, t]
        if u is not None:
            inputs.append(u)
        x = torch.cat(inputs, dim=-1)

        return self.net(x)


class ResNetODEFunc(AbstractODEFunc):
    """ODE function with residual connections."""

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        input_dim: int = 0,
    ):
        """Initialize ResNet ODE function.

        Args:
            state_dim: State dimension.
            hidden_dim: Hidden layer width (must equal state_dim for residuals).
            num_layers: Number of residual blocks.
            input_dim: Control input dimension.
        """
        super().__init__(state_dim, input_dim)
        raise NotImplementedError("TODO: Implement ResNetODEFunc")

    def forward(
        self,
        t: torch.Tensor,
        z: torch.Tensor,
        u: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute dz/dt with residual connections."""
        raise NotImplementedError("TODO: Implement ResNetODEFunc.forward()")


class HybridODEFunc(AbstractODEFunc):
    """Hybrid physics + neural correction ODE function."""

    def __init__(
        self,
        state_dim: int,
        physics_func: nn.Module,
        neural_func: AbstractODEFunc,
        correction_weight: float = 1.0,
    ):
        """Initialize hybrid ODE function.

        Args:
            state_dim: State dimension.
            physics_func: Known physics model.
            neural_func: Neural correction term.
            correction_weight: Weight for neural correction (0 = pure physics).
        """
        super().__init__(state_dim, neural_func.input_dim)
        self.physics_func = physics_func
        self.neural_func = neural_func
        self.correction_weight = correction_weight
        logger.debug(f"Initialized HybridODEFunc with correction_weight={correction_weight}")

    def forward(
        self,
        t: torch.Tensor,
        z: torch.Tensor,
        u: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute dz/dt = physics(z, t, u) + alpha * neural(z, t, u).

        Args:
            t: Time.
            z: State.
            u: Control.

        Returns:
            dz/dt with physics + neural correction.
        """
        physics_term = self.physics_func(t, z, u)
        neural_term = self.neural_func(t, z, u)
        return physics_term + self.correction_weight * neural_term


class PortHamiltonianODEFunc(AbstractODEFunc):
    """Port-Hamiltonian structure-preserving ODE function.

    Enforces dz/dt = (J - R) * grad_H(z) + B * u
    where J is skew-symmetric, R is positive semi-definite.
    """

    def __init__(self, state_dim: int, input_dim: int = 0):
        """Initialize Port-Hamiltonian ODE function.

        Args:
            state_dim: State dimension (must be even for J matrix structure).
            input_dim: Control input dimension.
        """
        super().__init__(state_dim, input_dim)
        raise NotImplementedError("TODO: Implement PortHamiltonianODEFunc")

    def forward(
        self,
        t: torch.Tensor,
        z: torch.Tensor,
        u: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute Port-Hamiltonian dynamics."""
        raise NotImplementedError("TODO: Implement PortHamiltonianODEFunc.forward()")


__all__ = [
    "AbstractODEFunc",
    "MLPODEFunc",
    "ResNetODEFunc",
    "HybridODEFunc",
    "PortHamiltonianODEFunc",
]
