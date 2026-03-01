"""ODE right-hand-side function networks."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import cast

import torch
from torch import nn

from reactor_twin.exceptions import ValidationError

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
                    raise ValidationError(f"Unknown activation: {activation}")

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
            z: State, shape (batch, state_dim) or (state_dim,).
            u: Control, shape (batch, input_dim).

        Returns:
            dz/dt, shape (batch, state_dim) or (state_dim,).
        """
        # Handle unbatched input
        squeezed = z.ndim == 1
        if squeezed:
            z = z.unsqueeze(0)
            if u is not None:
                u = u.unsqueeze(0)

        batch_size = z.shape[0]
        input_size = self.state_dim + self.input_dim + 1

        # Pre-allocate buffer and fill in-place to avoid torch.cat overhead
        x = z.new_empty(batch_size, input_size)
        x[:, : self.state_dim] = z
        if t.ndim == 0:
            x[:, self.state_dim] = t
        else:
            x[:, self.state_dim] = t.reshape(batch_size)
        if u is not None:
            x[:, self.state_dim + 1 :] = u

        out = cast(torch.Tensor, self.net(x))
        return out.squeeze(0) if squeezed else out


class ResNetODEFunc(AbstractODEFunc):
    """ODE function with residual connections.

    Each residual block computes: h = h + activation(linear(h)).
    An input projection maps [z, t, u] to hidden_dim, and an output
    projection maps hidden_dim back to state_dim.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        input_dim: int = 0,
        activation: str = "softplus",
    ):
        """Initialize ResNet ODE function.

        Args:
            state_dim: State dimension.
            hidden_dim: Hidden layer width.
            num_layers: Number of residual blocks.
            input_dim: Control input dimension.
            activation: Activation function name.
        """
        super().__init__(state_dim, input_dim)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        total_input = state_dim + input_dim + 1  # z + u + t
        self.input_proj = nn.Linear(total_input, hidden_dim)

        act_fn: nn.Module
        if activation == "softplus":
            act_fn = nn.Softplus()
        elif activation == "tanh":
            act_fn = nn.Tanh()
        elif activation == "relu":
            act_fn = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        blocks: list[nn.Module] = []
        for _ in range(num_layers):
            blocks.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), act_fn))
        self.blocks = nn.ModuleList(blocks)

        self.output_proj = nn.Linear(hidden_dim, state_dim)
        logger.debug(
            f"Initialized ResNetODEFunc: state_dim={state_dim}, "
            f"hidden={hidden_dim}, blocks={num_layers}"
        )

    def forward(
        self,
        t: torch.Tensor,
        z: torch.Tensor,
        u: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute dz/dt with residual connections."""
        batch_size = z.shape[0]
        if t.ndim == 0:
            t_expand = t.expand(batch_size, 1)
        else:
            t_expand = t.reshape(batch_size, 1)

        inputs = [z, t_expand]
        if u is not None:
            inputs.append(u)
        x = torch.cat(inputs, dim=-1)

        h = self.input_proj(x)
        for block in self.blocks:
            h = h + cast(torch.Tensor, block(h))
        return cast(torch.Tensor, self.output_proj(h))


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
        physics_term = cast(torch.Tensor, self.physics_func(t, z, u))
        neural_term = cast(torch.Tensor, self.neural_func(t, z, u))
        return physics_term + self.correction_weight * neural_term


class PortHamiltonianODEFunc(AbstractODEFunc):
    """Port-Hamiltonian structure-preserving ODE function.

    Enforces dz/dt = (J - R) * grad_H(z) + B * u
    where J is skew-symmetric, R is positive semi-definite, and
    H(z) is a learned Hamiltonian (energy) function.
    """

    def __init__(
        self,
        state_dim: int,
        input_dim: int = 0,
        hidden_dim: int = 64,
    ):
        """Initialize Port-Hamiltonian ODE function.

        Args:
            state_dim: State dimension.
            input_dim: Control input dimension.
            hidden_dim: Hidden dimension for Hamiltonian network.
        """
        super().__init__(state_dim, input_dim)

        # J parameter: skew-symmetric via J = A - A^T
        self.J_param = nn.Parameter(torch.randn(state_dim, state_dim) * 0.1)

        # R parameter: PSD via R = B B^T
        self.R_factor = nn.Parameter(torch.randn(state_dim, state_dim) * 0.1)

        # Input matrix B
        if input_dim > 0:
            self.B = nn.Parameter(torch.randn(state_dim, input_dim) * 0.1)
        else:
            self.register_buffer("B", torch.zeros(state_dim, 1))

        # Hamiltonian network H(z) -> scalar
        self.hamiltonian_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Softplus(),
            nn.Linear(hidden_dim, 1),
        )
        logger.debug(f"Initialized PortHamiltonianODEFunc: state_dim={state_dim}")

    def get_J(self) -> torch.Tensor:
        """Get skew-symmetric J matrix."""
        return self.J_param - self.J_param.T

    def get_R(self) -> torch.Tensor:
        """Get positive semi-definite R matrix."""
        return self.R_factor @ self.R_factor.T

    def hamiltonian(self, z: torch.Tensor) -> torch.Tensor:
        """Compute Hamiltonian H(z), shape (batch,)."""
        return cast(torch.Tensor, self.hamiltonian_net(z)).squeeze(-1)

    def forward(
        self,
        t: torch.Tensor,
        z: torch.Tensor,
        u: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute dz/dt = (J - R) grad_H(z) + B u."""
        # Do not detach z when it already requires grad — detaching severs the
        # autograd graph and breaks the adjoint method (∂f/∂z becomes zero).
        z_in = z if z.requires_grad else z.detach().requires_grad_(True)
        H = self.hamiltonian(z_in)
        grad_H = torch.autograd.grad(H.sum(), z_in, create_graph=True)[0]

        J = self.get_J()
        R = self.get_R()
        JR = J - R  # (state_dim, state_dim)

        dz_dt = torch.einsum("ij,bj->bi", JR, grad_H)

        if u is not None and self.input_dim > 0:
            dz_dt = dz_dt + torch.einsum("ij,bj->bi", self.B, u)

        return dz_dt


__all__ = [
    "AbstractODEFunc",
    "MLPODEFunc",
    "ResNetODEFunc",
    "HybridODEFunc",
    "PortHamiltonianODEFunc",
]
