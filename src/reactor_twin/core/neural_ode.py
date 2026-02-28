"""Standard Neural ODE implementation."""

from __future__ import annotations

import logging
from typing import Any, cast

import torch
from torchdiffeq import odeint, odeint_adjoint

from reactor_twin.core.base import AbstractNeuralDE
from reactor_twin.core.ode_func import AbstractODEFunc, MLPODEFunc
from reactor_twin.utils.registry import NEURAL_DE_REGISTRY

logger = logging.getLogger(__name__)


@NEURAL_DE_REGISTRY.register("neural_ode")
class NeuralODE(AbstractNeuralDE):
    """Standard Neural ODE for reactor dynamics.

    Parameterizes the derivative dz/dt = f_theta(z, t, u) where f_theta is a
    neural network, and solves the IVP with a black-box ODE integrator. The
    adjoint method enables O(1) memory backpropagation.

    Attributes:
        ode_func: ODE right-hand-side function.
        solver: ODE solver method name.
        atol: Absolute tolerance for solver.
        rtol: Relative tolerance for solver.
        adjoint: Whether to use adjoint method for backprop.
    """

    def __init__(
        self,
        state_dim: int,
        ode_func: AbstractODEFunc | None = None,
        solver: str = "dopri5",
        atol: float = 1e-6,
        rtol: float = 1e-3,
        adjoint: bool = True,
        input_dim: int = 0,
        output_dim: int | None = None,
        **ode_func_kwargs: Any,
    ):
        """Initialize Neural ODE.

        Args:
            state_dim: Dimension of latent state.
            ode_func: ODE function. If None, creates MLPODEFunc.
            solver: Solver method ('dopri5', 'euler', 'rk4', etc.).
            atol: Absolute tolerance.
            rtol: Relative tolerance.
            adjoint: Use adjoint method for memory-efficient backprop.
            input_dim: Dimension of external inputs/controls.
            output_dim: Dimension of observations. Defaults to state_dim.
            **ode_func_kwargs: Arguments for MLPODEFunc if ode_func is None.
        """
        super().__init__(state_dim, input_dim, output_dim)

        # Create ODE function if not provided
        if ode_func is None:
            ode_func = MLPODEFunc(
                state_dim=state_dim,
                input_dim=input_dim,
                **ode_func_kwargs,
            )
        self.ode_func = ode_func

        self.solver = solver
        self.atol = atol
        self.rtol = rtol
        self._integrate = odeint_adjoint if adjoint else odeint

        logger.info(
            f"Initialized NeuralODE: state_dim={state_dim}, solver={solver}, adjoint={adjoint}"
        )

    def forward(
        self,
        z0: torch.Tensor,
        t_span: torch.Tensor,
        controls: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Integrate ODE from z0 over t_span.

        Args:
            z0: Initial state, shape (batch, state_dim).
            t_span: Time points to evaluate at, shape (num_times,).
            controls: External inputs at each time, shape (batch, num_times, input_dim).
                Defaults to None.

        Returns:
            Trajectory z(t), shape (batch, num_times, state_dim).
        """
        # Controls: if provided, set as constant on the ODE function
        if controls is not None:
            if hasattr(self.ode_func, "_constant_controls"):
                self.ode_func._constant_controls = controls
            else:
                logger.warning(
                    "Controls provided but ODE function doesn't support "
                    "constant controls. Ignoring."
                )

        # Integrate
        z_trajectory = self._integrate(
            self.ode_func,
            z0,
            t_span,
            rtol=self.rtol,
            atol=self.atol,
            method=self.solver,
        )

        # Transpose to (batch, time, state)
        z_trajectory = cast(torch.Tensor, z_trajectory).transpose(0, 1)

        return z_trajectory

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        loss_weights: dict[str, float] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute multi-objective loss.

        Args:
            predictions: Model predictions, shape (batch, num_times, state_dim).
            targets: Ground truth, shape (batch, num_times, state_dim).
            loss_weights: Dictionary of loss component weights.
                Default: {'data': 1.0}.

        Returns:
            Dictionary with keys 'total' and 'data'.
        """
        if loss_weights is None:
            loss_weights = {"data": 1.0}

        # Data-fitting loss (MSE)
        data_loss = torch.mean((predictions - targets) ** 2)

        # Total loss (weighted sum)
        total_loss = loss_weights.get("data", 1.0) * data_loss

        return {
            "total": total_loss,
            "data": data_loss,
        }


__all__ = ["NeuralODE"]
