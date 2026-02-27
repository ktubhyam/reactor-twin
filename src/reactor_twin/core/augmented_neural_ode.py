"""Augmented Neural ODE with additional dimensions for expressivity.

Standard Neural ODEs can struggle with complex dynamics due to topological
constraints. Augmented Neural ODEs add extra "augmented" dimensions that
don't correspond to physical states but increase the expressivity of the
learned dynamics.

Reference: Dupont et al. (2019). "Augmented Neural ODEs." NeurIPS.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torchdiffeq import odeint, odeint_adjoint

from reactor_twin.core.base import AbstractNeuralDE
from reactor_twin.core.ode_func import AbstractODEFunc, MLPODEFunc
from reactor_twin.utils.registry import NEURAL_DE_REGISTRY

logger = logging.getLogger(__name__)


@NEURAL_DE_REGISTRY.register("augmented_neural_ode")
class AugmentedNeuralODE(AbstractNeuralDE):
    """Augmented Neural ODE with extra dimensions.

    Augments state space with additional dimensions:
        z_full = [z_physical, z_augmented]

    The ODE evolves in the full augmented space, but we only observe
    the physical dimensions in the output.

    Key insight: Adding dimensions allows ODE to "lift" trajectories into
    higher-dimensional space where they can cross without violating uniqueness.

    Attributes:
        augment_dim: Number of augmented dimensions.
        ode_func: ODE function operating on augmented state.
    """

    def __init__(
        self,
        state_dim: int,
        augment_dim: int,
        input_dim: int = 0,
        output_dim: int | None = None,
        ode_func: AbstractODEFunc | None = None,
        solver: str = "dopri5",
        atol: float = 1e-6,
        rtol: float = 1e-3,
        adjoint: bool = True,
        **ode_func_kwargs: Any,
    ):
        """Initialize Augmented Neural ODE.

        Args:
            state_dim: Dimension of physical state.
            augment_dim: Number of augmented dimensions to add.
            input_dim: Dimension of external inputs/controls.
            output_dim: Dimension of observations. Defaults to state_dim.
            ode_func: ODE function in augmented space. If None, creates MLPODEFunc.
            solver: ODE solver method.
            atol: Absolute tolerance.
            rtol: Relative tolerance.
            adjoint: Use adjoint method for backprop.
            **ode_func_kwargs: Arguments for MLPODEFunc if ode_func is None.
        """
        super().__init__(state_dim, input_dim, output_dim)
        self.augment_dim = augment_dim
        self.full_dim = state_dim + augment_dim

        # Create ODE function in augmented space
        if ode_func is None:
            ode_func = MLPODEFunc(
                state_dim=self.full_dim,
                input_dim=input_dim,
                **ode_func_kwargs,
            )
        self.ode_func = ode_func

        self.solver = solver
        self.atol = atol
        self.rtol = rtol
        self._integrate = odeint_adjoint if adjoint else odeint

        logger.info(
            f"Initialized AugmentedNeuralODE: "
            f"state_dim={state_dim}, augment_dim={augment_dim}, "
            f"full_dim={self.full_dim}"
        )

    def augment_state(self, z: torch.Tensor) -> torch.Tensor:
        """Augment physical state with zeros.

        Args:
            z: Physical state, shape (batch, state_dim).

        Returns:
            Augmented state, shape (batch, full_dim).
        """
        batch_size = z.shape[0]
        z_augmented = torch.zeros(
            batch_size,
            self.augment_dim,
            dtype=z.dtype,
            device=z.device,
        )
        z_full = torch.cat([z, z_augmented], dim=-1)
        return z_full

    def extract_physical(self, z_full: torch.Tensor) -> torch.Tensor:
        """Extract physical dimensions from augmented state.

        Args:
            z_full: Augmented state, shape (batch, full_dim) or (batch, time, full_dim).

        Returns:
            Physical state, shape (batch, state_dim) or (batch, time, state_dim).
        """
        return z_full[..., :self.state_dim]

    def forward(
        self,
        z0: torch.Tensor,
        t_span: torch.Tensor,
        controls: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through Augmented Neural ODE.

        Args:
            z0: Initial physical state, shape (batch, state_dim).
            t_span: Time points, shape (num_times,).
            controls: External inputs (optional).

        Returns:
            Physical state trajectory, shape (batch, num_times, state_dim).
        """
        # Augment initial state
        z0_full = self.augment_state(z0)

        # Integrate in augmented space
        z_trajectory_full = self._integrate(
            self.ode_func,
            z0_full,
            t_span,
            rtol=self.rtol,
            atol=self.atol,
            method=self.solver,
        )

        # Transpose to (batch, time, full_dim)
        z_trajectory_full = z_trajectory_full.transpose(0, 1)

        # Extract physical dimensions
        z_trajectory = self.extract_physical(z_trajectory_full)

        return z_trajectory

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        loss_weights: dict[str, float] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute loss on physical dimensions.

        Args:
            predictions: Model predictions, shape (batch, num_times, state_dim).
            targets: Ground truth, shape (batch, num_times, state_dim).
            loss_weights: Dictionary of loss weights.
                Default: {'data': 1.0, 'augment_reg': 0.01}.

        Returns:
            Dictionary with keys 'total', 'data', optionally 'augment_reg'.
        """
        if loss_weights is None:
            loss_weights = {"data": 1.0, "augment_reg": 0.01}

        # Data-fitting loss (MSE on physical dimensions)
        data_loss = torch.mean((predictions - targets) ** 2)

        # Augmented dimension regularization (encourage small augmented states)
        # This requires accessing the full augmented trajectory
        # For simplicity, we skip this in the base implementation
        # It would be added in the training loop if needed
        augment_reg = torch.tensor(0.0, device=predictions.device)

        # Total loss
        total_loss = (
            loss_weights.get("data", 1.0) * data_loss
            + loss_weights.get("augment_reg", 0.01) * augment_reg
        )

        losses = {
            "total": total_loss,
            "data": data_loss,
        }

        if augment_reg.item() > 0:
            losses["augment_reg"] = augment_reg

        return losses

    def get_augmented_trajectory(
        self,
        z0: torch.Tensor,
        t_span: torch.Tensor,
        controls: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Get full augmented trajectory (including augmented dimensions).

        Useful for visualization and analysis.

        Args:
            z0: Initial physical state, shape (batch, state_dim).
            t_span: Time points, shape (num_times,).
            controls: External inputs (optional).

        Returns:
            Full augmented trajectory, shape (batch, num_times, full_dim).
        """
        # Augment initial state
        z0_full = self.augment_state(z0)

        # Integrate in augmented space
        z_trajectory_full = self._integrate(
            self.ode_func,
            z0_full,
            t_span,
            rtol=self.rtol,
            atol=self.atol,
            method=self.solver,
        )

        # Transpose to (batch, time, full_dim)
        z_trajectory_full = z_trajectory_full.transpose(0, 1)

        return z_trajectory_full


__all__ = ["AugmentedNeuralODE"]
