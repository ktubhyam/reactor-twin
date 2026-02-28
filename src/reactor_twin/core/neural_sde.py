"""Neural Stochastic Differential Equation for uncertainty quantification.

Neural SDEs model stochastic dynamics:
    dz = f(z, t) dt + g(z, t) dW

where:
- f(z, t) is the drift (deterministic part)
- g(z, t) is the diffusion (stochastic part)
- dW is Brownian motion

Enables uncertainty quantification and robust predictions.

Reference: Li et al. (2020). "Scalable Gradients for Stochastic Differential Equations." AISTATS.
"""

from __future__ import annotations

import logging
from typing import Any, cast

import torch
from torch import nn

from reactor_twin.core.base import AbstractNeuralDE
from reactor_twin.core.ode_func import AbstractODEFunc, MLPODEFunc
from reactor_twin.utils.registry import NEURAL_DE_REGISTRY

logger = logging.getLogger(__name__)

# Check if torchsde is available
try:
    from torchsde import sdeint

    TORCHSDE_AVAILABLE = True
except ImportError:
    TORCHSDE_AVAILABLE = False
    logger.warning(
        "torchsde not installed. Neural SDE will not work. Install with: pip install torchsde"
    )


class SDEFunc(nn.Module):
    """SDE function with drift and diffusion terms."""

    def __init__(
        self,
        drift_func: AbstractODEFunc,
        diffusion_func: nn.Module | None = None,
        noise_type: str = "diagonal",
        sde_type: str = "ito",
    ):
        """Initialize SDE function.

        Args:
            drift_func: Drift term f(z, t).
            diffusion_func: Diffusion term g(z, t). If None, uses learnable diagonal.
            noise_type: 'diagonal', 'additive', 'scalar', or 'general'.
            sde_type: 'ito' or 'stratonovich'.
        """
        super().__init__()
        self.drift_func = drift_func
        self.noise_type = noise_type
        self.sde_type = sde_type

        self.diffusion_func: nn.Sequential | nn.Parameter
        # Create diffusion function if not provided
        if diffusion_func is None:
            state_dim = drift_func.state_dim
            if noise_type == "diagonal":
                # Learnable diagonal diffusion
                self.diffusion_func = nn.Sequential(
                    nn.Linear(state_dim, 64),
                    nn.Softplus(),
                    nn.Linear(64, state_dim),
                    nn.Softplus(),  # Ensure positive
                )
            elif noise_type == "additive":
                # Constant diagonal diffusion
                self.diffusion_func = nn.Parameter(torch.ones(state_dim) * 0.1)
            elif noise_type == "scalar":
                # Single scalar for all dimensions
                self.diffusion_func = nn.Parameter(torch.tensor(0.1))
            else:
                raise ValueError(f"Unsupported noise_type for auto-creation: {noise_type}")
        else:
            self.diffusion_func = diffusion_func  # type: ignore[assignment]

    def f(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Drift term f(z, t).

        Args:
            t: Time, shape () or (batch,).
            z: State, shape (batch, state_dim).

        Returns:
            Drift, shape (batch, state_dim).
        """
        return cast(torch.Tensor, self.drift_func(t, z))

    def g(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Diffusion term g(z, t).

        Args:
            t: Time, shape () or (batch,).
            z: State, shape (batch, state_dim).

        Returns:
            Diffusion matrix, shape depends on noise_type:
                - 'diagonal': (batch, state_dim)
                - 'additive': (batch, state_dim) (constant)
                - 'scalar': (batch, 1)
                - 'general': (batch, state_dim, noise_dim)
        """
        if self.noise_type == "diagonal":
            return cast(torch.Tensor, cast(nn.Sequential, self.diffusion_func)(z))
        elif self.noise_type == "additive":
            batch_size = z.shape[0]
            return cast(nn.Parameter, self.diffusion_func).expand(batch_size, -1)
        elif self.noise_type == "scalar":
            batch_size = z.shape[0]
            return cast(nn.Parameter, self.diffusion_func).expand(batch_size, 1)
        else:
            return cast(torch.Tensor, cast(nn.Sequential, self.diffusion_func)(t, z))


@NEURAL_DE_REGISTRY.register("neural_sde")
class NeuralSDE(AbstractNeuralDE):
    """Neural Stochastic Differential Equation.

    Models stochastic dynamics for uncertainty quantification:
        dz = f(z, t) dt + g(z, t) dW

    Can generate multiple sample paths from the same initial condition.

    Attributes:
        sde_func: SDE function with drift and diffusion.
        noise_type: Type of noise ('diagonal', 'additive', 'scalar').
        sde_type: 'ito' or 'stratonovich'.
    """

    def __init__(
        self,
        state_dim: int,
        drift_func: AbstractODEFunc | None = None,
        diffusion_func: nn.Module | None = None,
        noise_type: str = "diagonal",
        sde_type: str = "ito",
        method: str = "euler",
        dt: float = 1e-2,
        input_dim: int = 0,
        output_dim: int | None = None,
        **drift_func_kwargs: Any,
    ):
        """Initialize Neural SDE.

        Args:
            state_dim: Dimension of state space.
            drift_func: Drift function. If None, creates MLPODEFunc.
            diffusion_func: Diffusion function. If None, creates learnable diagonal.
            noise_type: 'diagonal', 'additive', 'scalar', or 'general'.
            sde_type: 'ito' or 'stratonovich'.
            method: SDE solver method ('euler', 'milstein', 'srk').
            dt: Time step for solver.
            input_dim: Dimension of external inputs/controls.
            output_dim: Dimension of observations. Defaults to state_dim.
            **drift_func_kwargs: Arguments for MLPODEFunc if drift_func is None.
        """
        super().__init__(state_dim, input_dim, output_dim)

        if not TORCHSDE_AVAILABLE:
            raise ImportError(
                "torchsde is required for Neural SDE. Install with: pip install torchsde"
            )

        # Create drift function if not provided
        if drift_func is None:
            drift_func = MLPODEFunc(
                state_dim=state_dim,
                input_dim=input_dim,
                **drift_func_kwargs,
            )

        # Create SDE function
        self.sde_func = SDEFunc(
            drift_func=drift_func,
            diffusion_func=diffusion_func,
            noise_type=noise_type,
            sde_type=sde_type,
        )

        self.noise_type = noise_type
        self.sde_type = sde_type
        self.method = method
        self.dt = dt

        logger.info(
            f"Initialized NeuralSDE: "
            f"state_dim={state_dim}, noise_type={noise_type}, "
            f"method={method}"
        )

    def forward(
        self,
        z0: torch.Tensor,
        t_span: torch.Tensor,
        controls: torch.Tensor | None = None,
        num_samples: int = 1,
    ) -> torch.Tensor:
        """Forward pass through Neural SDE.

        Args:
            z0: Initial state, shape (batch, state_dim).
            t_span: Time points, shape (num_times,).
            controls: External inputs (not yet supported).
            num_samples: Number of SDE sample paths to generate.

        Returns:
            Trajectory, shape (num_samples, batch, num_times, state_dim).
        """
        if controls is not None:
            raise NotImplementedError("Controls not yet supported for Neural SDE")

        # Generate multiple sample paths
        traj_list: list[torch.Tensor] = []
        for _ in range(num_samples):
            # Solve SDE
            z_trajectory = sdeint(
                self.sde_func,
                z0,
                t_span,
                method=self.method,
                dt=self.dt,
            )

            # Transpose to (batch, time, state_dim)
            z_trajectory = cast(torch.Tensor, z_trajectory).transpose(0, 1)
            traj_list.append(z_trajectory)

        # Stack samples: (num_samples, batch, time, state_dim)
        trajectories = torch.stack(traj_list, dim=0)

        # If single sample, remove sample dimension
        if num_samples == 1:
            trajectories = trajectories.squeeze(0)

        return trajectories

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        loss_weights: dict[str, float] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute loss for Neural SDE.

        For stochastic predictions, we average over samples.

        Args:
            predictions: Model predictions, shape (num_samples, batch, time, state_dim)
                or (batch, time, state_dim) if num_samples=1.
            targets: Ground truth, shape (batch, time, state_dim).
            loss_weights: Dictionary of loss weights.

        Returns:
            Dictionary with keys 'total', 'data'.
        """
        if loss_weights is None:
            loss_weights = {"data": 1.0}

        # Handle multi-sample predictions
        if predictions.ndim == 4:  # (num_samples, batch, time, state_dim)
            # Average predictions over samples
            predictions_mean = predictions.mean(dim=0)
        else:
            predictions_mean = predictions

        # Data-fitting loss (MSE)
        data_loss = torch.mean((predictions_mean - targets) ** 2)

        # Could add variance regularization here
        # variance_loss = predictions.var(dim=0).mean()

        return {
            "total": loss_weights.get("data", 1.0) * data_loss,
            "data": data_loss,
        }

    def predict_with_uncertainty(
        self,
        z0: torch.Tensor,
        t_span: torch.Tensor,
        num_samples: int = 100,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate predictions with uncertainty estimates.

        Args:
            z0: Initial state, shape (batch, state_dim).
            t_span: Time points, shape (num_times,).
            num_samples: Number of SDE samples for uncertainty estimation.

        Returns:
            Tuple of (mean, std), each shape (batch, num_times, state_dim).
        """
        self.eval()
        with torch.no_grad():
            # Generate samples
            samples = self.forward(z0, t_span, num_samples=num_samples)
            # samples: (num_samples, batch, num_times, state_dim)

            # Compute statistics
            mean = samples.mean(dim=0)  # (batch, time, state_dim)
            std = samples.std(dim=0)  # (batch, time, state_dim)

        return mean, std


__all__ = ["NeuralSDE", "SDEFunc"]
