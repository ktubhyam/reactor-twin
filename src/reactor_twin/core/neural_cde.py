"""Neural Controlled Differential Equation for irregular time series.

Neural CDEs extend Neural ODEs to handle irregular, asynchronous observations:
    dz/dt = f_theta(z(t)) * dX/dt

where X(t) is a "control path" (continuous interpolation of observations).

Key advantages:
- Handles missing data and irregular sampling
- Natural continuous-time representation
- Can incorporate raw sensor measurements directly

Reference: Kidger et al. (2020). "Neural Controlled Differential Equations
for Irregular Time Series." NeurIPS Spotlight.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from reactor_twin.core.base import AbstractNeuralDE
from reactor_twin.utils.registry import NEURAL_DE_REGISTRY

logger = logging.getLogger(__name__)

# Check if torchcde is available
try:
    import torchcde
    TORCHCDE_AVAILABLE = True
except ImportError:
    TORCHCDE_AVAILABLE = False
    logger.warning(
        "torchcde not installed. Neural CDE will not work. "
        "Install with: pip install torchcde"
    )


class CDEFunc(nn.Module):
    """CDE vector field function.

    Computes f_theta(z) which will be multiplied by dX/dt.
    """

    def __init__(
        self,
        state_dim: int,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
    ):
        """Initialize CDE function.

        Args:
            state_dim: Dimension of hidden state z.
            input_dim: Dimension of control/input X.
            hidden_dim: Hidden layer width.
            num_layers: Number of layers.
        """
        super().__init__()
        self.state_dim = state_dim
        self.input_dim = input_dim

        # Network outputs matrix: (state_dim, input_dim)
        layers: list[nn.Module] = []
        in_dim = state_dim

        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.Tanh())
            in_dim = hidden_dim

        # Final layer outputs flattened matrix
        layers.append(nn.Linear(in_dim, state_dim * input_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Compute vector field f_theta(z).

        Args:
            t: Time (not used, for compatibility).
            z: Hidden state, shape (batch, state_dim).

        Returns:
            Matrix f_theta(z), shape (batch, state_dim, input_dim).
        """
        batch_size = z.shape[0]
        output = self.net(z)  # (batch, state_dim * input_dim)
        output = output.view(batch_size, self.state_dim, self.input_dim)
        return output


@NEURAL_DE_REGISTRY.register("neural_cde")
class NeuralCDE(AbstractNeuralDE):
    """Neural Controlled Differential Equation.

    Models dynamics controlled by observed time series:
        dz/dt = f_theta(z(t)) * dX/dt

    where X(t) is a continuous path interpolating observations.

    Workflow:
    1. Interpolate irregular observations to get control path X(t)
    2. Solve CDE: z(t) = z0 + integral(f_theta(z) * dX/dt, 0, t)
    3. Read out predictions from z(t)

    Attributes:
        cde_func: CDE vector field function.
        interpolation: Type of interpolation ('linear', 'cubic').
        readout: Linear readout from hidden state to output.
    """

    def __init__(
        self,
        state_dim: int,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int | None = None,
        cde_hidden_dim: int = 64,
        cde_num_layers: int = 3,
        interpolation: str = "cubic",
        solver: str = "dopri5",
        atol: float = 1e-6,
        rtol: float = 1e-3,
        adjoint: bool = True,
    ):
        """Initialize Neural CDE.

        Args:
            state_dim: Dimension of hidden state z (internal).
            input_dim: Dimension of observations X.
            hidden_dim: Hidden dimension (for initial network).
            output_dim: Dimension of predictions. Defaults to input_dim.
            cde_hidden_dim: Hidden dimension for CDE function.
            cde_num_layers: Number of layers in CDE function.
            interpolation: 'linear' or 'cubic' spline interpolation.
            solver: ODE solver method.
            atol: Absolute tolerance.
            rtol: Relative tolerance.
            adjoint: Use adjoint method for backprop.
        """
        super().__init__(state_dim, input_dim, output_dim or input_dim)

        if not TORCHCDE_AVAILABLE:
            raise ImportError(
                "torchcde is required for Neural CDE. "
                "Install with: pip install torchcde"
            )

        # Initial projection from observations to hidden state
        self.initial_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

        # CDE vector field
        self.cde_func = CDEFunc(
            state_dim=state_dim,
            input_dim=input_dim,
            hidden_dim=cde_hidden_dim,
            num_layers=cde_num_layers,
        )

        # Readout from hidden state to output
        self.readout = nn.Linear(state_dim, self.output_dim)

        self.interpolation = interpolation
        self.solver = solver
        self.atol = atol
        self.rtol = rtol
        self.adjoint = adjoint

        logger.info(
            f"Initialized NeuralCDE: "
            f"state_dim={state_dim}, input_dim={input_dim}, "
            f"interpolation={interpolation}"
        )

    def forward(
        self,
        z0: torch.Tensor,
        t_span: torch.Tensor,
        controls: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through Neural CDE.

        Args:
            z0: Initial observations, shape (batch, input_dim).
                NOTE: For CDE, z0 is the observation, not hidden state.
            t_span: Time points, shape (num_times,).
            controls: Full observation sequence, shape (batch, num_times, input_dim).
                Required for Neural CDE (represents X(t)).

        Returns:
            Predictions, shape (batch, num_times, output_dim).
        """
        if controls is None:
            raise ValueError("Neural CDE requires 'controls' (observation sequence)")

        batch_size = controls.shape[0]
        num_times = controls.shape[1]

        # Interpolate observations to get continuous control path
        if self.interpolation == "linear":
            coeffs = torchcde.linear_interpolation_coeffs(controls, t_span)
            X = torchcde.LinearInterpolation(coeffs, t_span)
        elif self.interpolation == "cubic":
            coeffs = torchcde.natural_cubic_coeffs(controls, t_span)
            X = torchcde.CubicSpline(coeffs, t_span)
        else:
            raise ValueError(f"Unknown interpolation: {self.interpolation}")

        # Initial hidden state from first observation
        z_initial = self.initial_network(controls[:, 0, :])  # (batch, state_dim)

        # Solve CDE
        z_trajectory = torchcde.cdeint(
            X=X,
            func=self.cde_func,
            z0=z_initial,
            t=t_span,
            method=self.solver,
            rtol=self.rtol,
            atol=self.atol,
            adjoint=self.adjoint,
        )

        # z_trajectory: (batch, num_times, state_dim)

        # Readout to output space
        predictions = self.readout(z_trajectory)  # (batch, num_times, output_dim)

        return predictions

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        loss_weights: dict[str, float] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute loss for Neural CDE.

        Args:
            predictions: Model predictions, shape (batch, num_times, output_dim).
            targets: Ground truth, shape (batch, num_times, output_dim).
            loss_weights: Dictionary of loss weights.

        Returns:
            Dictionary with keys 'total', 'data'.
        """
        if loss_weights is None:
            loss_weights = {"data": 1.0}

        # Data-fitting loss (MSE)
        # Only compute loss at observed time points (non-NaN targets)
        mask = ~torch.isnan(targets)
        if mask.any():
            data_loss = torch.mean(((predictions - targets) ** 2)[mask])
        else:
            data_loss = torch.mean((predictions - targets) ** 2)

        return {
            "total": data_loss,
            "data": data_loss,
        }

    def forward_with_irregular_observations(
        self,
        observations: torch.Tensor,
        observation_times: torch.Tensor,
        prediction_times: torch.Tensor,
    ) -> torch.Tensor:
        """Handle irregularly-sampled observations.

        Args:
            observations: Observations at irregular times,
                shape (batch, num_obs, input_dim).
            observation_times: Times of observations, shape (batch, num_obs).
            prediction_times: Times to predict at, shape (num_pred,).

        Returns:
            Predictions at prediction_times, shape (batch, num_pred, output_dim).
        """
        batch_size = observations.shape[0]

        # For irregularly-sampled data, we need to:
        # 1. Sort observations by time
        # 2. Interpolate to create control path
        # 3. Solve CDE and evaluate at prediction_times

        # This is a placeholder - full implementation requires handling
        # variable-length sequences per batch element
        raise NotImplementedError(
            "Irregular observations not yet fully implemented. "
            "Use regular grid with forward() and controls argument."
        )


__all__ = ["NeuralCDE", "CDEFunc"]
