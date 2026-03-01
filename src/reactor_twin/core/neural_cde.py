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

from typing import cast

import torch
from torch import nn

from reactor_twin.core.base import AbstractNeuralDE
from reactor_twin.exceptions import ValidationError
from reactor_twin.utils.registry import NEURAL_DE_REGISTRY

logger = logging.getLogger(__name__)

# Check if torchcde is available
try:
    import torchcde

    TORCHCDE_AVAILABLE = True
except ImportError:
    TORCHCDE_AVAILABLE = False
    logger.warning(
        "torchcde not installed. Neural CDE will not work. Install with: pip install torchcde"
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

        for _i in range(num_layers - 1):
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
        output = cast(torch.Tensor, output).view(batch_size, self.state_dim, self.input_dim)
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
                "torchcde is required for Neural CDE. Install with: pip install torchcde"
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
            raise ValidationError("Neural CDE requires 'controls' (observation sequence)")

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
        predictions = cast(torch.Tensor, self.readout(cast(torch.Tensor, z_trajectory)))  # (batch, num_times, output_dim)

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

        Builds a unified time grid from the union of observation_times and
        prediction_times, places observations at their respective time indices,
        linearly interpolates to fill gaps, solves the CDE over the full grid,
        and returns predictions at prediction_times.

        Args:
            observations: Observations at irregular times,
                shape (batch, num_obs, input_dim).
            observation_times: Shared observation times, shape (num_obs,).
                Must be 1D — the same time grid for every batch element.
            prediction_times: Times to predict at, shape (num_pred,).

        Returns:
            Predictions at prediction_times, shape (batch, num_pred, output_dim).

        Raises:
            ValueError: If observation_times is not 1D.
        """
        if observation_times.ndim != 1:
            raise ValueError(
                "observation_times must be 1D (shared across batch elements). "
                f"Got shape {tuple(observation_times.shape)}. "
                "If observation times differ per batch element, pad with NaN and "
                "use a shared superset grid."
            )

        batch_size, num_obs, _ = observations.shape
        device = observations.device
        dtype = observations.dtype

        # Build unified time grid: sorted union of observation and prediction times
        all_times = torch.cat([observation_times.to(dtype), prediction_times.to(dtype)])
        unified_times, _ = torch.sort(torch.unique(all_times))
        n_unified = unified_times.shape[0]

        # Create observation tensor on unified grid, NaN-filled initially
        X_unified = torch.full(
            (batch_size, n_unified, self.input_dim),
            float("nan"),
            dtype=dtype,
            device=device,
        )

        # Place observations at their respective positions in unified_times
        for k in range(num_obs):
            t_k = observation_times[k].to(dtype)
            matches = (unified_times == t_k).nonzero(as_tuple=True)[0]
            if len(matches) > 0:
                X_unified[:, matches[0], :] = observations[:, k, :]

        # Fill NaN gaps by linear interpolation
        X_unified = _fill_nan_linear(X_unified)

        # Build control path interpolation
        if self.interpolation == "linear":
            coeffs = torchcde.linear_interpolation_coeffs(X_unified, unified_times)
            X_path = torchcde.LinearInterpolation(coeffs, unified_times)
        elif self.interpolation == "cubic":
            coeffs = torchcde.natural_cubic_coeffs(X_unified, unified_times)
            X_path = torchcde.CubicSpline(coeffs, unified_times)
        else:
            raise ValueError(f"Unknown interpolation: {self.interpolation}")

        # Initial hidden state from first observation
        z_initial = self.initial_network(observations[:, 0, :])  # (batch, state_dim)

        # Solve CDE over unified grid
        z_trajectory = torchcde.cdeint(
            X=X_path,
            func=self.cde_func,
            z0=z_initial,
            t=unified_times,
            method=self.solver,
            rtol=self.rtol,
            atol=self.atol,
            adjoint=self.adjoint,
        )
        # z_trajectory: (batch, n_unified, state_dim)

        # Readout
        predictions_unified = cast(
            torch.Tensor, self.readout(cast(torch.Tensor, z_trajectory))
        )  # (batch, n_unified, output_dim)

        # Extract at prediction_times indices.  Use searchsorted instead of float
        # equality to avoid fragile [0][0] access on nonzero() results.
        pred_indices = torch.searchsorted(
            unified_times.contiguous(), prediction_times.to(dtype).contiguous()
        )
        pred_indices = pred_indices.clamp(0, n_unified - 1)

        return predictions_unified[:, pred_indices, :]


def _fill_nan_linear(x: torch.Tensor) -> torch.Tensor:
    """Fill NaN values by linear interpolation along the time axis (dim 1).

    Leading/trailing NaNs are filled by constant extrapolation from the
    nearest known value.

    Args:
        x: Tensor of shape (batch, time, dim) with possible NaN entries.

    Returns:
        Tensor of same shape with NaNs replaced.
    """
    result = x.clone()
    batch, n_time, n_dim = result.shape
    t_idx = torch.arange(n_time, dtype=x.dtype, device=x.device)

    for d in range(n_dim):
        y = result[:, :, d]  # (batch, time)
        valid_mask = ~torch.isnan(y)  # (batch, time)

        for b in range(batch):
            valid = valid_mask[b]
            if valid.all() or not valid.any():
                continue

            known_t = t_idx[valid]
            known_v = y[b, valid]
            query_t = t_idx[~valid]

            # Clamp to known range for constant extrapolation at edges
            query_t_clamped = query_t.clamp(known_t[0], known_t[-1])

            right = torch.searchsorted(known_t.contiguous(), query_t_clamped.contiguous())
            right = right.clamp(1, len(known_t) - 1)
            left = right - 1

            t0, t1 = known_t[left], known_t[right]
            v0, v1 = known_v[left], known_v[right]
            alpha = ((query_t_clamped - t0) / (t1 - t0).clamp(min=1e-8))
            result[b, ~valid, d] = v0 + alpha * (v1 - v0)

    return result


__all__ = ["NeuralCDE", "CDEFunc"]
