"""Latent Neural ODE with encoder-decoder architecture.

For high-dimensional observation spaces, projects observations into a
low-dimensional latent space, evolves latent dynamics via Neural ODE,
then decodes back to observation space.

Architecture:
    observations -> encoder -> z0 (latent) -> Neural ODE -> z(t) -> decoder -> predictions

Reference: Rubanova et al. (2019). "Latent ODEs for Irregularly-Sampled Time Series." NeurIPS.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import nn
from torchdiffeq import odeint, odeint_adjoint

from reactor_twin.core.base import AbstractNeuralDE
from reactor_twin.core.ode_func import AbstractODEFunc, MLPODEFunc
from reactor_twin.utils.registry import NEURAL_DE_REGISTRY

logger = logging.getLogger(__name__)


class Encoder(nn.Module):
    """Encode observations to latent space.

    Can use RNN/GRU for sequential encoding or MLP for single-point encoding.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        encoder_type: str = "gru",
    ):
        """Initialize encoder.

        Args:
            input_dim: Dimension of observations.
            latent_dim: Dimension of latent space.
            hidden_dim: Hidden layer width.
            num_layers: Number of layers.
            encoder_type: 'gru', 'lstm', or 'mlp'.
        """
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder_type = encoder_type

        if encoder_type == "gru":
            self.rnn = nn.GRU(
                input_dim,
                hidden_dim,
                num_layers,
                batch_first=True,
            )
            self.fc_mean = nn.Linear(hidden_dim, latent_dim)
            self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        elif encoder_type == "lstm":
            self.rnn = nn.LSTM(
                input_dim,
                hidden_dim,
                num_layers,
                batch_first=True,
            )
            self.fc_mean = nn.Linear(hidden_dim, latent_dim)
            self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        elif encoder_type == "mlp":
            # Simple MLP encoder (for single time point)
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            self.fc_mean = nn.Linear(hidden_dim, latent_dim)
            self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode observations to latent distribution.

        Args:
            x: Observations, shape (batch, seq_len, input_dim) for RNN
               or (batch, input_dim) for MLP.

        Returns:
            Tuple of (z_mean, z_logvar), each shape (batch, latent_dim).
        """
        if self.encoder_type in ("gru", "lstm"):
            # Use last hidden state
            _, h_last = self.rnn(x)
            if self.encoder_type == "lstm":
                h_last = h_last[0]  # LSTM returns (h, c)
            h = h_last[-1]  # Last layer hidden state: (batch, hidden_dim)
        else:  # mlp
            h = self.net(x)

        z_mean = self.fc_mean(h)
        z_logvar = self.fc_logvar(h)

        return z_mean, z_logvar


class Decoder(nn.Module):
    """Decode latent states to observation space."""

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
    ):
        """Initialize decoder.

        Args:
            latent_dim: Dimension of latent space.
            output_dim: Dimension of observations.
            hidden_dim: Hidden layer width.
            num_layers: Number of layers.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        layers: list[nn.Module] = []
        in_dim = latent_dim
        for i in range(num_layers):
            out_dim = output_dim if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.ReLU())
            in_dim = hidden_dim

        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent states to observations.

        Args:
            z: Latent states, shape (batch, latent_dim) or (batch, time, latent_dim).

        Returns:
            Observations, same shape but with latent_dim -> output_dim.
        """
        return self.net(z)


@NEURAL_DE_REGISTRY.register("latent_neural_ode")
class LatentNeuralODE(AbstractNeuralDE):
    """Latent Neural ODE with encoder-decoder architecture.

    Workflow:
    1. Encode observations x0 to latent mean/logvar: (mu, sigma)
    2. Sample z0 ~ N(mu, sigma) via reparameterization trick
    3. Integrate latent ODE: z(t) = ODEsolve(f_theta, z0, t_span)
    4. Decode latent trajectory: x_pred(t) = decoder(z(t))

    Attributes:
        encoder: Encoder network (observations -> latent).
        decoder: Decoder network (latent -> observations).
        ode_func: ODE function in latent space.
        latent_dim: Dimension of latent space.
    """

    def __init__(
        self,
        state_dim: int,
        latent_dim: int,
        input_dim: int = 0,
        output_dim: int | None = None,
        ode_func: AbstractODEFunc | None = None,
        encoder_hidden_dim: int = 64,
        decoder_hidden_dim: int = 64,
        encoder_type: str = "gru",
        solver: str = "dopri5",
        atol: float = 1e-6,
        rtol: float = 1e-3,
        adjoint: bool = True,
        **ode_func_kwargs: Any,
    ):
        """Initialize Latent Neural ODE.

        Args:
            state_dim: Dimension of observation space.
            latent_dim: Dimension of latent space (< state_dim for compression).
            input_dim: Dimension of external inputs/controls.
            output_dim: Dimension of observations. Defaults to state_dim.
            ode_func: ODE function in latent space. If None, creates MLPODEFunc.
            encoder_hidden_dim: Hidden dimension for encoder.
            decoder_hidden_dim: Hidden dimension for decoder.
            encoder_type: 'gru', 'lstm', or 'mlp'.
            solver: ODE solver method.
            atol: Absolute tolerance.
            rtol: Relative tolerance.
            adjoint: Use adjoint method for backprop.
            **ode_func_kwargs: Arguments for MLPODEFunc if ode_func is None.
        """
        super().__init__(state_dim, input_dim, output_dim)
        self.latent_dim = latent_dim

        # Create encoder
        self.encoder = Encoder(
            input_dim=state_dim,
            latent_dim=latent_dim,
            hidden_dim=encoder_hidden_dim,
            encoder_type=encoder_type,
        )

        # Create decoder
        self.decoder = Decoder(
            latent_dim=latent_dim,
            output_dim=self.output_dim,
            hidden_dim=decoder_hidden_dim,
        )

        # Create ODE function in latent space
        if ode_func is None:
            ode_func = MLPODEFunc(
                state_dim=latent_dim,
                input_dim=input_dim,
                **ode_func_kwargs,
            )
        self.ode_func = ode_func

        self.solver = solver
        self.atol = atol
        self.rtol = rtol
        self._integrate = odeint_adjoint if adjoint else odeint

        logger.info(
            f"Initialized LatentNeuralODE: "
            f"state_dim={state_dim}, latent_dim={latent_dim}, "
            f"encoder={encoder_type}"
        )

    def encode(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode observations to latent distribution.

        Args:
            x: Observations, shape (batch, input_dim) or (batch, seq_len, input_dim).

        Returns:
            Tuple of (z_mean, z_logvar).
        """
        return self.encoder(x)

    def reparameterize(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """Reparameterization trick: z = mu + eps * sigma.

        Args:
            mu: Mean, shape (batch, latent_dim).
            logvar: Log variance, shape (batch, latent_dim).

        Returns:
            Sampled z, shape (batch, latent_dim).
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent states to observations.

        Args:
            z: Latent states, shape (batch, latent_dim) or (batch, time, latent_dim).

        Returns:
            Observations, shape (batch, output_dim) or (batch, time, output_dim).
        """
        return self.decoder(z)

    def forward(
        self,
        z0: torch.Tensor,
        t_span: torch.Tensor,
        controls: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through Latent Neural ODE.

        Note: z0 here is the *observation* at t=0, not latent state.
        We encode it to get the latent initial condition.

        Args:
            z0: Initial observations, shape (batch, state_dim).
            t_span: Time points, shape (num_times,).
            controls: External inputs (optional).

        Returns:
            Predicted observations, shape (batch, num_times, output_dim).
        """
        # Encode to latent space
        z_mean, z_logvar = self.encode(z0)

        # Sample latent initial condition
        z_latent = self.reparameterize(z_mean, z_logvar)

        # Integrate in latent space
        z_trajectory = self._integrate(
            self.ode_func,
            z_latent,
            t_span,
            rtol=self.rtol,
            atol=self.atol,
            method=self.solver,
        )

        # Transpose to (batch, time, latent_dim)
        z_trajectory = z_trajectory.transpose(0, 1)

        # Decode to observation space
        x_trajectory = self.decode(z_trajectory)

        return x_trajectory

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        loss_weights: dict[str, float] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute loss including reconstruction and KL divergence.

        Args:
            predictions: Model predictions, shape (batch, num_times, output_dim).
            targets: Ground truth, shape (batch, num_times, output_dim).
            loss_weights: Dictionary of loss weights.
                Default: {'reconstruction': 1.0, 'kl': 0.01}.

        Returns:
            Dictionary with keys 'total', 'reconstruction', 'kl'.
        """
        if loss_weights is None:
            loss_weights = {"reconstruction": 1.0, "kl": 0.01}

        # Reconstruction loss (MSE)
        reconstruction_loss = torch.mean((predictions - targets) ** 2)

        # KL divergence (computed from encoder output)
        # This requires encoding the initial observation again
        # For simplicity, we compute it on the first time point
        x0 = targets[:, 0, :]  # (batch, state_dim)
        z_mean, z_logvar = self.encode(x0)

        # KL(q(z) || p(z)) where p(z) = N(0, I)
        kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
        kl_loss = kl_loss / targets.size(0)  # Average over batch

        # Total loss
        total_loss = (
            loss_weights.get("reconstruction", 1.0) * reconstruction_loss
            + loss_weights.get("kl", 0.01) * kl_loss
        )

        return {
            "total": total_loss,
            "reconstruction": reconstruction_loss,
            "kl": kl_loss,
        }


__all__ = ["LatentNeuralODE", "Encoder", "Decoder"]
