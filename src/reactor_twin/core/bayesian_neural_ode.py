"""Bayesian Neural ODE with weight uncertainty via variational inference.

Uses Gaussian weight posteriors with the reparameterization trick for
differentiable sampling and analytic KL divergence.  Training minimises
the evidence lower bound (ELBO): data_loss + beta * KL.

Reference: Blundell et al. (2015). "Weight Uncertainty in Neural Networks."
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
from torch import nn
from torchdiffeq import odeint, odeint_adjoint

from reactor_twin.core.base import AbstractNeuralDE
from reactor_twin.core.ode_func import AbstractODEFunc
from reactor_twin.utils.registry import NEURAL_DE_REGISTRY

logger = logging.getLogger(__name__)


class BayesianLinear(nn.Module):
    """Linear layer with Gaussian weight posteriors.

    Each weight and bias has a mean (mu) and a log-variance (log_sigma)
    parameter.  Forward samples via the reparameterization trick:
        w = mu + sigma * epsilon,   epsilon ~ N(0, 1)
    """

    def __init__(self, in_features: int, out_features: int, prior_sigma: float = 1.0):
        super().__init__()
        if prior_sigma <= 0:
            raise ValueError(f"prior_sigma must be > 0, got {prior_sigma}")

        self.in_features = in_features
        self.out_features = out_features
        self.prior_sigma = prior_sigma

        # Posterior parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_log_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_log_sigma = nn.Parameter(torch.empty(out_features))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Xavier-like init for mu
        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        fan_in = self.in_features
        bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias_mu, -bound, bound)
        # Start with small sigma (log_sigma ~ -5 => sigma ~ 0.007)
        nn.init.constant_(self.weight_log_sigma, -5.0)
        nn.init.constant_(self.bias_log_sigma, -5.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Sample weights and apply linear transform."""
        weight_sigma = torch.exp(torch.clamp(self.weight_log_sigma, min=-20, max=10))
        bias_sigma = torch.exp(torch.clamp(self.bias_log_sigma, min=-20, max=10))

        # Reparameterization trick
        weight = self.weight_mu + weight_sigma * torch.randn_like(self.weight_mu)
        bias = self.bias_mu + bias_sigma * torch.randn_like(self.bias_mu)

        return nn.functional.linear(x, weight, bias)

    def kl_divergence(self) -> torch.Tensor:
        """Analytic KL(q(w) || p(w)) for Gaussian prior/posterior.

        KL = 0.5 * sum[ sigma_q^2/sigma_p^2 + (mu_p - mu_q)^2/sigma_p^2
                        - 1 + log(sigma_p^2/sigma_q^2) ]
        With zero-mean prior: mu_p = 0.
        """
        prior_var = self.prior_sigma**2

        # Weights
        weight_var = torch.exp(2.0 * torch.clamp(self.weight_log_sigma, min=-20, max=10))
        kl_w = 0.5 * torch.sum(
            weight_var / prior_var
            + self.weight_mu**2 / prior_var
            - 1.0
            - 2.0 * torch.clamp(self.weight_log_sigma, min=-20, max=10)
            + math.log(prior_var)
        )

        # Biases
        bias_var = torch.exp(2.0 * torch.clamp(self.bias_log_sigma, min=-20, max=10))
        kl_b = 0.5 * torch.sum(
            bias_var / prior_var
            + self.bias_mu**2 / prior_var
            - 1.0
            - 2.0 * torch.clamp(self.bias_log_sigma, min=-20, max=10)
            + math.log(prior_var)
        )

        return kl_w + kl_b


class BayesianMLPODEFunc(AbstractODEFunc):
    """MLP ODE function with Bayesian linear layers."""

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        input_dim: int = 0,
        activation: str = "softplus",
        prior_sigma: float = 1.0,
    ):
        super().__init__(state_dim, input_dim)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        total_input = state_dim + input_dim + 1  # z + u + t

        layers: list[nn.Module] = []
        for i in range(num_layers):
            in_f = total_input if i == 0 else hidden_dim
            out_f = state_dim if i == num_layers - 1 else hidden_dim
            layers.append(BayesianLinear(in_f, out_f, prior_sigma=prior_sigma))
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
        self._bayesian_layers = [m for m in self.net if isinstance(m, BayesianLinear)]

    def forward(
        self,
        t: torch.Tensor,
        z: torch.Tensor,
        u: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = z.shape[0]
        if t.ndim == 0:
            t_expand = t.expand(batch_size, 1)
        else:
            t_expand = t.reshape(batch_size, 1)

        inputs = [z, t_expand]
        if u is not None:
            inputs.append(u)
        x = torch.cat(inputs, dim=-1)
        return self.net(x)

    def kl_divergence(self) -> torch.Tensor:
        """Total KL divergence across all Bayesian layers."""
        kl = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer in self._bayesian_layers:
            kl = kl + layer.kl_divergence()
        return kl


@NEURAL_DE_REGISTRY.register("bayesian_neural_ode")
class BayesianNeuralODE(AbstractNeuralDE):
    """Bayesian Neural ODE with weight uncertainty.

    Uses variational inference (ELBO) to learn a posterior over
    ODE function weights, enabling principled uncertainty
    quantification through multiple forward passes.

    Attributes:
        ode_func: Bayesian ODE right-hand-side function.
        solver: ODE solver method.
        beta: KL divergence weight in ELBO.
    """

    def __init__(
        self,
        state_dim: int,
        ode_func: BayesianMLPODEFunc | None = None,
        solver: str = "dopri5",
        atol: float = 1e-6,
        rtol: float = 1e-3,
        adjoint: bool = False,
        beta: float = 1.0,
        input_dim: int = 0,
        output_dim: int | None = None,
        **ode_func_kwargs: Any,
    ):
        super().__init__(state_dim, input_dim, output_dim)

        if ode_func is None:
            ode_func = BayesianMLPODEFunc(
                state_dim=state_dim,
                input_dim=input_dim,
                **ode_func_kwargs,
            )
        self.ode_func = ode_func
        self.solver = solver
        self.atol = atol
        self.rtol = rtol
        self.beta = beta
        self._integrate = odeint_adjoint if adjoint else odeint

        logger.info(
            f"Initialized BayesianNeuralODE: state_dim={state_dim}, solver={solver}, beta={beta}"
        )

    def forward(
        self,
        z0: torch.Tensor,
        t_span: torch.Tensor,
        controls: torch.Tensor | None = None,
        num_samples: int = 1,
    ) -> torch.Tensor:
        """Forward pass with optional multi-sample uncertainty.

        Args:
            z0: Initial state, shape (batch, state_dim).
            t_span: Time points, shape (num_times,).
            controls: Not yet supported.
            num_samples: Number of forward passes for uncertainty.

        Returns:
            If num_samples == 1: shape (batch, num_times, state_dim).
            If num_samples > 1: shape (num_samples, batch, num_times, state_dim).
        """
        if num_samples == 1:
            z_traj = self._integrate(
                self.ode_func,
                z0,
                t_span,
                rtol=self.rtol,
                atol=self.atol,
                method=self.solver,
            )
            return z_traj.transpose(0, 1)  # (batch, time, state)

        trajectories = []
        for _ in range(num_samples):
            z_traj = self._integrate(
                self.ode_func,
                z0,
                t_span,
                rtol=self.rtol,
                atol=self.atol,
                method=self.solver,
            )
            trajectories.append(z_traj.transpose(0, 1))

        return torch.stack(trajectories, dim=0)  # (samples, batch, time, state)

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        loss_weights: dict[str, float] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute ELBO loss = data_loss + beta * KL.

        Args:
            predictions: Shape (batch, time, state) or (samples, batch, time, state).
            targets: Shape (batch, time, state).
            loss_weights: Optional weights.

        Returns:
            Dict with 'total', 'data', 'kl' keys.
        """
        if loss_weights is None:
            loss_weights = {"data": 1.0}

        # Handle multi-sample predictions: compute loss per sample then average
        if predictions.ndim == 4:
            per_sample_losses = []
            for s in range(predictions.shape[0]):
                per_sample_losses.append(torch.mean((predictions[s] - targets) ** 2))
            data_loss = torch.stack(per_sample_losses).mean()
        else:
            data_loss = torch.mean((predictions - targets) ** 2)

        kl_loss = self.ode_func.kl_divergence()

        total = loss_weights.get("data", 1.0) * data_loss + self.beta * kl_loss

        return {
            "total": total,
            "data": data_loss,
            "kl": kl_loss,
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
            num_samples: Number of forward passes.

        Returns:
            Tuple of (mean, std), each shape (batch, num_times, state_dim).
        """
        was_training = self.training
        self.eval()
        with torch.no_grad():
            samples = self.forward(z0, t_span, num_samples=num_samples)
            # samples: (num_samples, batch, time, state)
            mean = samples.mean(dim=0)
            std = samples.std(dim=0)
        if was_training:
            self.train()
        return mean, std


__all__ = ["BayesianLinear", "BayesianMLPODEFunc", "BayesianNeuralODE"]
