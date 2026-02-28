"""Hybrid mechanistic-neural ODE model.

Combines a known physics model (AbstractReactor) with a neural correction
term, enabling learning of unmodeled dynamics while respecting known
physics.  Uses torch.autograd.Function with finite-difference Jacobian
to make the non-differentiable reactor ODE compatible with backprop.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from torch import nn
from torchdiffeq import odeint, odeint_adjoint

from reactor_twin.core.base import AbstractNeuralDE
from reactor_twin.core.ode_func import AbstractODEFunc, MLPODEFunc
from reactor_twin.reactors.base import AbstractReactor
from reactor_twin.utils.registry import NEURAL_DE_REGISTRY

logger = logging.getLogger(__name__)


class _ReactorPhysicsAutograd(torch.autograd.Function):
    """Custom autograd for non-differentiable reactor ODE via finite differences."""

    @staticmethod
    def forward(ctx, z: torch.Tensor, t: torch.Tensor, reactor: Any, eps: float) -> torch.Tensor:
        ctx.reactor = reactor
        ctx.eps = eps
        ctx.save_for_backward(z, t)

        batch_size = z.shape[0]
        result = torch.zeros_like(z)

        t_val = t.item() if t.ndim == 0 else float(t.flatten()[0])

        for b in range(batch_size):
            z_np = z[b].detach().cpu().numpy().astype(np.float64)
            dz_np = reactor.ode_rhs(t_val, z_np)
            result[b] = torch.tensor(dz_np, dtype=z.dtype, device=z.device)

        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        z, t = ctx.saved_tensors
        reactor = ctx.reactor
        eps = ctx.eps

        batch_size, state_dim = z.shape
        t_val = t.item() if t.ndim == 0 else float(t.flatten()[0])

        # Central finite-difference Jacobian
        grad_z = torch.zeros_like(z)
        for b in range(batch_size):
            z_np = z[b].detach().cpu().numpy().astype(np.float64)
            jac = np.zeros((state_dim, state_dim), dtype=np.float64)
            for j in range(state_dim):
                z_fwd = z_np.copy()
                z_bwd = z_np.copy()
                z_fwd[j] += eps
                z_bwd[j] -= eps
                f_fwd = np.array(reactor.ode_rhs(t_val, z_fwd), dtype=np.float64)
                f_bwd = np.array(reactor.ode_rhs(t_val, z_bwd), dtype=np.float64)
                jac[:, j] = (f_fwd - f_bwd) / (2.0 * eps)

            # grad_z[b] = J^T @ grad_output[b]
            g = grad_output[b].detach().cpu().numpy().astype(np.float64)
            grad_z[b] = torch.tensor(jac.T @ g, dtype=z.dtype, device=z.device)

        return grad_z, None, None, None


class _ZeroPhysicsFunc(nn.Module):
    """Returns zeros for physics when no reactor is provided."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("_dummy", torch.zeros(1))

    def forward(
        self,
        t: torch.Tensor,
        z: torch.Tensor,
        u: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return torch.zeros_like(z)


class ReactorPhysicsFunc(nn.Module):
    """Wraps an AbstractReactor.ode_rhs as a differentiable torch Module.

    Uses finite-difference Jacobian via custom autograd.Function
    for gradient computation.
    """

    def __init__(self, reactor: AbstractReactor, eps: float = 1e-5):
        super().__init__()
        self.reactor = reactor
        self.eps = eps
        # Register a dummy parameter so this is recognized as nn.Module with device
        self.register_buffer("_dummy", torch.zeros(1))

    def forward(
        self,
        t: torch.Tensor,
        z: torch.Tensor,
        u: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return _ReactorPhysicsAutograd.apply(z, t, self.reactor, self.eps)


class _HybridODEFunc(nn.Module):
    """Combined physics + neural ODE function for torchdiffeq."""

    def __init__(
        self,
        physics_func: ReactorPhysicsFunc | _ZeroPhysicsFunc,
        neural_func: AbstractODEFunc,
        alpha: float = 1.0,
    ):
        super().__init__()
        self.physics_func = physics_func
        self.neural_func = neural_func
        self.alpha = alpha

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        physics = self.physics_func(t, z)
        neural = self.neural_func(t, z)
        return physics + self.alpha * neural


@NEURAL_DE_REGISTRY.register("hybrid_neural_ode")
class HybridNeuralODE(AbstractNeuralDE):
    """Hybrid mechanistic-neural ODE model.

    Combines known reactor physics with a neural correction:
        dz/dt = f_physics(z, t) + alpha * f_neural(z, t)

    The physics component wraps an AbstractReactor, and the neural
    component learns residual dynamics not captured by the model.

    Attributes:
        physics_func: Differentiable wrapper around reactor ODE.
        neural_func: Neural correction network.
        alpha: Correction weight (0 = pure physics).
        physics_reg_weight: Regularization weight for ||f_neural||^2.
    """

    def __init__(
        self,
        state_dim: int,
        reactor: AbstractReactor | None = None,
        neural_func: AbstractODEFunc | None = None,
        alpha: float = 1.0,
        physics_reg_weight: float = 0.01,
        solver: str = "dopri5",
        atol: float = 1e-6,
        rtol: float = 1e-3,
        adjoint: bool = False,
        input_dim: int = 0,
        output_dim: int | None = None,
        **neural_func_kwargs: Any,
    ):
        super().__init__(state_dim, input_dim, output_dim)

        # Physics function
        if reactor is not None:
            self.physics_func = ReactorPhysicsFunc(reactor)
        else:
            self.physics_func = _ZeroPhysicsFunc()

        # Neural correction function
        if neural_func is None:
            neural_func = MLPODEFunc(
                state_dim=state_dim,
                input_dim=input_dim,
                **neural_func_kwargs,
            )
        self.neural_func = neural_func

        self.alpha = alpha
        self.physics_reg_weight = physics_reg_weight
        self.solver = solver
        self.atol = atol
        self.rtol = rtol
        self._integrate = odeint_adjoint if adjoint else odeint

        # Combined ODE function
        self._hybrid_func = _HybridODEFunc(self.physics_func, self.neural_func, alpha)

        logger.info(
            f"Initialized HybridNeuralODE: state_dim={state_dim}, alpha={alpha}, solver={solver}"
        )

    def forward(
        self,
        z0: torch.Tensor,
        t_span: torch.Tensor,
        controls: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Integrate hybrid ODE.

        Args:
            z0: Initial state, shape (batch, state_dim).
            t_span: Time points, shape (num_times,).
            controls: Not yet supported.

        Returns:
            Trajectory, shape (batch, num_times, state_dim).
        """
        z_traj = self._integrate(
            self._hybrid_func,
            z0,
            t_span,
            rtol=self.rtol,
            atol=self.atol,
            method=self.solver,
        )
        return z_traj.transpose(0, 1)

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        loss_weights: dict[str, float] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute loss with physics regularization.

        Args:
            predictions: Shape (batch, time, state).
            targets: Shape (batch, time, state).
            loss_weights: Optional weights.

        Returns:
            Dict with 'total', 'data', 'physics_reg' keys.
        """
        if loss_weights is None:
            loss_weights = {"data": 1.0}

        data_loss = torch.mean((predictions - targets) ** 2)

        # Physics regularization: penalize ||f_neural||^2
        # Sample from multiple time steps across the trajectory
        num_time_steps = predictions.shape[1]
        num_samples = min(num_time_steps, 4)
        indices = torch.linspace(0, num_time_steps - 1, num_samples).long()

        reg_sum = torch.tensor(0.0, device=predictions.device)
        for idx in indices:
            z_sample = predictions[:, idx, :]
            t_sample = torch.tensor(float(idx), device=z_sample.device)
            neural_output = self.neural_func(t_sample, z_sample)
            reg_sum = reg_sum + torch.mean(neural_output**2)
        physics_reg = reg_sum / len(indices)

        total = loss_weights.get("data", 1.0) * data_loss + self.physics_reg_weight * physics_reg

        return {
            "total": total,
            "data": data_loss,
            "physics_reg": physics_reg,
        }

    def get_correction_magnitude(
        self,
        z0: torch.Tensor,
        t_span: torch.Tensor,
    ) -> torch.Tensor:
        """Compute ratio ||f_neural|| / ||f_total|| as diagnostic.

        Args:
            z0: Initial state, shape (batch, state_dim).
            t_span: Time points, shape (num_times,).

        Returns:
            Ratio scalar (averaged over batch and time).
        """
        was_training = self.training
        self.eval()
        with torch.no_grad():
            traj = self.forward(z0, t_span)
            # Evaluate at each trajectory point
            neural_norms = []
            total_norms = []
            for i in range(traj.shape[1]):
                z_i = traj[:, i, :]
                t_i = t_span[i] if i < len(t_span) else t_span[-1]
                t_tensor = (
                    t_i.clone().detach().to(dtype=z_i.dtype, device=z_i.device)
                    if isinstance(t_i, torch.Tensor)
                    else torch.tensor(t_i, dtype=z_i.dtype, device=z_i.device)
                )

                neural = self.neural_func(t_tensor, z_i)
                total = self._hybrid_func(t_tensor, z_i)

                neural_norms.append(torch.norm(neural, dim=-1))
                total_norms.append(torch.norm(total, dim=-1))

            neural_norm = torch.stack(neural_norms).mean()
            total_norm = torch.stack(total_norms).mean()

            # Avoid division by zero
            ratio = neural_norm / (total_norm + 1e-8)

        if was_training:
            self.train()

        return ratio


__all__ = ["ReactorPhysicsFunc", "HybridNeuralODE"]
