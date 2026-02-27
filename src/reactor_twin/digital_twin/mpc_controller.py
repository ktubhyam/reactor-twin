"""Model Predictive Control using a Neural ODE as the plant model.

Implements receding-horizon optimisation with differentiable Euler
rollouts through the learned dynamics.  The control sequence is
optimised via ``torch.optim.LBFGS`` with warm-starting for real-time
performance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import torch

from reactor_twin.core.base import AbstractNeuralDE

logger = logging.getLogger(__name__)


# ======================================================================
# Objective function
# ======================================================================


@dataclass
class MPCObjective:
    r"""Quadratic stage + terminal cost for MPC.

    .. math::
        J = \sum_{k=0}^{N-1} \bigl[(y_k - y_{\text{ref}})^\top Q\,(y_k - y_{\text{ref}})
            + u_k^\top R\, u_k\bigr]
            + (y_N - y_{\text{ref}})^\top Q_f\,(y_N - y_{\text{ref}})

    Attributes:
        Q: State-tracking weight matrix ``(state_dim, state_dim)``.
        R: Control effort weight matrix ``(input_dim, input_dim)``.
        Q_f: Terminal cost weight.  Defaults to ``Q``.
    """

    Q: torch.Tensor
    R: torch.Tensor
    Q_f: torch.Tensor | None = None

    def stage_cost(
        self,
        y: torch.Tensor,
        y_ref: torch.Tensor,
        u: torch.Tensor,
    ) -> torch.Tensor:
        """Compute single-step stage cost.

        Args:
            y: Predicted state, shape ``(state_dim,)``.
            y_ref: Reference, shape ``(state_dim,)``.
            u: Applied control, shape ``(input_dim,)``.

        Returns:
            Scalar cost tensor.
        """
        e = y - y_ref
        return e @ self.Q @ e + u @ self.R @ u

    def terminal_cost(
        self,
        y: torch.Tensor,
        y_ref: torch.Tensor,
    ) -> torch.Tensor:
        """Compute terminal cost.

        Args:
            y: Final predicted state, shape ``(state_dim,)``.
            y_ref: Reference, shape ``(state_dim,)``.

        Returns:
            Scalar cost tensor.
        """
        Q_f = self.Q_f if self.Q_f is not None else self.Q
        e = y - y_ref
        return e @ Q_f @ e

    def trajectory_cost(
        self,
        trajectory: torch.Tensor,
        y_ref: torch.Tensor,
        controls: torch.Tensor,
    ) -> torch.Tensor:
        """Compute total cost over a predicted horizon.

        Args:
            trajectory: Predicted states ``(horizon+1, state_dim)`` (includes z0).
            y_ref: Reference point ``(state_dim,)``.
            controls: Control sequence ``(horizon, input_dim)``.

        Returns:
            Scalar total cost.
        """
        horizon = controls.shape[0]
        cost = torch.tensor(0.0, device=trajectory.device)
        for k in range(horizon):
            cost = cost + self.stage_cost(trajectory[k + 1], y_ref, controls[k])
        cost = cost + self.terminal_cost(trajectory[-1], y_ref)
        return cost


# ======================================================================
# Control constraints
# ======================================================================


@dataclass
class ControlConstraints:
    """Box constraints on control inputs with optional soft output penalties.

    Attributes:
        u_min: Lower bounds on controls, shape ``(input_dim,)``.
        u_max: Upper bounds on controls, shape ``(input_dim,)``.
        y_min: Soft lower bounds on outputs (optional).
        y_max: Soft upper bounds on outputs (optional).
        penalty_weight: Weight for soft output constraint violations.
    """

    u_min: torch.Tensor
    u_max: torch.Tensor
    y_min: torch.Tensor | None = None
    y_max: torch.Tensor | None = None
    penalty_weight: float = 100.0

    def clamp_controls(self, u: torch.Tensor) -> torch.Tensor:
        """Apply box constraints to control inputs.

        Args:
            u: Raw controls, shape ``(..., input_dim)``.

        Returns:
            Clamped controls.
        """
        return torch.clamp(u, self.u_min, self.u_max)

    def output_penalty(self, y: torch.Tensor) -> torch.Tensor:
        """Soft penalty for output constraint violations.

        Args:
            y: Predicted output, shape ``(..., state_dim)``.

        Returns:
            Scalar penalty.
        """
        penalty = torch.tensor(0.0, device=y.device)
        if self.y_min is not None:
            violation = torch.relu(self.y_min - y)
            penalty = penalty + self.penalty_weight * (violation**2).sum()
        if self.y_max is not None:
            violation = torch.relu(y - self.y_max)
            penalty = penalty + self.penalty_weight * (violation**2).sum()
        return penalty


# ======================================================================
# MPC Controller
# ======================================================================


class MPCController:
    """Receding-horizon MPC using a Neural ODE plant model.

    Uses differentiable Euler rollouts through the learned ``ode_func``
    and optimises the control sequence via ``torch.optim.LBFGS``.

    Attributes:
        model: Neural DE whose ``ode_func.forward(t, z, u)`` provides
            the dynamics.
        horizon: Number of prediction steps.
        dt: Time step for Euler integration.
        objective: Cost function specification.
        constraints: Control/output constraints.
    """

    def __init__(
        self,
        model: AbstractNeuralDE,
        horizon: int = 10,
        dt: float = 0.01,
        objective: MPCObjective | None = None,
        constraints: ControlConstraints | None = None,
        max_iter: int = 20,
        device: str | torch.device = "cpu",
    ) -> None:
        """Initialize MPC controller.

        Args:
            model: Neural DE with ``ode_func`` that accepts ``(t, z, u)``.
            horizon: Prediction/control horizon.
            dt: Euler step size.
            objective: MPC cost function.  A default identity-weighted
                objective is created if ``None``.
            constraints: Box constraints.  ``None`` means unconstrained.
            max_iter: Maximum L-BFGS iterations per solve.
            device: Torch device.
        """
        self.model = model
        self.horizon = horizon
        self.dt = dt
        self.max_iter = max_iter
        self.device = torch.device(device)

        state_dim = model.state_dim
        input_dim = model.input_dim or 1

        if objective is None:
            objective = MPCObjective(
                Q=torch.eye(state_dim, device=self.device),
                R=0.01 * torch.eye(input_dim, device=self.device),
            )
        self.objective = objective
        self.constraints = constraints

        # Warm-start buffer
        self._u_prev: torch.Tensor | None = None

        logger.info(
            f"MPCController: horizon={horizon}, dt={dt}, "
            f"state_dim={state_dim}, input_dim={input_dim}"
        )

    # ------------------------------------------------------------------
    # Trajectory prediction
    # ------------------------------------------------------------------

    def _predict_trajectory(
        self,
        z0: torch.Tensor,
        controls: torch.Tensor,
        t_horizon: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Euler rollout through the Neural ODE dynamics.

        This bypasses ``NeuralODE.forward()`` (which doesn't support
        time-varying controls) and calls ``model.ode_func(t, z, u)``
        directly.

        Args:
            z0: Initial state, shape ``(state_dim,)``.
            controls: Control sequence, shape ``(horizon, input_dim)``.
            t_horizon: Time points (unused; included for interface
                consistency).

        Returns:
            Trajectory ``(horizon + 1, state_dim)`` including ``z0``.
        """
        trajectory = [z0]
        z = z0
        t = torch.tensor(0.0, device=self.device)

        for k in range(controls.shape[0]):
            u = controls[k]
            z_batch = z.unsqueeze(0)
            u_batch = u.unsqueeze(0)
            dzdt = self.model.ode_func(t, z_batch, u_batch).squeeze(0)
            z = z + dzdt * self.dt
            trajectory.append(z)
            t = t + self.dt

        return torch.stack(trajectory)  # (horizon+1, state_dim)

    # ------------------------------------------------------------------
    # Optimisation
    # ------------------------------------------------------------------

    def optimize(
        self,
        z0: torch.Tensor,
        y_ref: torch.Tensor,
        u_init: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Solve the MPC optimisation problem.

        Args:
            z0: Current state, shape ``(state_dim,)``.
            y_ref: Reference setpoint, shape ``(state_dim,)``.
            u_init: Initial guess for control sequence,
                shape ``(horizon, input_dim)``.  Uses warm-start from
                previous solve when ``None``.

        Returns:
            Dictionary with keys ``controls`` (optimised sequence),
            ``trajectory`` (predicted states), ``cost`` (final cost),
            and ``converged`` (bool).
        """
        input_dim = self.model.input_dim or 1
        z0 = z0.detach().to(self.device)
        y_ref = y_ref.detach().to(self.device)

        # Initial guess (warm start)
        if u_init is not None:
            u_seq = u_init.clone().detach().to(self.device).requires_grad_(True)
        elif self._u_prev is not None:
            # Shift previous solution left, repeat last element
            u_seq = torch.cat([self._u_prev[1:], self._u_prev[-1:]], dim=0)
            u_seq = u_seq.detach().requires_grad_(True)
        else:
            u_seq = torch.zeros(self.horizon, input_dim, device=self.device, requires_grad=True)

        optimizer = torch.optim.LBFGS(
            [u_seq],
            max_iter=self.max_iter,
            line_search_fn="strong_wolfe",
        )

        final_cost = torch.tensor(0.0)
        converged = False

        def closure() -> torch.Tensor:
            nonlocal final_cost
            optimizer.zero_grad()

            # Clamp controls
            u_clamped = u_seq
            if self.constraints is not None:
                u_clamped = self.constraints.clamp_controls(u_seq)

            traj = self._predict_trajectory(z0, u_clamped)
            cost = self.objective.trajectory_cost(traj, y_ref, u_clamped)

            # Soft output penalties
            if self.constraints is not None:
                for k in range(1, traj.shape[0]):
                    cost = cost + self.constraints.output_penalty(traj[k])

            cost.backward()
            final_cost = cost.detach()
            return cost

        try:
            optimizer.step(closure)
            converged = True
        except Exception as exc:
            logger.warning(f"MPC optimisation failed: {exc}")

        # Final trajectory with optimised controls
        u_opt = u_seq.detach()
        if self.constraints is not None:
            u_opt = self.constraints.clamp_controls(u_opt)

        with torch.no_grad():
            traj = self._predict_trajectory(z0, u_opt)

        self._u_prev = u_opt.clone()

        return {
            "controls": u_opt,
            "trajectory": traj,
            "cost": final_cost.item(),
            "converged": converged,
        }

    def step(
        self,
        z_current: torch.Tensor,
        y_ref: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Execute a single receding-horizon step.

        Solves the MPC problem and returns only the first control action
        (receding horizon principle).

        Args:
            z_current: Current state, shape ``(state_dim,)``.
            y_ref: Reference setpoint, shape ``(state_dim,)``.

        Returns:
            ``(u_applied, info)`` — the control to apply and the full
            optimisation result dictionary.
        """
        info = self.optimize(z_current, y_ref)
        u_applied = info["controls"][0]
        return u_applied, info


__all__ = [
    "MPCObjective",
    "ControlConstraints",
    "MPCController",
]
