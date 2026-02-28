"""Model Predictive Control using a Neural ODE as the plant model.

Implements receding-horizon optimisation with differentiable Euler
rollouts through the learned dynamics.  The control sequence is
optimised via ``torch.optim.LBFGS`` with warm-starting for real-time
performance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, cast

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

        def closure() -> float:
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

            cost.backward()  # type: ignore[no-untyped-call]
            final_cost = cost.detach()
            return cast(float, cost)

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


# ======================================================================
# Economic MPC
# ======================================================================


@dataclass
class EconomicObjective:
    """Economic objective for profit-maximising MPC.

    Unlike the quadratic setpoint-tracking objective, economic MPC
    directly optimises a monetary cost function:

    .. math::
        J = \\sum_{k=0}^{N-1} \\bigl[-\\text{revenue}(y_k) + \\text{cost}(u_k)\\bigr]

    Attributes:
        revenue_weights: Per-state revenue coefficients ``(state_dim,)``.
            Positive values mean that state is worth money (e.g. product
            concentration).
        cost_weights: Per-control cost coefficients ``(input_dim,)``.
            Positive values penalise control effort (e.g. heating cost).
        state_penalties: Optional quadratic penalties for safety bounds
            ``(state_dim,)``.  Applied as ``w_i * max(0, y_min_i - y_i)^2``.
    """

    revenue_weights: torch.Tensor
    cost_weights: torch.Tensor
    state_penalties: torch.Tensor | None = None
    y_min_safety: torch.Tensor | None = None
    y_max_safety: torch.Tensor | None = None

    def stage_cost(
        self,
        y: torch.Tensor,
        u: torch.Tensor,
    ) -> torch.Tensor:
        """Economic stage cost (negative profit).

        Args:
            y: Predicted state ``(state_dim,)``.
            u: Applied control ``(input_dim,)``.

        Returns:
            Scalar cost (lower is better).
        """
        revenue = (self.revenue_weights * y).sum()
        cost = (self.cost_weights * u.abs()).sum()
        stage = -revenue + cost

        # Safety penalties
        if self.state_penalties is not None:
            if self.y_min_safety is not None:
                violation = torch.relu(self.y_min_safety - y)
                stage = stage + (self.state_penalties * violation**2).sum()
            if self.y_max_safety is not None:
                violation = torch.relu(y - self.y_max_safety)
                stage = stage + (self.state_penalties * violation**2).sum()
        return stage

    def trajectory_cost(
        self,
        trajectory: torch.Tensor,
        controls: torch.Tensor,
    ) -> torch.Tensor:
        """Total economic cost over predicted horizon.

        Args:
            trajectory: States ``(horizon+1, state_dim)`` (includes z0).
            controls: Control sequence ``(horizon, input_dim)``.

        Returns:
            Scalar total cost.
        """
        horizon = controls.shape[0]
        cost = torch.tensor(0.0, device=trajectory.device)
        for k in range(horizon):
            cost = cost + self.stage_cost(trajectory[k + 1], controls[k])
        return cost


class EconomicMPC:
    """Economic MPC that maximises profit rather than tracking a setpoint.

    Uses the same Euler rollout and L-BFGS optimisation as
    :class:`MPCController`, but with an economic objective that directly
    optimises revenue minus cost.

    Example::

        econ_obj = EconomicObjective(
            revenue_weights=torch.tensor([0., 10., 0.]),  # product B is valuable
            cost_weights=torch.tensor([1.0]),              # heating cost
        )
        empc = EconomicMPC(model, horizon=20, dt=0.01, objective=econ_obj)
        u, info = empc.step(z_current)
    """

    def __init__(
        self,
        model: AbstractNeuralDE,
        horizon: int = 20,
        dt: float = 0.01,
        objective: EconomicObjective | None = None,
        constraints: ControlConstraints | None = None,
        max_iter: int = 30,
        device: str | torch.device = "cpu",
    ) -> None:
        self.model = model
        self.horizon = horizon
        self.dt = dt
        self.max_iter = max_iter
        self.device = torch.device(device)
        self.constraints = constraints
        self._u_prev: torch.Tensor | None = None

        input_dim = model.input_dim or 1
        state_dim = model.state_dim

        if objective is None:
            objective = EconomicObjective(
                revenue_weights=torch.ones(state_dim, device=self.device),
                cost_weights=0.01 * torch.ones(input_dim, device=self.device),
            )
        self.objective = objective

        logger.info(f"EconomicMPC: horizon={horizon}, dt={dt}")

    def _predict_trajectory(
        self,
        z0: torch.Tensor,
        controls: torch.Tensor,
    ) -> torch.Tensor:
        """Euler rollout (same as MPCController)."""
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
        return torch.stack(trajectory)

    def optimize(
        self,
        z0: torch.Tensor,
        u_init: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Solve the economic MPC problem.

        Args:
            z0: Current state ``(state_dim,)``.
            u_init: Optional initial guess ``(horizon, input_dim)``.

        Returns:
            Dict with ``controls``, ``trajectory``, ``cost``, ``converged``,
            and ``profit`` (negative of cost).
        """
        input_dim = self.model.input_dim or 1
        z0 = z0.detach().to(self.device)

        if u_init is not None:
            u_seq = u_init.clone().detach().to(self.device).requires_grad_(True)
        elif self._u_prev is not None:
            u_seq = torch.cat([self._u_prev[1:], self._u_prev[-1:]], dim=0)
            u_seq = u_seq.detach().requires_grad_(True)
        else:
            u_seq = torch.zeros(self.horizon, input_dim, device=self.device, requires_grad=True)

        optimizer = torch.optim.LBFGS(
            [u_seq], max_iter=self.max_iter, line_search_fn="strong_wolfe"
        )
        final_cost = torch.tensor(0.0)

        def closure() -> float:
            nonlocal final_cost
            optimizer.zero_grad()
            u_clamped = u_seq
            if self.constraints is not None:
                u_clamped = self.constraints.clamp_controls(u_seq)
            traj = self._predict_trajectory(z0, u_clamped)
            cost = self.objective.trajectory_cost(traj, u_clamped)
            if self.constraints is not None:
                for k in range(1, traj.shape[0]):
                    cost = cost + self.constraints.output_penalty(traj[k])
            cost.backward()  # type: ignore[no-untyped-call]
            final_cost = cost.detach()
            return cast(float, cost)

        converged = False
        try:
            optimizer.step(closure)
            converged = True
        except Exception as exc:
            logger.warning(f"EconomicMPC optimisation failed: {exc}")

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
            "profit": -final_cost.item(),
            "converged": converged,
        }

    def step(
        self,
        z_current: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Execute a single receding-horizon economic MPC step.

        Args:
            z_current: Current state ``(state_dim,)``.

        Returns:
            ``(u_applied, info)``
        """
        info = self.optimize(z_current)
        return info["controls"][0], info


# ======================================================================
# Stochastic MPC (using Neural SDE uncertainty)
# ======================================================================


class StochasticMPC:
    """Stochastic MPC using Neural SDE multi-sample rollouts.

    Draws ``n_samples`` stochastic trajectories from the Neural SDE
    and optimises the expected cost with chance constraints on state
    bounds.

    Example::

        smpc = StochasticMPC(sde_model, horizon=10, n_samples=16)
        u, info = smpc.step(z_current, y_ref)
    """

    def __init__(
        self,
        model: AbstractNeuralDE,
        horizon: int = 10,
        dt: float = 0.01,
        objective: MPCObjective | None = None,
        constraints: ControlConstraints | None = None,
        n_samples: int = 16,
        risk_alpha: float = 0.95,
        max_iter: int = 20,
        device: str | torch.device = "cpu",
    ) -> None:
        """Initialize Stochastic MPC.

        Args:
            model: Neural SDE (or any Neural DE) model.
            horizon: Prediction horizon.
            dt: Euler-Maruyama step size.
            objective: Cost function (defaults to identity-weighted quadratic).
            constraints: Control/output constraints.
            n_samples: Number of stochastic rollout samples.
            risk_alpha: CVaR confidence level for chance constraints (0.5-1.0).
            max_iter: L-BFGS iterations.
            device: Torch device.
        """
        self.model = model
        self.horizon = horizon
        self.dt = dt
        self.n_samples = n_samples
        self.risk_alpha = risk_alpha
        self.max_iter = max_iter
        self.device = torch.device(device)
        self.constraints = constraints
        self._u_prev: torch.Tensor | None = None

        state_dim = model.state_dim
        input_dim = model.input_dim or 1

        if objective is None:
            objective = MPCObjective(
                Q=torch.eye(state_dim, device=self.device),
                R=0.01 * torch.eye(input_dim, device=self.device),
            )
        self.objective = objective

        logger.info(
            f"StochasticMPC: horizon={horizon}, dt={dt}, "
            f"n_samples={n_samples}, risk_alpha={risk_alpha}"
        )

    def _stochastic_rollout(
        self,
        z0: torch.Tensor,
        controls: torch.Tensor,
    ) -> torch.Tensor:
        """Euler-Maruyama rollout with additive noise.

        Draws ``n_samples`` parallel trajectories.  If the model has a
        ``diffusion`` attribute (Neural SDE), it is used for the noise
        term.  Otherwise, a small Gaussian perturbation is added.

        Args:
            z0: Initial state ``(state_dim,)``.
            controls: ``(horizon, input_dim)``.

        Returns:
            Trajectories ``(n_samples, horizon+1, state_dim)``.
        """
        z = z0.unsqueeze(0).expand(self.n_samples, -1).clone()  # (n_samples, state_dim)
        trajectories = [z]

        t = torch.tensor(0.0, device=self.device)
        has_diffusion = hasattr(self.model, "ode_func") and hasattr(
            self.model.ode_func, "diffusion"
        )

        for k in range(controls.shape[0]):
            u = controls[k].unsqueeze(0).expand(self.n_samples, -1)
            dzdt = self.model.ode_func(t, z, u)  # (n_samples, state_dim)

            # Noise term
            dW = torch.randn_like(z) * (self.dt**0.5)
            if has_diffusion:
                sigma = self.model.ode_func.diffusion(t, z)
                noise = sigma * dW
            else:
                noise = 0.01 * dW  # small default noise

            z = z + dzdt * self.dt + noise
            trajectories.append(z)
            t = t + self.dt

        return torch.stack(trajectories, dim=1)  # (n_samples, horizon+1, state_dim)

    def optimize(
        self,
        z0: torch.Tensor,
        y_ref: torch.Tensor,
        u_init: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Solve the stochastic MPC problem.

        Minimises the expected cost across samples, plus a CVaR penalty
        on constraint violations.

        Args:
            z0: Current state ``(state_dim,)``.
            y_ref: Reference ``(state_dim,)``.
            u_init: Optional initial guess ``(horizon, input_dim)``.

        Returns:
            Dict with ``controls``, ``mean_trajectory``, ``cost``,
            ``converged``, ``std_trajectory``.
        """
        input_dim = self.model.input_dim or 1
        z0 = z0.detach().to(self.device)
        y_ref = y_ref.detach().to(self.device)

        if u_init is not None:
            u_seq = u_init.clone().detach().to(self.device).requires_grad_(True)
        elif self._u_prev is not None:
            u_seq = torch.cat([self._u_prev[1:], self._u_prev[-1:]], dim=0)
            u_seq = u_seq.detach().requires_grad_(True)
        else:
            u_seq = torch.zeros(self.horizon, input_dim, device=self.device, requires_grad=True)

        optimizer = torch.optim.LBFGS(
            [u_seq], max_iter=self.max_iter, line_search_fn="strong_wolfe"
        )
        final_cost = torch.tensor(0.0)

        def closure() -> float:
            nonlocal final_cost
            optimizer.zero_grad()

            u_clamped = u_seq
            if self.constraints is not None:
                u_clamped = self.constraints.clamp_controls(u_seq)

            # Multi-sample rollout
            trajs = self._stochastic_rollout(z0, u_clamped)  # (n_samples, H+1, state_dim)

            # Expected cost across samples
            costs = torch.zeros(self.n_samples, device=self.device)
            for s in range(self.n_samples):
                costs[s] = self.objective.trajectory_cost(trajs[s], y_ref, u_clamped)

            # CVaR: mean of worst (1-alpha) fraction
            n_worst = max(1, int((1.0 - self.risk_alpha) * self.n_samples))
            sorted_costs, _ = torch.sort(costs, descending=True)
            cvar = sorted_costs[:n_worst].mean()

            # Combined: expected cost + CVaR penalty
            expected_cost = costs.mean()
            cost = expected_cost + 0.5 * cvar

            # Output penalties on mean trajectory
            if self.constraints is not None:
                mean_traj = trajs.mean(dim=0)
                for k in range(1, mean_traj.shape[0]):
                    cost = cost + self.constraints.output_penalty(mean_traj[k])

            cost.backward()  # type: ignore[no-untyped-call]
            final_cost = cost.detach()
            return cast(float, cost)

        converged = False
        try:
            optimizer.step(closure)
            converged = True
        except Exception as exc:
            logger.warning(f"StochasticMPC optimisation failed: {exc}")

        u_opt = u_seq.detach()
        if self.constraints is not None:
            u_opt = self.constraints.clamp_controls(u_opt)

        with torch.no_grad():
            trajs = self._stochastic_rollout(z0, u_opt)
            mean_traj = trajs.mean(dim=0)
            std_traj = trajs.std(dim=0)

        self._u_prev = u_opt.clone()
        return {
            "controls": u_opt,
            "mean_trajectory": mean_traj,
            "std_trajectory": std_traj,
            "cost": final_cost.item(),
            "converged": converged,
        }

    def step(
        self,
        z_current: torch.Tensor,
        y_ref: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Execute a single stochastic MPC step.

        Args:
            z_current: Current state ``(state_dim,)``.
            y_ref: Reference setpoint ``(state_dim,)``.

        Returns:
            ``(u_applied, info)``
        """
        info = self.optimize(z_current, y_ref)
        return info["controls"][0], info


__all__ = [
    "MPCObjective",
    "ControlConstraints",
    "MPCController",
    "EconomicObjective",
    "EconomicMPC",
    "StochasticMPC",
]
