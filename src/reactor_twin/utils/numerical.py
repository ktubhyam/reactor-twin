"""Numerical utilities for reactor simulations and Neural DEs."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import torch
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

logger = logging.getLogger(__name__)


def integrate_ode(
    func: Callable[..., Any],
    t_span: tuple[float, float],
    y0: npt.NDArray[Any],
    t_eval: npt.NDArray[Any] | None = None,
    method: str = "RK45",
    **kwargs: Any,
) -> npt.NDArray[Any]:
    """Integrate ODE using scipy backend.

    Args:
        func: Right-hand side function dy/dt = func(t, y)
        t_span: Integration interval (t_start, t_end)
        y0: Initial condition, shape (n_states,)
        t_eval: Time points for solution output
        method: Integration method (RK45, Radau, BDF, etc.)
        **kwargs: Additional arguments for solve_ivp

    Returns:
        Solution array, shape (n_steps, n_states)
    """
    sol = solve_ivp(func, t_span, y0, method=method, t_eval=t_eval, **kwargs)
    if not sol.success:
        logger.warning(f"ODE integration warning: {sol.message}")
    return cast(npt.NDArray[Any], sol.y.T)  # (n_steps, n_states)


def finite_difference_jacobian(
    func: Callable[..., Any],
    x: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Compute Jacobian via central finite differences.

    Args:
        func: Function to differentiate, signature (x) -> y
        x: Input point, shape (n_inputs,)
        epsilon: Finite difference step size

    Returns:
        Jacobian matrix, shape (n_outputs, n_inputs)
    """
    n = x.shape[0]
    f0 = func(x)
    n_out = f0.shape[0]
    jac = x.new_empty(n_out, n)

    for i in range(n):
        e_i = torch.zeros_like(x)
        e_i[i] = epsilon
        f_plus = func(x + e_i)
        f_minus = func(x - e_i)
        jac[:, i] = (f_plus - f_minus) / (2.0 * epsilon)

    return jac


def detect_stiffness(
    eigenvalues: torch.Tensor | npt.NDArray[Any],
) -> tuple[bool, float]:
    """Detect stiffness from Jacobian eigenvalues.

    Args:
        eigenvalues: Eigenvalues of Jacobian matrix, shape (n_states,)

    Returns:
        Tuple of (is_stiff, stiffness_ratio)
            - is_stiff: True if stiffness ratio > 1000
            - stiffness_ratio: max(|lambda|) / min(|lambda|)
    """
    if isinstance(eigenvalues, torch.Tensor):
        eigs = eigenvalues.detach().cpu()
        magnitudes = torch.abs(eigs).float()
        max_mag = float(magnitudes.max().item())
        min_mag = float(magnitudes.min().item())
    else:
        magnitudes = np.abs(eigenvalues)
        max_mag = float(np.max(magnitudes))
        min_mag = float(np.min(magnitudes))

    if min_mag < 1e-15:
        ratio = float("inf")
    else:
        ratio = max_mag / min_mag

    return ratio > 1000.0, ratio


def runge_kutta_4(
    func: Callable[..., Any],
    t: float,
    y: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """Classic 4th-order Runge-Kutta step.

    Args:
        func: Right-hand side function dy/dt = func(t, y)
        t: Current time
        y: Current state, shape (batch, n_states)
        dt: Time step size

    Returns:
        Updated state, shape (batch, n_states)
    """
    k1 = func(t, y)
    k2 = func(t + 0.5 * dt, y + 0.5 * dt * k1)
    k3 = func(t + 0.5 * dt, y + 0.5 * dt * k2)
    k4 = func(t + dt, y + dt * k3)
    return cast(torch.Tensor, y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4))


def adaptive_step_size(
    error: torch.Tensor,
    dt: float,
    atol: float = 1e-6,
    rtol: float = 1e-5,
    safety_factor: float = 0.9,
) -> float:
    """Compute adaptive step size from error estimate.

    Uses the standard embedded RK formula: dt_new = safety * dt * (tol/err)^(1/p)
    with p=4 (for RK45-style methods).

    Args:
        error: Error estimate, shape (batch, n_states)
        dt: Current step size
        atol: Absolute tolerance
        rtol: Relative tolerance
        safety_factor: Safety factor for step size adjustment

    Returns:
        New step size
    """
    # Combined tolerance: use absolute tolerance as baseline
    tol = atol + rtol * error.abs()
    # Error ratio: max over all components
    err_ratio = (error.abs() / tol).max().item()
    if err_ratio < 1e-15:
        # Error is essentially zero — allow a large increase
        return dt * 5.0

    # RK4: order p=4, so exponent = 1/(p+1) = 0.2
    dt_new = safety_factor * dt * (1.0 / err_ratio) ** 0.2
    # Clamp growth/shrink to avoid instability
    dt_new = max(dt_new, 0.1 * dt)
    dt_new = min(dt_new, 5.0 * dt)
    return float(dt_new)


def interpolate_trajectory(
    t_coarse: torch.Tensor,
    y_coarse: torch.Tensor,
    t_fine: torch.Tensor,
    method: str = "cubic",
) -> torch.Tensor:
    """Interpolate trajectory to finer time grid.

    Args:
        t_coarse: Coarse time points, shape (n_coarse,)
        y_coarse: Coarse state values, shape (n_coarse, n_states)
        t_fine: Fine time points, shape (n_fine,)
        method: Interpolation method ("linear", "cubic")

    Returns:
        Interpolated states, shape (n_fine, n_states)
    """
    tc = t_coarse.detach().cpu().numpy()
    yc = y_coarse.detach().cpu().numpy()
    tf = t_fine.detach().cpu().numpy()

    n_states = yc.shape[1]
    result = np.empty((len(tf), n_states))

    for i in range(n_states):
        if method == "cubic":
            cs = CubicSpline(tc, yc[:, i])
            result[:, i] = cs(tf)
        elif method == "linear":
            result[:, i] = np.interp(tf, tc, yc[:, i])
        else:
            raise ValueError(f"Unknown interpolation method: {method}")

    return torch.tensor(result, dtype=y_coarse.dtype, device=y_coarse.device)
