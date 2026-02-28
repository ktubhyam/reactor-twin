"""Numerical utilities for reactor simulations and Neural DEs."""

from __future__ import annotations

import logging

import numpy as np
import torch
from scipy.integrate import solve_ivp

logger = logging.getLogger(__name__)


def integrate_ode(
    func: callable,
    t_span: tuple[float, float],
    y0: np.ndarray,
    t_eval: np.ndarray | None = None,
    method: str = "RK45",
    **kwargs,
) -> np.ndarray:
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

    Raises:
        NotImplementedError: TODO
    """
    raise NotImplementedError("TODO")


def finite_difference_jacobian(
    func: callable,
    x: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Compute Jacobian via finite differences.

    Args:
        func: Function to differentiate, signature (x) -> y
        x: Input point, shape (n_inputs,)
        epsilon: Finite difference step size

    Returns:
        Jacobian matrix, shape (n_outputs, n_inputs)

    Raises:
        NotImplementedError: TODO
    """
    raise NotImplementedError("TODO")


def detect_stiffness(
    eigenvalues: torch.Tensor | np.ndarray,
) -> tuple[bool, float]:
    """Detect stiffness from Jacobian eigenvalues.

    Args:
        eigenvalues: Eigenvalues of Jacobian matrix, shape (n_states,)

    Returns:
        Tuple of (is_stiff, stiffness_ratio)
            - is_stiff: True if stiffness ratio > 1000
            - stiffness_ratio: max(|lambda|) / min(|lambda|)

    Raises:
        NotImplementedError: TODO
    """
    raise NotImplementedError("TODO")


def runge_kutta_4(
    func: callable,
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

    Raises:
        NotImplementedError: TODO
    """
    raise NotImplementedError("TODO")


def adaptive_step_size(
    error: torch.Tensor,
    dt: float,
    atol: float = 1e-6,
    rtol: float = 1e-5,
    safety_factor: float = 0.9,
) -> float:
    """Compute adaptive step size from error estimate.

    Args:
        error: Error estimate, shape (batch, n_states)
        dt: Current step size
        atol: Absolute tolerance
        rtol: Relative tolerance
        safety_factor: Safety factor for step size adjustment

    Returns:
        New step size

    Raises:
        NotImplementedError: TODO
    """
    raise NotImplementedError("TODO")


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

    Raises:
        NotImplementedError: TODO
    """
    raise NotImplementedError("TODO")
