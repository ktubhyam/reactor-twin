"""Generate training data from reactor models using scipy integration."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from scipy.integrate import solve_ivp

from reactor_twin.reactors.base import AbstractReactor

logger = logging.getLogger(__name__)


class ReactorDataGenerator:
    """Generate training datasets from reactor models.

    Uses scipy.integrate.solve_ivp to solve reactor ODEs and generate
    ground-truth trajectories for training Neural DEs.

    Attributes:
        reactor: Reactor instance to generate data from.
        method: ODE solver method ('LSODA', 'Radau', 'BDF', etc.).
        rtol: Relative tolerance for solver.
        atol: Absolute tolerance for solver.
    """

    def __init__(
        self,
        reactor: AbstractReactor,
        method: str = "LSODA",
        rtol: float = 1e-6,
        atol: float = 1e-8,
    ):
        """Initialize data generator.

        Args:
            reactor: Reactor instance with ode_rhs method.
            method: Scipy solver method. LSODA recommended for stiff systems.
            rtol: Relative tolerance.
            atol: Absolute tolerance.
        """
        self.reactor = reactor
        self.method = method
        self.rtol = rtol
        self.atol = atol

        logger.info(f"Initialized ReactorDataGenerator: reactor={reactor.name}, method={method}")

    def generate_trajectory(
        self,
        t_span: tuple[float, float],
        t_eval: npt.NDArray[Any],
        y0: npt.NDArray[Any] | None = None,
        controls: npt.NDArray[Any] | None = None,
    ) -> dict[str, npt.NDArray[Any]]:
        """Generate single trajectory.

        Args:
            t_span: Time interval (t_start, t_end).
            t_eval: Time points to evaluate at, shape (num_times,).
            y0: Initial condition, shape (state_dim,). If None, uses reactor default.
            controls: Control inputs (optional), shape (input_dim,).

        Returns:
            Dictionary with keys:
                - 't': Time points, shape (num_times,)
                - 'y': State trajectory, shape (num_times, state_dim)
                - 'success': Boolean indicating if integration succeeded
        """
        if y0 is None:
            y0 = self.reactor.get_initial_state()

        # Define ODE function wrapper for controls
        def ode_func(t: float, y: npt.NDArray[Any]) -> npt.NDArray[Any]:
            return self.reactor.ode_rhs(t, y, controls)

        # Integrate
        sol = solve_ivp(
            ode_func,
            t_span,
            y0,
            method=self.method,
            t_eval=t_eval,
            rtol=self.rtol,
            atol=self.atol,
        )

        if not sol.success:
            logger.warning(f"Integration failed: {sol.message}")

        return {
            "t": sol.t,  # (num_times,)
            "y": sol.y.T,  # (num_times, state_dim)
            "success": sol.success,
        }

    def generate_batch(
        self,
        batch_size: int,
        t_span: tuple[float, float],
        t_eval: npt.NDArray[Any],
        initial_conditions: npt.NDArray[Any] | None = None,
        controls: npt.NDArray[Any] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Generate batch of trajectories.

        Args:
            batch_size: Number of trajectories to generate.
            t_span: Time interval (t_start, t_end).
            t_eval: Time points, shape (num_times,).
            initial_conditions: Initial conditions, shape (batch_size, state_dim).
                If None, generates random perturbations around reactor default.
            controls: Controls for each trajectory, shape (batch_size, input_dim).

        Returns:
            Dictionary with keys:
                - 'z0': Initial states, shape (batch_size, state_dim) (torch.Tensor)
                - 't_span': Time points, shape (num_times,) (torch.Tensor)
                - 'targets': Trajectories, shape (batch_size, num_times, state_dim) (torch.Tensor)
        """
        # Always compute default IC for retry fallback
        y0_default = self.reactor.get_initial_state()

        # Generate initial conditions if not provided
        if initial_conditions is None:
            # Perturb default IC with small noise
            initial_conditions = np.tile(y0_default, (batch_size, 1))
            noise = np.random.randn(batch_size, len(y0_default)) * 0.1
            initial_conditions = initial_conditions + noise
            initial_conditions = np.maximum(initial_conditions, 0)  # Keep non-negative

        trajectories = []
        for i in range(batch_size):
            y0 = initial_conditions[i]
            u = controls[i] if controls is not None else None

            result = self.generate_trajectory(t_span, t_eval, y0, u)

            if not result["success"]:
                logger.warning(f"Trajectory {i}/{batch_size} failed, retrying with default IC")
                y0_retry = y0_default.copy()
                retry = self.generate_trajectory(t_span, t_eval, y0_retry, u)
                if retry["success"]:
                    trajectories.append(retry["y"])
                    initial_conditions[i] = y0_retry
                else:
                    logger.warning(f"Retry also failed for trajectory {i}, using last valid")
                    trajectories.append(np.zeros((len(t_eval), self.reactor.state_dim)))
            else:
                trajectories.append(result["y"])

        # Convert to torch tensors
        z0 = torch.tensor(initial_conditions, dtype=torch.float32)
        t_span_torch = torch.tensor(t_eval, dtype=torch.float32)
        targets = torch.tensor(np.stack(trajectories), dtype=torch.float32)

        return {
            "z0": z0,  # (batch, state_dim)
            "t_span": t_span_torch,  # (num_times,)
            "targets": targets,  # (batch, num_times, state_dim)
        }

    def generate_dataset(
        self,
        num_trajectories: int,
        t_span: tuple[float, float],
        t_eval: npt.NDArray[Any],
        batch_size: int = 32,
    ) -> list[dict[str, torch.Tensor]]:
        """Generate full dataset as list of batches.

        Args:
            num_trajectories: Total number of trajectories.
            t_span: Time interval (t_start, t_end).
            t_eval: Time points, shape (num_times,).
            batch_size: Batch size.

        Returns:
            List of batch dictionaries.
        """
        num_batches = (num_trajectories + batch_size - 1) // batch_size
        dataset = []

        for i in range(num_batches):
            current_batch_size = min(batch_size, num_trajectories - i * batch_size)
            batch = self.generate_batch(current_batch_size, t_span, t_eval)
            dataset.append(batch)

            logger.debug(
                f"Generated batch {i + 1}/{num_batches}: {current_batch_size} trajectories"
            )

        logger.info(f"Generated dataset: {num_trajectories} trajectories, {num_batches} batches")
        return dataset


__all__ = ["ReactorDataGenerator"]
