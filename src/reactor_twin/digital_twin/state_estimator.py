"""Extended Kalman Filter state estimator for reactor digital twins.

Provides real-time state estimation by fusing neural ODE predictions with
noisy sensor measurements. Uses autograd Jacobians for the linearized
prediction step.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from reactor_twin.core.base import AbstractNeuralDE

logger = logging.getLogger(__name__)


class EKFStateEstimator:
    """Extended Kalman Filter using a Neural ODE as the process model.

    Fuses model predictions with noisy measurements to produce optimal
    state estimates. The Jacobian of the learned dynamics is computed
    via ``torch.func.jacrev`` for the EKF linearization step.

    Attributes:
        model: Neural DE whose ``ode_func`` provides the dynamics.
        state_dim: Dimension of the full state vector.
        obs_dim: Number of observed variables.
        obs_indices: Which state variables are measured.
        Q: Process noise covariance, shape ``(state_dim, state_dim)``.
        R: Measurement noise covariance, shape ``(obs_dim, obs_dim)``.
        P0: Initial error covariance, shape ``(state_dim, state_dim)``.
        dt: Discretization time step for Euler prediction.
        device: Torch device.
    """

    def __init__(
        self,
        model: AbstractNeuralDE,
        state_dim: int,
        obs_dim: int | None = None,
        obs_indices: list[int] | None = None,
        Q: torch.Tensor | float = 1e-4,
        R: torch.Tensor | float = 1e-2,
        P0: torch.Tensor | float = 1.0,
        dt: float = 0.01,
        device: str | torch.device = "cpu",
    ) -> None:
        """Configure EKF with process/measurement noise.

        Args:
            model: Neural DE with an ``ode_func`` attribute.
            state_dim: Dimension of the state vector.
            obs_dim: Number of observed variables. Inferred from
                ``obs_indices`` if not given.
            obs_indices: Indices of observable states. Defaults to all.
            Q: Process noise covariance. Scalar broadens to diagonal.
            R: Measurement noise covariance. Scalar broadens to diagonal.
            P0: Initial error covariance. Scalar broadens to diagonal.
            dt: Euler discretization step (seconds).
            device: Torch device for all tensors.
        """
        self.model = model
        self.state_dim = state_dim
        self.device = torch.device(device)
        self.dt = dt

        # Observable indices
        if obs_indices is None:
            obs_indices = list(range(state_dim))
        self.obs_indices = obs_indices
        self.obs_dim = obs_dim or len(obs_indices)

        # Covariance matrices
        self.Q = self._to_matrix(Q, state_dim).to(self.device)
        self.R = self._to_matrix(R, self.obs_dim).to(self.device)
        self.P0 = self._to_matrix(P0, state_dim).to(self.device)

        # Precompute observation matrix
        self._H = self._observation_matrix()

        logger.info(
            f"EKFStateEstimator: state_dim={state_dim}, obs_dim={self.obs_dim}, "
            f"dt={dt}"
        )

    @staticmethod
    def _to_matrix(val: torch.Tensor | float, dim: int) -> torch.Tensor:
        """Convert scalar or tensor to a ``(dim, dim)`` covariance matrix."""
        if isinstance(val, (int, float)):
            return torch.eye(dim) * val
        if val.ndim == 1:
            return torch.diag(val)
        return val

    # ------------------------------------------------------------------
    # Jacobian helpers
    # ------------------------------------------------------------------

    def _ode_func_wrapper(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Wrapper that calls ``model.ode_func`` with the signature expected by jacrev.

        ``torch.func.jacrev`` differentiates w.r.t. the first positional
        argument, so we put *z* first.
        """
        # ode_func.forward(t, z, u=None)
        return self.model.ode_func(t, z)

    def _compute_jacobian(
        self, z: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Compute the Jacobian df/dz via autograd.

        Tries ``torch.func.jacrev`` first (vectorized, fast) and falls
        back to ``torch.autograd.functional.jacobian`` if the model
        contains operations incompatible with ``torch.func``.

        Args:
            z: State vector, shape ``(state_dim,)`` (unbatched).
            t: Scalar time tensor.

        Returns:
            Jacobian F = df/dz, shape ``(state_dim, state_dim)``.
        """
        z = z.detach().requires_grad_(True)
        t = t.detach()
        try:
            jac_fn = torch.func.jacrev(self._ode_func_wrapper, argnums=0)
            F = jac_fn(z, t)
        except Exception:
            logger.debug("torch.func.jacrev failed; falling back to autograd.functional.jacobian")
            def _fn(z_in: torch.Tensor) -> torch.Tensor:
                return self.model.ode_func(t, z_in.unsqueeze(0)).squeeze(0)
            F = torch.autograd.functional.jacobian(_fn, z)
            F = F.detach()
        return F

    # ------------------------------------------------------------------
    # Observation matrix
    # ------------------------------------------------------------------

    def _observation_matrix(self) -> torch.Tensor:
        """Build the linear observation (selection) matrix H.

        Returns:
            H of shape ``(obs_dim, state_dim)`` with ones at observed indices.
        """
        H = torch.zeros(self.obs_dim, self.state_dim, device=self.device)
        for i, idx in enumerate(self.obs_indices):
            H[i, idx] = 1.0
        return H

    # ------------------------------------------------------------------
    # EKF predict / update
    # ------------------------------------------------------------------

    def predict_step(
        self,
        z_est: torch.Tensor,
        P: torch.Tensor,
        dt: float | None = None,
        controls: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """EKF time-update (prediction) step.

        Propagates the state estimate and covariance forward by one Euler
        step using the learned dynamics.

        Args:
            z_est: Current state estimate, shape ``(state_dim,)``.
            P: Current error covariance, shape ``(state_dim, state_dim)``.
            dt: Override time step. Uses ``self.dt`` if ``None``.
            controls: Control input, shape ``(input_dim,)``. *Currently unused.*

        Returns:
            ``(z_pred, P_pred)`` — predicted state and covariance.
        """
        dt = dt or self.dt
        t = torch.tensor(0.0, device=self.device)

        # Evaluate dynamics: dz/dt = f(t, z)
        z_batch = z_est.unsqueeze(0)  # (1, state_dim)
        with torch.no_grad():
            dzdt = self.model.ode_func(t, z_batch).squeeze(0)

        # Euler prediction
        z_pred = z_est + dzdt * dt

        # Jacobian for covariance propagation
        F = self._compute_jacobian(z_est, t)
        F_d = torch.eye(self.state_dim, device=self.device) + F * dt  # Discrete-time

        P_pred = F_d @ P @ F_d.T + self.Q

        return z_pred, P_pred

    def update_step(
        self,
        z_pred: torch.Tensor,
        P_pred: torch.Tensor,
        measurement: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """EKF measurement-update (correction) step.

        Args:
            z_pred: Predicted state, shape ``(state_dim,)``.
            P_pred: Predicted covariance, shape ``(state_dim, state_dim)``.
            measurement: Observation vector, shape ``(obs_dim,)``.

        Returns:
            ``(z_upd, P_upd, innovation)`` — updated state, covariance,
            and the measurement innovation vector.
        """
        H = self._H

        # Innovation
        y_pred = H @ z_pred
        innovation = measurement - y_pred

        # Innovation covariance
        S = H @ P_pred @ H.T + self.R

        # Kalman gain
        K = P_pred @ H.T @ torch.linalg.inv(S)

        # State update
        z_upd = z_pred + K @ innovation

        # Covariance update (Joseph form for numerical stability)
        I_KH = torch.eye(self.state_dim, device=self.device) - K @ H
        P_upd = I_KH @ P_pred @ I_KH.T + K @ self.R @ K.T

        return z_upd, P_upd, innovation

    # ------------------------------------------------------------------
    # Full filter pass
    # ------------------------------------------------------------------

    def filter(
        self,
        measurements: torch.Tensor,
        z0: torch.Tensor | None = None,
        t_span: torch.Tensor | None = None,
        controls: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run the full EKF over a sequence of measurements.

        Args:
            measurements: Observed data, shape ``(num_times, obs_dim)``.
            z0: Initial state estimate, shape ``(state_dim,)``.
                Defaults to zeros.
            t_span: Time points, shape ``(num_times,)``. Used to compute
                per-step ``dt``. If ``None``, uses constant ``self.dt``.
            controls: Control inputs, shape ``(num_times, input_dim)``.
                *Currently unused.*

        Returns:
            Dictionary with keys:

            - ``states``: Filtered state estimates ``(num_times, state_dim)``
            - ``covariances``: Error covariances ``(num_times, state_dim, state_dim)``
            - ``innovations``: Innovation vectors ``(num_times, obs_dim)``
            - ``kalman_gains``: Not stored (memory); see ``innovations``.
        """
        num_times = measurements.shape[0]
        measurements = measurements.to(self.device)

        if z0 is None:
            z0 = torch.zeros(self.state_dim, device=self.device)
        else:
            z0 = z0.to(self.device)

        # Storage
        states = torch.zeros(num_times, self.state_dim, device=self.device)
        covariances = torch.zeros(
            num_times, self.state_dim, self.state_dim, device=self.device
        )
        innovations = torch.zeros(num_times, self.obs_dim, device=self.device)

        z_est = z0.clone()
        P = self.P0.clone()

        for k in range(num_times):
            # Per-step dt
            if t_span is not None and k > 0:
                step_dt = (t_span[k] - t_span[k - 1]).item()
            else:
                step_dt = self.dt

            # Predict
            z_pred, P_pred = self.predict_step(z_est, P, dt=step_dt)

            # Update
            z_est, P, innov = self.update_step(z_pred, P_pred, measurements[k])

            states[k] = z_est
            covariances[k] = P
            innovations[k] = innov

        return {
            "states": states,
            "covariances": covariances,
            "innovations": innovations,
        }


__all__ = ["EKFStateEstimator"]
