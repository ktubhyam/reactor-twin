"""Symbolic regression for discovering reactor kinetics expressions.

Wraps PySR to find interpretable mathematical expressions that
approximate learned Neural ODE dynamics.

Optional dependency: ``pysr>=0.18`` in the ``[discovery]`` group.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

from reactor_twin.core.base import AbstractNeuralDE

logger = logging.getLogger(__name__)

try:
    from pysr import PySRRegressor

    PYSR_AVAILABLE = True
except ImportError:
    PYSR_AVAILABLE = False
    logger.debug("PySR not installed. Symbolic regression unavailable.")


class SymbolicRegressor:
    """Thin wrapper around PySR for discovering symbolic expressions.

    Attributes:
        model: Fitted PySR regressor (after calling ``fit``).
        feature_names: Names of input features.
    """

    def __init__(
        self,
        niterations: int = 40,
        binary_operators: list[str] | None = None,
        unary_operators: list[str] | None = None,
        feature_names: list[str] | None = None,
        **pysr_kwargs: Any,
    ):
        if not PYSR_AVAILABLE:
            raise ImportError(
                "PySR is required for symbolic regression. "
                "Install with: pip install reactor-twin[discovery]"
            )

        if binary_operators is None:
            binary_operators = ["+", "-", "*", "/"]
        if unary_operators is None:
            unary_operators = ["exp", "log", "sqrt", "square"]

        self.feature_names = feature_names
        self.model = PySRRegressor(
            niterations=niterations,
            binary_operators=binary_operators,
            unary_operators=unary_operators,
            **pysr_kwargs,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> SymbolicRegressor:
        """Fit symbolic regression model.

        Args:
            X: Input features, shape (N, n_features).
            y: Target values, shape (N,) or (N, 1).

        Returns:
            Self for chaining.
        """
        self.model.fit(X, y, variable_names=self.feature_names)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using best discovered expression.

        Args:
            X: Input features, shape (N, n_features).

        Returns:
            Predictions, shape (N,).
        """
        return self.model.predict(X)

    def get_expression(self, index: int = 0) -> str:
        """Get the i-th symbolic expression as a string.

        Args:
            index: Index in the Pareto front (0 = simplest).

        Returns:
            Human-readable expression string.
        """
        equations = self.model.equations_
        if equations is None or len(equations) == 0:
            return "No equations found"
        return str(equations.iloc[index]["equation"])

    def pareto_front(self) -> list[dict[str, Any]]:
        """Return the Pareto front of complexity vs. accuracy.

        Returns:
            List of dicts with 'complexity', 'loss', and 'equation' keys.
        """
        equations = self.model.equations_
        if equations is None or len(equations) == 0:
            return []
        return [
            {
                "complexity": int(row["complexity"]),
                "loss": float(row["loss"]),
                "equation": str(row["equation"]),
            }
            for _, row in equations.iterrows()
        ]


class SymbolicKineticsDiscovery:
    """Pipeline: trained NeuralODE -> extract (z, dz/dt) -> symbolic regression.

    Takes a trained Neural ODE model, extracts state-derivative pairs
    from trajectories, and runs symbolic regression to discover
    interpretable kinetic rate expressions.

    Attributes:
        model: Trained Neural ODE.
        state_labels: Names of state variables.
    """

    def __init__(
        self,
        model: AbstractNeuralDE,
        state_labels: list[str] | None = None,
    ):
        self.model = model
        self.state_labels = state_labels

    def extract_derivatives(
        self,
        z0: torch.Tensor,
        t_span: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract (state, derivative) pairs from model.

        Args:
            z0: Initial states, shape (batch, state_dim).
            t_span: Time points, shape (num_times,).

        Returns:
            Tuple of (Z, dZ_dt):
                - Z: States, shape (N, state_dim)
                - dZ_dt: Derivatives, shape (N, state_dim)
        """
        self.model.eval()
        z0_req = z0.detach().requires_grad_(False)

        with torch.no_grad():
            traj = self.model(z0_req, t_span)
            # traj: (batch, time, state_dim)

        # Compute derivatives using the ODE function
        batch, n_t, state_dim = traj.shape
        Z_list = []
        dZ_list = []

        for b in range(batch):
            for ti in range(n_t):
                z_point = traj[b, ti].unsqueeze(0)  # (1, state_dim)
                t_point = t_span[ti] if ti < len(t_span) else t_span[-1]
                t_tensor = t_point.clone().detach().to(dtype=z_point.dtype)

                # Get derivative from ODE func
                if hasattr(self.model, "ode_func"):
                    dz = self.model.ode_func(t_tensor, z_point)
                elif ti < n_t - 1:
                    # Fallback: finite difference
                    dt = (t_span[ti + 1] - t_span[ti]).item()
                    dz = (traj[b, ti + 1] - traj[b, ti]).unsqueeze(0) / max(dt, 1e-8)
                else:
                    continue

                Z_list.append(z_point.detach().cpu().numpy())
                dZ_list.append(dz.detach().cpu().numpy())

        Z = np.concatenate(Z_list, axis=0)  # (N, state_dim)
        dZ_dt = np.concatenate(dZ_list, axis=0)  # (N, state_dim)

        return Z, dZ_dt

    def discover(
        self,
        z0: torch.Tensor,
        t_span: torch.Tensor,
        target_state_index: int = 0,
        niterations: int = 40,
        **pysr_kwargs: Any,
    ) -> SymbolicRegressor:
        """Run full discovery pipeline for one state variable.

        Args:
            z0: Initial states, shape (batch, state_dim).
            t_span: Time points.
            target_state_index: Which dz/dt component to discover.
            niterations: PySR iterations.
            **pysr_kwargs: Additional PySR arguments.

        Returns:
            Fitted SymbolicRegressor with discovered expression.
        """
        Z, dZ_dt = self.extract_derivatives(z0, t_span)

        # Target: derivative of the chosen state variable
        y = dZ_dt[:, target_state_index]

        regressor = SymbolicRegressor(
            niterations=niterations,
            feature_names=self.state_labels,
            **pysr_kwargs,
        )
        regressor.fit(Z, y)

        logger.info(
            f"Discovered expression for state {target_state_index}: {regressor.get_expression()}"
        )

        return regressor

    def validate(
        self,
        regressor: SymbolicRegressor,
        Z_test: np.ndarray,
        dZ_test: np.ndarray,
        target_state_index: int = 0,
    ) -> dict[str, float]:
        """Validate discovered expression on test data.

        Args:
            regressor: Fitted symbolic regressor.
            Z_test: Test states, shape (N, state_dim).
            dZ_test: Test derivatives, shape (N, state_dim).
            target_state_index: Which state derivative to validate.

        Returns:
            Dict with 'mse' and 'r_squared' metrics.
        """
        y_true = dZ_test[:, target_state_index]
        y_pred = regressor.predict(Z_test)

        mse = float(np.mean((y_true - y_pred) ** 2))
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r_squared = 1.0 - ss_res / max(ss_tot, 1e-10)

        return {"mse": mse, "r_squared": float(r_squared)}


__all__ = ["SymbolicRegressor", "SymbolicKineticsDiscovery"]
