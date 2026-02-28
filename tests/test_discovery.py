"""Tests for symbolic regression discovery (requires PySR)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

pysr = pytest.importorskip("pysr")

from reactor_twin.core.neural_ode import NeuralODE
from reactor_twin.discovery.symbolic_regression import (
    SymbolicKineticsDiscovery,
    SymbolicRegressor,
)

# ── SymbolicRegressor ────────────────────────────────────────────────

class TestSymbolicRegressor:
    def test_instantiation(self):
        reg = SymbolicRegressor(niterations=5)
        assert reg.model is not None

    def test_custom_operators(self):
        reg = SymbolicRegressor(
            niterations=5,
            binary_operators=["+", "*"],
            unary_operators=["exp"],
        )
        assert reg.model is not None

    def test_feature_names(self):
        reg = SymbolicRegressor(
            niterations=5,
            feature_names=["x", "y"],
        )
        assert reg.feature_names == ["x", "y"]

    def test_fit_and_predict(self):
        """Fit on simple data and verify predictions are finite."""
        X = np.random.randn(100, 2)
        y = X[:, 0] + X[:, 1]
        reg = SymbolicRegressor(niterations=5, maxsize=10)
        reg.fit(X, y)
        pred = reg.predict(X)
        assert pred.shape == (100,)
        assert np.all(np.isfinite(pred))

    def test_get_expression(self):
        X = np.random.randn(100, 2)
        y = X[:, 0] + X[:, 1]
        reg = SymbolicRegressor(niterations=5, maxsize=10)
        reg.fit(X, y)
        expr = reg.get_expression()
        assert isinstance(expr, str)
        assert len(expr) > 0

    def test_pareto_front(self):
        X = np.random.randn(100, 2)
        y = X[:, 0] * X[:, 1]
        reg = SymbolicRegressor(niterations=5, maxsize=10)
        reg.fit(X, y)
        front = reg.pareto_front()
        assert isinstance(front, list)
        if len(front) > 0:
            assert "complexity" in front[0]
            assert "loss" in front[0]
            assert "equation" in front[0]

    def test_get_expression_before_fit(self):
        reg = SymbolicRegressor(niterations=5)
        expr = reg.get_expression()
        assert "No equations" in expr


# ── SymbolicKineticsDiscovery ────────────────────────────────────────

class TestSymbolicKineticsDiscovery:
    def test_extract_derivatives(self):
        model = NeuralODE(
            state_dim=2, solver="euler", adjoint=False,
            hidden_dim=16, num_layers=2,
        )
        discovery = SymbolicKineticsDiscovery(model, state_labels=["C_A", "C_B"])
        z0 = torch.randn(3, 2)
        t = torch.linspace(0, 0.1, 10)
        Z, dZ = discovery.extract_derivatives(z0, t)
        assert Z.ndim == 2
        assert dZ.ndim == 2
        assert Z.shape[1] == 2
        assert dZ.shape[1] == 2
        assert Z.shape[0] == dZ.shape[0]

    def test_extract_derivatives_finite(self):
        model = NeuralODE(
            state_dim=2, solver="euler", adjoint=False,
            hidden_dim=16, num_layers=2,
        )
        discovery = SymbolicKineticsDiscovery(model)
        z0 = torch.randn(2, 2) * 0.1
        t = torch.linspace(0, 0.05, 5)
        Z, dZ = discovery.extract_derivatives(z0, t)
        assert np.all(np.isfinite(Z))
        assert np.all(np.isfinite(dZ))

    def test_discover_returns_regressor(self):
        model = NeuralODE(
            state_dim=2, solver="euler", adjoint=False,
            hidden_dim=16, num_layers=2,
        )
        discovery = SymbolicKineticsDiscovery(model, state_labels=["C_A", "C_B"])
        z0 = torch.randn(5, 2)
        t = torch.linspace(0, 0.1, 20)
        reg = discovery.discover(z0, t, target_state_index=0, niterations=3, maxsize=10)
        assert isinstance(reg, SymbolicRegressor)
        assert isinstance(reg.get_expression(), str)

    def test_validate(self):
        model = NeuralODE(
            state_dim=2, solver="euler", adjoint=False,
            hidden_dim=16, num_layers=2,
        )
        discovery = SymbolicKineticsDiscovery(model)
        z0 = torch.randn(5, 2)
        t = torch.linspace(0, 0.1, 20)

        Z, dZ = discovery.extract_derivatives(z0, t)
        reg = SymbolicRegressor(niterations=3, maxsize=10)
        reg.fit(Z, dZ[:, 0])

        metrics = discovery.validate(reg, Z, dZ, target_state_index=0)
        assert "mse" in metrics
        assert "r_squared" in metrics
        assert np.isfinite(metrics["mse"])

    def test_discover_different_state_index(self):
        model = NeuralODE(
            state_dim=3, solver="euler", adjoint=False,
            hidden_dim=16, num_layers=2,
        )
        discovery = SymbolicKineticsDiscovery(model)
        z0 = torch.randn(5, 3)
        t = torch.linspace(0, 0.1, 15)
        reg = discovery.discover(z0, t, target_state_index=2, niterations=3, maxsize=10)
        assert isinstance(reg, SymbolicRegressor)
