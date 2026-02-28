"""Tests for reactor_twin.discovery.symbolic_regression without PySR."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

from reactor_twin.core.neural_ode import NeuralODE
from reactor_twin.core.ode_func import MLPODEFunc
from reactor_twin.discovery.symbolic_regression import SymbolicKineticsDiscovery


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(42)


@pytest.fixture
def model():
    ode_func = MLPODEFunc(state_dim=2, hidden_dim=16, num_layers=2)
    return NeuralODE(state_dim=2, ode_func=ode_func)


class TestSymbolicRegressorImportGuard:
    def test_raises_import_error_when_pysr_unavailable(self):
        with patch("reactor_twin.discovery.symbolic_regression.PYSR_AVAILABLE", False):
            from reactor_twin.discovery.symbolic_regression import SymbolicRegressor
            with pytest.raises(ImportError, match="PySR is required"):
                SymbolicRegressor()


class TestExtractDerivatives:
    def test_returns_arrays(self, model):
        discovery = SymbolicKineticsDiscovery(model)
        z0 = torch.randn(3, 2)
        t_span = torch.linspace(0, 1, 10)
        Z, dZ_dt = discovery.extract_derivatives(z0, t_span)
        assert isinstance(Z, np.ndarray)
        assert isinstance(dZ_dt, np.ndarray)

    def test_output_shapes(self, model):
        batch = 3
        n_t = 10
        discovery = SymbolicKineticsDiscovery(model)
        z0 = torch.randn(batch, 2)
        t_span = torch.linspace(0, 1, n_t)
        Z, dZ_dt = discovery.extract_derivatives(z0, t_span)
        assert Z.shape == (batch * n_t, 2)
        assert dZ_dt.shape == (batch * n_t, 2)

    def test_values_are_finite(self, model):
        discovery = SymbolicKineticsDiscovery(model)
        z0 = torch.randn(2, 2)
        t_span = torch.linspace(0, 0.5, 5)
        Z, dZ_dt = discovery.extract_derivatives(z0, t_span)
        assert np.all(np.isfinite(Z))
        assert np.all(np.isfinite(dZ_dt))

    def test_with_state_labels(self, model):
        discovery = SymbolicKineticsDiscovery(model, state_labels=["C_A", "C_B"])
        assert discovery.state_labels == ["C_A", "C_B"]

    def test_fallback_finite_difference(self):
        """Model without ode_func attribute should use finite difference."""
        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)
                self.state_dim = 2

            def forward(self, z0, t_span, **kwargs):
                batch = z0.shape[0]
                n_t = len(t_span)
                return z0.unsqueeze(1).expand(batch, n_t, 2)

            def eval(self):
                return self

        mock = MockModel()
        discovery = SymbolicKineticsDiscovery(mock)
        z0 = torch.randn(2, 2)
        t_span = torch.linspace(0, 1, 5)
        Z, dZ_dt = discovery.extract_derivatives(z0, t_span)
        # Should get N-1 points per batch (finite diff skips last)
        assert Z.shape[0] == 2 * 4  # batch * (n_t - 1)


# ---------------------------------------------------------------------------
# Tests that mock PySR as available to cover the PySR-dependent code paths
# ---------------------------------------------------------------------------


def _make_mock_pysr_regressor():
    """Create a mock PySRRegressor class that behaves enough like the real one."""
    mock_class = MagicMock()
    mock_instance = MagicMock()
    mock_class.return_value = mock_instance

    # Default: model has equations_ as a DataFrame
    mock_instance.equations_ = pd.DataFrame(
        {
            "equation": ["x0 + x1", "x0 * x1 + 0.5"],
            "complexity": [3, 5],
            "loss": [0.1, 0.01],
        }
    )
    mock_instance.fit.return_value = None
    mock_instance.predict.return_value = np.array([1.0, 2.0, 3.0])

    return mock_class, mock_instance


class TestSymbolicRegressorWithMockedPySR:
    """Tests for SymbolicRegressor with PySR mocked as available."""

    def test_init_default_operators(self):
        """Lines 52-58: __init__ when PySR is available with default operators."""
        mock_class, mock_instance = _make_mock_pysr_regressor()
        with patch(
            "reactor_twin.discovery.symbolic_regression.PYSR_AVAILABLE", True
        ), patch(
            "reactor_twin.discovery.symbolic_regression.PySRRegressor",
            mock_class,
            create=True,
        ):
            from reactor_twin.discovery.symbolic_regression import SymbolicRegressor

            reg = SymbolicRegressor(niterations=10, feature_names=["x0", "x1"])
            assert reg.feature_names == ["x0", "x1"]
            # PySRRegressor was called with default operators
            mock_class.assert_called_once_with(
                niterations=10,
                binary_operators=["+", "-", "*", "/"],
                unary_operators=["exp", "log", "sqrt", "square"],
            )

    def test_init_custom_operators(self):
        """Lines 52-58: __init__ with custom operators (skip defaults)."""
        mock_class, _ = _make_mock_pysr_regressor()
        with patch(
            "reactor_twin.discovery.symbolic_regression.PYSR_AVAILABLE", True
        ), patch(
            "reactor_twin.discovery.symbolic_regression.PySRRegressor",
            mock_class,
            create=True,
        ):
            from reactor_twin.discovery.symbolic_regression import SymbolicRegressor

            SymbolicRegressor(
                binary_operators=["+", "*"],
                unary_operators=["exp"],
            )
            mock_class.assert_called_once_with(
                niterations=40,
                binary_operators=["+", "*"],
                unary_operators=["exp"],
            )

    def test_fit(self):
        """Lines 75-76: fit calls self.model.fit."""
        mock_class, mock_instance = _make_mock_pysr_regressor()
        with patch(
            "reactor_twin.discovery.symbolic_regression.PYSR_AVAILABLE", True
        ), patch(
            "reactor_twin.discovery.symbolic_regression.PySRRegressor",
            mock_class,
            create=True,
        ):
            from reactor_twin.discovery.symbolic_regression import SymbolicRegressor

            reg = SymbolicRegressor(feature_names=["a", "b"])
            X = np.array([[1, 2], [3, 4]])
            y = np.array([1.0, 2.0])
            result = reg.fit(X, y)
            mock_instance.fit.assert_called_once_with(
                X, y, variable_names=["a", "b"]
            )
            assert result is reg  # returns self

    def test_predict(self):
        """Line 87: predict calls self.model.predict."""
        mock_class, mock_instance = _make_mock_pysr_regressor()
        with patch(
            "reactor_twin.discovery.symbolic_regression.PYSR_AVAILABLE", True
        ), patch(
            "reactor_twin.discovery.symbolic_regression.PySRRegressor",
            mock_class,
            create=True,
        ):
            from reactor_twin.discovery.symbolic_regression import SymbolicRegressor

            reg = SymbolicRegressor()
            X = np.array([[1, 2], [3, 4], [5, 6]])
            preds = reg.predict(X)
            mock_instance.predict.assert_called_once_with(X)
            np.testing.assert_array_equal(preds, np.array([1.0, 2.0, 3.0]))

    def test_get_expression(self):
        """Lines 98-101: get_expression reads model.equations_."""
        mock_class, mock_instance = _make_mock_pysr_regressor()
        with patch(
            "reactor_twin.discovery.symbolic_regression.PYSR_AVAILABLE", True
        ), patch(
            "reactor_twin.discovery.symbolic_regression.PySRRegressor",
            mock_class,
            create=True,
        ):
            from reactor_twin.discovery.symbolic_regression import SymbolicRegressor

            reg = SymbolicRegressor()
            # Default mock has equations
            assert reg.get_expression(0) == "x0 + x1"
            assert reg.get_expression(1) == "x0 * x1 + 0.5"

    def test_get_expression_no_equations(self):
        """Lines 99-100: get_expression with None equations."""
        mock_class, mock_instance = _make_mock_pysr_regressor()
        mock_instance.equations_ = None
        with patch(
            "reactor_twin.discovery.symbolic_regression.PYSR_AVAILABLE", True
        ), patch(
            "reactor_twin.discovery.symbolic_regression.PySRRegressor",
            mock_class,
            create=True,
        ):
            from reactor_twin.discovery.symbolic_regression import SymbolicRegressor

            reg = SymbolicRegressor()
            assert reg.get_expression() == "No equations found"

    def test_get_expression_empty_equations(self):
        """Lines 99-100: get_expression with empty DataFrame."""
        mock_class, mock_instance = _make_mock_pysr_regressor()
        mock_instance.equations_ = pd.DataFrame(
            columns=["equation", "complexity", "loss"]
        )
        with patch(
            "reactor_twin.discovery.symbolic_regression.PYSR_AVAILABLE", True
        ), patch(
            "reactor_twin.discovery.symbolic_regression.PySRRegressor",
            mock_class,
            create=True,
        ):
            from reactor_twin.discovery.symbolic_regression import SymbolicRegressor

            reg = SymbolicRegressor()
            assert reg.get_expression() == "No equations found"

    def test_pareto_front(self):
        """Lines 109-112: pareto_front reads model.equations_."""
        mock_class, mock_instance = _make_mock_pysr_regressor()
        with patch(
            "reactor_twin.discovery.symbolic_regression.PYSR_AVAILABLE", True
        ), patch(
            "reactor_twin.discovery.symbolic_regression.PySRRegressor",
            mock_class,
            create=True,
        ):
            from reactor_twin.discovery.symbolic_regression import SymbolicRegressor

            reg = SymbolicRegressor()
            front = reg.pareto_front()
            assert len(front) == 2
            assert front[0]["equation"] == "x0 + x1"
            assert front[0]["complexity"] == 3
            assert front[0]["loss"] == pytest.approx(0.1)
            assert front[1]["equation"] == "x0 * x1 + 0.5"

    def test_pareto_front_no_equations(self):
        """Lines 110: pareto_front with None equations."""
        mock_class, mock_instance = _make_mock_pysr_regressor()
        mock_instance.equations_ = None
        with patch(
            "reactor_twin.discovery.symbolic_regression.PYSR_AVAILABLE", True
        ), patch(
            "reactor_twin.discovery.symbolic_regression.PySRRegressor",
            mock_class,
            create=True,
        ):
            from reactor_twin.discovery.symbolic_regression import SymbolicRegressor

            reg = SymbolicRegressor()
            assert reg.pareto_front() == []

    def test_pareto_front_empty_equations(self):
        """Lines 110: pareto_front with empty DataFrame."""
        mock_class, mock_instance = _make_mock_pysr_regressor()
        mock_instance.equations_ = pd.DataFrame(
            columns=["equation", "complexity", "loss"]
        )
        with patch(
            "reactor_twin.discovery.symbolic_regression.PYSR_AVAILABLE", True
        ), patch(
            "reactor_twin.discovery.symbolic_regression.PySRRegressor",
            mock_class,
            create=True,
        ):
            from reactor_twin.discovery.symbolic_regression import SymbolicRegressor

            reg = SymbolicRegressor()
            assert reg.pareto_front() == []


class TestSymbolicKineticsDiscoveryWithMockedPySR:
    """Tests for discover() and validate() with PySR mocked."""

    @pytest.fixture
    def model(self):
        torch.manual_seed(42)
        ode_func = MLPODEFunc(state_dim=2, hidden_dim=16, num_layers=2)
        return NeuralODE(state_dim=2, ode_func=ode_func)

    def test_discover(self, model):
        """Lines 214-230: discover() calls extract_derivatives then SymbolicRegressor."""
        mock_class, mock_instance = _make_mock_pysr_regressor()

        with patch(
            "reactor_twin.discovery.symbolic_regression.PYSR_AVAILABLE", True
        ), patch(
            "reactor_twin.discovery.symbolic_regression.PySRRegressor",
            mock_class,
            create=True,
        ):
            discovery = SymbolicKineticsDiscovery(
                model, state_labels=["C_A", "C_B"]
            )
            z0 = torch.randn(3, 2)
            t_span = torch.linspace(0, 1, 5)

            regressor = discovery.discover(
                z0, t_span, target_state_index=0, niterations=5
            )

            # The regressor should have been fitted
            mock_instance.fit.assert_called_once()
            # Check that the fit call got the right shapes
            call_args = mock_instance.fit.call_args
            X_arg, y_arg = call_args[0]
            assert X_arg.shape[1] == 2  # state_dim
            assert y_arg.ndim == 1
            # Returns a SymbolicRegressor (from the module)
            assert hasattr(regressor, "model")
            assert hasattr(regressor, "feature_names")

    def test_discover_with_target_index(self, model):
        """Lines 214-230: discover with target_state_index=1."""
        mock_class, mock_instance = _make_mock_pysr_regressor()

        with patch(
            "reactor_twin.discovery.symbolic_regression.PYSR_AVAILABLE", True
        ), patch(
            "reactor_twin.discovery.symbolic_regression.PySRRegressor",
            mock_class,
            create=True,
        ):
            discovery = SymbolicKineticsDiscovery(model)
            z0 = torch.randn(2, 2)
            t_span = torch.linspace(0, 0.5, 4)

            discovery.discover(
                z0, t_span, target_state_index=1, niterations=3
            )
            # fit was called; y should come from column 1 of dZ_dt
            mock_instance.fit.assert_called_once()

    def test_validate(self, model):
        """Lines 250-258: validate() computes mse and r_squared."""
        mock_class, mock_instance = _make_mock_pysr_regressor()
        # Predict returns values close to the true derivatives
        mock_instance.predict.return_value = np.array([1.0, 2.0, 3.0])

        with patch(
            "reactor_twin.discovery.symbolic_regression.PYSR_AVAILABLE", True
        ), patch(
            "reactor_twin.discovery.symbolic_regression.PySRRegressor",
            mock_class,
            create=True,
        ):
            from reactor_twin.discovery.symbolic_regression import SymbolicRegressor

            reg = SymbolicRegressor()

            discovery = SymbolicKineticsDiscovery(model)
            Z_test = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
            dZ_test = np.array([[1.0, 0.5], [2.0, 1.0], [3.0, 1.5]])

            metrics = discovery.validate(
                reg, Z_test, dZ_test, target_state_index=0
            )

            assert "mse" in metrics
            assert "r_squared" in metrics
            # Perfect predictions: mse should be 0, r_squared should be 1
            assert metrics["mse"] == pytest.approx(0.0)
            assert metrics["r_squared"] == pytest.approx(1.0)

    def test_validate_imperfect_predictions(self, model):
        """Lines 250-258: validate() with imperfect predictions."""
        mock_class, mock_instance = _make_mock_pysr_regressor()
        # Predictions differ from truth
        mock_instance.predict.return_value = np.array([1.5, 2.5, 3.5])

        with patch(
            "reactor_twin.discovery.symbolic_regression.PYSR_AVAILABLE", True
        ), patch(
            "reactor_twin.discovery.symbolic_regression.PySRRegressor",
            mock_class,
            create=True,
        ):
            from reactor_twin.discovery.symbolic_regression import SymbolicRegressor

            reg = SymbolicRegressor()

            discovery = SymbolicKineticsDiscovery(model)
            Z_test = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
            dZ_test = np.array([[1.0, 0.5], [2.0, 1.0], [3.0, 1.5]])

            metrics = discovery.validate(
                reg, Z_test, dZ_test, target_state_index=0
            )

            assert metrics["mse"] > 0
            assert metrics["r_squared"] < 1.0
