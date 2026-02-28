"""Tests for sensitivity analysis (requires SALib)."""

from __future__ import annotations

import numpy as np
import pytest

SALib = pytest.importorskip("SALib")

from reactor_twin.reactors.systems import create_exothermic_cstr
from reactor_twin.utils.sensitivity import SensitivityAnalyzer

# ── helpers ──────────────────────────────────────────────────────────


def _make_analyzer() -> SensitivityAnalyzer:
    reactor = create_exothermic_cstr(isothermal=True)
    return SensitivityAnalyzer(
        reactor=reactor,
        param_names=["V", "F"],
        param_bounds=[[5.0, 20.0], [0.5, 5.0]],
    )


# ── Initialization ───────────────────────────────────────────────────


class TestSensitivityInit:
    def test_basic_init(self):
        analyzer = _make_analyzer()
        assert len(analyzer.param_names) == 2
        assert analyzer.problem["num_vars"] == 2

    def test_problem_structure(self):
        analyzer = _make_analyzer()
        assert "names" in analyzer.problem
        assert "bounds" in analyzer.problem
        assert "num_vars" in analyzer.problem


# ── Sobol Analysis ───────────────────────────────────────────────────


class TestSobolAnalysis:
    def test_sobol_returns_indices(self):
        analyzer = _make_analyzer()
        t_eval = np.linspace(0, 1.0, 20)
        result = analyzer.analyze(
            t_span=(0, 1.0),
            t_eval=t_eval,
            method="sobol",
            N=32,
        )
        assert "S1" in result
        assert "ST" in result
        assert result["method"] == "sobol"

    def test_sobol_S1_shape(self):
        analyzer = _make_analyzer()
        t_eval = np.linspace(0, 1.0, 20)
        result = analyzer.analyze(
            t_span=(0, 1.0),
            t_eval=t_eval,
            method="sobol",
            N=32,
        )
        assert len(result["S1"]) == 2  # Two parameters

    def test_sobol_ST_shape(self):
        analyzer = _make_analyzer()
        t_eval = np.linspace(0, 1.0, 20)
        result = analyzer.analyze(
            t_span=(0, 1.0),
            t_eval=t_eval,
            method="sobol",
            N=32,
        )
        assert len(result["ST"]) == 2

    def test_sobol_indices_finite(self):
        analyzer = _make_analyzer()
        t_eval = np.linspace(0, 1.0, 20)
        result = analyzer.analyze(
            t_span=(0, 1.0),
            t_eval=t_eval,
            method="sobol",
            N=64,
        )
        assert np.all(np.isfinite(result["S1"]))
        assert np.all(np.isfinite(result["ST"]))

    def test_sobol_param_names(self):
        analyzer = _make_analyzer()
        t_eval = np.linspace(0, 1.0, 20)
        result = analyzer.analyze(
            t_span=(0, 1.0),
            t_eval=t_eval,
            method="sobol",
            N=32,
        )
        assert result["param_names"] == ["V", "F"]


# ── Morris Analysis ──────────────────────────────────────────────────


class TestMorrisAnalysis:
    def test_morris_returns_indices(self):
        analyzer = _make_analyzer()
        t_eval = np.linspace(0, 1.0, 20)
        result = analyzer.analyze(
            t_span=(0, 1.0),
            t_eval=t_eval,
            method="morris",
            N=16,
        )
        assert "mu" in result
        assert "mu_star" in result
        assert "sigma" in result
        assert result["method"] == "morris"

    def test_morris_mu_star_shape(self):
        analyzer = _make_analyzer()
        t_eval = np.linspace(0, 1.0, 20)
        result = analyzer.analyze(
            t_span=(0, 1.0),
            t_eval=t_eval,
            method="morris",
            N=16,
        )
        assert len(result["mu_star"]) == 2


# ── Error Handling ───────────────────────────────────────────────────


class TestSensitivityErrors:
    def test_invalid_method_raises(self):
        analyzer = _make_analyzer()
        t_eval = np.linspace(0, 1.0, 20)
        with pytest.raises(ValueError, match="Unknown method"):
            analyzer.analyze(
                t_span=(0, 1.0),
                t_eval=t_eval,
                method="invalid",
                N=32,
            )

    def test_output_index(self):
        analyzer = _make_analyzer()
        t_eval = np.linspace(0, 1.0, 20)
        result = analyzer.analyze(
            t_span=(0, 1.0),
            t_eval=t_eval,
            method="sobol",
            output_index=1,
            N=32,
        )
        assert "S1" in result

    def test_output_index_out_of_bounds_raises(self):
        analyzer = _make_analyzer()
        t_eval = np.linspace(0, 1.0, 20)
        with pytest.raises(ValueError, match="output_index"):
            analyzer.analyze(
                t_span=(0, 1.0),
                t_eval=t_eval,
                method="sobol",
                output_index=99,
                N=32,
            )

    def test_output_index_out_of_bounds_evaluate_model(self):
        """Cover lines 113-115: output_index >= state_dim raises ValueError
        directly from _evaluate_model."""
        analyzer = _make_analyzer()
        t_eval = np.linspace(0, 1.0, 20)
        param_values = np.array([[10.0, 2.0]])  # 1 sample, 2 params
        with pytest.raises(ValueError, match="output_index.*out of bounds"):
            analyzer._evaluate_model(
                param_values, t_span=(0, 1.0), t_eval=t_eval, output_index=99
            )


class TestSensitivityNaNHandling:
    """Tests for NaN handling in _evaluate_model."""

    def test_nan_fraction_above_threshold_raises(self):
        """Cover line 124: RuntimeError when > 50% evaluations produce NaN."""
        from unittest.mock import patch

        analyzer = _make_analyzer()
        t_eval = np.linspace(0, 1.0, 20)
        # Create 4 parameter samples
        param_values = np.array([
            [10.0, 2.0],
            [15.0, 3.0],
            [10.0, 2.0],
            [15.0, 3.0],
        ])

        # Mock solve_ivp to always fail (produces NaN for > 50%)
        from unittest.mock import MagicMock

        mock_sol = MagicMock()
        mock_sol.success = False

        with patch(
            "scipy.integrate.solve_ivp", return_value=mock_sol
        ), pytest.raises(RuntimeError, match="More than 50%"):
            analyzer._evaluate_model(
                param_values, t_span=(0, 1.0), t_eval=t_eval, output_index=0
            )

    def test_partial_nan_replaced_with_median(self):
        """Cover lines 130-134: some NaN values replaced with median of valid results."""
        from unittest.mock import MagicMock, patch

        analyzer = _make_analyzer()
        t_eval = np.linspace(0, 1.0, 20)
        # 4 samples: need < 50% failure for replacement logic
        param_values = np.array([
            [10.0, 2.0],
            [15.0, 3.0],
            [10.0, 2.0],
            [15.0, 3.0],
        ])

        call_count = 0

        def mock_solve_ivp(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            mock_sol = MagicMock()
            if call_count == 2:
                # Make the 2nd call fail -> NaN
                mock_sol.success = False
            else:
                # Other calls succeed
                mock_sol.success = True
                mock_sol.y = np.array([[1.0] * len(t_eval), [2.0] * len(t_eval)])
            return mock_sol

        with patch(
            "scipy.integrate.solve_ivp", side_effect=mock_solve_ivp
        ):
            outputs = analyzer._evaluate_model(
                param_values, t_span=(0, 1.0), t_eval=t_eval, output_index=0
            )
            # All outputs should be finite (NaN replaced with median)
            assert np.all(np.isfinite(outputs))
            # The NaN entry should now be equal to median of valid results
            assert outputs[1] == np.median(outputs[[0, 2, 3]])

    def test_exception_in_evaluation_produces_nan(self):
        """Cover lines 113-115 (exception path): exception during solve_ivp
        sets output to NaN."""
        from unittest.mock import patch

        analyzer = _make_analyzer()
        t_eval = np.linspace(0, 1.0, 20)
        param_values = np.array([
            [10.0, 2.0],
            [15.0, 3.0],
            [10.0, 2.0],
            [15.0, 3.0],
        ])

        call_count = 0

        def mock_solve_ivp(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 3:
                raise RuntimeError("Simulated solver crash")
            from unittest.mock import MagicMock
            mock_sol = MagicMock()
            mock_sol.success = True
            mock_sol.y = np.array([[5.0] * len(t_eval), [6.0] * len(t_eval)])
            return mock_sol

        with patch(
            "scipy.integrate.solve_ivp", side_effect=mock_solve_ivp
        ):
            outputs = analyzer._evaluate_model(
                param_values, t_span=(0, 1.0), t_eval=t_eval, output_index=0
            )
            # NaN was replaced with median of valid values
            assert np.all(np.isfinite(outputs))


class TestSALibUnavailable:
    """Test the ImportError path when SALib is not available."""

    def test_salib_unavailable_raises_import_error(self):
        """Cover line 49: SALib not available raises ImportError on init."""
        from unittest.mock import patch

        # Temporarily set the module-level SALIB_AVAILABLE to False
        with patch("reactor_twin.utils.sensitivity.SALIB_AVAILABLE", False):
            reactor = create_exothermic_cstr(isothermal=True)
            with pytest.raises(ImportError, match="SALib is required"):
                SensitivityAnalyzer(
                    reactor=reactor,
                    param_names=["V", "F"],
                    param_bounds=[[5.0, 20.0], [0.5, 5.0]],
                )


class TestSALibAvailable:
    """Test the SALib available=True import path."""

    def test_salib_available_flag_is_true(self):
        """Cover lines 22-24: verify SALIB_AVAILABLE is True when SALib is installed."""
        from reactor_twin.utils.sensitivity import SALIB_AVAILABLE

        assert SALIB_AVAILABLE is True
