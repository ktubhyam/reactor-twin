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
