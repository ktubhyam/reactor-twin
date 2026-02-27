"""Smoke tests for all dashboard pages.

Verify that every page module can be imported and that key
globals / functions are accessible.  These tests are skipped if
streamlit is not installed.
"""

from __future__ import annotations

import importlib

import pytest

# Skip the entire module if streamlit is not installed
st = pytest.importorskip("streamlit")

# All 10 page module names (digit-prefixed -> importlib required)
PAGE_MODULES = [
    "reactor_twin.dashboard.pages.01_reactor_sim",
    "reactor_twin.dashboard.pages.02_phase_portrait",
    "reactor_twin.dashboard.pages.03_bifurcation",
    "reactor_twin.dashboard.pages.04_rtd_analysis",
    "reactor_twin.dashboard.pages.05_parameter_sweep",
    "reactor_twin.dashboard.pages.06_sensitivity",
    "reactor_twin.dashboard.pages.07_pareto",
    "reactor_twin.dashboard.pages.08_fault_monitor",
    "reactor_twin.dashboard.pages.09_model_validation",
    "reactor_twin.dashboard.pages.10_latent_explorer",
]


# ── app module ──────────────────────────────────────────────────────

class TestDashboardApp:
    """Test that the main dashboard app module can be imported."""

    def test_import_app(self) -> None:
        mod = importlib.import_module("reactor_twin.dashboard.app")
        assert hasattr(mod, "main")


# ── page modules ────────────────────────────────────────────────────

class TestDashboardPages:
    """Test that each dashboard page can be imported without error."""

    @pytest.mark.parametrize("module_name", PAGE_MODULES)
    def test_import_page(self, module_name: str) -> None:
        """Import the page module; success means no import-time crash."""
        mod = importlib.import_module(module_name)
        # The module object must exist
        assert mod is not None


# ── _safe_import_plotly helper ──────────────────────────────────────

class TestSafeImportPlotly:
    """Test the _safe_import_plotly helper from the reactor_sim page."""

    def test_safe_import_plotly_returns_module_or_none(self) -> None:
        mod = importlib.import_module(
            "reactor_twin.dashboard.pages.01_reactor_sim"
        )
        fn = getattr(mod, "_safe_import_plotly", None)
        if fn is None:
            pytest.skip("_safe_import_plotly not found in module")
        result = fn()
        # Should either return the plotly.graph_objects module or None
        if result is not None:
            assert hasattr(result, "Figure")
        # If result is None, plotly is simply not installed -- that is fine.
