"""Tests for Fluidized Bed Reactor."""

from __future__ import annotations

import numpy as np
import pytest

from reactor_twin.exceptions import ConfigurationError
from reactor_twin.reactors.fluidized_bed import FluidizedBedReactor
from reactor_twin.utils.registry import REACTOR_REGISTRY

# ── helpers ──────────────────────────────────────────────────────────


def _default_params(**overrides) -> dict:
    """Return valid default params for a 2-species fluidized bed."""
    params = {
        "u_mf": 0.05,
        "u_0": 0.20,
        "epsilon_mf": 0.45,
        "d_b": 0.05,
        "H_bed": 2.0,
        "A_bed": 1.0,
        "K_be": 0.5,
        "C_feed": [1.0, 0.0],
        "T_feed": 400.0,
        "gamma_b": 0.01,
    }
    params.update(overrides)
    return params


def _make_reactor(**overrides) -> FluidizedBedReactor:
    params = _default_params(**overrides)
    return FluidizedBedReactor(
        name="test_fb",
        num_species=2,
        params=params,
    )


# ── Initialization ───────────────────────────────────────────────────


class TestFluidizedBedInit:
    def test_basic_init(self):
        r = _make_reactor()
        assert r.name == "test_fb"
        assert r.num_species == 2

    def test_state_dim_isothermal(self):
        r = _make_reactor()
        # 2 bubble + 2 emulsion = 4
        assert r.state_dim == 4

    def test_state_dim_non_isothermal(self):
        params = _default_params(rho=1000.0, Cp=1.0)
        r = FluidizedBedReactor("test", 2, params, isothermal=False)
        assert r.state_dim == 5

    def test_missing_required_param_raises(self):
        params = _default_params()
        del params["u_mf"]
        with pytest.raises(ConfigurationError, match="u_mf"):
            FluidizedBedReactor("test", 2, params)

    def test_missing_K_be_raises(self):
        params = _default_params()
        del params["K_be"]
        with pytest.raises(ConfigurationError, match="K_be"):
            FluidizedBedReactor("test", 2, params)

    def test_u0_less_than_umf_raises(self):
        params = _default_params(u_0=0.03, u_mf=0.05)
        with pytest.raises(ConfigurationError, match="u_0"):
            FluidizedBedReactor("test", 2, params)

    def test_u0_equal_umf_raises(self):
        params = _default_params(u_0=0.05, u_mf=0.05)
        with pytest.raises(ConfigurationError, match="u_0"):
            FluidizedBedReactor("test", 2, params)

    def test_default_gamma_b(self):
        params = _default_params()
        del params["gamma_b"]
        r = FluidizedBedReactor("test", 2, params)
        assert r.params["gamma_b"] == 0.01

    def test_non_isothermal_missing_params_raises(self):
        params = _default_params()
        with pytest.raises(ConfigurationError, match="rho"):
            FluidizedBedReactor("test", 2, params, isothermal=False)

    def test_d_b_zero_raises(self):
        params = _default_params(d_b=0.0)
        with pytest.raises(ConfigurationError, match="d_b"):
            FluidizedBedReactor("test", 2, params)

    def test_epsilon_mf_invalid_raises(self):
        for val in [0.0, 1.0, -0.1, 1.5]:
            params = _default_params(epsilon_mf=val)
            with pytest.raises(ConfigurationError, match="epsilon_mf"):
                FluidizedBedReactor("test", 2, params)

    def test_K_be_negative_raises(self):
        params = _default_params(K_be=-0.5)
        with pytest.raises(ConfigurationError, match="K_be"):
            FluidizedBedReactor("test", 2, params)

    def test_params_not_mutated(self):
        """Constructor should not mutate the caller's dict."""
        params = _default_params()
        del params["gamma_b"]
        original_keys = set(params.keys())
        FluidizedBedReactor("test", 2, params)
        assert set(params.keys()) == original_keys
        assert "gamma_b" not in params


# ── Physics helpers ──────────────────────────────────────────────────


class TestFluidizedBedPhysics:
    def test_bubble_rise_velocity_positive(self):
        r = _make_reactor()
        u_b = r._bubble_rise_velocity()
        assert u_b > 0

    def test_bubble_rise_velocity_numeric(self):
        """Hand-calculated: u_b = (0.20 - 0.05) + 0.711*sqrt(9.81*0.05)."""
        r = _make_reactor()
        u_b = r._bubble_rise_velocity()
        expected = 0.15 + 0.711 * np.sqrt(9.81 * 0.05)
        np.testing.assert_allclose(u_b, expected, rtol=1e-10)

    def test_bubble_fraction_between_0_and_1(self):
        r = _make_reactor()
        delta = r._bubble_fraction()
        assert 0 < delta < 1

    def test_phase_volumes_positive(self):
        r = _make_reactor()
        V_b, V_e = r._phase_volumes()
        assert V_b > 0
        assert V_e > 0


# ── ODE RHS ──────────────────────────────────────────────────────────


class TestFluidizedBedODE:
    def test_ode_rhs_shape(self):
        r = _make_reactor()
        y0 = r.get_initial_state()
        dy = r.ode_rhs(0.0, y0)
        assert dy.shape == y0.shape

    def test_ode_rhs_finite(self):
        r = _make_reactor()
        y0 = r.get_initial_state()
        dy = r.ode_rhs(0.0, y0)
        assert np.all(np.isfinite(dy))

    def test_mass_transfer_direction(self):
        """Mass should transfer from high to low concentration phase."""
        r = _make_reactor()
        # Bubble has high conc, emulsion has low
        y = np.array([2.0, 0.0, 0.5, 0.0])
        dy = r.ode_rhs(0.0, y)
        # Emulsion should gain (K_be > 0 and C_b > C_e)
        assert dy[2] > 0  # C_e_0 increases

    def test_equal_concentrations_zero_transfer(self):
        """At equal bubble/emulsion conc, no inter-phase transfer."""
        r = _make_reactor()
        # Same concentration in both phases, equal to feed
        C_feed = np.array([1.0, 0.0])
        y = np.concatenate([C_feed, C_feed])
        dy = r.ode_rhs(0.0, y)
        # When C_b == C_e == C_feed: flow term = 0 and transfer term = 0
        # Only gamma_b * rates remains (which is 0 with no kinetics)
        np.testing.assert_array_almost_equal(dy, 0.0, decimal=10)

    def test_ode_rhs_with_controls(self):
        r = _make_reactor()
        y0 = r.get_initial_state()
        dy = r.ode_rhs(0.0, y0, u=np.array([1.0]))
        assert dy.shape == y0.shape

    def test_non_isothermal_ode_shape(self):
        params = _default_params(rho=1000.0, Cp=1.0)
        r = FluidizedBedReactor("test", 2, params, isothermal=False)
        y0 = r.get_initial_state()
        dy = r.ode_rhs(0.0, y0)
        assert dy.shape == (5,)

    def test_bubble_phase_uses_bubble_concentrations(self):
        """Verify bubble phase rates are computed from C_b, not C_e."""
        from unittest.mock import MagicMock

        kinetics = MagicMock()
        kinetics.compute_rates = MagicMock(return_value=np.array([0.1, -0.1]))

        params = _default_params()
        r = FluidizedBedReactor("test", 2, params, kinetics=kinetics)

        y = np.array([2.0, 0.5, 0.8, 0.2])  # C_b = [2, 0.5], C_e = [0.8, 0.2]
        r.ode_rhs(0.0, y)

        # compute_rates should be called twice: once for C_b, once for C_e
        assert kinetics.compute_rates.call_count == 2
        # First call should be with C_b = [2.0, 0.5]
        first_call_conc = kinetics.compute_rates.call_args_list[0][0][0]
        np.testing.assert_array_almost_equal(first_call_conc, [2.0, 0.5])
        # Second call should be with C_e = [0.8, 0.2]
        second_call_conc = kinetics.compute_rates.call_args_list[1][0][0]
        np.testing.assert_array_almost_equal(second_call_conc, [0.8, 0.2])


# ── Validate State ──────────────────────────────────────────────────


class TestFluidizedBedValidateState:
    def test_validate_state_both_phases_positive(self):
        r = _make_reactor()
        y = np.array([1.0, 0.5, 0.8, 0.2])
        assert r.validate_state(y)

    def test_validate_state_bubble_negative(self):
        r = _make_reactor()
        y = np.array([-0.1, 0.5, 0.8, 0.2])
        assert not r.validate_state(y)

    def test_validate_state_emulsion_negative(self):
        r = _make_reactor()
        y = np.array([1.0, 0.5, -0.1, 0.2])
        assert not r.validate_state(y)


# ── Initial State & Labels ───────────────────────────────────────────


class TestFluidizedBedState:
    def test_initial_state_shape(self):
        r = _make_reactor()
        y0 = r.get_initial_state()
        assert y0.shape == (4,)

    def test_initial_state_defaults_to_feed(self):
        r = _make_reactor()
        y0 = r.get_initial_state()
        np.testing.assert_array_almost_equal(y0[:2], [1.0, 0.0])
        np.testing.assert_array_almost_equal(y0[2:], [1.0, 0.0])

    def test_custom_initial_state(self):
        params = _default_params(C_b_initial=[0.8, 0.1], C_e_initial=[0.5, 0.05])
        r = FluidizedBedReactor("test", 2, params)
        y0 = r.get_initial_state()
        np.testing.assert_array_almost_equal(y0[:2], [0.8, 0.1])
        np.testing.assert_array_almost_equal(y0[2:], [0.5, 0.05])

    def test_state_labels(self):
        r = _make_reactor()
        labels = r.get_state_labels()
        assert labels == ["C_b_0", "C_b_1", "C_e_0", "C_e_1"]

    def test_state_labels_non_isothermal(self):
        params = _default_params(rho=1000.0, Cp=1.0)
        r = FluidizedBedReactor("test", 2, params, isothermal=False)
        labels = r.get_state_labels()
        assert labels[-1] == "T"


# ── Serialization ────────────────────────────────────────────────────


class TestFluidizedBedSerialization:
    def test_to_dict(self):
        r = _make_reactor()
        d = r.to_dict()
        assert d["type"] == "FluidizedBedReactor"
        assert d["state_dim"] == 4

    def test_from_dict_roundtrip(self):
        r = _make_reactor()
        d = r.to_dict()
        r2 = FluidizedBedReactor.from_dict(d)
        assert r2.name == r.name
        assert r2.state_dim == r.state_dim

    def test_repr(self):
        r = _make_reactor()
        s = repr(r)
        assert "FluidizedBedReactor" in s


# ── Registry ─────────────────────────────────────────────────────────


class TestFluidizedBedRegistry:
    def test_registered(self):
        assert "fluidized_bed" in REACTOR_REGISTRY

    def test_registry_get(self):
        cls = REACTOR_REGISTRY.get("fluidized_bed")
        assert cls is FluidizedBedReactor


# ── Integration (scipy) ─────────────────────────────────────────────


class TestFluidizedBedIntegration:
    def test_scipy_integration(self):
        from scipy.integrate import solve_ivp

        r = _make_reactor()
        y0 = r.get_initial_state()
        sol = solve_ivp(r.ode_rhs, [0, 1.0], y0, method="RK45", max_step=0.1)
        assert sol.success
        assert np.all(np.isfinite(sol.y))
