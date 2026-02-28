"""Tests for Membrane Reactor."""

from __future__ import annotations

import numpy as np
import pytest

from reactor_twin.exceptions import ConfigurationError
from reactor_twin.reactors.membrane import MembraneReactor
from reactor_twin.utils.registry import REACTOR_REGISTRY

# ── helpers ──────────────────────────────────────────────────────────


def _default_params(**overrides) -> dict:
    """Return valid default params for a 2-species membrane reactor."""
    params = {
        "V_ret": 10.0,
        "V_perm": 5.0,
        "F_ret": 1.0,
        "F_perm": 0.5,
        "A_membrane": 2.0,
        "Q": [0.1, 0.05],
        "permeating_species_indices": [0, 1],
        "permeation_law": "linear",
        "C_ret_feed": [1.0, 0.5],
        "T_feed": 350.0,
    }
    params.update(overrides)
    return params


def _make_reactor(**overrides) -> MembraneReactor:
    params = _default_params(**overrides)
    return MembraneReactor(
        name="test_membrane",
        num_species=2,
        params=params,
    )


# ── Initialization ───────────────────────────────────────────────────


class TestMembraneInit:
    def test_basic_init(self):
        r = _make_reactor()
        assert r.name == "test_membrane"
        assert r.num_species == 2
        assert r.num_permeating == 2

    def test_state_dim_isothermal(self):
        r = _make_reactor()
        # 2 retentate + 2 permeate = 4
        assert r.state_dim == 4

    def test_state_dim_non_isothermal(self):
        params = _default_params(rho=1000.0, Cp=4.18, UA=100.0, T_coolant=300.0)
        r = MembraneReactor("test", 2, params, isothermal=False)
        # 2 ret + 2 perm + 1 T = 5
        assert r.state_dim == 5

    def test_partial_permeation(self):
        """Only one species permeates."""
        params = _default_params(
            permeating_species_indices=[0],
            Q=[0.1],
        )
        r = MembraneReactor("test", 2, params)
        assert r.num_permeating == 1
        assert r.state_dim == 3  # 2 ret + 1 perm

    def test_missing_required_param_raises(self):
        params = _default_params()
        del params["V_ret"]
        with pytest.raises(ConfigurationError, match="V_ret"):
            MembraneReactor("test", 2, params)

    def test_missing_V_perm_raises(self):
        params = _default_params()
        del params["V_perm"]
        with pytest.raises(ConfigurationError, match="V_perm"):
            MembraneReactor("test", 2, params)

    def test_missing_A_membrane_raises(self):
        params = _default_params()
        del params["A_membrane"]
        with pytest.raises(ConfigurationError, match="A_membrane"):
            MembraneReactor("test", 2, params)

    def test_invalid_permeation_law_raises(self):
        params = _default_params(permeation_law="invalid")
        with pytest.raises(ConfigurationError, match="permeation_law"):
            MembraneReactor("test", 2, params)

    def test_non_isothermal_missing_params_raises(self):
        params = _default_params()
        with pytest.raises(ConfigurationError, match="rho"):
            MembraneReactor("test", 2, params, isothermal=False)

    def test_Q_length_mismatch_raises(self):
        params = _default_params(Q=[0.1])  # Only 1 Q but 2 permeating species
        with pytest.raises(ConfigurationError, match="len.*Q"):
            MembraneReactor("test", 2, params)

    def test_species_index_out_of_bounds_raises(self):
        params = _default_params(permeating_species_indices=[0, 5])
        with pytest.raises(ConfigurationError, match="permeating_species_indices"):
            MembraneReactor("test", 2, params)

    def test_C_ret_feed_length_mismatch_raises(self):
        params = _default_params(C_ret_feed=[1.0])  # Only 1 but 2 species
        with pytest.raises(ConfigurationError, match="C_ret_feed"):
            MembraneReactor("test", 2, params)


# ── ODE RHS ──────────────────────────────────────────────────────────


class TestMembraneODE:
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

    def test_linear_permeation_direction(self):
        """Flux goes from high to low concentration."""
        r = _make_reactor()
        # Retentate has feed conc, permeate starts at 0
        y = np.array([1.0, 0.5, 0.0, 0.0])  # C_ret > C_perm
        dy = r.ode_rhs(0.0, y)
        # Check permeate increases
        assert dy[2] > 0  # C_perm_0 should increase (flux in)
        assert dy[3] > 0  # C_perm_1 should increase (flux in)

    def test_zero_flux_at_equilibrium(self):
        """No permeation when concentrations are equal."""
        r = _make_reactor()
        # Set retentate = permeate = feed (also zero dilution driving force)
        y = np.array([1.0, 0.5, 1.0, 0.5])
        dy = r.ode_rhs(0.0, y)
        # At equal concentrations with linear law, permeation flux = 0
        # The only driving force is the sweep flow on permeate side
        # dC_perm = (F_perm/V_perm)*(0 - C_perm) + J/V_perm
        # J = 0 at equal conc, so dC_perm = -(F_perm/V_perm)*C_perm < 0
        assert dy[2] < 0  # sweep removes permeate

    def test_sievert_law_sqrt(self):
        """Sievert's law uses sqrt of concentrations."""
        params = _default_params(permeation_law="sievert")
        r = MembraneReactor("test", 2, params)
        y = np.array([4.0, 1.0, 1.0, 0.25])
        dy = r.ode_rhs(0.0, y)
        assert np.all(np.isfinite(dy))

    def test_sievert_handles_zero_concentration(self):
        """Sievert's law with zero concentration should not produce NaN."""
        params = _default_params(permeation_law="sievert")
        r = MembraneReactor("test", 2, params)
        y = np.array([0.0, 0.0, 0.0, 0.0])
        dy = r.ode_rhs(0.0, y)
        assert np.all(np.isfinite(dy))

    def test_non_isothermal_ode_shape(self):
        params = _default_params(rho=1000.0, Cp=4.18, UA=100.0, T_coolant=300.0)
        r = MembraneReactor("test", 2, params, isothermal=False)
        y0 = r.get_initial_state()
        dy = r.ode_rhs(0.0, y0)
        assert dy.shape == (5,)  # 2 ret + 2 perm + 1 T

    def test_ode_rhs_with_controls(self):
        """Controls parameter is accepted (even if unused)."""
        r = _make_reactor()
        y0 = r.get_initial_state()
        dy = r.ode_rhs(0.0, y0, u=np.array([1.0]))
        assert dy.shape == y0.shape

    def test_mass_balance_conservation(self):
        """Flux leaving retentate must equal flux entering permeate."""
        r = _make_reactor()
        y = np.array([2.0, 1.0, 0.5, 0.25])
        dy = r.ode_rhs(0.0, y)
        V_ret, V_perm = 10.0, 5.0
        F_ret, F_perm = 1.0, 0.5
        C_ret_feed = np.array([1.0, 0.5])
        perm_idx = np.array([0, 1])

        # For each permeating species:
        # dC_ret[i]*V_ret = F_ret*(C_feed[i]-C_ret[i]) + rates[i]*V_ret - J[i]
        # dC_perm[j]*V_perm = F_perm*(0-C_perm[j]) + J[j]
        # So J[i] = dC_perm[j]*V_perm - F_perm*(0-C_perm[j])
        #         = dC_perm[j]*V_perm + F_perm*C_perm[j]
        # And from retentate side (no kinetics):
        # J[i] = F_ret*(C_feed[i]-C_ret[i]) - dC_ret[i]*V_ret
        for k, idx in enumerate(perm_idx):
            J_from_perm = dy[2 + k] * V_perm + F_perm * y[2 + k]
            J_from_ret = F_ret * (C_ret_feed[idx] - y[idx]) - dy[idx] * V_ret
            np.testing.assert_allclose(J_from_perm, J_from_ret, rtol=1e-10)


# ── Validate State ──────────────────────────────────────────────────


class TestMembraneValidateState:
    def test_validate_state_positive(self):
        r = _make_reactor()
        y = np.array([1.0, 0.5, 0.1, 0.05])
        assert r.validate_state(y)

    def test_validate_state_retentate_negative(self):
        r = _make_reactor()
        y = np.array([-0.1, 0.5, 0.1, 0.05])
        assert not r.validate_state(y)

    def test_validate_state_permeate_negative(self):
        r = _make_reactor()
        y = np.array([1.0, 0.5, -0.1, 0.05])
        assert not r.validate_state(y)

    def test_validate_state_both_phases(self):
        """Validate checks both retentate and permeate."""
        r = _make_reactor()
        # All positive
        assert r.validate_state(np.array([1.0, 0.5, 0.1, 0.05]))
        # Permeate negative
        assert not r.validate_state(np.array([1.0, 0.5, 0.1, -0.05]))


# ── Initial State & Labels ───────────────────────────────────────────


class TestMembraneState:
    def test_initial_state_shape(self):
        r = _make_reactor()
        y0 = r.get_initial_state()
        assert y0.shape == (4,)

    def test_initial_state_values(self):
        r = _make_reactor()
        y0 = r.get_initial_state()
        np.testing.assert_array_almost_equal(y0[:2], [1.0, 0.5])
        np.testing.assert_array_almost_equal(y0[2:], [0.0, 0.0])

    def test_custom_initial_state(self):
        params = _default_params(C_ret_initial=[0.8, 0.3], C_perm_initial=[0.1, 0.05])
        r = MembraneReactor("test", 2, params)
        y0 = r.get_initial_state()
        np.testing.assert_array_almost_equal(y0[:2], [0.8, 0.3])
        np.testing.assert_array_almost_equal(y0[2:], [0.1, 0.05])

    def test_state_labels(self):
        r = _make_reactor()
        labels = r.get_state_labels()
        assert labels == ["C_ret_0", "C_ret_1", "C_perm_0", "C_perm_1"]

    def test_state_labels_non_isothermal(self):
        params = _default_params(rho=1000.0, Cp=4.18, UA=100.0, T_coolant=300.0)
        r = MembraneReactor("test", 2, params, isothermal=False)
        labels = r.get_state_labels()
        assert labels[-1] == "T"
        assert len(labels) == 5


# ── Serialization ────────────────────────────────────────────────────


class TestMembraneSerialization:
    def test_to_dict(self):
        r = _make_reactor()
        d = r.to_dict()
        assert d["name"] == "test_membrane"
        assert d["type"] == "MembraneReactor"
        assert d["num_species"] == 2
        assert d["state_dim"] == 4

    def test_from_dict_roundtrip(self):
        r = _make_reactor()
        d = r.to_dict()
        r2 = MembraneReactor.from_dict(d)
        assert r2.name == r.name
        assert r2.num_species == r.num_species
        assert r2.state_dim == r.state_dim

    def test_repr(self):
        r = _make_reactor()
        s = repr(r)
        assert "MembraneReactor" in s
        assert "test_membrane" in s


# ── Registry ─────────────────────────────────────────────────────────


class TestMembraneRegistry:
    def test_registered_in_reactor_registry(self):
        assert "membrane" in REACTOR_REGISTRY

    def test_registry_get_returns_class(self):
        cls = REACTOR_REGISTRY.get("membrane")
        assert cls is MembraneReactor

    def test_registry_instantiation(self):
        cls = REACTOR_REGISTRY.get("membrane")
        params = _default_params()
        r = cls(name="from_registry", num_species=2, params=params)
        assert isinstance(r, MembraneReactor)


# ── Integration (scipy) ─────────────────────────────────────────────


class TestMembraneIntegration:
    def test_scipy_integration(self):
        """Ensure ODE can be integrated with scipy."""
        from scipy.integrate import solve_ivp

        r = _make_reactor()
        y0 = r.get_initial_state()
        sol = solve_ivp(r.ode_rhs, [0, 1.0], y0, method="RK45", max_step=0.1)
        assert sol.success
        assert sol.y.shape[0] == 4
        assert np.all(np.isfinite(sol.y))

    def test_permeate_increases_over_time(self):
        """Permeate concentrations should increase from zero."""
        from scipy.integrate import solve_ivp

        r = _make_reactor()
        y0 = r.get_initial_state()
        t_eval = np.linspace(0, 5.0, 50)
        sol = solve_ivp(r.ode_rhs, [0, 5.0], y0, t_eval=t_eval, method="RK45")
        assert sol.success
        # Permeate species 0 should be higher at end than start
        assert sol.y[2, -1] > sol.y[2, 0]
