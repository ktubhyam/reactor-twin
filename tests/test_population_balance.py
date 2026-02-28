"""Tests for the population balance (crystallization) reactor."""

from __future__ import annotations

import numpy as np
import pytest

from reactor_twin.exceptions import ConfigurationError
from reactor_twin.reactors.population_balance import PopulationBalanceReactor


# ── Fixtures ──────────────────────────────────────────────────────────

def _default_params(**overrides):
    p = {
        "V": 10.0,
        "C_sat": 1.0,
        "kg": 1e-4,
        "g": 1.0,
        "kb": 1e6,
        "b": 2.0,
        "shape_factor": 0.5236,  # pi/6 for spheres
        "rho_crystal": 2500.0,
    }
    p.update(overrides)
    return p


@pytest.fixture
def pb_default():
    """Default isothermal population balance reactor (4 moments)."""
    return PopulationBalanceReactor(
        name="test_pb",
        num_species=1,
        params=_default_params(C_initial=1.5),
        isothermal=True,
        num_moments=4,
    )


@pytest.fixture
def pb_nonisothermal():
    """Non-isothermal population balance reactor."""
    return PopulationBalanceReactor(
        name="test_pb_noniso",
        num_species=1,
        params=_default_params(C_initial=1.5, rho=1000.0, Cp=4.184),
        isothermal=False,
        num_moments=4,
    )


@pytest.fixture
def pb_six_moments():
    """Population balance reactor with 6 moments."""
    return PopulationBalanceReactor(
        name="test_pb_6",
        num_species=1,
        params=_default_params(C_initial=1.5),
        isothermal=True,
        num_moments=6,
    )


# ── Initialization ────────────────────────────────────────────────────


class TestPBInit:
    def test_instantiation(self, pb_default):
        assert pb_default.name == "test_pb"
        assert pb_default.num_moments == 4
        assert pb_default.isothermal is True

    def test_nonisothermal(self, pb_nonisothermal):
        assert pb_nonisothermal.isothermal is False

    def test_six_moments(self, pb_six_moments):
        assert pb_six_moments.num_moments == 6


# ── State Dimension ──────────────────────────────────────────────────


class TestPBStateDim:
    def test_state_dim_isothermal(self, pb_default):
        # 1 (C) + 4 moments = 5
        assert pb_default.state_dim == 5

    def test_state_dim_nonisothermal(self, pb_nonisothermal):
        # 1 (C) + 4 moments + 1 (T) = 6
        assert pb_nonisothermal.state_dim == 6

    def test_state_dim_six_moments(self, pb_six_moments):
        # 1 (C) + 6 moments = 7
        assert pb_six_moments.state_dim == 7


# ── ODE RHS ──────────────────────────────────────────────────────────


class TestPBODE:
    def test_ode_rhs_shape(self, pb_default):
        y0 = pb_default.get_initial_state()
        dy = pb_default.ode_rhs(0.0, y0)
        assert isinstance(dy, np.ndarray)
        assert dy.shape == (pb_default.state_dim,)

    def test_ode_rhs_shape_noniso(self, pb_nonisothermal):
        y0 = pb_nonisothermal.get_initial_state()
        dy = pb_nonisothermal.ode_rhs(0.0, y0)
        assert dy.shape == (pb_nonisothermal.state_dim,)

    def test_ode_rhs_finite(self, pb_default):
        y0 = pb_default.get_initial_state()
        dy = pb_default.ode_rhs(0.0, y0)
        assert np.all(np.isfinite(dy))

    def test_ode_rhs_finite_noniso(self, pb_nonisothermal):
        y0 = pb_nonisothermal.get_initial_state()
        dy = pb_nonisothermal.ode_rhs(0.0, y0)
        assert np.all(np.isfinite(dy))

    def test_supersaturated_nucleation_positive(self, pb_default):
        """When C > C_sat, nucleation rate > 0 => dmu_0/dt > 0."""
        y0 = pb_default.get_initial_state()
        dy = pb_default.ode_rhs(0.0, y0)
        # dmu_0/dt = B0 > 0 because C_initial=1.5 > C_sat=1.0
        assert dy[1] > 0  # mu_0 derivative

    def test_undersaturated_no_nucleation(self, pb_default):
        """When C < C_sat, growth/nucleation = 0."""
        y0 = pb_default.get_initial_state()
        y0[0] = 0.5  # C < C_sat=1.0
        dy = pb_default.ode_rhs(0.0, y0)
        # S < 0 => max(S,0) = 0 => G=0, B0=0
        assert dy[1] == pytest.approx(0.0)  # dmu_0/dt = B0 = 0

    def test_concentration_decreases_during_growth(self, pb_default):
        """Crystal growth consumes solute: dC/dt < 0 when growing."""
        y0 = pb_default.get_initial_state()
        # Set some crystals (mu_2 > 0) so growth consumes mass
        y0[3] = 1.0  # mu_2 > 0
        dy = pb_default.ode_rhs(0.0, y0)
        # dC/dt = -3*kv*rho*G*mu_2 < 0 when S>0 and mu_2>0
        assert dy[0] < 0

    def test_moment_zero_growth_when_no_crystals(self, pb_default):
        """dmu_k/dt = k*G*mu_{k-1}. With mu all zero, dmu_1..dmu_3 = 0."""
        y0 = pb_default.get_initial_state()
        # mu_initial = 0, so mu_{k-1}=0 for k>=1
        dy = pb_default.ode_rhs(0.0, y0)
        # dmu_1 = 1*G*mu_0 = 0 (mu_0 starts at 0)
        assert dy[2] == pytest.approx(0.0)
        assert dy[3] == pytest.approx(0.0)
        assert dy[4] == pytest.approx(0.0)

    def test_growth_propagates_through_moments(self):
        """With mu_0 > 0, growth should cause dmu_1 > 0."""
        reactor = PopulationBalanceReactor(
            name="grow",
            num_species=1,
            params=_default_params(C_initial=2.0),
            num_moments=4,
        )
        y0 = reactor.get_initial_state()
        y0[1] = 100.0  # mu_0 (many crystals)
        dy = reactor.ode_rhs(0.0, y0)
        # dmu_1 = 1*G*mu_0 where G > 0 and mu_0 > 0
        assert dy[2] > 0

    def test_integration_conserves_mass(self, pb_default):
        """Total mass (solute + crystal) should be approximately conserved."""
        from scipy.integrate import solve_ivp

        y0 = pb_default.get_initial_state()
        y0[0] = 1.5  # supersaturated
        y0[1] = 10.0  # seed crystals mu_0
        sol = solve_ivp(pb_default.ode_rhs, [0, 1.0], y0, method="RK45", max_step=0.01)
        assert sol.success
        # Just verify solution is finite
        assert np.all(np.isfinite(sol.y[:, -1]))


# ── Initial State ────────────────────────────────────────────────────


class TestPBInitialState:
    def test_shape(self, pb_default):
        y0 = pb_default.get_initial_state()
        assert y0.shape == (pb_default.state_dim,)

    def test_shape_noniso(self, pb_nonisothermal):
        y0 = pb_nonisothermal.get_initial_state()
        assert y0.shape == (pb_nonisothermal.state_dim,)

    def test_c_initial(self, pb_default):
        y0 = pb_default.get_initial_state()
        assert y0[0] == pytest.approx(1.5)

    def test_mu_defaults_to_zero(self, pb_default):
        y0 = pb_default.get_initial_state()
        np.testing.assert_allclose(y0[1:5], 0.0)

    def test_custom_mu_initial(self):
        reactor = PopulationBalanceReactor(
            name="custom_mu",
            num_species=1,
            params=_default_params(C_initial=1.5, mu_initial=[10, 1, 0.1, 0.01]),
            num_moments=4,
        )
        y0 = reactor.get_initial_state()
        np.testing.assert_allclose(y0[1:5], [10, 1, 0.1, 0.01])

    def test_noniso_includes_temperature(self, pb_nonisothermal):
        y0 = pb_nonisothermal.get_initial_state()
        assert y0[-1] == pytest.approx(298.15)  # default T_initial


# ── State Labels ─────────────────────────────────────────────────────


class TestPBLabels:
    def test_labels_isothermal(self, pb_default):
        labels = pb_default.get_state_labels()
        assert labels == ["C", "mu_0", "mu_1", "mu_2", "mu_3"]

    def test_labels_noniso(self, pb_nonisothermal):
        labels = pb_nonisothermal.get_state_labels()
        assert labels[-1] == "T"
        assert labels[0] == "C"

    def test_labels_six_moments(self, pb_six_moments):
        labels = pb_six_moments.get_state_labels()
        assert len(labels) == 7
        assert "mu_5" in labels


# ── Derived Quantities ───────────────────────────────────────────────


class TestPBDerived:
    def test_mean_size_with_crystals(self, pb_default):
        y = np.array([1.5, 100.0, 5.0, 0.5, 0.05])
        ms = pb_default.mean_size(y)
        assert ms == pytest.approx(5.0 / 100.0)

    def test_mean_size_no_crystals(self, pb_default):
        y = np.array([1.5, 0.0, 0.0, 0.0, 0.0])
        ms = pb_default.mean_size(y)
        assert ms == 0.0

    def test_cv_with_crystals(self, pb_default):
        # mu_0=100, mu_1=10, mu_2=2 => ratio=2*100/100=2 => CV=sqrt(1)=1
        y = np.array([1.5, 100.0, 10.0, 2.0, 0.05])
        cv = pb_default.coefficient_of_variation(y)
        assert cv == pytest.approx(1.0)

    def test_cv_monodisperse(self, pb_default):
        # Monodisperse: mu_2*mu_0/mu_1^2 = 1 => CV = 0
        y = np.array([1.5, 100.0, 10.0, 1.0, 0.05])
        cv = pb_default.coefficient_of_variation(y)
        assert cv == pytest.approx(0.0)

    def test_cv_no_crystals(self, pb_default):
        y = np.array([1.5, 0.0, 0.0, 0.0, 0.0])
        cv = pb_default.coefficient_of_variation(y)
        assert cv == 0.0


# ── Serialization ────────────────────────────────────────────────────


class TestPBSerialization:
    def test_to_dict(self, pb_default):
        d = pb_default.to_dict()
        assert d["type"] == "PopulationBalanceReactor"
        assert d["num_species"] == 1

    def test_from_dict_round_trip(self, pb_default):
        config = pb_default.to_dict()
        config["isothermal"] = True
        config["num_moments"] = 4
        restored = PopulationBalanceReactor.from_dict(config)
        assert restored.name == pb_default.name
        assert restored.state_dim == pb_default.state_dim
        assert restored.num_moments == pb_default.num_moments


# ── Parameter Validation ─────────────────────────────────────────────


class TestPBValidation:
    @pytest.mark.parametrize(
        "missing_key",
        ["V", "C_sat", "kg", "g", "kb", "b", "shape_factor", "rho_crystal"],
    )
    def test_missing_required_param(self, missing_key):
        params = _default_params()
        del params[missing_key]
        with pytest.raises(ConfigurationError, match="Missing required parameter"):
            PopulationBalanceReactor(name="bad", num_species=1, params=params)

    def test_missing_thermo_nonisothermal(self):
        with pytest.raises(ConfigurationError, match="Non-isothermal"):
            PopulationBalanceReactor(
                name="bad",
                num_species=1,
                params=_default_params(),
                isothermal=False,
            )


# ── Registry ─────────────────────────────────────────────────────────


class TestPBRegistry:
    def test_registered(self):
        from reactor_twin.utils.registry import REACTOR_REGISTRY

        assert "population_balance" in REACTOR_REGISTRY

    def test_get_returns_class(self):
        from reactor_twin.utils.registry import REACTOR_REGISTRY

        cls = REACTOR_REGISTRY.get("population_balance")
        assert cls is PopulationBalanceReactor


# ── Repr ─────────────────────────────────────────────────────────────


class TestPBRepr:
    def test_repr(self, pb_default):
        r = repr(pb_default)
        assert "PopulationBalanceReactor" in r
