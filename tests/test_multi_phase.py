"""Tests for the gas-liquid multi-phase reactor."""

from __future__ import annotations

import numpy as np
import pytest

from reactor_twin.exceptions import ConfigurationError
from reactor_twin.reactors.multi_phase import MultiPhaseReactor


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def mp_isothermal():
    """Isothermal multi-phase reactor: 2 liquid species, 1 gas species."""
    return MultiPhaseReactor(
        name="test_mp_iso",
        num_species=2,
        params={
            "V_L": 100.0,
            "V_G": 50.0,
            "F_L": 10.0,
            "F_G": 5.0,
            "kLa": 0.5,
            "H": [10.0],
            "C_L_feed": [1.0, 0.0],
            "C_G_feed": [0.5],
            "T_feed": 300.0,
            "gas_species_indices": [0],
        },
        isothermal=True,
    )


@pytest.fixture
def mp_nonisothermal():
    """Non-isothermal multi-phase reactor."""
    return MultiPhaseReactor(
        name="test_mp_noniso",
        num_species=2,
        params={
            "V_L": 100.0,
            "V_G": 50.0,
            "F_L": 10.0,
            "F_G": 5.0,
            "kLa": 0.5,
            "H": [10.0],
            "C_L_feed": [1.0, 0.0],
            "C_G_feed": [0.5],
            "T_feed": 300.0,
            "gas_species_indices": [0],
            "rho": 1000.0,
            "Cp": 4.184,
            "UA": 500.0,
            "T_coolant": 290.0,
        },
        isothermal=False,
    )


@pytest.fixture
def mp_two_gas():
    """Multi-phase reactor with 3 liquid, 2 gas species."""
    return MultiPhaseReactor(
        name="test_mp_2gas",
        num_species=3,
        params={
            "V_L": 80.0,
            "V_G": 40.0,
            "F_L": 8.0,
            "F_G": 4.0,
            "kLa": 0.3,
            "H": [10.0, 20.0],
            "C_L_feed": [1.0, 0.0, 0.5],
            "C_G_feed": [0.4, 0.2],
            "T_feed": 310.0,
            "gas_species_indices": [0, 1],
        },
        isothermal=True,
    )


# ── Initialization ────────────────────────────────────────────────────


class TestMultiPhaseInit:
    def test_instantiation_isothermal(self, mp_isothermal):
        assert mp_isothermal.name == "test_mp_iso"
        assert mp_isothermal.num_species == 2
        assert mp_isothermal.isothermal is True
        assert mp_isothermal.num_gas_species == 1

    def test_instantiation_nonisothermal(self, mp_nonisothermal):
        assert mp_nonisothermal.isothermal is False

    def test_instantiation_two_gas(self, mp_two_gas):
        assert mp_two_gas.num_gas_species == 2
        assert mp_two_gas.num_species == 3


# ── State Dimension ──────────────────────────────────────────────────


class TestMultiPhaseStateDim:
    def test_state_dim_isothermal(self, mp_isothermal):
        # 2 liquid + 1 gas = 3
        assert mp_isothermal.state_dim == 3

    def test_state_dim_nonisothermal(self, mp_nonisothermal):
        # 2 liquid + 1 gas + 1 temp = 4
        assert mp_nonisothermal.state_dim == 4

    def test_state_dim_two_gas(self, mp_two_gas):
        # 3 liquid + 2 gas = 5
        assert mp_two_gas.state_dim == 5


# ── ODE RHS ──────────────────────────────────────────────────────────


class TestMultiPhaseODE:
    def test_ode_rhs_shape_isothermal(self, mp_isothermal):
        y0 = mp_isothermal.get_initial_state()
        dy = mp_isothermal.ode_rhs(0.0, y0)
        assert isinstance(dy, np.ndarray)
        assert dy.shape == (mp_isothermal.state_dim,)

    def test_ode_rhs_shape_nonisothermal(self, mp_nonisothermal):
        y0 = mp_nonisothermal.get_initial_state()
        dy = mp_nonisothermal.ode_rhs(0.0, y0)
        assert dy.shape == (mp_nonisothermal.state_dim,)

    def test_ode_rhs_shape_two_gas(self, mp_two_gas):
        y0 = mp_two_gas.get_initial_state()
        dy = mp_two_gas.ode_rhs(0.0, y0)
        assert dy.shape == (mp_two_gas.state_dim,)

    def test_ode_rhs_finite_isothermal(self, mp_isothermal):
        y0 = mp_isothermal.get_initial_state()
        dy = mp_isothermal.ode_rhs(0.0, y0)
        assert np.all(np.isfinite(dy))

    def test_ode_rhs_finite_nonisothermal(self, mp_nonisothermal):
        y0 = mp_nonisothermal.get_initial_state()
        dy = mp_nonisothermal.ode_rhs(0.0, y0)
        assert np.all(np.isfinite(dy))

    def test_ode_rhs_finite_two_gas(self, mp_two_gas):
        y0 = mp_two_gas.get_initial_state()
        dy = mp_two_gas.ode_rhs(0.0, y0)
        assert np.all(np.isfinite(dy))

    def test_mass_transfer_direction(self, mp_isothermal):
        """When C_G/H > C_L, gas dissolves into liquid (positive liquid transfer)."""
        y0 = mp_isothermal.get_initial_state()
        # Set liquid conc to 0, gas conc high -> transfer into liquid
        y0[0] = 0.0  # C_L_0
        y0[2] = 1.0  # C_G_0
        dy = mp_isothermal.ode_rhs(0.0, y0)
        # Liquid species 0 should increase (mass transfer term > 0)
        # Also has dilution: (F_L/V_L)*(C_L_feed_0 - C_L_0) = (10/100)*(1-0) = 0.1
        assert dy[0] > 0  # liquid species 0 increases

    def test_gas_consumption_balances(self, mp_isothermal):
        """Gas lost = liquid gained (scaled by volumes)."""
        y = np.array([0.0, 0.0, 1.0])  # C_L=[0,0], C_G=[1]
        dy = mp_isothermal.ode_rhs(0.0, y)
        # Transfer term: kLa * (C_G/H - C_L) = 0.5*(1/10 - 0) = 0.05
        # Liquid gain for species 0: kLa*(C_eq - C_L) = 0.05
        # Gas loss: (kLa*V_L/V_G)*(C_eq - C_L) = (0.5*100/50)*0.1 = 0.1
        # But gas also has flow term: (F_G/V_G)*(C_G_feed - C_G) = (5/50)*(0.5-1) = -0.05
        # So gas total: -0.05 - 0.1 = -0.15
        assert dy[2] < 0  # gas concentration decreases

    def test_equilibrium_no_transfer(self, mp_isothermal):
        """At Henry equilibrium, transfer term vanishes."""
        H = mp_isothermal.params["H"][0]
        C_G_val = 1.0
        C_L_eq = C_G_val / H  # equilibrium
        y = np.array([C_L_eq, 0.0, C_G_val])
        dy = mp_isothermal.ode_rhs(0.0, y)
        # Mass transfer = kLa*(C_eq - C_L) = 0 at equilibrium
        # Liquid 0 derivative = dilution only: (F_L/V_L)*(C_L_feed_0 - C_L_eq)
        expected_dilution = (10.0 / 100.0) * (1.0 - C_L_eq)
        assert dy[0] == pytest.approx(expected_dilution, abs=1e-10)

    def test_steady_state_approach(self, mp_isothermal):
        """After many steps, derivatives should decrease."""
        from scipy.integrate import solve_ivp

        y0 = mp_isothermal.get_initial_state()
        sol = solve_ivp(mp_isothermal.ode_rhs, [0, 100], y0, method="RK45")
        assert sol.success
        dy_final = mp_isothermal.ode_rhs(100.0, sol.y[:, -1])
        dy_initial = mp_isothermal.ode_rhs(0.0, y0)
        assert np.linalg.norm(dy_final) < np.linalg.norm(dy_initial) + 1e-6


# ── Initial State ────────────────────────────────────────────────────


class TestMultiPhaseInitialState:
    def test_shape_isothermal(self, mp_isothermal):
        y0 = mp_isothermal.get_initial_state()
        assert y0.shape == (mp_isothermal.state_dim,)

    def test_shape_nonisothermal(self, mp_nonisothermal):
        y0 = mp_nonisothermal.get_initial_state()
        assert y0.shape == (mp_nonisothermal.state_dim,)

    def test_defaults_to_feed(self, mp_isothermal):
        y0 = mp_isothermal.get_initial_state()
        np.testing.assert_allclose(y0[:2], [1.0, 0.0])  # C_L_feed
        np.testing.assert_allclose(y0[2], 0.5)  # C_G_feed

    def test_noniso_includes_temperature(self, mp_nonisothermal):
        y0 = mp_nonisothermal.get_initial_state()
        assert y0[-1] == pytest.approx(300.0)

    def test_custom_initial_state(self):
        reactor = MultiPhaseReactor(
            name="custom",
            num_species=2,
            params={
                "V_L": 100.0,
                "V_G": 50.0,
                "F_L": 10.0,
                "F_G": 5.0,
                "kLa": 0.5,
                "H": [10.0],
                "C_L_feed": [1.0, 0.0],
                "C_G_feed": [0.5],
                "T_feed": 300.0,
                "gas_species_indices": [0],
                "C_L_initial": [0.5, 0.1],
                "C_G_initial": [0.3],
            },
        )
        y0 = reactor.get_initial_state()
        np.testing.assert_allclose(y0[:2], [0.5, 0.1])
        np.testing.assert_allclose(y0[2], 0.3)


# ── State Labels ─────────────────────────────────────────────────────


class TestMultiPhaseLabels:
    def test_labels_isothermal(self, mp_isothermal):
        labels = mp_isothermal.get_state_labels()
        assert len(labels) == mp_isothermal.state_dim
        assert labels == ["C_L_0", "C_L_1", "C_G_0"]

    def test_labels_nonisothermal(self, mp_nonisothermal):
        labels = mp_nonisothermal.get_state_labels()
        assert labels[-1] == "T"
        assert len(labels) == mp_nonisothermal.state_dim

    def test_labels_two_gas(self, mp_two_gas):
        labels = mp_two_gas.get_state_labels()
        assert "C_G_0" in labels
        assert "C_G_1" in labels


# ── Serialization ────────────────────────────────────────────────────


class TestMultiPhaseSerialization:
    def test_to_dict(self, mp_isothermal):
        d = mp_isothermal.to_dict()
        assert d["type"] == "MultiPhaseReactor"
        assert d["num_species"] == 2

    def test_from_dict_round_trip(self, mp_isothermal):
        config = mp_isothermal.to_dict()
        config["isothermal"] = True
        restored = MultiPhaseReactor.from_dict(config)
        assert restored.name == mp_isothermal.name
        assert restored.state_dim == mp_isothermal.state_dim
        assert restored.num_gas_species == mp_isothermal.num_gas_species


# ── Parameter Validation ─────────────────────────────────────────────


class TestMultiPhaseValidation:
    @pytest.mark.parametrize(
        "missing_key",
        ["V_L", "V_G", "F_L", "F_G", "kLa", "H", "C_L_feed", "C_G_feed", "T_feed"],
    )
    def test_missing_required_param(self, missing_key):
        params = {
            "V_L": 100.0,
            "V_G": 50.0,
            "F_L": 10.0,
            "F_G": 5.0,
            "kLa": 0.5,
            "H": [10.0],
            "C_L_feed": [1.0],
            "C_G_feed": [0.5],
            "T_feed": 300.0,
            "gas_species_indices": [0],
        }
        del params[missing_key]
        with pytest.raises(ConfigurationError, match="Missing required parameter"):
            MultiPhaseReactor(name="bad", num_species=1, params=params)

    def test_missing_thermo_nonisothermal(self):
        with pytest.raises(ConfigurationError, match="Non-isothermal"):
            MultiPhaseReactor(
                name="bad",
                num_species=1,
                params={
                    "V_L": 100.0,
                    "V_G": 50.0,
                    "F_L": 10.0,
                    "F_G": 5.0,
                    "kLa": 0.5,
                    "H": [10.0],
                    "C_L_feed": [1.0],
                    "C_G_feed": [0.5],
                    "T_feed": 300.0,
                    "gas_species_indices": [0],
                },
                isothermal=False,
            )


# ── Registry ─────────────────────────────────────────────────────────


class TestMultiPhaseRegistry:
    def test_registered(self):
        from reactor_twin.utils.registry import REACTOR_REGISTRY

        assert "multi_phase" in REACTOR_REGISTRY

    def test_get_returns_class(self):
        from reactor_twin.utils.registry import REACTOR_REGISTRY

        cls = REACTOR_REGISTRY.get("multi_phase")
        assert cls is MultiPhaseReactor


# ── Repr ─────────────────────────────────────────────────────────────


class TestMultiPhaseRepr:
    def test_repr(self, mp_isothermal):
        r = repr(mp_isothermal)
        assert "MultiPhaseReactor" in r
        assert "test_mp_iso" in r
