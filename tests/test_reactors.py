"""Tests for reactor types: CSTR, Batch, SemiBatch, PFR, and benchmark systems."""

from __future__ import annotations

import numpy as np
import pytest

from reactor_twin.reactors import (
    BatchReactor,
    CSTRReactor,
    PlugFlowReactor,
    SemiBatchReactor,
)
from reactor_twin.reactors.kinetics.arrhenius import ArrheniusKinetics
from reactor_twin.reactors.systems import (
    create_bioreactor_cstr,
    create_consecutive_cstr,
    create_exothermic_cstr,
    create_parallel_cstr,
    create_van_de_vusse_cstr,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cstr_isothermal():
    """Create a simple isothermal CSTR (no kinetics)."""
    return CSTRReactor(
        name="test_cstr_iso",
        num_species=2,
        params={
            "V": 100.0,
            "F": 10.0,
            "C_feed": [1.0, 0.0],
            "T_feed": 350.0,
        },
        isothermal=True,
    )


@pytest.fixture
def cstr_nonisothermal():
    """Create a non-isothermal CSTR (no kinetics)."""
    return CSTRReactor(
        name="test_cstr_noniso",
        num_species=2,
        params={
            "V": 100.0,
            "F": 10.0,
            "C_feed": [1.0, 0.0],
            "T_feed": 350.0,
            "rho": 1000.0,
            "Cp": 4.184,
            "UA": 500.0,
            "T_coolant": 300.0,
        },
        isothermal=False,
    )


@pytest.fixture
def cstr_with_kinetics():
    """Create an isothermal CSTR with Arrhenius kinetics A -> B."""
    kinetics = ArrheniusKinetics(
        name="simple_A_to_B",
        num_reactions=1,
        params={
            "k0": np.array([1e10]),
            "Ea": np.array([50000.0]),
            "stoich": np.array([[-1, 1]]),
            "orders": np.array([[1, 0]]),
        },
    )
    return CSTRReactor(
        name="test_cstr_kinetics",
        num_species=2,
        params={
            "V": 100.0,
            "F": 10.0,
            "C_feed": [1.0, 0.0],
            "T_feed": 350.0,
        },
        kinetics=kinetics,
        isothermal=True,
    )


@pytest.fixture
def batch_isothermal():
    """Create a simple isothermal, constant-volume batch reactor."""
    return BatchReactor(
        name="test_batch_iso",
        num_species=2,
        params={
            "V": 50.0,
            "T": 350.0,
            "C_initial": [1.0, 0.0],
        },
        isothermal=True,
        constant_volume=True,
    )


@pytest.fixture
def batch_nonisothermal():
    """Create a non-isothermal, constant-volume batch reactor."""
    return BatchReactor(
        name="test_batch_noniso",
        num_species=2,
        params={
            "V": 50.0,
            "T": 350.0,
            "C_initial": [1.0, 0.0],
            "rho": 1000.0,
            "Cp": 4.184,
        },
        isothermal=False,
        constant_volume=True,
    )


@pytest.fixture
def batch_variable_volume():
    """Create an isothermal, variable-volume batch reactor (gas-phase)."""
    return BatchReactor(
        name="test_batch_varV",
        num_species=2,
        params={
            "V": 50.0,
            "T": 350.0,
            "C_initial": [1.0, 0.0],
        },
        isothermal=True,
        constant_volume=False,
    )


@pytest.fixture
def batch_noniso_variable_volume():
    """Create a non-isothermal, variable-volume batch reactor."""
    return BatchReactor(
        name="test_batch_all",
        num_species=2,
        params={
            "V": 50.0,
            "T": 350.0,
            "C_initial": [1.0, 0.0],
            "rho": 1000.0,
            "Cp": 4.184,
        },
        isothermal=False,
        constant_volume=False,
    )


@pytest.fixture
def semi_batch_isothermal():
    """Create a simple isothermal semi-batch reactor."""
    return SemiBatchReactor(
        name="test_semi_iso",
        num_species=2,
        params={
            "V": 50.0,
            "T": 350.0,
            "F_in": 1.0,
            "C_in": [2.0, 0.0],
        },
        isothermal=True,
    )


@pytest.fixture
def semi_batch_nonisothermal():
    """Create a non-isothermal semi-batch reactor."""
    return SemiBatchReactor(
        name="test_semi_noniso",
        num_species=2,
        params={
            "V": 50.0,
            "T": 350.0,
            "F_in": 1.0,
            "C_in": [2.0, 0.0],
            "rho": 1000.0,
            "Cp": 4.184,
            "T_in": 340.0,
        },
        isothermal=False,
    )


@pytest.fixture
def pfr_small():
    """Create a PFR with 5 spatial cells."""
    return PlugFlowReactor(
        name="test_pfr",
        num_species=2,
        params={
            "L": 1.0,
            "u": 0.1,
            "D": 0.01,
            "C_in": [1.0, 0.0],
            "T": 350.0,
        },
        num_cells=5,
    )


@pytest.fixture
def pfr_default_cells():
    """Create a PFR with default num_cells (50)."""
    return PlugFlowReactor(
        name="test_pfr_default",
        num_species=2,
        params={
            "L": 2.0,
            "u": 0.5,
            "D": 0.001,
            "C_in": [1.0, 0.0],
            "T": 350.0,
        },
    )


# ===========================================================================
# CSTR Tests
# ===========================================================================


class TestCSTRReactor:
    """Tests for the Continuous Stirred-Tank Reactor."""

    # -- Initialization --

    def test_instantiation_isothermal(self, cstr_isothermal):
        assert cstr_isothermal.name == "test_cstr_iso"
        assert cstr_isothermal.num_species == 2
        assert cstr_isothermal.isothermal is True
        assert cstr_isothermal.kinetics is None

    def test_instantiation_nonisothermal(self, cstr_nonisothermal):
        assert cstr_nonisothermal.name == "test_cstr_noniso"
        assert cstr_nonisothermal.isothermal is False

    def test_instantiation_with_kinetics(self, cstr_with_kinetics):
        assert cstr_with_kinetics.kinetics is not None
        assert cstr_with_kinetics.kinetics.name == "simple_A_to_B"

    # -- State dimension --

    def test_state_dim_isothermal(self, cstr_isothermal):
        """Isothermal CSTR: state_dim == num_species."""
        assert cstr_isothermal.state_dim == 2

    def test_state_dim_nonisothermal(self, cstr_nonisothermal):
        """Non-isothermal CSTR: state_dim == num_species + 1 (temperature)."""
        assert cstr_nonisothermal.state_dim == 3

    # -- ODE right-hand side --

    def test_ode_rhs_shape_isothermal(self, cstr_isothermal):
        y0 = cstr_isothermal.get_initial_state()
        dy = cstr_isothermal.ode_rhs(0.0, y0)
        assert isinstance(dy, np.ndarray)
        assert dy.shape == (cstr_isothermal.state_dim,)

    def test_ode_rhs_shape_nonisothermal(self, cstr_nonisothermal):
        y0 = cstr_nonisothermal.get_initial_state()
        dy = cstr_nonisothermal.ode_rhs(0.0, y0)
        assert isinstance(dy, np.ndarray)
        assert dy.shape == (cstr_nonisothermal.state_dim,)

    def test_ode_rhs_finite(self, cstr_isothermal):
        y0 = cstr_isothermal.get_initial_state()
        dy = cstr_isothermal.ode_rhs(0.0, y0)
        assert np.all(np.isfinite(dy))

    def test_ode_rhs_with_kinetics(self, cstr_with_kinetics):
        """With kinetics attached, ode_rhs should include reaction source terms."""
        y0 = cstr_with_kinetics.get_initial_state()
        dy_kin = cstr_with_kinetics.ode_rhs(0.0, y0)
        assert dy_kin.shape == (cstr_with_kinetics.state_dim,)
        assert np.all(np.isfinite(dy_kin))

    def test_ode_rhs_no_kinetics_only_dilution(self, cstr_isothermal):
        """Without kinetics, ode_rhs should only reflect dilution: (F/V)*(C_feed - C)."""
        V = cstr_isothermal.params["V"]
        F = cstr_isothermal.params["F"]
        C_feed = np.array(cstr_isothermal.params["C_feed"])
        y0 = cstr_isothermal.get_initial_state()
        C0 = y0[: cstr_isothermal.num_species]
        dy = cstr_isothermal.ode_rhs(0.0, y0)
        expected = (F / V) * (C_feed - C0)
        np.testing.assert_allclose(dy, expected, atol=1e-12)

    def test_ode_rhs_with_control_input(self, cstr_isothermal):
        """When u is provided, the first element overrides flow rate F."""
        y0 = cstr_isothermal.get_initial_state()
        u = np.array([20.0])  # Override flow rate
        dy = cstr_isothermal.ode_rhs(0.0, y0, u=u)
        assert dy.shape == (cstr_isothermal.state_dim,)
        assert np.all(np.isfinite(dy))

    # -- Initial state --

    def test_get_initial_state_shape_isothermal(self, cstr_isothermal):
        y0 = cstr_isothermal.get_initial_state()
        assert y0.shape == (cstr_isothermal.state_dim,)

    def test_get_initial_state_shape_nonisothermal(self, cstr_nonisothermal):
        y0 = cstr_nonisothermal.get_initial_state()
        assert y0.shape == (cstr_nonisothermal.state_dim,)

    def test_get_initial_state_defaults_to_feed(self, cstr_isothermal):
        """Without C_initial in params, initial state defaults to C_feed."""
        y0 = cstr_isothermal.get_initial_state()
        C_feed = np.array(cstr_isothermal.params["C_feed"])
        np.testing.assert_allclose(y0, C_feed)

    def test_get_initial_state_uses_c_initial(self):
        """When C_initial is provided, it is used instead of C_feed."""
        reactor = CSTRReactor(
            name="ci_cstr",
            num_species=2,
            params={
                "V": 100.0,
                "F": 10.0,
                "C_feed": [1.0, 0.0],
                "T_feed": 350.0,
                "C_initial": [0.5, 0.1],
            },
            isothermal=True,
        )
        y0 = reactor.get_initial_state()
        np.testing.assert_allclose(y0, [0.5, 0.1])

    def test_get_initial_state_noniso_includes_temperature(self, cstr_nonisothermal):
        y0 = cstr_nonisothermal.get_initial_state()
        # Last element is temperature (defaults to T_feed)
        assert y0[-1] == pytest.approx(cstr_nonisothermal.params["T_feed"])

    # -- State labels --

    def test_get_state_labels_isothermal(self, cstr_isothermal):
        labels = cstr_isothermal.get_state_labels()
        assert len(labels) == cstr_isothermal.state_dim
        assert labels == ["C_0", "C_1"]
        assert "T" not in labels

    def test_get_state_labels_nonisothermal(self, cstr_nonisothermal):
        labels = cstr_nonisothermal.get_state_labels()
        assert len(labels) == cstr_nonisothermal.state_dim
        assert labels[-1] == "T"

    # -- Observable indices --

    def test_get_observable_indices(self, cstr_isothermal):
        indices = cstr_isothermal.get_observable_indices()
        assert indices == list(range(cstr_isothermal.state_dim))

    # -- Validate state --

    def test_validate_state_valid(self, cstr_isothermal):
        y = np.array([0.5, 0.3])
        assert cstr_isothermal.validate_state(y) is True

    def test_validate_state_negative_concentration(self, cstr_isothermal):
        y = np.array([-0.1, 0.3])
        assert cstr_isothermal.validate_state(y) is False

    # -- Serialization --

    def test_to_dict_contains_required_keys(self, cstr_isothermal):
        d = cstr_isothermal.to_dict()
        assert d["name"] == "test_cstr_iso"
        assert d["type"] == "CSTRReactor"
        assert d["num_species"] == 2
        assert d["state_dim"] == 2
        assert "params" in d

    def test_from_dict_round_trip_isothermal(self, cstr_isothermal):
        config = cstr_isothermal.to_dict()
        config["isothermal"] = True
        restored = CSTRReactor.from_dict(config)
        assert restored.name == cstr_isothermal.name
        assert restored.num_species == cstr_isothermal.num_species
        assert restored.state_dim == cstr_isothermal.state_dim
        assert restored.isothermal is True

    def test_from_dict_round_trip_nonisothermal(self, cstr_nonisothermal):
        config = cstr_nonisothermal.to_dict()
        config["isothermal"] = False
        restored = CSTRReactor.from_dict(config)
        assert restored.name == cstr_nonisothermal.name
        assert restored.state_dim == cstr_nonisothermal.state_dim
        assert restored.isothermal is False

    # -- Parameter validation --

    def test_missing_required_param_V(self):
        with pytest.raises(ValueError, match="Missing required parameter"):
            CSTRReactor(
                name="bad",
                num_species=2,
                params={"F": 10.0, "C_feed": [1.0, 0.0], "T_feed": 350.0},
                isothermal=True,
            )

    def test_missing_required_param_F(self):
        with pytest.raises(ValueError, match="Missing required parameter"):
            CSTRReactor(
                name="bad",
                num_species=2,
                params={"V": 100.0, "C_feed": [1.0, 0.0], "T_feed": 350.0},
                isothermal=True,
            )

    def test_missing_required_param_C_feed(self):
        with pytest.raises(ValueError, match="Missing required parameter"):
            CSTRReactor(
                name="bad",
                num_species=2,
                params={"V": 100.0, "F": 10.0, "T_feed": 350.0},
                isothermal=True,
            )

    def test_missing_required_param_T_feed(self):
        with pytest.raises(ValueError, match="Missing required parameter"):
            CSTRReactor(
                name="bad",
                num_species=2,
                params={"V": 100.0, "F": 10.0, "C_feed": [1.0, 0.0]},
                isothermal=True,
            )

    def test_missing_thermo_param_nonisothermal(self):
        """Non-isothermal CSTR requires rho, Cp, UA, T_coolant."""
        with pytest.raises(ValueError, match="Non-isothermal CSTR requires"):
            CSTRReactor(
                name="bad",
                num_species=2,
                params={
                    "V": 100.0,
                    "F": 10.0,
                    "C_feed": [1.0, 0.0],
                    "T_feed": 350.0,
                    # Missing rho, Cp, UA, T_coolant
                },
                isothermal=False,
            )

    # -- Repr --

    def test_repr(self, cstr_isothermal):
        r = repr(cstr_isothermal)
        assert "CSTRReactor" in r
        assert "test_cstr_iso" in r


# ===========================================================================
# Batch Reactor Tests
# ===========================================================================


class TestBatchReactor:
    """Tests for the batch reactor."""

    # -- Initialization --

    def test_instantiation(self, batch_isothermal):
        assert batch_isothermal.name == "test_batch_iso"
        assert batch_isothermal.num_species == 2
        assert batch_isothermal.isothermal is True
        assert batch_isothermal.constant_volume is True

    # -- State dimension --

    def test_state_dim_isothermal_constant_volume(self, batch_isothermal):
        """Isothermal + constant volume: state_dim == num_species."""
        assert batch_isothermal.state_dim == 2

    def test_state_dim_nonisothermal_constant_volume(self, batch_nonisothermal):
        """Non-isothermal + constant volume: state_dim == num_species + 1 (T)."""
        assert batch_nonisothermal.state_dim == 3

    def test_state_dim_isothermal_variable_volume(self, batch_variable_volume):
        """Isothermal + variable volume: state_dim == num_species + 1 (V)."""
        assert batch_variable_volume.state_dim == 3

    def test_state_dim_noniso_variable_volume(self, batch_noniso_variable_volume):
        """Non-isothermal + variable volume: state_dim == num_species + 2 (V + T)."""
        assert batch_noniso_variable_volume.state_dim == 4

    # -- ODE right-hand side --

    def test_ode_rhs_shape_isothermal(self, batch_isothermal):
        y0 = batch_isothermal.get_initial_state()
        dy = batch_isothermal.ode_rhs(0.0, y0)
        assert isinstance(dy, np.ndarray)
        assert dy.shape == (batch_isothermal.state_dim,)

    def test_ode_rhs_shape_nonisothermal(self, batch_nonisothermal):
        y0 = batch_nonisothermal.get_initial_state()
        dy = batch_nonisothermal.ode_rhs(0.0, y0)
        assert dy.shape == (batch_nonisothermal.state_dim,)

    def test_ode_rhs_shape_variable_volume(self, batch_variable_volume):
        y0 = batch_variable_volume.get_initial_state()
        dy = batch_variable_volume.ode_rhs(0.0, y0)
        assert dy.shape == (batch_variable_volume.state_dim,)

    def test_ode_rhs_finite(self, batch_isothermal):
        y0 = batch_isothermal.get_initial_state()
        dy = batch_isothermal.ode_rhs(0.0, y0)
        assert np.all(np.isfinite(dy))

    def test_ode_rhs_no_kinetics_zero_rates(self, batch_isothermal):
        """Without kinetics, batch reactor rates are all zero (closed system, no reaction)."""
        y0 = batch_isothermal.get_initial_state()
        dy = batch_isothermal.ode_rhs(0.0, y0)
        np.testing.assert_allclose(dy, 0.0, atol=1e-15)

    # -- Initial state --

    def test_get_initial_state_shape(self, batch_isothermal):
        y0 = batch_isothermal.get_initial_state()
        assert y0.shape == (batch_isothermal.state_dim,)

    def test_get_initial_state_uses_c_initial(self, batch_isothermal):
        y0 = batch_isothermal.get_initial_state()
        np.testing.assert_allclose(y0, [1.0, 0.0])

    def test_get_initial_state_noniso_includes_temperature(self, batch_nonisothermal):
        y0 = batch_nonisothermal.get_initial_state()
        assert y0[-1] == pytest.approx(batch_nonisothermal.params["T"])

    def test_get_initial_state_variable_volume(self, batch_variable_volume):
        y0 = batch_variable_volume.get_initial_state()
        # [C_0, C_1, V]
        assert y0.shape == (3,)
        assert y0[2] == pytest.approx(50.0)  # V_initial defaults to V

    # -- State labels --

    def test_get_state_labels_isothermal(self, batch_isothermal):
        labels = batch_isothermal.get_state_labels()
        assert len(labels) == batch_isothermal.state_dim
        assert "V" not in labels
        assert "T" not in labels

    def test_get_state_labels_nonisothermal(self, batch_nonisothermal):
        labels = batch_nonisothermal.get_state_labels()
        assert labels[-1] == "T"
        assert "V" not in labels  # constant volume

    def test_get_state_labels_variable_volume(self, batch_variable_volume):
        labels = batch_variable_volume.get_state_labels()
        assert "V" in labels
        assert "T" not in labels  # isothermal

    def test_get_state_labels_noniso_variable(self, batch_noniso_variable_volume):
        labels = batch_noniso_variable_volume.get_state_labels()
        assert "V" in labels
        assert "T" in labels

    # -- Validate state --

    def test_validate_state_valid(self, batch_isothermal):
        assert batch_isothermal.validate_state(np.array([0.5, 0.3])) is True

    def test_validate_state_negative(self, batch_isothermal):
        assert batch_isothermal.validate_state(np.array([-0.1, 0.3])) is False

    # -- Serialization --

    def test_to_dict(self, batch_isothermal):
        d = batch_isothermal.to_dict()
        assert d["type"] == "BatchReactor"
        assert d["num_species"] == 2

    def test_from_dict_round_trip(self, batch_isothermal):
        config = batch_isothermal.to_dict()
        config["isothermal"] = True
        config["constant_volume"] = True
        restored = BatchReactor.from_dict(config)
        assert restored.name == batch_isothermal.name
        assert restored.state_dim == batch_isothermal.state_dim

    # -- Parameter validation --

    def test_missing_required_param_V(self):
        with pytest.raises(ValueError, match="Missing required parameter"):
            BatchReactor(
                name="bad",
                num_species=2,
                params={"T": 350.0},
            )

    def test_missing_required_param_T(self):
        with pytest.raises(ValueError, match="Missing required parameter"):
            BatchReactor(
                name="bad",
                num_species=2,
                params={"V": 50.0},
            )

    def test_missing_thermo_param_nonisothermal(self):
        """Non-isothermal batch requires rho, Cp."""
        with pytest.raises(ValueError, match="Non-isothermal batch reactor requires"):
            BatchReactor(
                name="bad",
                num_species=2,
                params={"V": 50.0, "T": 350.0},
                isothermal=False,
            )


# ===========================================================================
# SemiBatch Reactor Tests
# ===========================================================================


class TestSemiBatchReactor:
    """Tests for the semi-batch reactor."""

    # -- Initialization --

    def test_instantiation(self, semi_batch_isothermal):
        assert semi_batch_isothermal.name == "test_semi_iso"
        assert semi_batch_isothermal.num_species == 2
        assert semi_batch_isothermal.isothermal is True

    # -- State dimension --

    def test_state_dim_isothermal(self, semi_batch_isothermal):
        """Semi-batch always includes volume: state_dim == num_species + 1."""
        assert semi_batch_isothermal.state_dim == 3

    def test_state_dim_nonisothermal(self, semi_batch_nonisothermal):
        """Non-isothermal semi-batch: state_dim == num_species + 1 (V) + 1 (T)."""
        assert semi_batch_nonisothermal.state_dim == 4

    # -- ODE right-hand side --

    def test_ode_rhs_shape_isothermal(self, semi_batch_isothermal):
        y0 = semi_batch_isothermal.get_initial_state()
        dy = semi_batch_isothermal.ode_rhs(0.0, y0)
        assert isinstance(dy, np.ndarray)
        assert dy.shape == (semi_batch_isothermal.state_dim,)

    def test_ode_rhs_shape_nonisothermal(self, semi_batch_nonisothermal):
        y0 = semi_batch_nonisothermal.get_initial_state()
        dy = semi_batch_nonisothermal.ode_rhs(0.0, y0)
        assert dy.shape == (semi_batch_nonisothermal.state_dim,)

    def test_ode_rhs_finite(self, semi_batch_isothermal):
        y0 = semi_batch_isothermal.get_initial_state()
        dy = semi_batch_isothermal.ode_rhs(0.0, y0)
        assert np.all(np.isfinite(dy))

    def test_ode_rhs_volume_increases(self, semi_batch_isothermal):
        """In a semi-batch reactor, dV/dt == F_in > 0."""
        y0 = semi_batch_isothermal.get_initial_state()
        dy = semi_batch_isothermal.ode_rhs(0.0, y0)
        # Volume derivative is at index num_species
        dV_dt = dy[semi_batch_isothermal.num_species]
        F_in = semi_batch_isothermal.params["F_in"]
        assert dV_dt == pytest.approx(F_in)

    def test_ode_rhs_with_control_input(self, semi_batch_isothermal):
        y0 = semi_batch_isothermal.get_initial_state()
        u = np.array([5.0])  # Override F_in
        dy = semi_batch_isothermal.ode_rhs(0.0, y0, u=u)
        # Volume derivative should use the control input
        dV_dt = dy[semi_batch_isothermal.num_species]
        assert dV_dt == pytest.approx(5.0)

    # -- Initial state --

    def test_get_initial_state_shape(self, semi_batch_isothermal):
        y0 = semi_batch_isothermal.get_initial_state()
        assert y0.shape == (semi_batch_isothermal.state_dim,)

    def test_get_initial_state_includes_volume(self, semi_batch_isothermal):
        y0 = semi_batch_isothermal.get_initial_state()
        # Volume at index num_species defaults to V param
        V0 = y0[semi_batch_isothermal.num_species]
        assert V0 == pytest.approx(semi_batch_isothermal.params["V"])

    # -- State labels --

    def test_get_state_labels_isothermal(self, semi_batch_isothermal):
        labels = semi_batch_isothermal.get_state_labels()
        assert len(labels) == semi_batch_isothermal.state_dim
        assert "V" in labels
        assert "T" not in labels

    def test_get_state_labels_nonisothermal(self, semi_batch_nonisothermal):
        labels = semi_batch_nonisothermal.get_state_labels()
        assert "V" in labels
        assert "T" in labels

    # -- Validate state --

    def test_validate_state_valid(self, semi_batch_isothermal):
        y = np.array([0.5, 0.3, 50.0])
        assert semi_batch_isothermal.validate_state(y) is True

    def test_validate_state_negative_concentration(self, semi_batch_isothermal):
        y = np.array([-0.1, 0.3, 50.0])
        assert semi_batch_isothermal.validate_state(y) is False

    # -- Serialization --

    def test_to_dict(self, semi_batch_isothermal):
        d = semi_batch_isothermal.to_dict()
        assert d["type"] == "SemiBatchReactor"
        assert d["num_species"] == 2

    def test_from_dict_round_trip(self, semi_batch_isothermal):
        config = semi_batch_isothermal.to_dict()
        config["isothermal"] = True
        restored = SemiBatchReactor.from_dict(config)
        assert restored.name == semi_batch_isothermal.name
        assert restored.state_dim == semi_batch_isothermal.state_dim

    # -- Parameter validation --

    def test_missing_required_param_F_in(self):
        with pytest.raises(ValueError, match="Missing required parameter"):
            SemiBatchReactor(
                name="bad",
                num_species=2,
                params={"V": 50.0, "T": 350.0, "C_in": [2.0, 0.0]},
            )

    def test_missing_required_param_C_in(self):
        with pytest.raises(ValueError, match="Missing required parameter"):
            SemiBatchReactor(
                name="bad",
                num_species=2,
                params={"V": 50.0, "T": 350.0, "F_in": 1.0},
            )

    def test_missing_thermo_nonisothermal(self):
        """Non-isothermal semi-batch requires rho, Cp, T_in."""
        with pytest.raises(ValueError, match="Non-isothermal semi-batch"):
            SemiBatchReactor(
                name="bad",
                num_species=2,
                params={"V": 50.0, "T": 350.0, "F_in": 1.0, "C_in": [2.0, 0.0]},
                isothermal=False,
            )


# ===========================================================================
# Plug Flow Reactor Tests
# ===========================================================================


class TestPlugFlowReactor:
    """Tests for the plug flow reactor (Method of Lines)."""

    # -- Initialization --

    def test_instantiation(self, pfr_small):
        assert pfr_small.name == "test_pfr"
        assert pfr_small.num_species == 2
        assert pfr_small.num_cells == 5

    def test_default_num_cells(self, pfr_default_cells):
        """Default num_cells should be 50."""
        assert pfr_default_cells.num_cells == 50

    def test_attributes(self, pfr_small):
        assert pfr_small.length == pytest.approx(1.0)
        assert pfr_small.velocity == pytest.approx(0.1)
        assert pfr_small.dispersion == pytest.approx(0.01)
        assert pfr_small.dz == pytest.approx(1.0 / 5)

    # -- State dimension --

    def test_state_dim(self, pfr_small):
        """PFR: state_dim == num_species * num_cells."""
        assert pfr_small.state_dim == 2 * 5

    def test_state_dim_default(self, pfr_default_cells):
        assert pfr_default_cells.state_dim == 2 * 50

    # -- ODE right-hand side --

    def test_ode_rhs_shape(self, pfr_small):
        y0 = pfr_small.get_initial_state()
        dy = pfr_small.ode_rhs(0.0, y0)
        assert isinstance(dy, np.ndarray)
        assert dy.shape == (pfr_small.state_dim,)

    def test_ode_rhs_finite(self, pfr_small):
        y0 = pfr_small.get_initial_state()
        dy = pfr_small.ode_rhs(0.0, y0)
        assert np.all(np.isfinite(dy))

    def test_ode_rhs_uniform_ic_no_kinetics(self, pfr_small):
        """With uniform initial condition equal to C_in and no kinetics,
        advection and dispersion terms should be near zero (steady state)."""
        y0 = pfr_small.get_initial_state()
        dy = pfr_small.ode_rhs(0.0, y0)
        # All cells start at C_in, no reaction -> derivatives should be ~0
        np.testing.assert_allclose(dy, 0.0, atol=1e-10)

    # -- Initial state --

    def test_get_initial_state_shape(self, pfr_small):
        y0 = pfr_small.get_initial_state()
        assert y0.shape == (pfr_small.state_dim,)

    def test_get_initial_state_tiled(self, pfr_small):
        """Initial state repeats C_in across all cells.

        Layout: [C_0,cell_0,...,C_0,cell_N-1, C_1,cell_0,...,C_1,cell_N-1].
        """
        y0 = pfr_small.get_initial_state()
        C_in = np.array(pfr_small.params["C_in"])
        expected = np.repeat(C_in, pfr_small.num_cells)
        np.testing.assert_allclose(y0, expected)

    # -- State labels --

    def test_get_state_labels_count(self, pfr_small):
        labels = pfr_small.get_state_labels()
        assert len(labels) == pfr_small.state_dim

    def test_get_state_labels_format(self, pfr_small):
        labels = pfr_small.get_state_labels()
        assert labels[0] == "C_0_cell_0"
        assert labels[1] == "C_0_cell_1"

    # -- Outlet concentrations --

    def test_get_outlet_concentrations_shape(self, pfr_small):
        y0 = pfr_small.get_initial_state()
        outlet = pfr_small.get_outlet_concentrations(y0)
        assert outlet.shape == (pfr_small.num_species,)

    def test_get_outlet_concentrations_values(self, pfr_small):
        """With uniform IC, outlet == C_in."""
        y0 = pfr_small.get_initial_state()
        outlet = pfr_small.get_outlet_concentrations(y0)
        C_in = np.array(pfr_small.params["C_in"])
        np.testing.assert_allclose(outlet, C_in)

    # -- Axial profile --

    def test_get_axial_profile_shapes(self, pfr_small):
        y0 = pfr_small.get_initial_state()
        z_pos, C_profiles = pfr_small.get_axial_profile(y0)
        assert z_pos.shape == (pfr_small.num_cells,)
        assert C_profiles.shape == (pfr_small.num_species, pfr_small.num_cells)

    def test_get_axial_profile_z_range(self, pfr_small):
        y0 = pfr_small.get_initial_state()
        z_pos, _ = pfr_small.get_axial_profile(y0)
        assert z_pos[0] == pytest.approx(0.0)
        assert z_pos[-1] == pytest.approx(pfr_small.length)

    # -- Serialization --

    def test_to_dict(self, pfr_small):
        d = pfr_small.to_dict()
        assert d["type"] == "PlugFlowReactor"

    def test_from_dict_round_trip(self, pfr_small):
        config = pfr_small.to_dict()
        config["num_cells"] = pfr_small.num_cells
        restored = PlugFlowReactor.from_dict(config)
        assert restored.name == pfr_small.name
        assert restored.state_dim == pfr_small.state_dim
        assert restored.num_cells == pfr_small.num_cells

    # -- Parameter validation --

    def test_missing_required_param_L(self):
        with pytest.raises(ValueError, match="Missing required parameter"):
            PlugFlowReactor(
                name="bad",
                num_species=2,
                params={"u": 0.1, "D": 0.01, "C_in": [1.0, 0.0], "T": 350.0},
            )

    def test_missing_required_param_u(self):
        with pytest.raises(ValueError, match="Missing required parameter"):
            PlugFlowReactor(
                name="bad",
                num_species=2,
                params={"L": 1.0, "D": 0.01, "C_in": [1.0, 0.0], "T": 350.0},
            )


# ===========================================================================
# Parametrized tests across all reactor types
# ===========================================================================


@pytest.mark.parametrize(
    "reactor_cls, kwargs",
    [
        (
            CSTRReactor,
            {
                "name": "p_cstr",
                "num_species": 2,
                "params": {"V": 100, "F": 10, "C_feed": [1.0, 0.0], "T_feed": 350.0},
                "isothermal": True,
            },
        ),
        (
            BatchReactor,
            {
                "name": "p_batch",
                "num_species": 2,
                "params": {"V": 50, "T": 350.0, "C_initial": [1.0, 0.0]},
                "isothermal": True,
            },
        ),
        (
            SemiBatchReactor,
            {
                "name": "p_semi",
                "num_species": 2,
                "params": {"V": 50, "T": 350.0, "F_in": 1.0, "C_in": [2.0, 0.0]},
                "isothermal": True,
            },
        ),
        (
            PlugFlowReactor,
            {
                "name": "p_pfr",
                "num_species": 2,
                "params": {
                    "L": 1.0,
                    "u": 0.1,
                    "D": 0.01,
                    "C_in": [1.0, 0.0],
                    "T": 350.0,
                },
                "num_cells": 5,
            },
        ),
    ],
    ids=["CSTR", "Batch", "SemiBatch", "PFR"],
)
class TestAllReactors:
    """Parametrized tests that run against every reactor type."""

    def test_ode_rhs_returns_numpy_array(self, reactor_cls, kwargs):
        reactor = reactor_cls(**kwargs)
        y0 = reactor.get_initial_state()
        dy = reactor.ode_rhs(0.0, y0)
        assert isinstance(dy, np.ndarray)
        assert dy.shape == y0.shape

    def test_initial_state_is_finite(self, reactor_cls, kwargs):
        reactor = reactor_cls(**kwargs)
        y0 = reactor.get_initial_state()
        assert np.all(np.isfinite(y0))

    def test_ode_rhs_is_finite(self, reactor_cls, kwargs):
        reactor = reactor_cls(**kwargs)
        y0 = reactor.get_initial_state()
        dy = reactor.ode_rhs(0.0, y0)
        assert np.all(np.isfinite(dy))

    def test_state_labels_count(self, reactor_cls, kwargs):
        reactor = reactor_cls(**kwargs)
        labels = reactor.get_state_labels()
        assert len(labels) == reactor.state_dim

    def test_observable_indices_within_bounds(self, reactor_cls, kwargs):
        reactor = reactor_cls(**kwargs)
        indices = reactor.get_observable_indices()
        assert all(0 <= i < reactor.state_dim for i in indices)

    def test_to_dict_contains_required_keys(self, reactor_cls, kwargs):
        reactor = reactor_cls(**kwargs)
        d = reactor.to_dict()
        for key in ["name", "type", "num_species", "state_dim", "params"]:
            assert key in d

    def test_repr_contains_class_name(self, reactor_cls, kwargs):
        reactor = reactor_cls(**kwargs)
        r = repr(reactor)
        assert reactor_cls.__name__ in r


# ===========================================================================
# Benchmark Systems Tests
# ===========================================================================


class TestBenchmarkSystems:
    """Tests for pre-configured benchmark reaction systems."""

    # -- Exothermic A->B CSTR --

    def test_create_exothermic_cstr_nonisothermal(self):
        reactor = create_exothermic_cstr(name="exo_test", isothermal=False)
        assert isinstance(reactor, CSTRReactor)
        assert reactor.isothermal is False
        assert reactor.num_species == 2
        assert reactor.state_dim == 3  # 2 species + temperature
        assert reactor.kinetics is not None

    def test_create_exothermic_cstr_isothermal(self):
        reactor = create_exothermic_cstr(name="exo_iso", isothermal=True)
        assert isinstance(reactor, CSTRReactor)
        assert reactor.isothermal is True
        assert reactor.state_dim == 2

    def test_exothermic_cstr_ode_rhs(self):
        reactor = create_exothermic_cstr(isothermal=False)
        y0 = reactor.get_initial_state()
        dy = reactor.ode_rhs(0.0, y0)
        assert dy.shape == (reactor.state_dim,)
        assert np.all(np.isfinite(dy))

    def test_exothermic_cstr_initial_state(self):
        reactor = create_exothermic_cstr(isothermal=False)
        y0 = reactor.get_initial_state()
        assert y0.shape == (reactor.state_dim,)
        # C_A_initial = 0.5, C_B_initial = 0.0, T_initial = 350.0
        np.testing.assert_allclose(y0, [0.5, 0.0, 350.0])

    # -- Van de Vusse CSTR --

    def test_create_van_de_vusse_cstr(self):
        reactor = create_van_de_vusse_cstr(name="vdv_test", isothermal=True)
        assert isinstance(reactor, CSTRReactor)
        assert reactor.isothermal is True
        assert reactor.num_species == 4  # A, B, C, D
        assert reactor.state_dim == 4
        assert reactor.kinetics is not None

    def test_van_de_vusse_cstr_ode_rhs(self):
        reactor = create_van_de_vusse_cstr()
        y0 = reactor.get_initial_state()
        dy = reactor.ode_rhs(0.0, y0)
        assert dy.shape == (reactor.state_dim,)
        assert np.all(np.isfinite(dy))

    def test_van_de_vusse_has_three_reactions(self):
        reactor = create_van_de_vusse_cstr()
        assert reactor.kinetics.num_reactions == 3

    # -- Consecutive A->B->C CSTR --

    def test_create_consecutive_cstr_isothermal(self):
        reactor = create_consecutive_cstr(name="cons_test", isothermal=True)
        assert isinstance(reactor, CSTRReactor)
        assert reactor.num_species == 3
        y0 = reactor.get_initial_state()
        dy = reactor.ode_rhs(0.0, y0)
        assert dy.shape == (reactor.state_dim,)
        assert np.all(np.isfinite(dy))

    # -- Parallel A->B, A->C CSTR --

    def test_create_parallel_cstr_isothermal(self):
        reactor = create_parallel_cstr(name="par_test", isothermal=True)
        assert isinstance(reactor, CSTRReactor)
        assert reactor.num_species == 3
        y0 = reactor.get_initial_state()
        dy = reactor.ode_rhs(0.0, y0)
        assert dy.shape == (reactor.state_dim,)
        assert np.all(np.isfinite(dy))

    # -- Bioreactor CSTR --

    def test_create_bioreactor_cstr_isothermal(self):
        reactor = create_bioreactor_cstr(name="bio_test", isothermal=True)
        assert isinstance(reactor, CSTRReactor)
        assert reactor.isothermal is True
        assert reactor.num_species == 3  # S, X, P
        assert reactor.state_dim == 3

    def test_create_bioreactor_cstr_nonisothermal(self):
        reactor = create_bioreactor_cstr(name="bio_noniso", isothermal=False)
        assert isinstance(reactor, CSTRReactor)
        assert reactor.isothermal is False
        assert reactor.state_dim == 4  # 3 species + T

    def test_bioreactor_cstr_ode_rhs(self):
        reactor = create_bioreactor_cstr(isothermal=True)
        y0 = reactor.get_initial_state()
        dy = reactor.ode_rhs(0.0, y0)
        assert dy.shape == (reactor.state_dim,)
        assert np.all(np.isfinite(dy))

    def test_bioreactor_initial_state(self):
        reactor = create_bioreactor_cstr(isothermal=True)
        y0 = reactor.get_initial_state()
        # C_initial = [10.0, 0.5, 0.0]
        np.testing.assert_allclose(y0, [10.0, 0.5, 0.0])

    def test_exothermic_default_name(self):
        reactor = create_exothermic_cstr()
        assert reactor.name == "exothermic_ab_cstr"

    def test_van_de_vusse_default_name(self):
        reactor = create_van_de_vusse_cstr()
        assert reactor.name == "van_de_vusse_cstr"

    def test_bioreactor_default_name(self):
        reactor = create_bioreactor_cstr()
        assert reactor.name == "bioreactor_cstr"
