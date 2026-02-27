"""Tests for kinetics models: Arrhenius, Michaelis-Menten, PowerLaw,
Langmuir-Hinshelwood, Reversible, and Monod."""

from __future__ import annotations

import numpy as np
import pytest

from reactor_twin.reactors.kinetics import (
    ArrheniusKinetics,
    LangmuirHinshelwoodKinetics,
    MichaelisMentenKinetics,
    MonodKinetics,
    PowerLawKinetics,
    ReversibleKinetics,
)
from reactor_twin.utils.constants import R_GAS


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def arrhenius_a_to_b():
    """Simple first-order A -> B with Arrhenius kinetics."""
    return ArrheniusKinetics(
        name="A_to_B",
        num_reactions=1,
        params={
            "k0": np.array([1e10]),
            "Ea": np.array([50000.0]),
            "stoich": np.array([[-1, 1]]),
            "orders": np.array([[1, 0]]),
        },
    )


@pytest.fixture
def arrhenius_two_reactions():
    """Two-reaction system: A->B, B->C with Arrhenius kinetics."""
    return ArrheniusKinetics(
        name="A_B_C",
        num_reactions=2,
        params={
            "k0": np.array([1e8, 5e7]),
            "Ea": np.array([40000.0, 60000.0]),
            "stoich": np.array([
                [-1, 1, 0],  # A -> B
                [0, -1, 1],  # B -> C
            ]),
            "orders": np.array([
                [1, 0, 0],  # first order in A
                [0, 1, 0],  # first order in B
            ]),
        },
    )


@pytest.fixture
def arrhenius_no_orders():
    """Arrhenius kinetics without explicit orders (defaults from stoich)."""
    return ArrheniusKinetics(
        name="default_orders",
        num_reactions=1,
        params={
            "k0": np.array([1e10]),
            "Ea": np.array([50000.0]),
            "stoich": np.array([[-1, 1]]),
            # No "orders" key -- should default to stoich-based orders
        },
    )


@pytest.fixture
def mm_kinetics():
    """Michaelis-Menten kinetics: S -> P (single reaction, 2 species)."""
    return MichaelisMentenKinetics(
        name="mm_s_to_p",
        num_reactions=1,
        params={
            "V_max": np.array([10.0]),
            "K_m": np.array([0.5]),
            "substrate_indices": np.array([0]),
            "stoich": np.array([[-1, 1]]),
        },
    )


@pytest.fixture
def mm_with_inhibition():
    """Michaelis-Menten kinetics with competitive inhibition: S -> P, I is inhibitor."""
    return MichaelisMentenKinetics(
        name="mm_inhibited",
        num_reactions=1,
        params={
            "V_max": np.array([10.0]),
            "K_m": np.array([0.5]),
            "substrate_indices": np.array([0]),
            "stoich": np.array([[-1, 1, 0]]),  # S, P, I
            "K_i": np.array([1.0]),
            "inhibitor_indices": np.array([2]),
        },
    )


@pytest.fixture
def power_law_second_order():
    """Power law kinetics: A -> B, second order in A."""
    return PowerLawKinetics(
        name="pl_a2b",
        num_reactions=1,
        params={
            "k": np.array([0.5]),
            "orders": np.array([[2.0, 0.0]]),
            "stoich": np.array([[-1, 1]]),
        },
    )


@pytest.fixture
def power_law_with_temperature():
    """Power law kinetics with temperature dependence via A and E_a."""
    return PowerLawKinetics(
        name="pl_temp",
        num_reactions=1,
        params={
            "k": np.array([1.0]),  # Baseline (used if A/E_a absent)
            "orders": np.array([[1.0, 0.0]]),
            "stoich": np.array([[-1, 1]]),
            "A": np.array([1e8]),
            "E_a": np.array([40000.0]),
        },
    )


@pytest.fixture
def lh_kinetics():
    """Langmuir-Hinshelwood kinetics: A -> B on catalyst surface."""
    return LangmuirHinshelwoodKinetics(
        name="lh_a_to_b",
        num_reactions=1,
        params={
            "k": np.array([5.0]),
            "K_ads": np.array([[2.0, 0.0]]),
            "orders_num": np.array([[1.0, 0.0]]),
            "orders_den": np.array([1.0]),
            "stoich": np.array([[-1, 1]]),
        },
    )


@pytest.fixture
def lh_kinetics_dual_site():
    """Langmuir-Hinshelwood kinetics with dual-site mechanism (denominator exponent=2)."""
    return LangmuirHinshelwoodKinetics(
        name="lh_dual",
        num_reactions=1,
        params={
            "k": np.array([5.0]),
            "K_ads": np.array([[2.0, 1.0]]),
            "orders_num": np.array([[1.0, 1.0]]),
            "orders_den": np.array([2.0]),
            "stoich": np.array([[-1, -1]]),  # A + B -> products
        },
    )


@pytest.fixture
def reversible_a_b():
    """Reversible reaction A <-> B with explicit k_r."""
    return ReversibleKinetics(
        name="rev_a_b",
        num_reactions=1,
        params={
            "k_f": np.array([1.0]),
            "k_r": np.array([0.5]),
            "orders_f": np.array([[1.0, 0.0]]),
            "orders_r": np.array([[0.0, 1.0]]),
            "stoich": np.array([[-1, 1]]),
        },
    )


@pytest.fixture
def reversible_keq():
    """Reversible reaction A <-> B using K_eq instead of k_r."""
    return ReversibleKinetics(
        name="rev_keq",
        num_reactions=1,
        params={
            "k_f": np.array([1.0]),
            "K_eq": np.array([2.0]),
            "orders_f": np.array([[1.0, 0.0]]),
            "orders_r": np.array([[0.0, 1.0]]),
            "stoich": np.array([[-1, 1]]),
        },
    )


@pytest.fixture
def reversible_with_temperature():
    """Reversible kinetics with temperature-dependent rate constants."""
    return ReversibleKinetics(
        name="rev_temp",
        num_reactions=1,
        params={
            "k_f": np.array([1.0]),
            "k_r": np.array([0.5]),
            "orders_f": np.array([[1.0, 0.0]]),
            "orders_r": np.array([[0.0, 1.0]]),
            "stoich": np.array([[-1, 1]]),
            "A_f": np.array([1e8]),
            "E_a_f": np.array([50000.0]),
            "A_r": np.array([5e7]),
            "E_a_r": np.array([60000.0]),
        },
    )


@pytest.fixture
def monod_kinetics():
    """Monod growth kinetics with 3 species: substrate, biomass, product."""
    return MonodKinetics(
        name="monod_test",
        num_species=3,
        mu_max=0.5,
        K_s=0.1,
        Y_xs=0.5,
        q_p=0.2,
        substrate_idx=0,
        biomass_idx=1,
        product_idx=2,
    )


@pytest.fixture
def monod_no_product():
    """Monod kinetics without product formation (q_p=0)."""
    return MonodKinetics(
        name="monod_no_prod",
        num_species=3,
        mu_max=0.5,
        K_s=0.1,
        Y_xs=0.5,
        q_p=0.0,
        substrate_idx=0,
        biomass_idx=1,
        product_idx=2,
    )


# ===========================================================================
# Arrhenius Kinetics Tests
# ===========================================================================


class TestArrheniusKinetics:
    """Tests for the Arrhenius kinetics model."""

    # -- Initialization --

    def test_instantiation(self, arrhenius_a_to_b):
        assert arrhenius_a_to_b.name == "A_to_B"
        assert arrhenius_a_to_b.num_reactions == 1
        np.testing.assert_array_equal(arrhenius_a_to_b.k0, [1e10])
        np.testing.assert_array_equal(arrhenius_a_to_b.Ea, [50000.0])

    def test_default_orders_from_stoich(self, arrhenius_no_orders):
        """When orders are not supplied, they should default to -min(stoich, 0)."""
        # stoich = [[-1, 1]]; orders should be [[1, 0]] (reactant only)
        expected = np.array([[1, 0]])
        np.testing.assert_array_equal(arrhenius_no_orders.orders, expected)

    def test_missing_k0_raises(self):
        with pytest.raises(ValueError, match="Missing required parameter"):
            ArrheniusKinetics(
                name="bad",
                num_reactions=1,
                params={
                    "Ea": np.array([50000.0]),
                    "stoich": np.array([[-1, 1]]),
                },
            )

    def test_missing_Ea_raises(self):
        with pytest.raises(ValueError, match="Missing required parameter"):
            ArrheniusKinetics(
                name="bad",
                num_reactions=1,
                params={
                    "k0": np.array([1e10]),
                    "stoich": np.array([[-1, 1]]),
                },
            )

    def test_missing_stoich_raises(self):
        with pytest.raises(ValueError, match="Missing required parameter"):
            ArrheniusKinetics(
                name="bad",
                num_reactions=1,
                params={
                    "k0": np.array([1e10]),
                    "Ea": np.array([50000.0]),
                },
            )

    # -- compute_rates --

    def test_compute_rates_shape(self, arrhenius_a_to_b):
        C = np.array([1.0, 0.0])
        rates = arrhenius_a_to_b.compute_rates(C, 350.0)
        assert rates.shape == (2,)  # num_species

    def test_compute_rates_two_reactions_shape(self, arrhenius_two_reactions):
        C = np.array([1.0, 0.5, 0.0])
        rates = arrhenius_two_reactions.compute_rates(C, 350.0)
        assert rates.shape == (3,)  # 3 species

    def test_rate_sign_a_consumed_b_produced(self, arrhenius_a_to_b):
        """For A->B: A is consumed (negative rate), B is produced (positive rate)."""
        C = np.array([1.0, 0.0])
        rates = arrhenius_a_to_b.compute_rates(C, 350.0)
        assert rates[0] < 0  # A consumed
        assert rates[1] > 0  # B produced

    def test_stoichiometry_conservation(self, arrhenius_a_to_b):
        """For A->B, net rate of A + net rate of B == 0 (mass balance)."""
        C = np.array([1.0, 0.0])
        rates = arrhenius_a_to_b.compute_rates(C, 350.0)
        np.testing.assert_allclose(rates[0] + rates[1], 0.0, atol=1e-15)

    def test_higher_temperature_faster_rate(self, arrhenius_a_to_b):
        """Rate magnitude increases with temperature (Arrhenius behavior)."""
        C = np.array([1.0, 0.0])
        rates_low = arrhenius_a_to_b.compute_rates(C, 300.0)
        rates_high = arrhenius_a_to_b.compute_rates(C, 400.0)
        assert rates_high[0] < rates_low[0]  # A consumed faster at higher T
        assert rates_high[1] > rates_low[1]  # B produced faster at higher T

    def test_arrhenius_rate_formula(self, arrhenius_a_to_b):
        """Verify rate matches k0 * exp(-Ea/(R*T)) * C_A for first order A->B."""
        C = np.array([2.0, 0.5])
        T = 400.0
        k = 1e10 * np.exp(-50000.0 / (R_GAS * T))
        expected_rate_A = -k * C[0]
        expected_rate_B = k * C[0]
        rates = arrhenius_a_to_b.compute_rates(C, T)
        np.testing.assert_allclose(rates[0], expected_rate_A, rtol=1e-10)
        np.testing.assert_allclose(rates[1], expected_rate_B, rtol=1e-10)

    def test_zero_concentration_zero_rate(self, arrhenius_a_to_b):
        """Zero reactant concentration should give zero rates."""
        C = np.array([0.0, 0.0])
        rates = arrhenius_a_to_b.compute_rates(C, 350.0)
        np.testing.assert_allclose(rates, 0.0, atol=1e-15)

    # -- validate_parameters --

    def test_validate_parameters_valid(self, arrhenius_a_to_b):
        assert arrhenius_a_to_b.validate_parameters() is True

    def test_validate_parameters_negative_k0(self):
        kin = ArrheniusKinetics(
            name="bad_k0",
            num_reactions=1,
            params={
                "k0": np.array([-1.0]),  # Invalid
                "Ea": np.array([50000.0]),
                "stoich": np.array([[-1, 1]]),
            },
        )
        assert kin.validate_parameters() is False

    def test_validate_parameters_negative_Ea(self):
        kin = ArrheniusKinetics(
            name="bad_Ea",
            num_reactions=1,
            params={
                "k0": np.array([1e10]),
                "Ea": np.array([-1000.0]),  # Invalid
                "stoich": np.array([[-1, 1]]),
            },
        )
        assert kin.validate_parameters() is False

    # -- Serialization --

    def test_to_dict(self, arrhenius_a_to_b):
        d = arrhenius_a_to_b.to_dict()
        assert d["name"] == "A_to_B"
        assert d["type"] == "ArrheniusKinetics"
        assert d["num_reactions"] == 1

    def test_from_dict_round_trip(self, arrhenius_a_to_b):
        config = arrhenius_a_to_b.to_dict()
        restored = ArrheniusKinetics.from_dict(config)
        assert restored.name == arrhenius_a_to_b.name
        assert restored.num_reactions == arrhenius_a_to_b.num_reactions


# ===========================================================================
# Michaelis-Menten Kinetics Tests
# ===========================================================================


class TestMichaelisMentenKinetics:
    """Tests for Michaelis-Menten enzyme kinetics."""

    # -- Initialization --

    def test_instantiation(self, mm_kinetics):
        assert mm_kinetics.name == "mm_s_to_p"
        assert mm_kinetics.num_reactions == 1
        np.testing.assert_array_equal(mm_kinetics.V_max, [10.0])
        np.testing.assert_array_equal(mm_kinetics.K_m, [0.5])

    def test_missing_V_max_raises(self):
        with pytest.raises(ValueError, match="Missing required parameter"):
            MichaelisMentenKinetics(
                name="bad",
                num_reactions=1,
                params={
                    "K_m": np.array([0.5]),
                    "substrate_indices": np.array([0]),
                    "stoich": np.array([[-1, 1]]),
                },
            )

    def test_missing_K_m_raises(self):
        with pytest.raises(ValueError, match="Missing required parameter"):
            MichaelisMentenKinetics(
                name="bad",
                num_reactions=1,
                params={
                    "V_max": np.array([10.0]),
                    "substrate_indices": np.array([0]),
                    "stoich": np.array([[-1, 1]]),
                },
            )

    def test_missing_substrate_indices_raises(self):
        with pytest.raises(ValueError, match="Missing required parameter"):
            MichaelisMentenKinetics(
                name="bad",
                num_reactions=1,
                params={
                    "V_max": np.array([10.0]),
                    "K_m": np.array([0.5]),
                    "stoich": np.array([[-1, 1]]),
                },
            )

    # -- compute_rates --

    def test_compute_rates_shape(self, mm_kinetics):
        C = np.array([1.0, 0.0])
        rates = mm_kinetics.compute_rates(C, 310.0)
        assert rates.shape == (2,)

    def test_rate_sign(self, mm_kinetics):
        """Substrate consumed (negative), product formed (positive)."""
        C = np.array([1.0, 0.0])
        rates = mm_kinetics.compute_rates(C, 310.0)
        assert rates[0] < 0  # substrate consumed
        assert rates[1] > 0  # product formed

    def test_mm_rate_formula(self, mm_kinetics):
        """Verify rate = V_max * S / (K_m + S)."""
        S = 2.0
        C = np.array([S, 0.0])
        expected_rxn_rate = 10.0 * S / (0.5 + S)  # V_max=10, K_m=0.5
        rates = mm_kinetics.compute_rates(C, 310.0)
        # Product rate = +r, substrate rate = -r
        np.testing.assert_allclose(rates[1], expected_rxn_rate, rtol=1e-10)
        np.testing.assert_allclose(rates[0], -expected_rxn_rate, rtol=1e-10)

    def test_saturation_at_high_substrate(self, mm_kinetics):
        """At high [S], rate approaches V_max."""
        C_high = np.array([1000.0, 0.0])
        rates = mm_kinetics.compute_rates(C_high, 310.0)
        # Product rate should approach V_max = 10.0
        assert rates[1] > 0.99 * 10.0
        assert rates[1] <= 10.0 + 1e-6

    def test_half_rate_at_km(self, mm_kinetics):
        """At [S] == K_m, rate == V_max / 2."""
        K_m = 0.5
        C = np.array([K_m, 0.0])
        rates = mm_kinetics.compute_rates(C, 310.0)
        np.testing.assert_allclose(rates[1], 10.0 / 2.0, rtol=1e-10)

    def test_zero_substrate_zero_rate(self, mm_kinetics):
        """Zero substrate gives zero rate."""
        C = np.array([0.0, 0.0])
        rates = mm_kinetics.compute_rates(C, 310.0)
        np.testing.assert_allclose(rates, 0.0, atol=1e-15)

    # -- Inhibition --

    def test_inhibition_reduces_rate(self, mm_with_inhibition):
        """Competitive inhibitor should reduce the reaction rate."""
        C_no_inh = np.array([1.0, 0.0, 0.0])
        C_with_inh = np.array([1.0, 0.0, 5.0])
        rate_no = mm_with_inhibition.compute_rates(C_no_inh, 310.0)
        rate_yes = mm_with_inhibition.compute_rates(C_with_inh, 310.0)
        assert rate_yes[1] < rate_no[1]

    def test_inhibition_formula(self, mm_with_inhibition):
        """Verify r = V_max*S/(K_m*(1+I/K_i)+S) for competitive inhibition."""
        S = 1.0
        I_conc = 2.0
        C = np.array([S, 0.0, I_conc])
        # V_max=10, K_m=0.5, K_i=1.0
        # Standard rate: V_max*S/(K_m+S) = 10*1/(0.5+1) = 6.667
        # Inhibition factor: 1 + I/K_i = 1 + 2/1 = 3
        # The code applies: rate / inhibition_factor
        # rate = V_max*S/(K_m+S) = 10*1/1.5 = 6.667
        # rate / 3 = 2.222
        rates = mm_with_inhibition.compute_rates(C, 310.0)
        standard_rate = 10.0 * S / (0.5 + S)
        inhibition_factor = 1 + I_conc / 1.0
        expected = standard_rate / inhibition_factor
        np.testing.assert_allclose(rates[1], expected, rtol=1e-10)

    # -- validate_parameters --

    def test_validate_parameters_valid(self, mm_kinetics):
        assert mm_kinetics.validate_parameters() is True

    # -- Serialization --

    def test_to_dict(self, mm_kinetics):
        d = mm_kinetics.to_dict()
        assert d["type"] == "MichaelisMentenKinetics"

    def test_from_dict_round_trip(self, mm_kinetics):
        config = mm_kinetics.to_dict()
        restored = MichaelisMentenKinetics.from_dict(config)
        assert restored.name == mm_kinetics.name
        assert restored.num_reactions == mm_kinetics.num_reactions


# ===========================================================================
# Power Law Kinetics Tests
# ===========================================================================


class TestPowerLawKinetics:
    """Tests for power law kinetics."""

    # -- Initialization --

    def test_instantiation(self, power_law_second_order):
        assert power_law_second_order.name == "pl_a2b"
        assert power_law_second_order.num_reactions == 1
        np.testing.assert_array_equal(power_law_second_order.k, [0.5])

    def test_missing_k_raises(self):
        with pytest.raises(ValueError, match="Missing required parameter"):
            PowerLawKinetics(
                name="bad",
                num_reactions=1,
                params={
                    "orders": np.array([[1.0, 0.0]]),
                    "stoich": np.array([[-1, 1]]),
                },
            )

    def test_missing_orders_raises(self):
        with pytest.raises(ValueError, match="Missing required parameter"):
            PowerLawKinetics(
                name="bad",
                num_reactions=1,
                params={
                    "k": np.array([0.5]),
                    "stoich": np.array([[-1, 1]]),
                },
            )

    def test_missing_stoich_raises(self):
        with pytest.raises(ValueError, match="Missing required parameter"):
            PowerLawKinetics(
                name="bad",
                num_reactions=1,
                params={
                    "k": np.array([0.5]),
                    "orders": np.array([[1.0, 0.0]]),
                },
            )

    # -- compute_rates --

    def test_compute_rates_shape(self, power_law_second_order):
        C = np.array([1.0, 0.0])
        rates = power_law_second_order.compute_rates(C, 350.0)
        assert rates.shape == (2,)

    def test_second_order_scaling(self, power_law_second_order):
        """Second order: doubling concentration quadruples rate."""
        C1 = np.array([1.0, 0.0])
        C2 = np.array([2.0, 0.0])
        rate1 = power_law_second_order.compute_rates(C1, 350.0)
        rate2 = power_law_second_order.compute_rates(C2, 350.0)
        ratio = rate2[1] / rate1[1]
        np.testing.assert_allclose(ratio, 4.0, rtol=1e-6)

    def test_rate_formula(self, power_law_second_order):
        """Verify r = k * C_A^order for power law."""
        C_A = 3.0
        C = np.array([C_A, 0.0])
        # k=0.5, order=2 -> r = 0.5 * 3^2 = 4.5
        # stoich = [-1, 1], so rate_A = -4.5, rate_B = +4.5
        rates = power_law_second_order.compute_rates(C, 350.0)
        np.testing.assert_allclose(rates[0], -0.5 * 9.0, rtol=1e-10)
        np.testing.assert_allclose(rates[1], 0.5 * 9.0, rtol=1e-10)

    def test_temperature_dependence(self, power_law_with_temperature):
        """When A and E_a are provided, rate uses Arrhenius T dependence."""
        C = np.array([1.0, 0.0])
        rates_low = power_law_with_temperature.compute_rates(C, 300.0)
        rates_high = power_law_with_temperature.compute_rates(C, 400.0)
        # Higher T -> faster rate
        assert rates_high[1] > rates_low[1]

    def test_temperature_formula(self, power_law_with_temperature):
        """Verify rate uses k(T) = A * exp(-E_a/(R*T)) when A and E_a are provided."""
        C_A = 2.0
        C = np.array([C_A, 0.0])
        T = 380.0
        k_T = 1e8 * np.exp(-40000.0 / (R_GAS * T))
        expected_r = k_T * C_A ** 1.0  # order=1
        rates = power_law_with_temperature.compute_rates(C, T)
        np.testing.assert_allclose(rates[1], expected_r, rtol=1e-10)

    # -- validate_parameters --

    def test_validate_parameters_valid(self, power_law_second_order):
        assert power_law_second_order.validate_parameters() is True

    def test_validate_parameters_negative_k(self):
        kin = PowerLawKinetics(
            name="bad_k",
            num_reactions=1,
            params={
                "k": np.array([-0.5]),
                "orders": np.array([[1.0, 0.0]]),
                "stoich": np.array([[-1, 1]]),
            },
        )
        assert kin.validate_parameters() is False

    # -- Serialization --

    def test_to_dict(self, power_law_second_order):
        d = power_law_second_order.to_dict()
        assert d["type"] == "PowerLawKinetics"

    def test_from_dict_round_trip(self, power_law_second_order):
        config = power_law_second_order.to_dict()
        restored = PowerLawKinetics.from_dict(config)
        assert restored.name == power_law_second_order.name


# ===========================================================================
# Langmuir-Hinshelwood Kinetics Tests
# ===========================================================================


class TestLangmuirHinshelwoodKinetics:
    """Tests for Langmuir-Hinshelwood surface kinetics."""

    # -- Initialization --

    def test_instantiation(self, lh_kinetics):
        assert lh_kinetics.name == "lh_a_to_b"
        assert lh_kinetics.num_reactions == 1
        np.testing.assert_array_equal(lh_kinetics.k, [5.0])

    def test_missing_k_raises(self):
        with pytest.raises(ValueError, match="Missing required parameter"):
            LangmuirHinshelwoodKinetics(
                name="bad",
                num_reactions=1,
                params={
                    "K_ads": np.array([[2.0, 0.0]]),
                    "orders_num": np.array([[1.0, 0.0]]),
                    "orders_den": np.array([1.0]),
                    "stoich": np.array([[-1, 1]]),
                },
            )

    def test_missing_K_ads_raises(self):
        with pytest.raises(ValueError, match="Missing required parameter"):
            LangmuirHinshelwoodKinetics(
                name="bad",
                num_reactions=1,
                params={
                    "k": np.array([5.0]),
                    "orders_num": np.array([[1.0, 0.0]]),
                    "orders_den": np.array([1.0]),
                    "stoich": np.array([[-1, 1]]),
                },
            )

    # -- compute_rates --

    def test_compute_rates_shape(self, lh_kinetics):
        C = np.array([1.0, 0.0])
        rates = lh_kinetics.compute_rates(C, 350.0)
        assert rates.shape == (2,)

    def test_rate_formula_single_site(self, lh_kinetics):
        """Verify r = k*C_A^1 / (1 + K_A*C_A)^1."""
        C_A = 2.0
        C = np.array([C_A, 0.0])
        T = 350.0
        # k=5, K_ads_A=2, orders_num_A=1, orders_den=1
        numerator = 5.0 * C_A ** 1.0
        denominator = (1.0 + 2.0 * C_A) ** 1.0
        expected_r = numerator / denominator
        rates = lh_kinetics.compute_rates(C, T)
        np.testing.assert_allclose(rates[1], expected_r, rtol=1e-10)
        np.testing.assert_allclose(rates[0], -expected_r, rtol=1e-10)

    def test_saturation_effect(self, lh_kinetics):
        """At very high concentration, adsorption term dominates and rate saturates."""
        C_low = np.array([0.1, 0.0])
        C_high = np.array([10.0, 0.0])
        rate_low = lh_kinetics.compute_rates(C_low, 350.0)
        rate_high = lh_kinetics.compute_rates(C_high, 350.0)
        # Rate per unit concentration decreases (saturation)
        specific_low = abs(rate_low[0]) / C_low[0]
        specific_high = abs(rate_high[0]) / C_high[0]
        assert specific_high < specific_low

    def test_dual_site_denominator_squared(self, lh_kinetics_dual_site):
        """Verify dual-site mechanism uses (1+K*C)^2 denominator."""
        C = np.array([1.0, 1.0])
        T = 350.0
        # k=5, K_ads=[[2,1]], orders_num=[[1,1]], orders_den=[2]
        numerator = 5.0 * (1.0 ** 1.0) * (1.0 ** 1.0)
        denominator = (1.0 + 2.0 * 1.0 + 1.0 * 1.0) ** 2.0
        expected_r = numerator / denominator
        rates = lh_kinetics_dual_site.compute_rates(C, T)
        # stoich = [-1, -1], so both species consumed
        np.testing.assert_allclose(rates[0], -expected_r, rtol=1e-10)
        np.testing.assert_allclose(rates[1], -expected_r, rtol=1e-10)

    # -- validate_parameters --

    def test_validate_parameters_valid(self, lh_kinetics):
        assert lh_kinetics.validate_parameters() is True

    def test_validate_parameters_negative_k(self):
        kin = LangmuirHinshelwoodKinetics(
            name="bad",
            num_reactions=1,
            params={
                "k": np.array([-1.0]),
                "K_ads": np.array([[2.0, 0.0]]),
                "orders_num": np.array([[1.0, 0.0]]),
                "orders_den": np.array([1.0]),
                "stoich": np.array([[-1, 1]]),
            },
        )
        assert kin.validate_parameters() is False

    # -- Serialization --

    def test_to_dict(self, lh_kinetics):
        d = lh_kinetics.to_dict()
        assert d["type"] == "LangmuirHinshelwoodKinetics"

    def test_from_dict_round_trip(self, lh_kinetics):
        config = lh_kinetics.to_dict()
        restored = LangmuirHinshelwoodKinetics.from_dict(config)
        assert restored.name == lh_kinetics.name


# ===========================================================================
# Reversible Kinetics Tests
# ===========================================================================


class TestReversibleKinetics:
    """Tests for reversible reaction kinetics."""

    # -- Initialization --

    def test_instantiation_with_kr(self, reversible_a_b):
        assert reversible_a_b.name == "rev_a_b"
        assert reversible_a_b.num_reactions == 1
        np.testing.assert_array_equal(reversible_a_b.k_f, [1.0])
        np.testing.assert_array_equal(reversible_a_b.k_r, [0.5])
        assert reversible_a_b.K_eq is None

    def test_instantiation_with_keq(self, reversible_keq):
        np.testing.assert_array_equal(reversible_keq.K_eq, [2.0])
        # k_r should be computed as k_f / K_eq = 1.0 / 2.0 = 0.5
        np.testing.assert_allclose(reversible_keq.k_r, [0.5])

    def test_missing_both_kr_and_keq_raises(self):
        with pytest.raises(ValueError, match="Must provide either"):
            ReversibleKinetics(
                name="bad",
                num_reactions=1,
                params={
                    "k_f": np.array([1.0]),
                    "orders_f": np.array([[1.0, 0.0]]),
                    "orders_r": np.array([[0.0, 1.0]]),
                    "stoich": np.array([[-1, 1]]),
                },
            )

    def test_missing_kf_raises(self):
        with pytest.raises(ValueError, match="Missing required parameter"):
            ReversibleKinetics(
                name="bad",
                num_reactions=1,
                params={
                    "k_r": np.array([0.5]),
                    "orders_f": np.array([[1.0, 0.0]]),
                    "orders_r": np.array([[0.0, 1.0]]),
                    "stoich": np.array([[-1, 1]]),
                },
            )

    # -- compute_rates --

    def test_compute_rates_shape(self, reversible_a_b):
        C = np.array([1.0, 0.5])
        rates = reversible_a_b.compute_rates(C, 350.0)
        assert rates.shape == (2,)

    def test_net_rate_sign_forward_dominant(self, reversible_a_b):
        """When only reactant is present, forward reaction dominates."""
        C = np.array([1.0, 0.0])
        rates = reversible_a_b.compute_rates(C, 350.0)
        assert rates[0] < 0  # A consumed
        assert rates[1] > 0  # B produced

    def test_net_rate_sign_reverse_dominant(self, reversible_a_b):
        """When product greatly exceeds equilibrium, reverse reaction dominates."""
        # K_eq = k_f/k_r = 2.0. At [A]=0.1, [B]=10, reverse dominates.
        C = np.array([0.1, 10.0])
        rates = reversible_a_b.compute_rates(C, 350.0)
        assert rates[0] > 0  # A produced (reverse reaction)
        assert rates[1] < 0  # B consumed

    def test_equilibrium_zero_net_rate(self, reversible_a_b):
        """At equilibrium (k_f*C_A == k_r*C_B), net rates should be zero."""
        # k_f=1, k_r=0.5 -> K_eq=2 -> at equilibrium C_B/C_A=2
        C_eq = np.array([1.0, 2.0])
        rates = reversible_a_b.compute_rates(C_eq, 350.0)
        np.testing.assert_allclose(rates, 0.0, atol=1e-10)

    def test_rate_formula(self, reversible_a_b):
        """Verify r = k_f*C_A - k_r*C_B."""
        C_A, C_B = 2.0, 1.5
        C = np.array([C_A, C_B])
        expected_r = 1.0 * C_A - 0.5 * C_B  # k_f=1, k_r=0.5
        rates = reversible_a_b.compute_rates(C, 350.0)
        np.testing.assert_allclose(rates[0], -expected_r, rtol=1e-10)
        np.testing.assert_allclose(rates[1], expected_r, rtol=1e-10)

    def test_keq_gives_same_rates_as_kr(self, reversible_a_b, reversible_keq):
        """ReversibleKinetics initialized with K_eq should give same rates as k_r."""
        C = np.array([1.5, 0.8])
        T = 350.0
        rates_kr = reversible_a_b.compute_rates(C, T)
        rates_keq = reversible_keq.compute_rates(C, T)
        np.testing.assert_allclose(rates_kr, rates_keq, rtol=1e-10)

    # -- Temperature dependence --

    def test_temperature_affects_rates(self, reversible_with_temperature):
        """With A_f/E_a_f/A_r/E_a_r, rate should change with temperature."""
        C = np.array([1.0, 0.5])
        rates_low = reversible_with_temperature.compute_rates(C, 300.0)
        rates_high = reversible_with_temperature.compute_rates(C, 400.0)
        # Rates should differ at different temperatures
        assert not np.allclose(rates_low, rates_high)

    # -- get_equilibrium_constant --

    def test_get_equilibrium_constant_from_kr(self, reversible_a_b):
        """K_eq = k_f / k_r = 1.0 / 0.5 = 2.0."""
        K_eq = reversible_a_b.get_equilibrium_constant(350.0)
        np.testing.assert_allclose(K_eq, [2.0], rtol=1e-6)

    def test_get_equilibrium_constant_from_keq(self, reversible_keq):
        """When K_eq is stored directly, return it."""
        K_eq = reversible_keq.get_equilibrium_constant(350.0)
        np.testing.assert_allclose(K_eq, [2.0], rtol=1e-6)

    def test_get_equilibrium_constant_temperature_dependent(
        self, reversible_with_temperature
    ):
        """K_eq(T) = k_f(T)/k_r(T) should vary with temperature."""
        K_300 = reversible_with_temperature.get_equilibrium_constant(300.0)
        K_400 = reversible_with_temperature.get_equilibrium_constant(400.0)
        # K_eq changes with T when activation energies differ
        assert not np.allclose(K_300, K_400)

    def test_get_equilibrium_constant_formula(self, reversible_with_temperature):
        """Verify K_eq(T) = A_f*exp(-Ea_f/RT) / (A_r*exp(-Ea_r/RT))."""
        T = 380.0
        k_f_T = 1e8 * np.exp(-50000.0 / (R_GAS * T))
        k_r_T = 5e7 * np.exp(-60000.0 / (R_GAS * T))
        expected_K = k_f_T / k_r_T
        K = reversible_with_temperature.get_equilibrium_constant(T)
        np.testing.assert_allclose(K, [expected_K], rtol=1e-10)

    # -- validate_parameters --

    def test_validate_parameters_valid(self, reversible_a_b):
        assert reversible_a_b.validate_parameters() is True

    def test_validate_parameters_negative_kf(self):
        kin = ReversibleKinetics(
            name="bad",
            num_reactions=1,
            params={
                "k_f": np.array([-1.0]),
                "k_r": np.array([0.5]),
                "orders_f": np.array([[1.0, 0.0]]),
                "orders_r": np.array([[0.0, 1.0]]),
                "stoich": np.array([[-1, 1]]),
            },
        )
        assert kin.validate_parameters() is False

    def test_validate_parameters_negative_kr(self):
        kin = ReversibleKinetics(
            name="bad",
            num_reactions=1,
            params={
                "k_f": np.array([1.0]),
                "k_r": np.array([-0.5]),
                "orders_f": np.array([[1.0, 0.0]]),
                "orders_r": np.array([[0.0, 1.0]]),
                "stoich": np.array([[-1, 1]]),
            },
        )
        assert kin.validate_parameters() is False

    # -- Serialization --

    def test_to_dict(self, reversible_a_b):
        d = reversible_a_b.to_dict()
        assert d["type"] == "ReversibleKinetics"

    def test_from_dict_round_trip(self, reversible_a_b):
        config = reversible_a_b.to_dict()
        restored = ReversibleKinetics.from_dict(config)
        assert restored.name == reversible_a_b.name
        assert restored.num_reactions == reversible_a_b.num_reactions


# ===========================================================================
# Monod Kinetics Tests
# ===========================================================================


class TestMonodKinetics:
    """Tests for Monod growth kinetics for bioreactor systems."""

    # -- Initialization --

    def test_instantiation(self, monod_kinetics):
        assert monod_kinetics.name == "monod_test"
        assert monod_kinetics.mu_max == pytest.approx(0.5)
        assert monod_kinetics.K_s == pytest.approx(0.1)
        assert monod_kinetics.Y_xs == pytest.approx(0.5)
        assert monod_kinetics.q_p == pytest.approx(0.2)
        assert monod_kinetics.substrate_idx == 0
        assert monod_kinetics.biomass_idx == 1
        assert monod_kinetics.product_idx == 2

    def test_num_reactions(self, monod_kinetics):
        """With product formation (q_p > 0), num_reactions should be 2."""
        assert monod_kinetics.num_reactions == 2

    def test_num_reactions_no_product(self, monod_no_product):
        """Without product formation (q_p == 0), compute_rates should still work."""
        # q_p=0 but product_idx is not None -> num_reactions=2
        assert monod_no_product.num_reactions == 2

    # -- compute_rates --

    def test_compute_rates_shape(self, monod_kinetics):
        C = np.array([5.0, 1.0, 0.0])
        rates = monod_kinetics.compute_rates(C, 310.0)
        assert rates.shape == (3,)

    def test_substrate_consumed(self, monod_kinetics):
        """Substrate rate should be negative (consumed for growth)."""
        C = np.array([5.0, 1.0, 0.0])
        rates = monod_kinetics.compute_rates(C, 310.0)
        assert rates[0] < 0  # dS/dt = -mu*X/Y_xs < 0

    def test_biomass_grows(self, monod_kinetics):
        """Biomass rate should be positive (cell growth)."""
        C = np.array([5.0, 1.0, 0.0])
        rates = monod_kinetics.compute_rates(C, 310.0)
        assert rates[1] > 0  # dX/dt = mu*X > 0

    def test_product_formed(self, monod_kinetics):
        """Product rate should be positive (q_p > 0)."""
        C = np.array([5.0, 1.0, 0.0])
        rates = monod_kinetics.compute_rates(C, 310.0)
        assert rates[2] > 0  # dP/dt = q_p*X > 0

    def test_no_product_formation_when_qp_zero(self, monod_no_product):
        """When q_p==0, product rate should be zero."""
        C = np.array([5.0, 1.0, 0.0])
        rates = monod_no_product.compute_rates(C, 310.0)
        assert rates[2] == pytest.approx(0.0)

    def test_rate_formulas(self, monod_kinetics):
        """Verify:
            mu = mu_max * S / (K_s + S)
            dS/dt = -mu * X / Y_xs
            dX/dt = mu * X
            dP/dt = q_p * X
        """
        S, X, P = 5.0, 2.0, 0.5
        C = np.array([S, X, P])
        mu = 0.5 * S / (0.1 + S)
        expected = np.array([
            -mu * X / 0.5,  # dS/dt
            mu * X,         # dX/dt
            0.2 * X,        # dP/dt
        ])
        rates = monod_kinetics.compute_rates(C, 310.0)
        np.testing.assert_allclose(rates, expected, rtol=1e-10)

    def test_zero_substrate_zero_growth(self, monod_kinetics):
        """Zero substrate => mu=0 => no growth, no substrate consumption."""
        C = np.array([0.0, 1.0, 0.0])
        rates = monod_kinetics.compute_rates(C, 310.0)
        assert rates[0] == pytest.approx(0.0)  # no substrate consumption
        assert rates[1] == pytest.approx(0.0)  # no growth

    def test_zero_biomass_zero_rates(self, monod_kinetics):
        """Zero biomass => all rates zero (nothing to consume or grow)."""
        C = np.array([5.0, 0.0, 0.0])
        rates = monod_kinetics.compute_rates(C, 310.0)
        np.testing.assert_allclose(rates, 0.0, atol=1e-15)

    def test_saturation_at_high_substrate(self, monod_kinetics):
        """At high [S], mu approaches mu_max; growth rate plateaus."""
        C_low = np.array([0.01, 1.0, 0.0])
        C_med = np.array([1.0, 1.0, 0.0])
        C_high = np.array([1000.0, 1.0, 0.0])
        rate_low = monod_kinetics.compute_rates(C_low, 310.0)
        rate_med = monod_kinetics.compute_rates(C_med, 310.0)
        rate_high = monod_kinetics.compute_rates(C_high, 310.0)
        # Growth rate increases from low to medium
        assert rate_med[1] > rate_low[1]
        # Gap narrows at high substrate (saturation)
        gap_low_med = rate_med[1] - rate_low[1]
        gap_med_high = rate_high[1] - rate_med[1]
        assert gap_med_high < gap_low_med

    def test_substrate_consumption_greater_than_biomass_growth(self, monod_kinetics):
        """Since Y_xs < 1, substrate consumed is more than biomass produced (per mole)."""
        C = np.array([5.0, 1.0, 0.0])
        rates = monod_kinetics.compute_rates(C, 310.0)
        # |dS/dt| = mu*X/Y_xs > |dX/dt| = mu*X since Y_xs=0.5 < 1
        assert abs(rates[0]) > abs(rates[1])

    # -- get_specific_growth_rate --

    def test_get_specific_growth_rate(self, monod_kinetics):
        """mu(S) = mu_max * S / (K_s + S)."""
        S = 5.0
        mu = monod_kinetics.get_specific_growth_rate(S)
        expected = 0.5 * 5.0 / (0.1 + 5.0)
        assert mu == pytest.approx(expected, rel=1e-10)

    def test_specific_growth_rate_zero_substrate(self, monod_kinetics):
        mu = monod_kinetics.get_specific_growth_rate(0.0)
        assert mu == pytest.approx(0.0)

    def test_specific_growth_rate_at_ks(self, monod_kinetics):
        """At S == K_s, mu == mu_max / 2."""
        mu = monod_kinetics.get_specific_growth_rate(0.1)  # K_s = 0.1
        assert mu == pytest.approx(0.5 / 2.0, rel=1e-10)

    def test_specific_growth_rate_high_substrate(self, monod_kinetics):
        """At very high [S], mu approaches mu_max."""
        mu = monod_kinetics.get_specific_growth_rate(10000.0)
        assert mu == pytest.approx(0.5, rel=1e-3)

    def test_specific_growth_rate_negative_substrate_clamped(self, monod_kinetics):
        """Negative substrate should be clamped to zero."""
        mu = monod_kinetics.get_specific_growth_rate(-1.0)
        assert mu == pytest.approx(0.0)

    # -- Serialization --

    def test_from_dict(self, monod_kinetics):
        config = {
            "name": "monod_from_dict",
            "num_species": 3,
            "params": {
                "mu_max": 0.5,
                "K_s": 0.1,
                "Y_xs": 0.5,
                "q_p": 0.2,
            },
        }
        restored = MonodKinetics.from_dict(config)
        assert restored.name == "monod_from_dict"
        assert restored.mu_max == pytest.approx(0.5)


# ===========================================================================
# Cross-model parametrized tests
# ===========================================================================


@pytest.mark.parametrize(
    "kinetics_fixture",
    [
        "arrhenius_a_to_b",
        "mm_kinetics",
        "power_law_second_order",
        "lh_kinetics",
        "reversible_a_b",
    ],
)
class TestAllStoredKinetics:
    """Parametrized tests that run against all stoichiometry-based kinetics models."""

    def test_rates_are_finite(self, kinetics_fixture, request):
        """All kinetics should return finite rates for reasonable inputs."""
        kinetics = request.getfixturevalue(kinetics_fixture)
        num_species = kinetics.stoich.shape[1]
        C = np.ones(num_species) * 0.5
        rates = kinetics.compute_rates(C, 350.0)
        assert np.all(np.isfinite(rates))

    def test_rates_shape_matches_species(self, kinetics_fixture, request):
        """Output shape should be (num_species,)."""
        kinetics = request.getfixturevalue(kinetics_fixture)
        num_species = kinetics.stoich.shape[1]
        C = np.ones(num_species)
        rates = kinetics.compute_rates(C, 350.0)
        assert rates.shape == (num_species,)

    def test_validate_returns_bool(self, kinetics_fixture, request):
        """validate_parameters should return a boolean."""
        kinetics = request.getfixturevalue(kinetics_fixture)
        result = kinetics.validate_parameters()
        assert isinstance(result, bool)

    def test_to_dict_has_required_keys(self, kinetics_fixture, request):
        kinetics = request.getfixturevalue(kinetics_fixture)
        d = kinetics.to_dict()
        assert "name" in d
        assert "type" in d
        assert "num_reactions" in d
        assert "params" in d

    def test_repr_contains_class_name(self, kinetics_fixture, request):
        kinetics = request.getfixturevalue(kinetics_fixture)
        r = repr(kinetics)
        assert kinetics.__class__.__name__ in r


# ===========================================================================
# Registry integration tests
# ===========================================================================


class TestKineticsRegistry:
    """Verify that all kinetics models are properly registered."""

    def test_arrhenius_registered(self):
        from reactor_twin.utils.registry import KINETICS_REGISTRY

        assert "arrhenius" in KINETICS_REGISTRY
        cls = KINETICS_REGISTRY.get("arrhenius")
        assert cls is ArrheniusKinetics

    def test_michaelis_menten_registered(self):
        from reactor_twin.utils.registry import KINETICS_REGISTRY

        assert "michaelis_menten" in KINETICS_REGISTRY
        cls = KINETICS_REGISTRY.get("michaelis_menten")
        assert cls is MichaelisMentenKinetics

    def test_power_law_registered(self):
        from reactor_twin.utils.registry import KINETICS_REGISTRY

        assert "power_law" in KINETICS_REGISTRY
        cls = KINETICS_REGISTRY.get("power_law")
        assert cls is PowerLawKinetics

    def test_langmuir_hinshelwood_registered(self):
        from reactor_twin.utils.registry import KINETICS_REGISTRY

        assert "langmuir_hinshelwood" in KINETICS_REGISTRY
        cls = KINETICS_REGISTRY.get("langmuir_hinshelwood")
        assert cls is LangmuirHinshelwoodKinetics

    def test_reversible_registered(self):
        from reactor_twin.utils.registry import KINETICS_REGISTRY

        assert "reversible" in KINETICS_REGISTRY
        cls = KINETICS_REGISTRY.get("reversible")
        assert cls is ReversibleKinetics

    def test_monod_registered(self):
        from reactor_twin.utils.registry import KINETICS_REGISTRY

        assert "monod" in KINETICS_REGISTRY
        cls = KINETICS_REGISTRY.get("monod")
        assert cls is MonodKinetics


# ===========================================================================
# R_GAS constant test
# ===========================================================================


class TestConstants:
    """Verify the gas constant value."""

    def test_r_gas_value(self):
        assert R_GAS == pytest.approx(8.314, rel=1e-4)
