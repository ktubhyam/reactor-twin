"""Tests for benchmark reaction systems."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from reactor_twin.reactors import CSTRReactor
from reactor_twin.reactors.systems import (
    create_bioreactor_cstr,
    create_consecutive_cstr,
    create_exothermic_cstr,
    create_parallel_cstr,
    create_van_de_vusse_cstr,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def simulate_reactor(reactor, t_end=10.0, num_points=100):
    """Simulate a reactor with solve_ivp and return the solution."""
    y0 = reactor.get_initial_state()
    t_eval = np.linspace(0, t_end, num_points)

    sol = solve_ivp(
        reactor.ode_rhs,
        [0, t_end],
        y0,
        t_eval=t_eval,
        method="LSODA",
        rtol=1e-6,
        atol=1e-8,
    )
    return sol


# ---------------------------------------------------------------------------
# Exothermic A -> B Tests
# ---------------------------------------------------------------------------

class TestExothermicSystem:
    """Tests for the exothermic A -> B CSTR benchmark."""

    def test_creates_cstr(self):
        reactor = create_exothermic_cstr()
        assert isinstance(reactor, CSTRReactor)

    def test_isothermal_mode(self):
        reactor = create_exothermic_cstr(isothermal=True)
        assert reactor.isothermal is True
        assert reactor.state_dim == 2

    def test_nonisothermal_mode(self):
        reactor = create_exothermic_cstr(isothermal=False)
        assert reactor.isothermal is False
        assert reactor.state_dim == 3

    def test_simulation_isothermal(self):
        reactor = create_exothermic_cstr(isothermal=True)
        sol = simulate_reactor(reactor, t_end=5.0)
        assert sol.success, f"Integration failed: {sol.message}"
        # All concentrations should remain non-negative
        concentrations = sol.y[:2, :]
        assert np.all(concentrations >= -1e-10), "Negative concentrations found"

    def test_simulation_nonisothermal(self):
        reactor = create_exothermic_cstr(isothermal=False)
        sol = simulate_reactor(reactor, t_end=5.0)
        assert sol.success, f"Integration failed: {sol.message}"
        # Temperature should be positive
        temperature = sol.y[2, :]
        assert np.all(temperature > 0), "Temperature went negative"

    def test_steady_state_reasonable(self):
        """After long enough simulation, state should be physically reasonable."""
        reactor = create_exothermic_cstr(isothermal=True)
        sol = simulate_reactor(reactor, t_end=20.0, num_points=200)
        assert sol.success

        # Final state: concentrations should be non-negative
        y_final = sol.y[:, -1]
        assert np.all(y_final[:2] >= -1e-10)
        # A + B should be approximately equal to feed (mass balance)
        total_conc_final = y_final[0] + y_final[1]
        feed_total = sum(reactor.params["C_feed"])
        assert total_conc_final <= feed_total + 0.1, (
            f"Total concentration {total_conc_final} exceeds feed {feed_total}"
        )


# ---------------------------------------------------------------------------
# Van de Vusse Tests
# ---------------------------------------------------------------------------

class TestVanDeVusseSystem:
    """Tests for the Van de Vusse CSTR benchmark."""

    def test_creates_cstr(self):
        reactor = create_van_de_vusse_cstr()
        assert isinstance(reactor, CSTRReactor)

    def test_has_four_species(self):
        reactor = create_van_de_vusse_cstr()
        assert reactor.num_species == 4

    def test_simulation_runs(self):
        reactor = create_van_de_vusse_cstr()
        sol = simulate_reactor(reactor, t_end=0.01)  # short time (fast kinetics)
        assert sol.success, f"Integration failed: {sol.message}"

    def test_concentrations_non_negative(self):
        reactor = create_van_de_vusse_cstr()
        sol = simulate_reactor(reactor, t_end=0.01, num_points=50)
        assert sol.success
        assert np.all(sol.y >= -1e-8), "Negative concentrations found"


# ---------------------------------------------------------------------------
# Bioreactor Tests
# ---------------------------------------------------------------------------

class TestBioreactorSystem:
    """Tests for the bioreactor CSTR benchmark with Monod kinetics."""

    def test_creates_cstr(self):
        reactor = create_bioreactor_cstr()
        assert isinstance(reactor, CSTRReactor)

    def test_has_three_species(self):
        reactor = create_bioreactor_cstr()
        assert reactor.num_species == 3  # S, X, P

    def test_simulation_runs(self):
        reactor = create_bioreactor_cstr()
        sol = simulate_reactor(reactor, t_end=50.0)
        assert sol.success, f"Integration failed: {sol.message}"

    def test_biomass_grows(self):
        """Biomass should increase from initial condition."""
        reactor = create_bioreactor_cstr()
        sol = simulate_reactor(reactor, t_end=50.0)
        assert sol.success
        # Biomass is species index 1
        X_initial = sol.y[1, 0]
        X_final = sol.y[1, -1]
        # With substrate available and dilution rate < mu_max, biomass should grow
        assert X_final > X_initial, "Biomass did not grow"

    def test_steady_state_physically_reasonable(self):
        """At steady state, all concentrations should be non-negative and finite."""
        reactor = create_bioreactor_cstr()
        sol = simulate_reactor(reactor, t_end=200.0, num_points=500)
        assert sol.success
        y_final = sol.y[:, -1]
        assert np.all(y_final >= -1e-8), "Negative concentrations at steady state"
        assert np.all(np.isfinite(y_final)), "Non-finite values at steady state"


# ---------------------------------------------------------------------------
# Consecutive Reactions Tests
# ---------------------------------------------------------------------------

class TestConsecutiveSystem:
    """Tests for the consecutive reactions A->B->C CSTR benchmark."""

    def test_creates_cstr(self):
        reactor = create_consecutive_cstr()
        assert isinstance(reactor, CSTRReactor)

    def test_has_three_species(self):
        reactor = create_consecutive_cstr()
        assert reactor.num_species == 3

    def test_simulation_runs(self):
        reactor = create_consecutive_cstr()
        sol = simulate_reactor(reactor, t_end=5.0)
        assert sol.success, f"Integration failed: {sol.message}"

    def test_concentrations_non_negative(self):
        reactor = create_consecutive_cstr()
        sol = simulate_reactor(reactor, t_end=5.0)
        assert sol.success
        assert np.all(sol.y >= -1e-8), "Negative concentrations found"


# ---------------------------------------------------------------------------
# Parallel Reactions Tests
# ---------------------------------------------------------------------------

class TestParallelSystem:
    """Tests for the parallel reactions A->B, A->C CSTR benchmark."""

    def test_creates_cstr(self):
        reactor = create_parallel_cstr()
        assert isinstance(reactor, CSTRReactor)

    def test_has_three_species(self):
        reactor = create_parallel_cstr()
        assert reactor.num_species == 3

    def test_simulation_runs(self):
        reactor = create_parallel_cstr()
        sol = simulate_reactor(reactor, t_end=5.0)
        assert sol.success, f"Integration failed: {sol.message}"

    def test_concentrations_non_negative(self):
        reactor = create_parallel_cstr()
        sol = simulate_reactor(reactor, t_end=5.0)
        assert sol.success
        assert np.all(sol.y >= -1e-8), "Negative concentrations found"


# ---------------------------------------------------------------------------
# Parametrized creation tests (isothermal only to avoid param issues)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "factory",
    [
        create_exothermic_cstr,
        create_van_de_vusse_cstr,
        create_bioreactor_cstr,
        create_consecutive_cstr,
        create_parallel_cstr,
    ],
    ids=["exothermic", "van_de_vusse", "bioreactor", "consecutive", "parallel"],
)
class TestBenchmarkCommon:
    """Common tests for all benchmark systems that use proper CSTRReactor params."""

    def test_returns_cstr(self, factory):
        reactor = factory()
        assert isinstance(reactor, CSTRReactor)

    def test_has_kinetics(self, factory):
        reactor = factory()
        assert reactor.kinetics is not None

    def test_initial_state_finite(self, factory):
        reactor = factory()
        y0 = reactor.get_initial_state()
        assert np.all(np.isfinite(y0))

    def test_ode_rhs_finite(self, factory):
        reactor = factory()
        y0 = reactor.get_initial_state()
        dy = reactor.ode_rhs(0.0, y0)
        assert np.all(np.isfinite(dy))


# ---------------------------------------------------------------------------
# Consecutive Reactions Utility Functions
# ---------------------------------------------------------------------------

class TestConsecutiveUtilities:
    """Tests for consecutive reaction utility functions."""

    def test_optimal_residence_time_positive(self):
        from reactor_twin.reactors.systems.consecutive import compute_optimal_residence_time

        tau = compute_optimal_residence_time(k1=1.0, k2=1.0)
        assert tau > 0
        assert np.isfinite(tau)

    def test_optimal_residence_time_formula(self):
        from reactor_twin.reactors.systems.consecutive import compute_optimal_residence_time

        k1, k2 = 4.0, 9.0
        tau = compute_optimal_residence_time(k1, k2)
        expected = 1.0 / np.sqrt(k1 * k2)
        assert tau == pytest.approx(expected)

    def test_steady_state_sum_conservation(self):
        from reactor_twin.reactors.systems.consecutive import (
            compute_steady_state_concentrations,
        )

        C_A_in = 2.0
        C_A, C_B, C_C = compute_steady_state_concentrations(C_A_in, k1=1.0, k2=0.5, tau=1.0)
        # Mass balance: C_A + C_B + C_C = C_A_in
        assert C_A + C_B + C_C == pytest.approx(C_A_in, abs=1e-10)

    def test_steady_state_concentrations_non_negative(self):
        from reactor_twin.reactors.systems.consecutive import (
            compute_steady_state_concentrations,
        )

        C_A, C_B, C_C = compute_steady_state_concentrations(3.0, k1=2.0, k2=1.0, tau=0.5)
        assert C_A >= 0
        assert C_B >= 0
        assert C_C >= 0

    def test_species_names(self):
        from reactor_twin.reactors.systems.consecutive import get_consecutive_species_names

        names = get_consecutive_species_names()
        assert len(names) == 3


# ---------------------------------------------------------------------------
# Parallel Reactions Utility Functions
# ---------------------------------------------------------------------------

class TestParallelUtilities:
    """Tests for parallel reaction utility functions."""

    def test_selectivity_positive(self):
        from reactor_twin.reactors.systems.parallel import compute_selectivity

        S = compute_selectivity(C_A=1.0, k1=1.0, k2=0.5, n1=1.0, n2=2.0)
        assert S > 0

    def test_selectivity_zero_concentration(self):
        from reactor_twin.reactors.systems.parallel import compute_selectivity

        S = compute_selectivity(C_A=0.0, k1=1.0, k2=0.5, n1=1.0, n2=2.0)
        assert S == 0.0

    def test_selectivity_ratio(self):
        from reactor_twin.reactors.systems.parallel import compute_selectivity

        # S = (k1 * C_A^n1) / (k2 * C_A^n2 + eps)
        S = compute_selectivity(C_A=1.0, k1=2.0, k2=1.0, n1=1.0, n2=1.0)
        assert S == pytest.approx(2.0, rel=1e-6)

    def test_yield_bounds(self):
        from reactor_twin.reactors.systems.parallel import compute_yield

        y = compute_yield(C_A_in=3.0, C_A=1.0, C_B=1.5)
        assert 0.0 <= y <= 1.0

    def test_yield_no_consumption(self):
        from reactor_twin.reactors.systems.parallel import compute_yield

        y = compute_yield(C_A_in=3.0, C_A=3.0, C_B=0.0)
        assert y == 0.0

    def test_yield_formula(self):
        from reactor_twin.reactors.systems.parallel import compute_yield

        # yield = C_B / (C_A_in - C_A)
        y = compute_yield(C_A_in=4.0, C_A=2.0, C_B=1.0)
        assert y == pytest.approx(0.5)

    def test_species_names(self):
        from reactor_twin.reactors.systems.parallel import get_parallel_species_names

        names = get_parallel_species_names()
        assert len(names) == 3
