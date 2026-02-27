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
        lambda t, y: reactor.ode_rhs(t, y),
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
