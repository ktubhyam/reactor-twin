"""Tests for real-data benchmark reactors (Williams-Otto and Penicillin)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Add repo root to path so benchmarks/ is importable
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


class TestWilliamsOtto:
    def test_reactor_ode_shape(self):
        from benchmarks.real_data.williams_otto import WilliamsOttoReactor

        reactor = WilliamsOttoReactor()
        y0 = reactor.get_initial_state()
        dy = reactor.ode_rhs(0.0, y0)
        assert dy.shape == (6,)

    def test_initial_state(self):
        from benchmarks.real_data.williams_otto import WilliamsOttoReactor

        reactor = WilliamsOttoReactor()
        y0 = reactor.get_initial_state()
        assert y0.shape == (6,)
        assert y0[5] == 350.0  # initial temperature

    def test_generate_data_shapes(self):
        from benchmarks.real_data.williams_otto import generate_synthetic_data

        data = generate_synthetic_data(n_trajectories=5, n_points=20, t_end=10.0)
        assert data["t"].shape == (20,)
        assert data["y0"].shape[1] == 6
        assert data["trajectories"].shape[1] == 20
        assert data["trajectories"].shape[2] == 6

    def test_state_labels(self):
        from benchmarks.real_data.williams_otto import WilliamsOttoReactor

        reactor = WilliamsOttoReactor()
        labels = reactor.get_state_labels()
        assert len(labels) == 6
        assert "C_A" in labels

    def test_with_controls(self):
        from benchmarks.real_data.williams_otto import WilliamsOttoReactor

        reactor = WilliamsOttoReactor()
        y0 = reactor.get_initial_state()
        u = np.array([12.0, 8.0, 340.0])
        dy = reactor.ode_rhs(0.0, y0, u=u)
        assert dy.shape == (6,)


class TestPenicillin:
    def test_reactor_ode_shape(self):
        from benchmarks.real_data.penicillin import PenicillinReactor

        reactor = PenicillinReactor()
        y0 = reactor.get_initial_state()
        dy = reactor.ode_rhs(0.0, y0)
        assert dy.shape == (4,)

    def test_initial_state(self):
        from benchmarks.real_data.penicillin import PenicillinReactor

        reactor = PenicillinReactor()
        y0 = reactor.get_initial_state()
        assert y0.shape == (4,)
        assert y0[0] == 1.5  # initial biomass

    def test_generate_data_shapes(self):
        from benchmarks.real_data.penicillin import generate_synthetic_data

        data = generate_synthetic_data(n_trajectories=5, n_points=20, t_end=50.0)
        assert data["t"].shape == (20,)
        assert data["y0"].shape[1] == 4
        assert data["trajectories"].shape[1] == 20
        assert data["trajectories"].shape[2] == 4

    def test_volume_increases(self):
        """In a fed-batch, volume should increase over time."""
        from benchmarks.real_data.penicillin import generate_synthetic_data

        data = generate_synthetic_data(n_trajectories=1, n_points=50, t_end=100.0)
        V = data["trajectories"][0, :, 3]
        assert V[-1] > V[0]  # volume increases due to feed

    def test_with_controls(self):
        from benchmarks.real_data.penicillin import PenicillinReactor

        reactor = PenicillinReactor()
        y0 = reactor.get_initial_state()
        u = np.array([0.08])
        dy = reactor.ode_rhs(0.0, y0, u=u)
        assert dy.shape == (4,)
