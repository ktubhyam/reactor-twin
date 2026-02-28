"""Tests for EconomicMPC and StochasticMPC."""

from __future__ import annotations

import pytest
import torch

from reactor_twin.core.neural_ode import NeuralODE
from reactor_twin.core.ode_func import MLPODEFunc
from reactor_twin.digital_twin.mpc_controller import (
    ControlConstraints,
    EconomicMPC,
    EconomicObjective,
    StochasticMPC,
)


@pytest.fixture
def simple_model():
    ode_func = MLPODEFunc(state_dim=2, hidden_dim=16, num_layers=2, input_dim=1)
    return NeuralODE(state_dim=2, ode_func=ode_func, input_dim=1)


class TestEconomicObjective:
    def test_stage_cost(self):
        obj = EconomicObjective(
            revenue_weights=torch.tensor([0.0, 10.0]),
            cost_weights=torch.tensor([1.0]),
        )
        y = torch.tensor([0.5, 2.0])
        u = torch.tensor([0.3])
        cost = obj.stage_cost(y, u)
        # revenue = 0*0.5 + 10*2 = 20; cost = 1*0.3 = 0.3; total = -20 + 0.3 = -19.7
        assert cost.item() == pytest.approx(-19.7, abs=1e-5)

    def test_trajectory_cost(self):
        obj = EconomicObjective(
            revenue_weights=torch.tensor([0.0, 1.0]),
            cost_weights=torch.tensor([0.1]),
        )
        traj = torch.ones(4, 2)  # 3-step horizon, 4 points
        controls = torch.ones(3, 1) * 0.5
        cost = obj.trajectory_cost(traj, controls)
        assert isinstance(cost, torch.Tensor)


class TestEconomicMPC:
    def test_optimize(self, simple_model):
        empc = EconomicMPC(simple_model, horizon=5, dt=0.01)
        z0 = torch.tensor([1.0, 0.5])
        result = empc.optimize(z0)
        assert "controls" in result
        assert "profit" in result
        assert result["controls"].shape == (5, 1)

    def test_step(self, simple_model):
        empc = EconomicMPC(simple_model, horizon=3, dt=0.01)
        z0 = torch.tensor([1.0, 0.5])
        u, info = empc.step(z0)
        assert u.shape == (1,)
        assert info["converged"]

    def test_with_constraints(self, simple_model):
        constraints = ControlConstraints(
            u_min=torch.tensor([-1.0]),
            u_max=torch.tensor([1.0]),
        )
        empc = EconomicMPC(simple_model, horizon=3, dt=0.01, constraints=constraints)
        z0 = torch.tensor([1.0, 0.5])
        u, info = empc.step(z0)
        assert u.min() >= -1.0
        assert u.max() <= 1.0

    def test_warm_start(self, simple_model):
        empc = EconomicMPC(simple_model, horizon=3, dt=0.01)
        z0 = torch.tensor([1.0, 0.5])
        empc.step(z0)
        _, info2 = empc.step(z0)  # uses warm start
        assert info2["converged"]


class TestStochasticMPC:
    def test_optimize(self, simple_model):
        smpc = StochasticMPC(simple_model, horizon=3, dt=0.01, n_samples=4)
        z0 = torch.tensor([1.0, 0.5])
        y_ref = torch.tensor([0.5, 1.0])
        result = smpc.optimize(z0, y_ref)
        assert "controls" in result
        assert "mean_trajectory" in result
        assert "std_trajectory" in result
        assert result["controls"].shape == (3, 1)

    def test_step(self, simple_model):
        smpc = StochasticMPC(simple_model, horizon=3, dt=0.01, n_samples=4)
        z0 = torch.tensor([1.0, 0.5])
        y_ref = torch.tensor([0.5, 1.0])
        u, info = smpc.step(z0, y_ref)
        assert u.shape == (1,)

    def test_mean_trajectory_shape(self, simple_model):
        smpc = StochasticMPC(simple_model, horizon=5, dt=0.01, n_samples=8)
        z0 = torch.tensor([1.0, 0.5])
        y_ref = torch.tensor([0.5, 1.0])
        result = smpc.optimize(z0, y_ref)
        assert result["mean_trajectory"].shape == (6, 2)  # horizon+1, state_dim
        assert result["std_trajectory"].shape == (6, 2)

    def test_with_constraints(self, simple_model):
        constraints = ControlConstraints(
            u_min=torch.tensor([-0.5]),
            u_max=torch.tensor([0.5]),
            y_min=torch.tensor([0.0, 0.0]),
        )
        smpc = StochasticMPC(
            simple_model, horizon=3, dt=0.01, n_samples=4, constraints=constraints
        )
        z0 = torch.tensor([1.0, 0.5])
        y_ref = torch.tensor([0.5, 1.0])
        u, info = smpc.step(z0, y_ref)
        assert u.min() >= -0.5
        assert u.max() <= 0.5
