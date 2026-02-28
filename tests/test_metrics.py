"""Tests for reactor_twin.utils.metrics."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from reactor_twin.utils.metrics import (
    energy_balance_error,
    gibbs_monotonicity_score,
    mass_balance_error,
    positivity_violations,
    relative_rmse,
    rmse,
    rollout_divergence,
    stoichiometric_error,
)


class TestRMSE:
    def test_identical(self):
        y = np.array([[1.0, 2.0], [3.0, 4.0]])
        assert rmse(y, y) == pytest.approx(0.0)

    def test_known_value(self):
        y_pred = np.array([1.0, 2.0, 3.0])
        y_true = np.array([1.0, 2.0, 4.0])
        expected = np.sqrt(1 / 3)
        assert rmse(y_pred, y_true) == pytest.approx(expected, rel=1e-6)

    def test_torch_input(self):
        y = torch.tensor([1.0, 2.0, 3.0])
        assert rmse(y, y) == pytest.approx(0.0)

    def test_symmetry(self):
        a = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0])
        assert rmse(a, b) == pytest.approx(rmse(b, a))


class TestRelativeRMSE:
    def test_identical(self):
        y = np.array([1.0, 2.0, 3.0])
        assert relative_rmse(y, y) == pytest.approx(0.0)

    def test_known_percentage(self):
        y_true = np.array([10.0, 10.0])
        y_pred = np.array([11.0, 9.0])
        # RMSE = 1.0, mean|y_true| = 10.0, relative = 10%
        assert relative_rmse(y_pred, y_true) == pytest.approx(10.0, rel=1e-6)


class TestMassBalanceError:
    def test_perfect_balance(self):
        # A -> B: stoichiometry = [[-1, 1]]
        S = torch.tensor([[-1.0, 1.0]])
        initial = torch.tensor([[1.0, 0.0]])
        # Change is in row space of S: [delta, -delta] -> error = 0
        state = torch.tensor([[0.5, 0.5]])
        err = mass_balance_error(state, S, initial)
        assert err.item() == pytest.approx(0.0, abs=1e-5)

    def test_violation(self):
        S = torch.tensor([[-1.0, 1.0]])
        initial = torch.tensor([[1.0, 0.0]])
        # Change [0, 1] is NOT in row space of [[-1, 1]]
        state = torch.tensor([[1.0, 1.0]])  # delta = [0, 1], not proportional to [-1,1]
        err = mass_balance_error(state, S, initial)
        assert err.item() > 0.1

    def test_batch(self):
        S = torch.tensor([[-1.0, 1.0]])
        initial = torch.tensor([[1.0, 0.0], [2.0, 0.0]])
        state = torch.tensor([[0.5, 0.5], [1.0, 1.0]])
        err = mass_balance_error(state, S, initial)
        assert err.shape == (2,)


class TestEnergyBalanceError:
    def test_shape(self):
        T = torch.tensor([[300.0, 310.0, 320.0]])
        C = torch.randn(1, 3, 2)
        dH = torch.tensor([1000.0, -500.0])
        err = energy_balance_error(T, C, dH)
        assert err.shape == (1,)


class TestPositivityViolations:
    def test_no_violations(self):
        state = torch.ones(2, 10, 3)
        count, max_v = positivity_violations(state)
        assert count == 0
        assert max_v == 0.0

    def test_with_violations(self):
        state = torch.tensor([[[1.0, -0.5, 2.0]]])
        count, max_v = positivity_violations(state)
        assert count == 1
        assert max_v == pytest.approx(0.5)

    def test_custom_threshold(self):
        state = torch.tensor([[[0.05, 0.5, 1.0]]])
        count, _ = positivity_violations(state, threshold=0.1)
        assert count == 1  # 0.05 < 0.1


class TestStoichiometricError:
    def test_consistent_trajectory(self):
        # If concentration changes are proportional to stoichiometry, error = 0
        S = torch.tensor([[-1.0, 1.0]])
        # Trajectory where dC = [-0.1, 0.1] each step (in row space of S)
        C = torch.zeros(1, 5, 2)
        for i in range(5):
            C[0, i, 0] = 1.0 - 0.1 * i
            C[0, i, 1] = 0.1 * i
        err = stoichiometric_error(C, S)
        assert err.shape == (1, 5)
        assert err[0, 0].item() == 0.0  # first step always 0
        assert err[0, 1:].max().item() < 1e-5

    def test_inconsistent_trajectory(self):
        S = torch.tensor([[-1.0, 1.0]])
        # dC = [-0.1, 0.2] — not in row space
        C = torch.tensor([[[1.0, 0.0], [0.9, 0.2], [0.8, 0.4]]])
        err = stoichiometric_error(C, S)
        assert err[0, 1:].min().item() > 0.01


class TestGibbsMonotonicity:
    def test_monotone_decreasing(self):
        G = torch.tensor([10.0, 9.0, 8.0, 7.0])
        assert gibbs_monotonicity_score(G) == pytest.approx(100.0)

    def test_monotone_increasing(self):
        G = torch.tensor([1.0, 2.0, 3.0, 4.0])
        assert gibbs_monotonicity_score(G) == pytest.approx(0.0)

    def test_mixed(self):
        G = torch.tensor([10.0, 9.0, 11.0, 8.0])
        score = gibbs_monotonicity_score(G)
        assert 0.0 < score < 100.0

    def test_single_point(self):
        assert gibbs_monotonicity_score(torch.tensor([5.0])) == 100.0


class TestRolloutDivergence:
    def test_no_divergence(self):
        short = torch.ones(10, 2)
        long = torch.ones(20, 2)
        ratio = rollout_divergence(short, long, 2.0)
        assert ratio == pytest.approx(1.0, rel=0.01)

    def test_diverging(self):
        short = torch.ones(10, 2)
        long = 3.0 * torch.ones(20, 2)
        ratio = rollout_divergence(short, long, 2.0)
        assert ratio > 2.0
