"""Tests for reactor_twin.utils.numerical."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from reactor_twin.utils.numerical import (
    adaptive_step_size,
    detect_stiffness,
    finite_difference_jacobian,
    integrate_ode,
    interpolate_trajectory,
    runge_kutta_4,
)


class TestIntegrateODE:
    def test_exponential_decay(self):
        """dy/dt = -y, y(0) = 1 -> y(t) = exp(-t)."""
        def rhs(t, y):
            return -y
        t_eval = np.linspace(0, 2, 50)
        result = integrate_ode(rhs, (0.0, 2.0), np.array([1.0]), t_eval=t_eval)
        expected = np.exp(-t_eval)
        np.testing.assert_allclose(result[:, 0], expected, atol=1e-3)

    def test_shape(self):
        def rhs(t, y):
            return np.array([-y[0], y[1]])
        t_eval = np.linspace(0, 1, 20)
        result = integrate_ode(rhs, (0.0, 1.0), np.array([1.0, 0.5]), t_eval=t_eval)
        assert result.shape == (20, 2)

    def test_stiff_bdf(self):
        """BDF method for stiff ODE."""
        def rhs(t, y):
            return -1000 * y
        t_eval = np.linspace(0, 0.01, 20)
        result = integrate_ode(
            rhs, (0.0, 0.01), np.array([1.0]),
            t_eval=t_eval, method="BDF"
        )
        assert result[-1, 0] < 1e-3


class TestFiniteDifferenceJacobian:
    def test_linear(self):
        """Jacobian of f(x) = Ax is A."""
        A = torch.tensor([[2.0, 1.0], [0.0, 3.0]])

        def f(x):
            return A @ x

        x = torch.tensor([1.0, 1.0])
        J = finite_difference_jacobian(f, x)
        torch.testing.assert_close(J, A, atol=0.1, rtol=0.1)

    def test_scalar(self):
        """Jacobian of f(x) = x^2 is 2x."""
        def f(x):
            return x**2

        x = torch.tensor([3.0], dtype=torch.float64)
        J = finite_difference_jacobian(f, x)
        assert J.item() == pytest.approx(6.0, abs=1e-4)


class TestDetectStiffness:
    def test_stiff_system(self):
        eigs = np.array([-1000.0, -0.01])
        is_stiff, ratio = detect_stiffness(eigs)
        assert is_stiff is True
        assert ratio == pytest.approx(100000.0)

    def test_non_stiff(self):
        eigs = np.array([-2.0, -1.0])
        is_stiff, ratio = detect_stiffness(eigs)
        assert is_stiff is False
        assert ratio == pytest.approx(2.0)

    def test_torch_input(self):
        eigs = torch.tensor([-500.0, -0.1])
        is_stiff, ratio = detect_stiffness(eigs)
        assert is_stiff is True

    def test_zero_eigenvalue(self):
        eigs = np.array([0.0, -1.0])
        is_stiff, ratio = detect_stiffness(eigs)
        assert ratio == float("inf")


class TestRungeKutta4:
    def test_exponential_decay(self):
        """dy/dt = -y, y(0) = 1 -> y(dt) ≈ exp(-dt)."""
        def f(t, y):
            return -y

        y0 = torch.tensor([[1.0]])
        dt = 0.01
        y1 = runge_kutta_4(f, 0.0, y0, dt)
        expected = np.exp(-dt)
        assert y1.item() == pytest.approx(expected, abs=1e-8)

    def test_batch(self):
        def f(t, y):
            return -y

        y0 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        y1 = runge_kutta_4(f, 0.0, y0, 0.01)
        assert y1.shape == (2, 2)


class TestAdaptiveStepSize:
    def test_small_error_increases_step(self):
        error = torch.tensor([[1e-10]])
        dt_new = adaptive_step_size(error, dt=0.01)
        assert dt_new > 0.01

    def test_large_error_decreases_step(self):
        error = torch.tensor([[1.0]])
        dt_new = adaptive_step_size(error, dt=0.01)
        assert dt_new < 0.01

    def test_growth_clamped(self):
        error = torch.tensor([[1e-15]])
        dt_new = adaptive_step_size(error, dt=0.01)
        assert dt_new <= 0.05  # max 5x growth


class TestInterpolateTrajectory:
    def test_linear(self):
        t_c = torch.tensor([0.0, 1.0, 2.0])
        y_c = torch.tensor([[0.0], [1.0], [2.0]])
        t_f = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0])
        y_f = interpolate_trajectory(t_c, y_c, t_f, method="linear")
        expected = torch.tensor([[0.0], [0.5], [1.0], [1.5], [2.0]])
        torch.testing.assert_close(y_f, expected, atol=1e-5, rtol=1e-5)

    def test_cubic(self):
        t_c = torch.linspace(0, 2 * np.pi, 20)
        y_c = torch.sin(t_c).unsqueeze(-1)
        t_f = torch.linspace(0, 2 * np.pi, 100)
        y_f = interpolate_trajectory(t_c, y_c, t_f, method="cubic")
        expected = torch.sin(t_f).unsqueeze(-1)
        # Cubic interpolation should be very accurate
        assert (y_f - expected).abs().max().item() < 0.01

    def test_multidim(self):
        t_c = torch.tensor([0.0, 1.0, 2.0])
        y_c = torch.tensor([[0.0, 10.0], [1.0, 20.0], [2.0, 30.0]])
        t_f = torch.tensor([0.5, 1.5])
        y_f = interpolate_trajectory(t_c, y_c, t_f, method="linear")
        assert y_f.shape == (2, 2)
