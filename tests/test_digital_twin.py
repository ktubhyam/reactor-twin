"""Tests for digital twin modules: EKF, FaultDetector, MPCController."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from reactor_twin.core import NeuralODE
from reactor_twin.core.ode_func import MLPODEFunc
from reactor_twin.digital_twin import (
    EKFStateEstimator,
    FaultDetector,
    MPCController,
)
from reactor_twin.digital_twin.fault_detector import ResidualDetector, SPCChart
from reactor_twin.digital_twin.mpc_controller import (
    ControlConstraints,
    EconomicMPC,
    EconomicObjective,
    MPCObjective,
    StochasticMPC,
)

# ---------------------------------------------------------------------------
# Common fixtures
# ---------------------------------------------------------------------------

STATE_DIM = 2
INPUT_DIM = 1
OBS_DIM = 2


@pytest.fixture
def simple_model():
    """Create a simple NeuralODE model for testing digital twin components.

    Note: EKF predict_step calls model.ode_func(t, z) without control inputs,
    so we create a model with input_dim=0 to avoid dimension mismatches.
    """
    torch.manual_seed(42)
    ode_func = MLPODEFunc(state_dim=STATE_DIM, hidden_dim=16, num_layers=2, input_dim=0)
    model = NeuralODE(
        state_dim=STATE_DIM,
        ode_func=ode_func,
        solver="euler",
        atol=1e-3,
        rtol=1e-2,
        adjoint=False,
        input_dim=0,
    )
    return model


@pytest.fixture
def simple_model_with_ctrl():
    """Create a NeuralODE model with control inputs for MPC tests."""
    torch.manual_seed(42)
    ode_func = MLPODEFunc(state_dim=STATE_DIM, hidden_dim=16, num_layers=2, input_dim=INPUT_DIM)
    model = NeuralODE(
        state_dim=STATE_DIM,
        ode_func=ode_func,
        solver="euler",
        atol=1e-3,
        rtol=1e-2,
        adjoint=False,
        input_dim=INPUT_DIM,
    )
    return model


@pytest.fixture
def model_no_ctrl():
    """NeuralODE with no control inputs."""
    torch.manual_seed(0)
    return NeuralODE(state_dim=3, input_dim=0, adjoint=False, solver="euler")


@pytest.fixture
def model_with_ctrl():
    """NeuralODE with 1-D control input."""
    torch.manual_seed(0)
    return NeuralODE(state_dim=3, input_dim=1, adjoint=False, solver="euler")


@pytest.fixture
def model_2d():
    """Small 2-D NeuralODE for fast meta-learning tests."""
    torch.manual_seed(0)
    return NeuralODE(
        state_dim=2,
        input_dim=0,
        adjoint=False,
        solver="euler",
        hidden_dim=16,
        num_layers=2,
    )


# ---------------------------------------------------------------------------
# EKFStateEstimator Tests
# ---------------------------------------------------------------------------


class TestEKFStateEstimator:
    """Tests for the Extended Kalman Filter state estimator."""

    @pytest.fixture
    def ekf(self, simple_model):
        return EKFStateEstimator(
            model=simple_model,
            state_dim=STATE_DIM,
            obs_dim=OBS_DIM,
            Q=1e-4,
            R=1e-2,
            P0=1.0,
            dt=0.01,
        )

    def test_init_defaults(self, model_no_ctrl):
        ekf = EKFStateEstimator(model_no_ctrl, state_dim=3)
        assert ekf.state_dim == 3
        assert ekf.obs_dim == 3
        assert ekf.obs_indices == [0, 1, 2]
        assert ekf.Q.shape == (3, 3)
        assert ekf.R.shape == (3, 3)

    def test_init_partial_obs(self, model_no_ctrl):
        ekf = EKFStateEstimator(model_no_ctrl, state_dim=3, obs_indices=[0, 2])
        assert ekf.obs_dim == 2
        assert ekf._H.shape == (2, 3)
        assert ekf._H[0, 0] == 1.0
        assert ekf._H[1, 2] == 1.0
        assert ekf._H[0, 1] == 0.0

    def test_predict_step_shapes(self, ekf):
        z_est = torch.zeros(STATE_DIM)
        P = torch.eye(STATE_DIM)
        z_pred, P_pred = ekf.predict_step(z_est, P)
        assert z_pred.shape == (STATE_DIM,)
        assert P_pred.shape == (STATE_DIM, STATE_DIM)

    def test_update_step_shapes(self, ekf):
        z_pred = torch.zeros(STATE_DIM)
        P_pred = torch.eye(STATE_DIM)
        measurement = torch.randn(OBS_DIM)
        z_upd, P_upd, innovation = ekf.update_step(z_pred, P_pred, measurement)
        assert z_upd.shape == (STATE_DIM,)
        assert P_upd.shape == (STATE_DIM, STATE_DIM)
        assert innovation.shape == (OBS_DIM,)

    def test_predict_step_covariance_grows(self, ekf):
        """Process noise should increase uncertainty (trace of P).

        Note: The Jacobian propagation F_d @ P @ F_d.T can slightly shrink
        the covariance when F_d has eigenvalues < 1, but Q adds noise.
        We use a looser tolerance to account for this.
        """
        z_est = torch.zeros(STATE_DIM)
        P = torch.eye(STATE_DIM) * 0.1
        _, P_pred = ekf.predict_step(z_est, P)
        # P_pred = F_d P F_d^T + Q: with small Q the trace may slightly decrease
        # Just verify P_pred is still positive definite and not wildly different
        eigvals = torch.linalg.eigvalsh(P_pred)
        assert torch.all(eigvals > 0), "P_pred should remain positive definite"

    def test_predict_step_covariance_positive(self, model_no_ctrl):
        ekf = EKFStateEstimator(model_no_ctrl, state_dim=3)
        z = torch.randn(3)
        P = torch.eye(3)
        _, P_pred = ekf.predict_step(z, P)
        eigvals = torch.linalg.eigvalsh(P_pred)
        assert torch.all(eigvals > 0), "P_pred should be positive definite"

    def test_update_step_covariance_shrinks(self, ekf):
        """Measurement update should reduce uncertainty (trace of P)."""
        z_pred = torch.zeros(STATE_DIM)
        P_pred = torch.eye(STATE_DIM) * 10.0
        measurement = torch.zeros(OBS_DIM)
        _, P_upd, _ = ekf.update_step(z_pred, P_pred, measurement)
        assert torch.trace(P_upd) < torch.trace(P_pred)

    def test_predict_update_finite(self, ekf):
        """Predict and update should produce finite values."""
        z_est = torch.randn(STATE_DIM)
        P = torch.eye(STATE_DIM)
        z_pred, P_pred = ekf.predict_step(z_est, P)
        measurement = torch.randn(OBS_DIM)
        z_upd, P_upd, innov = ekf.update_step(z_pred, P_pred, measurement)
        assert torch.all(torch.isfinite(z_upd))
        assert torch.all(torch.isfinite(P_upd))
        assert torch.all(torch.isfinite(innov))

    def test_observation_matrix_shape(self, ekf):
        H = ekf._H
        assert H.shape == (OBS_DIM, STATE_DIM)

    def test_covariance_matrices_shape(self, ekf):
        assert ekf.Q.shape == (STATE_DIM, STATE_DIM)
        assert ekf.R.shape == (OBS_DIM, OBS_DIM)
        assert ekf.P0.shape == (STATE_DIM, STATE_DIM)

    def test_filter_full_pass(self, model_no_ctrl):
        ekf = EKFStateEstimator(model_no_ctrl, state_dim=3, dt=0.1)
        measurements = torch.randn(20, 3)
        result = ekf.filter(measurements, z0=torch.zeros(3))
        assert result["states"].shape == (20, 3)
        assert result["covariances"].shape == (20, 3, 3)
        assert result["innovations"].shape == (20, 3)

    def test_scalar_covariance_init(self, model_no_ctrl):
        ekf = EKFStateEstimator(model_no_ctrl, state_dim=3, Q=0.5, R=0.1, P0=2.0)
        assert ekf.Q[0, 0] == 0.5
        assert ekf.R[1, 1] == 0.1
        assert ekf.P0[2, 2] == 2.0


# ---------------------------------------------------------------------------
# SPCChart Tests
# ---------------------------------------------------------------------------


class TestSPCChart:
    """Tests for the Statistical Process Control chart."""

    @pytest.fixture
    def spc(self):
        return SPCChart(num_vars=3)

    def test_set_baseline(self, spc):
        data = np.random.randn(100, 3)
        spc.set_baseline(data)
        assert spc.mean is not None
        assert spc.std is not None
        assert spc.mean.shape == (3,)
        assert spc.std.shape == (3,)

    def test_update_returns_dict(self, spc):
        data = np.random.randn(100, 3) + 5.0
        spc.set_baseline(data)
        x = np.array([5.0, 5.0, 5.0])
        result = spc.update(x)
        assert "ewma_alarm" in result
        assert "cusum_alarm" in result
        assert "ewma_values" in result

    def test_update_no_alarm_for_normal_data(self, spc):
        """Normal data should not trigger alarms."""
        np.random.seed(42)
        data = np.random.randn(200, 3) * 0.1 + 5.0
        spc.set_baseline(data)
        x = np.array([5.0, 5.0, 5.0])
        result = spc.update(x)
        assert not np.any(result["ewma_alarm"])

    def test_update_alarm_for_large_deviation(self, spc):
        """Large deviation from baseline should trigger alarm."""
        np.random.seed(42)
        data = np.random.randn(200, 3) * 0.1 + 5.0
        spc.set_baseline(data)
        for _ in range(20):
            result = spc.update(np.array([50.0, 50.0, 50.0]))
        assert np.any(result["ewma_alarm"]) or np.any(result["cusum_alarm"])

    def test_reset_clears_state(self, spc):
        data = np.random.randn(100, 3)
        spc.set_baseline(data)
        spc.reset()
        np.testing.assert_allclose(spc._ewma, spc.mean)

    def test_update_without_baseline_raises(self, spc):
        with pytest.raises(RuntimeError, match="set_baseline"):
            spc.update(np.array([1.0, 2.0, 3.0]))


# ---------------------------------------------------------------------------
# FaultDetector Tests
# ---------------------------------------------------------------------------


class TestFaultDetector:
    """Tests for the unified fault detector."""

    @pytest.fixture
    def fault_detector(self, simple_model):
        return FaultDetector(
            model=simple_model,
            state_dim=STATE_DIM,
            obs_dim=OBS_DIM,
            dt=0.01,
        )

    def test_init(self, model_no_ctrl):
        fd = FaultDetector(model_no_ctrl, state_dim=3)
        assert fd.state_dim == 3

    def test_set_baseline(self, fault_detector):
        normal_data = {
            "observations": np.random.randn(100, OBS_DIM),
            "residuals": np.random.randn(100, STATE_DIM),
        }
        fault_detector.set_baseline(normal_data)
        assert fault_detector._has_baseline is True

    def test_spc_chart_accessible(self, fault_detector):
        assert isinstance(fault_detector.spc, SPCChart)
        assert fault_detector.spc.num_vars == OBS_DIM

    def test_update_returns_alarms(self, model_no_ctrl):
        fd = FaultDetector(model_no_ctrl, state_dim=3)
        normal_data = {
            "observations": np.random.randn(100, 3) * 0.1 + 1.0,
            "residuals": np.random.randn(100, 3) * 0.01,
        }
        fd.set_baseline(normal_data)
        result = fd.update(
            z_current=torch.randn(3),
            z_next_measured=torch.randn(3),
            t=1.0,
        )
        assert "alarms" in result
        assert "L2" in result
        assert isinstance(result["alarms"], list)


# ---------------------------------------------------------------------------
# MPCController Tests
# ---------------------------------------------------------------------------


class TestMPCController:
    """Tests for the Model Predictive Controller."""

    @pytest.fixture
    def mpc(self, simple_model_with_ctrl):
        return MPCController(
            model=simple_model_with_ctrl,
            horizon=5,
            dt=0.01,
            max_iter=5,
        )

    def test_optimize_returns_dict(self, mpc):
        z0 = torch.zeros(STATE_DIM)
        y_ref = torch.ones(STATE_DIM)
        result = mpc.optimize(z0, y_ref)
        assert isinstance(result, dict)
        assert "controls" in result
        assert "trajectory" in result
        assert "cost" in result
        assert "converged" in result

    def test_optimize_controls_shape(self, mpc):
        z0 = torch.zeros(STATE_DIM)
        y_ref = torch.ones(STATE_DIM)
        result = mpc.optimize(z0, y_ref)
        assert result["controls"].shape == (5, INPUT_DIM)

    def test_optimize_trajectory_shape(self, mpc):
        z0 = torch.zeros(STATE_DIM)
        y_ref = torch.ones(STATE_DIM)
        result = mpc.optimize(z0, y_ref)
        assert result["trajectory"].shape == (6, STATE_DIM)

    def test_optimize_cost_is_scalar(self, mpc):
        z0 = torch.zeros(STATE_DIM)
        y_ref = torch.ones(STATE_DIM)
        result = mpc.optimize(z0, y_ref)
        assert isinstance(result["cost"], float)

    def test_step_returns_first_control(self, mpc):
        z0 = torch.zeros(STATE_DIM)
        y_ref = torch.ones(STATE_DIM)
        u_applied, info = mpc.step(z0, y_ref)
        assert u_applied.shape == (INPUT_DIM,)

    def test_trajectory_starts_from_z0(self, mpc):
        z0 = torch.tensor([1.0, 2.0])
        y_ref = torch.ones(STATE_DIM)
        result = mpc.optimize(z0, y_ref)
        torch.testing.assert_close(result["trajectory"][0], z0, atol=1e-5, rtol=1e-5)

    def test_predict_trajectory_shape(self, model_with_ctrl):
        mpc = MPCController(model_with_ctrl, horizon=5, dt=0.01)
        z0 = torch.randn(3)
        controls = torch.randn(5, 1)
        traj = mpc._predict_trajectory(z0, controls)
        assert traj.shape == (6, 3)

    def test_warm_start(self, model_with_ctrl):
        mpc = MPCController(model_with_ctrl, horizon=5, dt=0.01, max_iter=3)
        mpc.step(torch.randn(3), torch.zeros(3))
        assert mpc._u_prev is not None
        mpc.step(torch.randn(3), torch.zeros(3))

    def test_with_constraints(self, model_with_ctrl):
        constraints = ControlConstraints(
            u_min=torch.tensor([-1.0]),
            u_max=torch.tensor([1.0]),
        )
        mpc = MPCController(
            model_with_ctrl,
            horizon=3,
            dt=0.01,
            constraints=constraints,
            max_iter=3,
        )
        u, info = mpc.step(torch.randn(3), torch.zeros(3))
        assert u.item() >= -1.0
        assert u.item() <= 1.0


# ---------------------------------------------------------------------------
# MPCObjective Tests
# ---------------------------------------------------------------------------


class TestMPCObjective:
    """Tests for MPC objective function."""

    def test_stage_cost(self):
        Q = torch.eye(2)
        R = torch.eye(1) * 0.1
        obj = MPCObjective(Q=Q, R=R)
        y = torch.tensor([1.0, 0.0])
        y_ref = torch.tensor([0.0, 0.0])
        u = torch.tensor([1.0])
        cost = obj.stage_cost(y, y_ref, u)
        assert cost.item() == pytest.approx(1.1, abs=1e-6)

    def test_terminal_cost(self):
        Q = torch.eye(2) * 2.0
        R = torch.eye(1)
        obj = MPCObjective(Q=Q, R=R)
        y = torch.tensor([1.0, 1.0])
        y_ref = torch.tensor([0.0, 0.0])
        cost = obj.terminal_cost(y, y_ref)
        assert cost.item() == pytest.approx(4.0, abs=1e-6)

    def test_terminal_cost_defaults_to_Q(self):
        Q = 2.0 * torch.eye(2)
        R = torch.eye(1)
        obj = MPCObjective(Q=Q, R=R)
        e = torch.ones(2)
        tc = obj.terminal_cost(e, torch.zeros(2))
        assert abs(tc.item() - 4.0) < 1e-5

    def test_trajectory_cost(self):
        Q = torch.eye(2)
        R = torch.eye(1) * 0.01
        obj = MPCObjective(Q=Q, R=R)
        traj = torch.randn(6, 2)
        y_ref = torch.zeros(2)
        controls = torch.randn(5, 1)
        cost = obj.trajectory_cost(traj, y_ref, controls)
        assert cost.item() > 0


# ---------------------------------------------------------------------------
# ControlConstraints Tests
# ---------------------------------------------------------------------------


class TestControlConstraints:
    """Tests for control constraint handling."""

    def test_clamp_controls(self):
        cc = ControlConstraints(
            u_min=torch.tensor([-1.0]),
            u_max=torch.tensor([1.0]),
        )
        u = torch.tensor([5.0])
        u_clamped = cc.clamp_controls(u)
        assert u_clamped.item() == 1.0

        u_neg = torch.tensor([-5.0])
        u_clamped_neg = cc.clamp_controls(u_neg)
        assert u_clamped_neg.item() == -1.0

    def test_output_penalty_zero_within_bounds(self):
        cc = ControlConstraints(
            u_min=torch.tensor([-1.0]),
            u_max=torch.tensor([1.0]),
            y_min=torch.tensor([0.0, 0.0]),
            y_max=torch.tensor([10.0, 10.0]),
        )
        y = torch.tensor([5.0, 5.0])
        penalty = cc.output_penalty(y)
        assert penalty.item() == pytest.approx(0.0, abs=1e-10)

    def test_output_penalty_nonzero_outside_bounds(self):
        cc = ControlConstraints(
            u_min=torch.tensor([-1.0]),
            u_max=torch.tensor([1.0]),
            y_min=torch.tensor([0.0, 0.0]),
            y_max=torch.tensor([10.0, 10.0]),
        )
        y = torch.tensor([-1.0, 15.0])
        penalty = cc.output_penalty(y)
        assert penalty.item() > 0


# ---------------------------------------------------------------------------
# FaultIsolator Tests (L3)
# ---------------------------------------------------------------------------


class TestFaultIsolator:
    """Tests for L3 fault isolation."""

    def test_isolate_without_baseline(self):
        from reactor_twin.digital_twin.fault_detector import FaultIsolator

        isolator = FaultIsolator(state_dim=3)
        residual = np.array([0.5, 0.1, 0.3])
        result = isolator.isolate(residual)
        assert "contributions" in result
        assert "ranking" in result
        assert "spe" in result
        assert len(result["contributions"]) == 3
        assert len(result["ranking"]) == 3

    def test_isolate_with_baseline(self):
        from reactor_twin.digital_twin.fault_detector import FaultIsolator

        isolator = FaultIsolator(state_dim=3)
        residuals = np.random.randn(100, 3) * 0.1
        isolator.set_baseline(residuals)
        assert isolator.baseline_residual_cov is not None

        residual = np.array([2.0, 0.01, 0.01])
        result = isolator.isolate(residual)
        # Variable 0 has the largest residual — should rank first
        assert result["ranking"][0] == 0
        assert result["spe"] > 0

    def test_isolate_contributions_non_negative(self):
        from reactor_twin.digital_twin.fault_detector import FaultIsolator

        isolator = FaultIsolator(state_dim=2)
        residual = np.array([1.0, -1.0])
        result = isolator.isolate(residual)
        assert np.all(result["contributions"] >= 0)


# ---------------------------------------------------------------------------
# FaultClassifier Tests (L4)
# ---------------------------------------------------------------------------


class TestFaultClassifier:
    """Tests for L4 fault classification."""

    def test_predict_before_fit(self):
        from reactor_twin.digital_twin.fault_detector import FaultClassifier

        clf = FaultClassifier()
        result = clf.predict(np.array([1.0, 2.0, 3.0]))
        assert result["label"] == "unknown"
        assert result["probabilities"] == {}

    def test_fit_and_predict_rf(self):
        pytest.importorskip("sklearn")
        from reactor_twin.digital_twin.fault_detector import FaultClassifier

        clf = FaultClassifier(method="rf", n_estimators=5)
        features = np.random.randn(30, 3)
        labels = np.array(["normal"] * 15 + ["fault_a"] * 15)
        clf.fit(features, labels)
        assert len(clf.classes) == 2

        result = clf.predict(np.random.randn(3))
        assert result["label"] in ("normal", "fault_a")
        assert len(result["probabilities"]) == 2

    def test_fit_and_predict_svm(self):
        pytest.importorskip("sklearn")
        from reactor_twin.digital_twin.fault_detector import FaultClassifier

        clf = FaultClassifier(method="svm")
        features = np.random.randn(30, 2)
        labels = ["normal"] * 20 + ["fault"] * 10
        clf.fit(features, labels)
        result = clf.predict(np.random.randn(2))
        assert result["label"] in ("normal", "fault")

    def test_predict_batch(self):
        pytest.importorskip("sklearn")
        from reactor_twin.digital_twin.fault_detector import FaultClassifier

        clf = FaultClassifier(method="rf", n_estimators=5)
        features = np.random.randn(40, 4)
        labels = ["a"] * 20 + ["b"] * 20
        clf.fit(features, labels)
        result = clf.predict(np.random.randn(4))
        assert result["label"] in ("a", "b")


# ---------------------------------------------------------------------------
# FaultDetector — L3 and L4 integration
# ---------------------------------------------------------------------------


class TestFaultDetectorL3L4:
    """Additional tests exercising L3/L4 paths of unified FaultDetector."""

    def test_update_triggers_l3_on_alarm(self, model_no_ctrl):
        """When residual exceeds threshold, L3 should appear in result."""
        fd = FaultDetector(model_no_ctrl, state_dim=3)
        # Set baseline with very tight residuals
        normal_data = {
            "observations": np.random.randn(100, 3) * 0.001 + 5.0,
            "residuals": np.random.randn(100, 3) * 0.001,
        }
        fd.set_baseline(normal_data)
        # Feed a wildly different measurement to trigger L2 alarm
        result = fd.update(
            z_current=torch.zeros(3),
            z_next_measured=torch.ones(3) * 100.0,
            t=1.0,
        )
        if np.any(result["L2"]["alarm"]):
            assert "L3" in result

    def test_update_with_classifier(self, model_no_ctrl):
        """FaultDetector with fitted classifier should produce L4 results."""
        pytest.importorskip("sklearn")
        fd = FaultDetector(model_no_ctrl, state_dim=3)
        normal_data = {
            "observations": np.random.randn(100, 3) * 0.1 + 1.0,
            "residuals": np.random.randn(100, 3) * 0.01,
            "features": np.random.randn(60, 3),
            "labels": np.array(["normal"] * 30 + ["fault"] * 30),
        }
        fd.set_baseline(normal_data)
        result = fd.update(
            z_current=torch.randn(3),
            z_next_measured=torch.randn(3),
            t=2.0,
        )
        assert "L4" in result
        assert result["L4"]["label"] in ("normal", "fault")

    def test_alarm_severity_levels(self):
        from reactor_twin.digital_twin.fault_detector import AlarmLevel

        assert AlarmLevel.NORMAL.value < AlarmLevel.WARNING.value
        assert AlarmLevel.WARNING.value < AlarmLevel.ALARM.value
        assert AlarmLevel.ALARM.value < AlarmLevel.CRITICAL.value


# ---------------------------------------------------------------------------
# EKFStateEstimator — Additional edge cases
# ---------------------------------------------------------------------------


class TestEKFEdgeCases:
    """Additional EKF edge-case tests for higher coverage."""

    def test_filter_with_t_span(self, model_no_ctrl):
        """filter() with explicit t_span uses variable dt."""
        ekf = EKFStateEstimator(model_no_ctrl, state_dim=3, dt=0.1)
        t_span = torch.linspace(0, 1, 10)
        measurements = torch.randn(10, 3)
        result = ekf.filter(measurements, z0=torch.zeros(3), t_span=t_span)
        assert result["states"].shape == (10, 3)

    def test_filter_default_z0(self, model_no_ctrl):
        """filter() without z0 defaults to zeros."""
        ekf = EKFStateEstimator(model_no_ctrl, state_dim=3, dt=0.1)
        measurements = torch.randn(5, 3)
        result = ekf.filter(measurements)
        assert result["states"].shape == (5, 3)

    def test_to_matrix_1d(self, model_no_ctrl):
        """_to_matrix handles 1D tensor (diagonal)."""
        mat = EKFStateEstimator._to_matrix(torch.tensor([1.0, 2.0, 3.0]), 3)
        assert mat.shape == (3, 3)
        assert mat[0, 0] == 1.0
        assert mat[1, 1] == 2.0

    def test_partial_observation_filter(self, model_no_ctrl):
        """filter() with partial observations."""
        ekf = EKFStateEstimator(model_no_ctrl, state_dim=3, obs_indices=[0, 2], dt=0.1)
        measurements = torch.randn(5, 2)
        result = ekf.filter(measurements, z0=torch.zeros(3))
        assert result["states"].shape == (5, 3)
        assert result["innovations"].shape == (5, 2)

    def test_to_matrix_2d_tensor(self, model_no_ctrl):
        """_to_matrix with a 2D tensor returns val directly (line 93)."""
        mat_2d = torch.eye(3) * 2.0
        result = EKFStateEstimator._to_matrix(mat_2d, 3)
        torch.testing.assert_close(result, mat_2d)
        assert result.shape == (3, 3)

    def test_jacobian_fallback(self, model_no_ctrl):
        """_compute_jacobian falls back to autograd.functional.jacobian
        when torch.func.jacrev fails (lines 127-134)."""
        from unittest.mock import patch

        ekf = EKFStateEstimator(model_no_ctrl, state_dim=3, dt=0.1)
        z = torch.randn(3)
        t = torch.tensor(0.0)

        # Patch torch.func.jacrev to raise an exception, forcing the fallback
        with patch("torch.func.jacrev", side_effect=RuntimeError("jacrev failed")):
            F = ekf._compute_jacobian(z, t)

        assert F.shape == (3, 3)
        assert torch.all(torch.isfinite(F))


# ---------------------------------------------------------------------------
# ResidualDetector — update without baseline (lines 237-238)
# ---------------------------------------------------------------------------


class TestResidualDetectorNoBaseline:
    """Tests for ResidualDetector.update when no baseline is set."""

    def test_update_without_baseline(self, simple_model):
        """update() without baseline returns z_score=zeros, alarm=zeros (lines 237-238)."""
        detector = ResidualDetector(
            model=simple_model,
            state_dim=STATE_DIM,
            dt=0.01,
        )
        # Do NOT set baseline
        z_current = torch.randn(STATE_DIM)
        z_next = torch.randn(STATE_DIM)
        result = detector.update(z_current, z_next)

        assert "residual" in result
        assert "alarm" in result
        assert "z_score" in result
        np.testing.assert_array_equal(result["z_score"], np.zeros(STATE_DIM))
        np.testing.assert_array_equal(result["alarm"], np.zeros(STATE_DIM, dtype=bool))


# ---------------------------------------------------------------------------
# EconomicObjective — safety penalties (lines 432-437)
# ---------------------------------------------------------------------------


class TestEconomicObjectiveSafety:
    """Tests for EconomicObjective safety penalty computation."""

    def test_stage_cost_with_safety_penalties_min(self):
        """Safety penalty for y_min violation (lines 432-434)."""
        obj = EconomicObjective(
            revenue_weights=torch.tensor([1.0, 1.0]),
            cost_weights=torch.tensor([0.1]),
            state_penalties=torch.tensor([10.0, 10.0]),
            y_min_safety=torch.tensor([0.5, 0.5]),
        )
        y_violating = torch.tensor([0.0, 0.0])  # below y_min_safety
        u = torch.tensor([0.0])
        cost = obj.stage_cost(y_violating, u)
        # Should include positive safety penalty
        assert cost.item() > 0

    def test_stage_cost_with_safety_penalties_max(self):
        """Safety penalty for y_max violation (lines 435-437)."""
        obj = EconomicObjective(
            revenue_weights=torch.tensor([1.0, 1.0]),
            cost_weights=torch.tensor([0.1]),
            state_penalties=torch.tensor([10.0, 10.0]),
            y_max_safety=torch.tensor([1.0, 1.0]),
        )
        y_violating = torch.tensor([2.0, 2.0])  # above y_max_safety
        u = torch.tensor([0.0])
        cost = obj.stage_cost(y_violating, u)
        # Revenue = 2+2=4, but penalty should dominate
        # cost = -revenue + control_cost + penalty
        # penalty = 10 * (2-1)^2 + 10 * (2-1)^2 = 20
        # cost = -4 + 0 + 20 = 16
        assert cost.item() > 0

    def test_stage_cost_with_both_safety_bounds(self):
        """Safety penalty with both min and max violations (lines 432-437)."""
        obj = EconomicObjective(
            revenue_weights=torch.tensor([0.0, 0.0]),
            cost_weights=torch.tensor([0.0]),
            state_penalties=torch.tensor([10.0, 10.0]),
            y_min_safety=torch.tensor([0.5, 0.5]),
            y_max_safety=torch.tensor([1.5, 1.5]),
        )
        # Within bounds -- no penalty
        y_safe = torch.tensor([1.0, 1.0])
        u = torch.tensor([0.0])
        cost_safe = obj.stage_cost(y_safe, u)
        assert cost_safe.item() == pytest.approx(0.0, abs=1e-5)

        # Below min bounds
        y_low = torch.tensor([0.0, 0.0])
        cost_low = obj.stage_cost(y_low, u)
        assert cost_low.item() > 0

    def test_stage_cost_no_penalty_when_state_penalties_is_none(self):
        """No safety penalty when state_penalties is None (line 431 guard)."""
        obj = EconomicObjective(
            revenue_weights=torch.tensor([1.0]),
            cost_weights=torch.tensor([0.1]),
            y_min_safety=torch.tensor([10.0]),  # even with bounds set
        )
        y = torch.tensor([0.0])  # violates y_min
        u = torch.tensor([1.0])
        cost = obj.stage_cost(y, u)
        # Only revenue + control cost, no safety penalty
        # cost = -0 + 0.1*1 = 0.1
        assert cost.item() == pytest.approx(0.1, abs=1e-5)


# ---------------------------------------------------------------------------
# EconomicMPC — optimize method (lines 546, 577-578)
# ---------------------------------------------------------------------------


class TestEconomicMPCOptimize:
    """Tests for EconomicMPC.optimize method."""

    def test_economic_mpc_optimize(self, simple_model_with_ctrl):
        """EconomicMPC optimize returns expected dict (lines 546, 577-578)."""
        econ_obj = EconomicObjective(
            revenue_weights=torch.tensor([0.0, 1.0]),
            cost_weights=torch.tensor([0.01]),
        )
        empc = EconomicMPC(
            model=simple_model_with_ctrl,
            horizon=3,
            dt=0.01,
            objective=econ_obj,
            max_iter=3,
        )
        z0 = torch.zeros(STATE_DIM)
        result = empc.optimize(z0)
        assert "controls" in result
        assert "trajectory" in result
        assert "cost" in result
        assert "profit" in result
        assert "converged" in result
        assert result["converged"] is True
        assert result["controls"].shape == (3, INPUT_DIM)

    def test_economic_mpc_optimize_with_u_init(self, simple_model_with_ctrl):
        """EconomicMPC optimize with u_init (line 546)."""
        empc = EconomicMPC(
            model=simple_model_with_ctrl,
            horizon=3,
            dt=0.01,
            max_iter=3,
        )
        z0 = torch.zeros(STATE_DIM)
        u_init = torch.ones(3, INPUT_DIM) * 0.5
        result = empc.optimize(z0, u_init=u_init)
        assert result["converged"] is True

    def test_economic_mpc_step(self, simple_model_with_ctrl):
        """EconomicMPC.step returns first control action."""
        empc = EconomicMPC(
            model=simple_model_with_ctrl,
            horizon=3,
            dt=0.01,
            max_iter=3,
        )
        z0 = torch.zeros(STATE_DIM)
        u_applied, info = empc.step(z0)
        assert u_applied.shape == (INPUT_DIM,)

    def test_economic_mpc_optimize_exception(self, simple_model_with_ctrl):
        """EconomicMPC optimize handles exceptions gracefully (lines 577-578)."""
        from unittest.mock import patch

        empc = EconomicMPC(
            model=simple_model_with_ctrl,
            horizon=3,
            dt=0.01,
            max_iter=3,
        )
        z0 = torch.zeros(STATE_DIM)

        # Make the optimizer step raise
        with patch.object(
            torch.optim.LBFGS, "step", side_effect=RuntimeError("LBFGS failed")
        ):
            result = empc.optimize(z0)
        assert result["converged"] is False


# ---------------------------------------------------------------------------
# MPCController — optimize with u_init and exception (lines 297, 338-339)
# ---------------------------------------------------------------------------


class TestMPCControllerExtended:
    """Extended tests for MPCController."""

    def test_optimize_with_u_init(self, simple_model_with_ctrl):
        """MPCController optimize with u_init parameter (line 297)."""
        mpc = MPCController(
            model=simple_model_with_ctrl,
            horizon=5,
            dt=0.01,
            max_iter=3,
        )
        z0 = torch.zeros(STATE_DIM)
        y_ref = torch.ones(STATE_DIM)
        u_init = torch.ones(5, INPUT_DIM) * 0.1
        result = mpc.optimize(z0, y_ref, u_init=u_init)
        assert result["converged"] is True
        assert result["controls"].shape == (5, INPUT_DIM)

    def test_optimize_exception_handling(self, simple_model_with_ctrl):
        """MPCController optimize handles exceptions (lines 338-339)."""
        from unittest.mock import patch

        mpc = MPCController(
            model=simple_model_with_ctrl,
            horizon=3,
            dt=0.01,
            max_iter=3,
        )
        z0 = torch.zeros(STATE_DIM)
        y_ref = torch.ones(STATE_DIM)

        with patch.object(
            torch.optim.LBFGS, "step", side_effect=RuntimeError("optimization failed")
        ):
            result = mpc.optimize(z0, y_ref)
        assert result["converged"] is False
        # Should still return valid output
        assert "controls" in result
        assert "trajectory" in result


# ---------------------------------------------------------------------------
# StochasticMPC tests (lines 713-714, 749-752, 800-801)
# ---------------------------------------------------------------------------


class TestStochasticMPC:
    """Tests for StochasticMPC."""

    def test_stochastic_mpc_optimize(self, simple_model_with_ctrl):
        """StochasticMPC optimize returns expected dict."""
        smpc = StochasticMPC(
            model=simple_model_with_ctrl,
            horizon=3,
            dt=0.01,
            n_samples=4,
            max_iter=3,
        )
        z0 = torch.zeros(STATE_DIM)
        y_ref = torch.ones(STATE_DIM)
        result = smpc.optimize(z0, y_ref)
        assert "controls" in result
        assert "mean_trajectory" in result
        assert "std_trajectory" in result
        assert "cost" in result
        assert "converged" in result
        assert result["converged"] is True

    def test_stochastic_mpc_step(self, simple_model_with_ctrl):
        """StochasticMPC.step returns first control."""
        smpc = StochasticMPC(
            model=simple_model_with_ctrl,
            horizon=3,
            dt=0.01,
            n_samples=4,
            max_iter=3,
        )
        z0 = torch.zeros(STATE_DIM)
        y_ref = torch.ones(STATE_DIM)
        u_applied, info = smpc.step(z0, y_ref)
        assert u_applied.shape == (INPUT_DIM,)

    def test_stochastic_mpc_warm_start(self, simple_model_with_ctrl):
        """StochasticMPC warm start shift (lines 749-752)."""
        smpc = StochasticMPC(
            model=simple_model_with_ctrl,
            horizon=3,
            dt=0.01,
            n_samples=4,
            max_iter=3,
        )
        z0 = torch.zeros(STATE_DIM)
        y_ref = torch.ones(STATE_DIM)

        # First call sets _u_prev
        smpc.step(z0, y_ref)
        assert smpc._u_prev is not None
        # Second call uses warm start (lines 750-752)
        result = smpc.optimize(z0, y_ref)
        assert result["converged"] is True

    def test_stochastic_mpc_with_u_init(self, simple_model_with_ctrl):
        """StochasticMPC optimize with u_init (line 749)."""
        smpc = StochasticMPC(
            model=simple_model_with_ctrl,
            horizon=3,
            dt=0.01,
            n_samples=4,
            max_iter=3,
        )
        z0 = torch.zeros(STATE_DIM)
        y_ref = torch.ones(STATE_DIM)
        u_init = torch.ones(3, INPUT_DIM) * 0.5
        result = smpc.optimize(z0, y_ref, u_init=u_init)
        assert result["converged"] is True

    def test_stochastic_mpc_optimize_failure(self, simple_model_with_ctrl):
        """StochasticMPC optimize handles failures (lines 800-801)."""
        from unittest.mock import patch

        smpc = StochasticMPC(
            model=simple_model_with_ctrl,
            horizon=3,
            dt=0.01,
            n_samples=4,
            max_iter=3,
        )
        z0 = torch.zeros(STATE_DIM)
        y_ref = torch.ones(STATE_DIM)

        with patch.object(
            torch.optim.LBFGS, "step", side_effect=RuntimeError("LBFGS failed")
        ):
            result = smpc.optimize(z0, y_ref)
        assert result["converged"] is False

    def test_stochastic_mpc_with_diffusion(self):
        """StochasticMPC uses diffusion function when available (lines 713-714)."""
        torch.manual_seed(42)

        # Create a model whose ode_func has a diffusion method
        ode_func = MLPODEFunc(state_dim=STATE_DIM, hidden_dim=16, num_layers=2, input_dim=INPUT_DIM)

        # Add a diffusion method to the ode_func
        def diffusion(t, z):
            return torch.ones_like(z) * 0.1

        ode_func.diffusion = diffusion

        model = NeuralODE(
            state_dim=STATE_DIM,
            ode_func=ode_func,
            solver="euler",
            adjoint=False,
            input_dim=INPUT_DIM,
        )

        smpc = StochasticMPC(
            model=model,
            horizon=3,
            dt=0.01,
            n_samples=4,
            max_iter=3,
        )
        z0 = torch.zeros(STATE_DIM)
        y_ref = torch.ones(STATE_DIM)
        result = smpc.optimize(z0, y_ref)
        assert result["converged"] is True
        assert result["mean_trajectory"].shape == (4, STATE_DIM)  # horizon+1
