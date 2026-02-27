"""Tests for Phase 5 Digital Twin modules.

Covers: EKFStateEstimator, FaultDetector (L1-L4), MPCController,
OnlineAdapter, and ReptileMetaLearner.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from reactor_twin.core.neural_ode import NeuralODE
from reactor_twin.digital_twin.state_estimator import EKFStateEstimator
from reactor_twin.digital_twin.fault_detector import (
    AlarmLevel,
    FaultClassifier,
    FaultDetector,
    FaultIsolator,
    ResidualDetector,
    SPCChart,
)
from reactor_twin.digital_twin.mpc_controller import (
    ControlConstraints,
    MPCController,
    MPCObjective,
)
from reactor_twin.digital_twin.online_adapter import (
    ElasticWeightConsolidation,
    OnlineAdapter,
    ReplayBuffer,
)
from reactor_twin.digital_twin.meta_learner import ReptileMetaLearner


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def model_no_ctrl() -> NeuralODE:
    """NeuralODE with no control inputs."""
    torch.manual_seed(0)
    return NeuralODE(state_dim=3, input_dim=0, adjoint=False, solver="euler")


@pytest.fixture
def model_with_ctrl() -> NeuralODE:
    """NeuralODE with 1-D control input."""
    torch.manual_seed(0)
    return NeuralODE(state_dim=3, input_dim=1, adjoint=False, solver="euler")


@pytest.fixture
def model_2d() -> NeuralODE:
    """Small 2-D NeuralODE for fast meta-learning tests."""
    torch.manual_seed(0)
    return NeuralODE(state_dim=2, input_dim=0, adjoint=False, solver="euler",
                     hidden_dim=16, num_layers=2)


# =====================================================================
# EKF State Estimator
# =====================================================================

class TestEKFStateEstimator:
    """Tests for EKFStateEstimator."""

    def test_init_defaults(self, model_no_ctrl: NeuralODE) -> None:
        ekf = EKFStateEstimator(model_no_ctrl, state_dim=3)
        assert ekf.state_dim == 3
        assert ekf.obs_dim == 3
        assert ekf.obs_indices == [0, 1, 2]
        assert ekf.Q.shape == (3, 3)
        assert ekf.R.shape == (3, 3)

    def test_init_partial_obs(self, model_no_ctrl: NeuralODE) -> None:
        ekf = EKFStateEstimator(model_no_ctrl, state_dim=3, obs_indices=[0, 2])
        assert ekf.obs_dim == 2
        assert ekf._H.shape == (2, 3)
        assert ekf._H[0, 0] == 1.0
        assert ekf._H[1, 2] == 1.0
        assert ekf._H[0, 1] == 0.0

    def test_predict_step_shapes(self, model_no_ctrl: NeuralODE) -> None:
        ekf = EKFStateEstimator(model_no_ctrl, state_dim=3)
        z = torch.randn(3)
        P = torch.eye(3)
        z_pred, P_pred = ekf.predict_step(z, P)
        assert z_pred.shape == (3,)
        assert P_pred.shape == (3, 3)

    def test_predict_step_covariance_positive(self, model_no_ctrl: NeuralODE) -> None:
        ekf = EKFStateEstimator(model_no_ctrl, state_dim=3)
        z = torch.randn(3)
        P = torch.eye(3)
        _, P_pred = ekf.predict_step(z, P)
        eigvals = torch.linalg.eigvalsh(P_pred)
        assert torch.all(eigvals > 0), "P_pred should be positive definite"

    def test_update_step_shapes(self, model_no_ctrl: NeuralODE) -> None:
        ekf = EKFStateEstimator(model_no_ctrl, state_dim=3, obs_indices=[0, 1])
        z_pred = torch.randn(3)
        P_pred = torch.eye(3)
        meas = torch.randn(2)
        z_upd, P_upd, innov = ekf.update_step(z_pred, P_pred, meas)
        assert z_upd.shape == (3,)
        assert P_upd.shape == (3, 3)
        assert innov.shape == (2,)

    def test_update_step_reduces_covariance(self, model_no_ctrl: NeuralODE) -> None:
        ekf = EKFStateEstimator(model_no_ctrl, state_dim=3)
        z_pred = torch.randn(3)
        P_pred = torch.eye(3) * 2.0
        meas = z_pred + torch.randn(3) * 0.01  # close measurement
        _, P_upd, _ = ekf.update_step(z_pred, P_pred, meas)
        assert P_upd.trace() < P_pred.trace(), "Update should reduce uncertainty"

    def test_filter_full_pass(self, model_no_ctrl: NeuralODE) -> None:
        ekf = EKFStateEstimator(model_no_ctrl, state_dim=3, dt=0.1)
        measurements = torch.randn(20, 3)
        result = ekf.filter(measurements, z0=torch.zeros(3))
        assert result["states"].shape == (20, 3)
        assert result["covariances"].shape == (20, 3, 3)
        assert result["innovations"].shape == (20, 3)

    def test_filter_with_t_span(self, model_no_ctrl: NeuralODE) -> None:
        ekf = EKFStateEstimator(model_no_ctrl, state_dim=3)
        measurements = torch.randn(10, 3)
        t_span = torch.linspace(0, 1, 10)
        result = ekf.filter(measurements, z0=torch.zeros(3), t_span=t_span)
        assert result["states"].shape == (10, 3)

    def test_scalar_covariance_init(self, model_no_ctrl: NeuralODE) -> None:
        ekf = EKFStateEstimator(model_no_ctrl, state_dim=3, Q=0.5, R=0.1, P0=2.0)
        assert ekf.Q[0, 0] == 0.5
        assert ekf.R[1, 1] == 0.1
        assert ekf.P0[2, 2] == 2.0


# =====================================================================
# SPC Chart (Fault Detection Level 1)
# =====================================================================

class TestSPCChart:
    """Tests for SPCChart (EWMA + CUSUM)."""

    def test_set_baseline(self) -> None:
        spc = SPCChart(num_vars=3)
        data = np.random.randn(100, 3) * 0.1 + 1.0
        spc.set_baseline(data)
        assert spc.mean is not None
        assert spc.std is not None
        np.testing.assert_allclose(spc.mean, 1.0, atol=0.05)

    def test_update_normal(self) -> None:
        spc = SPCChart(num_vars=2)
        data = np.random.randn(200, 2) * 0.1 + 5.0
        spc.set_baseline(data)
        result = spc.update(np.array([5.0, 5.0]))
        assert not np.any(result["ewma_alarm"])
        assert not np.any(result["cusum_alarm"])

    def test_update_fault_detection(self) -> None:
        spc = SPCChart(num_vars=2, ewma_lambda=0.3, cusum_h=4.0)
        normal = np.random.randn(200, 2) * 0.1 + 5.0
        spc.set_baseline(normal)
        # Feed many shifted observations
        for _ in range(50):
            result = spc.update(np.array([6.0, 5.0]))  # shift on var 0
        assert np.any(result["ewma_alarm"]) or np.any(result["cusum_alarm"])

    def test_update_without_baseline_raises(self) -> None:
        spc = SPCChart(num_vars=2)
        with pytest.raises(RuntimeError, match="set_baseline"):
            spc.update(np.array([1.0, 1.0]))

    def test_reset(self) -> None:
        spc = SPCChart(num_vars=2)
        data = np.random.randn(100, 2)
        spc.set_baseline(data)
        spc.update(np.array([10.0, 10.0]))
        spc.reset()
        np.testing.assert_array_equal(spc._ewma, spc.mean)


# =====================================================================
# Residual Detector (Level 2)
# =====================================================================

class TestResidualDetector:
    """Tests for ResidualDetector."""

    def test_compute_residual(self, model_no_ctrl: NeuralODE) -> None:
        rd = ResidualDetector(model_no_ctrl, state_dim=3, dt=0.01)
        z_curr = torch.randn(3)
        z_next = torch.randn(3)
        residual = rd.compute_residual(z_curr, z_next)
        assert residual.shape == (3,)

    def test_update_without_baseline(self, model_no_ctrl: NeuralODE) -> None:
        rd = ResidualDetector(model_no_ctrl, state_dim=3)
        result = rd.update(torch.randn(3), torch.randn(3))
        assert not np.any(result["alarm"])  # no baseline => no alarm

    def test_update_with_baseline(self, model_no_ctrl: NeuralODE) -> None:
        rd = ResidualDetector(model_no_ctrl, state_dim=3, threshold_sigma=2.0)
        baseline = np.random.randn(100, 3) * 0.01
        rd.set_baseline(baseline)
        result = rd.update(torch.randn(3), torch.randn(3))
        assert "residual" in result
        assert "alarm" in result
        assert "z_score" in result


# =====================================================================
# Fault Isolator (Level 3)
# =====================================================================

class TestFaultIsolator:
    """Tests for FaultIsolator."""

    def test_isolate_without_baseline(self) -> None:
        fi = FaultIsolator(state_dim=3)
        result = fi.isolate(np.array([1.0, 0.1, 0.01]))
        assert result["ranking"][0] == 0  # largest contribution first

    def test_isolate_with_baseline(self) -> None:
        fi = FaultIsolator(state_dim=3)
        residuals = np.random.randn(100, 3) * 0.1
        fi.set_baseline(residuals)
        result = fi.isolate(np.array([2.0, 0.0, 0.0]))
        assert result["contributions"].shape == (3,)
        assert "spe" in result
        assert result["spe"] > 0


# =====================================================================
# Fault Classifier (Level 4)
# =====================================================================

class TestFaultClassifier:
    """Tests for FaultClassifier."""

    def test_predict_without_fit(self) -> None:
        fc = FaultClassifier(method="rf")
        result = fc.predict(np.array([1.0, 2.0]))
        assert result["label"] == "unknown"

    def test_fit_and_predict(self) -> None:
        fc = FaultClassifier(method="rf", n_estimators=5)
        rng = np.random.default_rng(42)
        features = rng.normal(size=(50, 3))
        labels = np.array(["normal"] * 25 + ["fault_a"] * 25)
        fc.fit(features, labels)
        result = fc.predict(features[0])
        assert result["label"] in ["normal", "fault_a"]
        assert len(result["probabilities"]) == 2

    def test_svm_classifier(self) -> None:
        fc = FaultClassifier(method="svm")
        rng = np.random.default_rng(42)
        features = rng.normal(size=(40, 2))
        labels = np.array(["a"] * 20 + ["b"] * 20)
        fc.fit(features, labels)
        result = fc.predict(features[10])
        assert result["label"] in ["a", "b"]


# =====================================================================
# Unified Fault Detector
# =====================================================================

class TestFaultDetector:
    """Tests for unified FaultDetector."""

    def test_init(self, model_no_ctrl: NeuralODE) -> None:
        fd = FaultDetector(model_no_ctrl, state_dim=3)
        assert fd.state_dim == 3

    def test_set_baseline(self, model_no_ctrl: NeuralODE) -> None:
        fd = FaultDetector(model_no_ctrl, state_dim=3)
        normal_data = {
            "observations": np.random.randn(100, 3) * 0.1 + 1.0,
            "residuals": np.random.randn(100, 3) * 0.01,
        }
        fd.set_baseline(normal_data)
        assert fd._has_baseline

    def test_update_returns_alarms(self, model_no_ctrl: NeuralODE) -> None:
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


# =====================================================================
# MPC Objective
# =====================================================================

class TestMPCObjective:
    """Tests for MPCObjective cost functions."""

    def test_stage_cost(self) -> None:
        Q = torch.eye(2)
        R = 0.1 * torch.eye(1)
        obj = MPCObjective(Q=Q, R=R)
        y = torch.tensor([1.0, 2.0])
        y_ref = torch.tensor([1.0, 1.0])
        u = torch.tensor([0.5])
        cost = obj.stage_cost(y, y_ref, u)
        expected = 1.0 + 0.1 * 0.25  # (2-1)^2 + 0.1*0.5^2
        assert abs(cost.item() - expected) < 1e-5

    def test_trajectory_cost(self) -> None:
        Q = torch.eye(2)
        R = torch.eye(1) * 0.01
        obj = MPCObjective(Q=Q, R=R)
        traj = torch.randn(6, 2)  # 5 steps + z0
        y_ref = torch.zeros(2)
        controls = torch.randn(5, 1)
        cost = obj.trajectory_cost(traj, y_ref, controls)
        assert cost.item() > 0

    def test_terminal_cost_defaults_to_Q(self) -> None:
        Q = 2.0 * torch.eye(2)
        R = torch.eye(1)
        obj = MPCObjective(Q=Q, R=R)
        e = torch.ones(2)
        tc = obj.terminal_cost(e, torch.zeros(2))
        assert abs(tc.item() - 4.0) < 1e-5  # 2*(1^2+1^2)


# =====================================================================
# Control Constraints
# =====================================================================

class TestControlConstraints:
    """Tests for ControlConstraints."""

    def test_clamp(self) -> None:
        cc = ControlConstraints(
            u_min=torch.tensor([-1.0]),
            u_max=torch.tensor([1.0]),
        )
        u = torch.tensor([2.0])
        assert cc.clamp_controls(u).item() == 1.0
        assert cc.clamp_controls(torch.tensor([-3.0])).item() == -1.0

    def test_output_penalty_no_bounds(self) -> None:
        cc = ControlConstraints(
            u_min=torch.tensor([0.0]),
            u_max=torch.tensor([1.0]),
        )
        assert cc.output_penalty(torch.tensor([0.5])).item() == 0.0

    def test_output_penalty_violation(self) -> None:
        cc = ControlConstraints(
            u_min=torch.tensor([0.0]),
            u_max=torch.tensor([1.0]),
            y_min=torch.tensor([0.0]),
            y_max=torch.tensor([1.0]),
            penalty_weight=10.0,
        )
        pen = cc.output_penalty(torch.tensor([2.0]))
        assert pen.item() > 0  # violates y_max


# =====================================================================
# MPC Controller
# =====================================================================

class TestMPCController:
    """Tests for MPCController."""

    def test_predict_trajectory_shape(self, model_with_ctrl: NeuralODE) -> None:
        mpc = MPCController(model_with_ctrl, horizon=5, dt=0.01)
        z0 = torch.randn(3)
        controls = torch.randn(5, 1)
        traj = mpc._predict_trajectory(z0, controls)
        assert traj.shape == (6, 3)  # horizon+1 x state_dim

    def test_optimize(self, model_with_ctrl: NeuralODE) -> None:
        mpc = MPCController(model_with_ctrl, horizon=5, dt=0.01, max_iter=5)
        z0 = torch.randn(3)
        y_ref = torch.zeros(3)
        result = mpc.optimize(z0, y_ref)
        assert "controls" in result
        assert "trajectory" in result
        assert "cost" in result
        assert result["controls"].shape == (5, 1)

    def test_step(self, model_with_ctrl: NeuralODE) -> None:
        mpc = MPCController(model_with_ctrl, horizon=3, dt=0.01, max_iter=3)
        u, info = mpc.step(torch.randn(3), torch.zeros(3))
        assert u.shape == (1,)
        assert info["converged"]

    def test_warm_start(self, model_with_ctrl: NeuralODE) -> None:
        mpc = MPCController(model_with_ctrl, horizon=5, dt=0.01, max_iter=3)
        mpc.step(torch.randn(3), torch.zeros(3))
        assert mpc._u_prev is not None
        # Second step should use warm start
        mpc.step(torch.randn(3), torch.zeros(3))

    def test_with_constraints(self, model_with_ctrl: NeuralODE) -> None:
        constraints = ControlConstraints(
            u_min=torch.tensor([-1.0]),
            u_max=torch.tensor([1.0]),
        )
        mpc = MPCController(
            model_with_ctrl, horizon=3, dt=0.01,
            constraints=constraints, max_iter=3,
        )
        u, info = mpc.step(torch.randn(3), torch.zeros(3))
        assert u.item() >= -1.0
        assert u.item() <= 1.0


# =====================================================================
# Replay Buffer
# =====================================================================

class TestReplayBuffer:
    """Tests for ReplayBuffer."""

    def test_add_and_len(self) -> None:
        buf = ReplayBuffer(capacity=10)
        assert len(buf) == 0
        buf.add(torch.randn(3), torch.linspace(0, 1, 5), torch.randn(1, 5, 3))
        assert len(buf) == 1

    def test_capacity_overflow(self) -> None:
        buf = ReplayBuffer(capacity=3)
        for i in range(5):
            buf.add(torch.randn(3), torch.linspace(0, 1, 5), torch.randn(1, 5, 3))
        assert len(buf) == 3  # FIFO, oldest dropped

    def test_sample(self) -> None:
        buf = ReplayBuffer(capacity=50)
        for _ in range(10):
            buf.add(torch.randn(1, 3), torch.linspace(0, 1, 5), torch.randn(1, 5, 3))
        batch = buf.sample(4)
        assert "z0" in batch
        assert "t_span" in batch
        assert "targets" in batch
        assert batch["z0"].shape[0] == 4
        assert batch["targets"].shape[0] == 4


# =====================================================================
# Elastic Weight Consolidation
# =====================================================================

class TestEWC:
    """Tests for ElasticWeightConsolidation."""

    def test_penalty_before_consolidation(self, model_no_ctrl: NeuralODE) -> None:
        ewc = ElasticWeightConsolidation(model_no_ctrl, ewc_lambda=10.0)
        assert ewc.penalty().item() == 0.0

    def test_consolidation_without_data(self, model_no_ctrl: NeuralODE) -> None:
        ewc = ElasticWeightConsolidation(model_no_ctrl, ewc_lambda=10.0)
        ewc.consolidate()
        assert ewc._consolidated
        # Penalty should be zero right after consolidation (theta == theta*)
        assert abs(ewc.penalty().item()) < 1e-6

    def test_penalty_after_parameter_change(self, model_no_ctrl: NeuralODE) -> None:
        ewc = ElasticWeightConsolidation(model_no_ctrl, ewc_lambda=100.0)
        ewc.consolidate()
        # Perturb parameters
        with torch.no_grad():
            for p in model_no_ctrl.parameters():
                p.add_(torch.randn_like(p) * 0.1)
        penalty = ewc.penalty()
        assert penalty.item() > 0


# =====================================================================
# Online Adapter
# =====================================================================

class TestOnlineAdapter:
    """Tests for OnlineAdapter."""

    def test_add_experience(self, model_no_ctrl: NeuralODE) -> None:
        adapter = OnlineAdapter(model_no_ctrl, lr=1e-4)
        adapter.add_experience(
            torch.randn(1, 3), torch.linspace(0, 1, 10), torch.randn(1, 10, 3),
        )
        assert len(adapter.replay_buffer) == 1

    def test_adapt(self, model_no_ctrl: NeuralODE) -> None:
        adapter = OnlineAdapter(model_no_ctrl, lr=1e-3, buffer_capacity=50)
        # Add some replay experiences
        for _ in range(5):
            adapter.add_experience(
                torch.randn(1, 3), torch.linspace(0, 1, 10), torch.randn(1, 10, 3),
            )
        new_data = {
            "z0": torch.randn(2, 3),
            "t_span": torch.linspace(0, 1, 10),
            "targets": torch.randn(2, 10, 3),
        }
        losses = adapter.adapt(new_data, num_steps=3, batch_size=4)
        assert len(losses) == 3
        assert all(isinstance(l, float) for l in losses)

    def test_consolidate(self, model_no_ctrl: NeuralODE) -> None:
        adapter = OnlineAdapter(model_no_ctrl, lr=1e-4)
        adapter.consolidate()
        assert adapter.ewc._consolidated


# =====================================================================
# Reptile Meta-Learner
# =====================================================================

class TestReptileMetaLearner:
    """Tests for ReptileMetaLearner."""

    def test_init(self, model_2d: NeuralODE) -> None:
        meta = ReptileMetaLearner(model_2d, meta_lr=0.01, inner_lr=0.01, inner_steps=2)
        assert meta.meta_lr == 0.01
        assert meta.inner_steps == 2

    def test_meta_step(self, model_2d: NeuralODE) -> None:
        from reactor_twin.reactors.systems import create_exothermic_cstr
        from reactor_twin.training.data_generator import ReactorDataGenerator

        reactor = create_exothermic_cstr(isothermal=True)
        # state_dim=2 matches model_2d
        gen = ReactorDataGenerator(reactor)

        meta = ReptileMetaLearner(model_2d, meta_lr=0.01, inner_lr=0.01, inner_steps=2)

        # Save params before
        params_before = {n: p.clone() for n, p in model_2d.named_parameters()}

        disp = meta.meta_step([gen], t_span=(0, 1.0),
                              t_eval=np.linspace(0, 1, 10), batch_size=4)
        assert disp > 0  # parameters should have moved

        # At least one parameter should have changed
        any_changed = False
        for n, p in model_2d.named_parameters():
            if not torch.allclose(p, params_before[n]):
                any_changed = True
                break
        assert any_changed

    def test_fine_tune(self, model_2d: NeuralODE) -> None:
        from reactor_twin.reactors.systems import create_exothermic_cstr
        from reactor_twin.training.data_generator import ReactorDataGenerator

        reactor = create_exothermic_cstr(isothermal=True)
        gen = ReactorDataGenerator(reactor)

        meta = ReptileMetaLearner(model_2d, inner_lr=0.01)
        losses = meta.fine_tune(gen, t_span=(0, 1.0),
                                t_eval=np.linspace(0, 1, 10),
                                num_steps=3, batch_size=4)
        assert len(losses) == 3


# =====================================================================
# Integration: top-level imports
# =====================================================================

class TestTopLevelImports:
    """Verify all digital twin classes are importable from top-level."""

    def test_import_ekf(self) -> None:
        from reactor_twin import EKFStateEstimator
        assert EKFStateEstimator is not None

    def test_import_fault_detector(self) -> None:
        from reactor_twin import FaultDetector
        assert FaultDetector is not None

    def test_import_mpc(self) -> None:
        from reactor_twin import MPCController
        assert MPCController is not None

    def test_import_adapter(self) -> None:
        from reactor_twin import OnlineAdapter
        assert OnlineAdapter is not None

    def test_import_meta(self) -> None:
        from reactor_twin import ReptileMetaLearner
        assert ReptileMetaLearner is not None

    def test_digital_twin_registry(self) -> None:
        from reactor_twin import DIGITAL_TWIN_REGISTRY
        assert DIGITAL_TWIN_REGISTRY is not None
