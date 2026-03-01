"""Tests for Neural Controlled Differential Equation (neural_cde.py)."""

from __future__ import annotations

import pytest
import torch

torchcde = pytest.importorskip("torchcde")

from reactor_twin.core.neural_cde import CDEFunc, NeuralCDE
from reactor_twin.exceptions import ValidationError

# ══════════════════════════════════════════════════════════════════════
# CDEFunc Tests
# ══════════════════════════════════════════════════════════════════════


class TestCDEFunc:
    """Tests for the CDE vector-field function."""

    def test_forward_shape(self):
        func = CDEFunc(state_dim=4, input_dim=3, hidden_dim=16, num_layers=2)
        z = torch.randn(2, 4)
        t = torch.tensor(0.0)
        out = func(t, z)
        # output: (batch, state_dim, input_dim)
        assert out.shape == (2, 4, 3)

    def test_forward_finite(self):
        func = CDEFunc(state_dim=4, input_dim=3, hidden_dim=16, num_layers=2)
        z = torch.randn(2, 4)
        t = torch.tensor(0.0)
        out = func(t, z)
        assert torch.all(torch.isfinite(out))

    def test_gradient_flow(self):
        func = CDEFunc(state_dim=3, input_dim=2, hidden_dim=16, num_layers=2)
        z = torch.randn(2, 3, requires_grad=True)
        t = torch.tensor(0.0)
        out = func(t, z)
        loss = out.sum()
        loss.backward()
        assert z.grad is not None
        assert z.grad.abs().sum() > 0

    def test_different_layer_counts(self):
        for n in [2, 3, 4]:
            func = CDEFunc(state_dim=3, input_dim=2, hidden_dim=16, num_layers=n)
            z = torch.randn(1, 3)
            t = torch.tensor(0.0)
            out = func(t, z)
            assert out.shape == (1, 3, 2)

    def test_batch_independence(self):
        """Each batch element should be processed independently."""
        func = CDEFunc(state_dim=3, input_dim=2, hidden_dim=16, num_layers=2)
        z = torch.randn(4, 3)
        t = torch.tensor(0.0)
        out_batch = func(t, z)
        out_single = func(t, z[0:1])
        torch.testing.assert_close(out_batch[0:1], out_single, atol=1e-5, rtol=1e-5)


# ══════════════════════════════════════════════════════════════════════
# NeuralCDE Tests
# ══════════════════════════════════════════════════════════════════════


class TestNeuralCDEInstantiation:
    def test_basic_instantiation(self):
        model = NeuralCDE(state_dim=4, input_dim=3)
        assert model.state_dim == 4
        assert model.input_dim == 3
        assert model.output_dim == 3  # defaults to input_dim

    def test_torchcde_required(self):
        # torchcde is available (importorskip), just verify construction
        model = NeuralCDE(state_dim=4, input_dim=3)
        assert model is not None

    def test_custom_output_dim(self):
        model = NeuralCDE(state_dim=4, input_dim=3, output_dim=2)
        assert model.output_dim == 2

    def test_attributes(self):
        model = NeuralCDE(
            state_dim=4, input_dim=3, interpolation="linear", solver="euler"
        )
        assert model.interpolation == "linear"
        assert model.solver == "euler"

    def test_has_initial_network(self):
        model = NeuralCDE(state_dim=4, input_dim=3)
        assert hasattr(model, "initial_network")

    def test_has_cde_func(self):
        model = NeuralCDE(state_dim=4, input_dim=3)
        assert hasattr(model, "cde_func")
        assert isinstance(model.cde_func, CDEFunc)

    def test_has_readout(self):
        model = NeuralCDE(state_dim=4, input_dim=3)
        assert hasattr(model, "readout")


class TestNeuralCDEForward:
    def _make_model_and_data(self, state_dim=4, input_dim=3, batch=2, time=5):
        model = NeuralCDE(
            state_dim=state_dim,
            input_dim=input_dim,
            interpolation="linear",
            solver="euler",
            adjoint=False,
        )
        z0 = torch.randn(batch, input_dim)
        t_span = torch.linspace(0, 1, time)
        controls = torch.randn(batch, time, input_dim)
        return model, z0, t_span, controls

    def test_forward_shape(self):
        model, z0, t_span, controls = self._make_model_and_data()
        out = model(z0, t_span, controls=controls)
        # output: (batch, time, output_dim=input_dim)
        assert out.shape == (2, 5, 3)

    def test_controls_required(self):
        model = NeuralCDE(state_dim=4, input_dim=3)
        z0 = torch.randn(2, 3)
        t_span = torch.linspace(0, 1, 5)
        with pytest.raises(ValidationError, match="requires 'controls'"):
            model(z0, t_span, controls=None)

    def test_linear_interpolation(self):
        model = NeuralCDE(
            state_dim=4,
            input_dim=3,
            interpolation="linear",
            solver="euler",
            adjoint=False,
        )
        z0 = torch.randn(2, 3)
        t_span = torch.linspace(0, 1, 5)
        controls = torch.randn(2, 5, 3)
        out = model(z0, t_span, controls=controls)
        assert out.shape == (2, 5, 3)
        assert torch.all(torch.isfinite(out))

    def test_cubic_interpolation(self):
        model = NeuralCDE(
            state_dim=4,
            input_dim=3,
            interpolation="cubic",
            solver="euler",
            adjoint=False,
        )
        z0 = torch.randn(2, 3)
        t_span = torch.linspace(0, 1, 5)
        controls = torch.randn(2, 5, 3)
        out = model(z0, t_span, controls=controls)
        assert out.shape == (2, 5, 3)

    def test_forward_values_finite(self):
        model, z0, t_span, controls = self._make_model_and_data()
        out = model(z0, t_span, controls=controls)
        assert torch.all(torch.isfinite(out))


class TestNeuralCDELoss:
    def _make_model(self):
        return NeuralCDE(
            state_dim=4,
            input_dim=3,
            interpolation="linear",
            solver="euler",
            adjoint=False,
        )

    def test_compute_loss_keys(self):
        model = self._make_model()
        preds = torch.randn(2, 5, 3)
        targets = torch.randn(2, 5, 3)
        losses = model.compute_loss(preds, targets)
        assert "total" in losses
        assert "data" in losses

    def test_compute_loss_scalar(self):
        model = self._make_model()
        preds = torch.randn(2, 5, 3)
        targets = torch.randn(2, 5, 3)
        losses = model.compute_loss(preds, targets)
        assert losses["total"].ndim == 0

    def test_loss_zero_for_identical(self):
        model = self._make_model()
        data = torch.randn(2, 5, 3)
        losses = model.compute_loss(data, data)
        assert losses["total"].item() == pytest.approx(0.0, abs=1e-6)

    def test_nan_masking(self):
        model = self._make_model()
        preds = torch.randn(2, 5, 3)
        targets = torch.randn(2, 5, 3)
        # Mask some targets as NaN
        targets[0, 2, :] = float("nan")
        targets[1, 4, :] = float("nan")
        losses = model.compute_loss(preds, targets)
        # Loss should still be finite
        assert torch.isfinite(losses["total"])


class TestNeuralCDEIrregular:
    def test_forward_with_irregular_raises_for_2d_times(self):
        """2D observation_times (per-batch) must raise ValueError."""
        model = NeuralCDE(state_dim=4, input_dim=3)
        obs = torch.randn(2, 5, 3)
        obs_times = torch.randn(2, 5)  # 2D — not supported
        pred_times = torch.linspace(0, 1, 10)
        with pytest.raises(ValueError, match="observation_times must be 1D"):
            model.forward_with_irregular_observations(obs, obs_times, pred_times)

    def test_forward_with_irregular_observations_shape(self):
        """Irregularly-sampled observations produce correct output shape."""
        torch.manual_seed(0)
        model = NeuralCDE(state_dim=4, input_dim=3, interpolation="linear")
        batch, num_obs, input_dim = 2, 5, 3
        obs = torch.randn(batch, num_obs, input_dim)
        obs_times = torch.tensor([0.0, 0.2, 0.5, 0.7, 1.0])
        pred_times = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
        out = model.forward_with_irregular_observations(obs, obs_times, pred_times)
        assert out.shape == (batch, len(pred_times), model.output_dim)

    def test_forward_with_irregular_observations_pred_at_obs_times(self):
        """Predicting at observation times gives finite outputs."""
        torch.manual_seed(1)
        model = NeuralCDE(state_dim=4, input_dim=2, interpolation="linear")
        obs = torch.randn(3, 4, 2)
        obs_times = torch.tensor([0.0, 0.3, 0.6, 1.0])
        out = model.forward_with_irregular_observations(obs, obs_times, obs_times)
        assert out.shape == (3, 4, model.output_dim)
        assert torch.all(torch.isfinite(out))


class TestNeuralCDEGradients:
    def test_gradient_flow(self):
        model = NeuralCDE(
            state_dim=4,
            input_dim=3,
            interpolation="linear",
            solver="euler",
            adjoint=False,
        )
        z0 = torch.randn(2, 3)
        t_span = torch.linspace(0, 1, 5)
        controls = torch.randn(2, 5, 3)
        out = model(z0, t_span, controls=controls)
        loss = out.sum()
        loss.backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        assert has_grad


class TestNeuralCDERegistry:
    def test_registered(self):
        from reactor_twin.utils.registry import NEURAL_DE_REGISTRY

        assert "neural_cde" in NEURAL_DE_REGISTRY

    def test_get_returns_class(self):
        from reactor_twin.utils.registry import NEURAL_DE_REGISTRY

        cls = NEURAL_DE_REGISTRY.get("neural_cde")
        assert cls is NeuralCDE
