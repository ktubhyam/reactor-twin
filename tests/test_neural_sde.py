"""Tests for Neural Stochastic Differential Equation (neural_sde.py)."""

from __future__ import annotations

import pytest
import torch

torchsde = pytest.importorskip("torchsde")

from reactor_twin.core.neural_sde import NeuralSDE, SDEFunc
from reactor_twin.core.ode_func import MLPODEFunc

# ── Helpers ──────────────────────────────────────────────────────────

def _make_drift(state_dim: int = 3) -> MLPODEFunc:
    return MLPODEFunc(state_dim=state_dim, hidden_dim=16, num_layers=2)


# ══════════════════════════════════════════════════════════════════════
# SDEFunc Tests
# ══════════════════════════════════════════════════════════════════════


class TestSDEFunc:
    """Tests for the low-level SDEFunc (drift + diffusion)."""

    # -- Diagonal noise --

    def test_drift_shape_diagonal(self):
        drift = _make_drift()
        sde = SDEFunc(drift, noise_type="diagonal")
        z = torch.randn(4, 3)
        t = torch.tensor(0.0)
        out = sde.f(t, z)
        assert out.shape == (4, 3)

    def test_diffusion_shape_diagonal(self):
        drift = _make_drift()
        sde = SDEFunc(drift, noise_type="diagonal")
        z = torch.randn(4, 3)
        t = torch.tensor(0.0)
        out = sde.g(t, z)
        assert out.shape == (4, 3)

    def test_drift_finite_diagonal(self):
        drift = _make_drift()
        sde = SDEFunc(drift, noise_type="diagonal")
        z = torch.randn(4, 3)
        t = torch.tensor(0.0)
        out = sde.f(t, z)
        assert torch.all(torch.isfinite(out))

    def test_diffusion_finite_diagonal(self):
        drift = _make_drift()
        sde = SDEFunc(drift, noise_type="diagonal")
        z = torch.randn(4, 3)
        t = torch.tensor(0.0)
        out = sde.g(t, z)
        assert torch.all(torch.isfinite(out))

    # -- Additive noise --

    def test_diffusion_shape_additive(self):
        drift = _make_drift()
        sde = SDEFunc(drift, noise_type="additive")
        z = torch.randn(4, 3)
        t = torch.tensor(0.0)
        out = sde.g(t, z)
        assert out.shape == (4, 3)

    def test_diffusion_constant_additive(self):
        drift = _make_drift()
        sde = SDEFunc(drift, noise_type="additive")
        z1 = torch.randn(4, 3)
        z2 = torch.randn(4, 3)
        t = torch.tensor(0.0)
        g1 = sde.g(t, z1)
        g2 = sde.g(t, z2)
        # Additive: same diffusion regardless of state
        torch.testing.assert_close(g1, g2)

    # -- Scalar noise --

    def test_diffusion_shape_scalar(self):
        drift = _make_drift()
        sde = SDEFunc(drift, noise_type="scalar")
        z = torch.randn(4, 3)
        t = torch.tensor(0.0)
        out = sde.g(t, z)
        assert out.shape == (4, 1)

    # -- Custom diffusion --

    def test_custom_diffusion_func(self):
        drift = _make_drift()
        custom_diff = torch.nn.Sequential(
            torch.nn.Linear(3, 16),
            torch.nn.Softplus(),
            torch.nn.Linear(16, 3),
            torch.nn.Softplus(),
        )
        sde = SDEFunc(drift, diffusion_func=custom_diff, noise_type="diagonal")
        z = torch.randn(4, 3)
        t = torch.tensor(0.0)
        out = sde.g(t, z)
        assert out.shape == (4, 3)

    # -- Attributes --

    def test_noise_type_attribute(self):
        drift = _make_drift()
        sde = SDEFunc(drift, noise_type="additive")
        assert sde.noise_type == "additive"

    def test_sde_type_attribute(self):
        drift = _make_drift()
        sde = SDEFunc(drift, sde_type="stratonovich")
        assert sde.sde_type == "stratonovich"

    def test_default_sde_type(self):
        drift = _make_drift()
        sde = SDEFunc(drift)
        assert sde.sde_type == "ito"


# ══════════════════════════════════════════════════════════════════════
# NeuralSDE Tests
# ══════════════════════════════════════════════════════════════════════


class TestNeuralSDEInstantiation:
    def test_default_instantiation(self):
        model = NeuralSDE(state_dim=3)
        assert model.state_dim == 3
        assert model.noise_type == "diagonal"
        assert model.sde_type == "ito"

    def test_custom_drift(self):
        drift = _make_drift(4)
        model = NeuralSDE(state_dim=4, drift_func=drift)
        assert model.state_dim == 4

    def test_torchsde_required(self):
        # If torchsde is available (it is, since we importorskip'd it),
        # just verify construction works
        model = NeuralSDE(state_dim=2)
        assert model is not None

    def test_custom_noise_type(self):
        model = NeuralSDE(state_dim=3, noise_type="additive")
        assert model.noise_type == "additive"

    def test_custom_sde_type(self):
        model = NeuralSDE(state_dim=3, sde_type="stratonovich")
        assert model.sde_type == "stratonovich"


class TestNeuralSDEForward:
    def test_forward_shape_single_sample(self):
        model = NeuralSDE(state_dim=3, method="euler", dt=0.1)
        z0 = torch.randn(2, 3)
        t_span = torch.linspace(0, 1, 5)
        out = model(z0, t_span, num_samples=1)
        assert out.shape == (2, 5, 3)

    def test_forward_shape_multi_sample(self):
        model = NeuralSDE(state_dim=3, method="euler", dt=0.1)
        z0 = torch.randn(2, 3)
        t_span = torch.linspace(0, 1, 5)
        out = model(z0, t_span, num_samples=3)
        assert out.shape == (3, 2, 5, 3)

    def test_forward_values_finite(self):
        model = NeuralSDE(state_dim=2, method="euler", dt=0.1)
        z0 = torch.randn(2, 2)
        t_span = torch.linspace(0, 0.5, 3)
        out = model(z0, t_span, num_samples=1)
        assert torch.all(torch.isfinite(out))

    def test_different_seeds_different_paths(self):
        model = NeuralSDE(state_dim=2, method="euler", dt=0.1)
        z0 = torch.randn(1, 2)
        t_span = torch.linspace(0, 1, 5)
        out = model(z0, t_span, num_samples=5)
        # Different samples should generally differ
        # Check that not all samples are identical
        diffs = (out[0] - out[1]).abs().sum()
        # With stochastic noise, should be non-zero (very high probability)
        assert diffs > 0

    def test_controls_raise_not_implemented(self):
        model = NeuralSDE(state_dim=2)
        z0 = torch.randn(2, 2)
        t_span = torch.linspace(0, 1, 3)
        controls = torch.randn(2, 3, 1)
        with pytest.raises(NotImplementedError, match="Controls not yet supported"):
            model(z0, t_span, controls=controls)


class TestNeuralSDELoss:
    def test_compute_loss_keys(self):
        model = NeuralSDE(state_dim=2, method="euler", dt=0.1)
        preds = torch.randn(2, 5, 2)
        targets = torch.randn(2, 5, 2)
        losses = model.compute_loss(preds, targets)
        assert "total" in losses
        assert "data" in losses

    def test_compute_loss_scalar(self):
        model = NeuralSDE(state_dim=2, method="euler", dt=0.1)
        preds = torch.randn(2, 5, 2)
        targets = torch.randn(2, 5, 2)
        losses = model.compute_loss(preds, targets)
        assert losses["total"].ndim == 0

    def test_loss_zero_for_identical(self):
        model = NeuralSDE(state_dim=2, method="euler", dt=0.1)
        data = torch.randn(2, 5, 2)
        losses = model.compute_loss(data, data)
        assert losses["total"].item() == pytest.approx(0.0, abs=1e-6)

    def test_loss_multi_sample_averaging(self):
        model = NeuralSDE(state_dim=2, method="euler", dt=0.1)
        # 4D predictions: (num_samples, batch, time, state_dim)
        preds = torch.randn(3, 2, 5, 2)
        targets = torch.randn(2, 5, 2)
        losses = model.compute_loss(preds, targets)
        assert losses["total"].item() > 0


class TestNeuralSDEUncertainty:
    def test_predict_with_uncertainty_shapes(self):
        model = NeuralSDE(state_dim=2, method="euler", dt=0.1)
        z0 = torch.randn(2, 2)
        t_span = torch.linspace(0, 0.5, 4)
        mean, std = model.predict_with_uncertainty(z0, t_span, num_samples=5)
        assert mean.shape == (2, 4, 2)
        assert std.shape == (2, 4, 2)

    def test_mean_finite(self):
        model = NeuralSDE(state_dim=2, method="euler", dt=0.1)
        z0 = torch.randn(1, 2)
        t_span = torch.linspace(0, 0.5, 3)
        mean, std = model.predict_with_uncertainty(z0, t_span, num_samples=5)
        assert torch.all(torch.isfinite(mean))

    def test_std_nonneg(self):
        model = NeuralSDE(state_dim=2, method="euler", dt=0.1)
        z0 = torch.randn(1, 2)
        t_span = torch.linspace(0, 0.5, 3)
        mean, std = model.predict_with_uncertainty(z0, t_span, num_samples=10)
        assert torch.all(std >= 0)

    def test_std_varies_over_time(self):
        model = NeuralSDE(state_dim=2, method="euler", dt=0.05)
        z0 = torch.randn(1, 2)
        t_span = torch.linspace(0, 1.0, 10)
        mean, std = model.predict_with_uncertainty(z0, t_span, num_samples=20)
        # Std at t=0 should be ~0 (all paths start from same z0)
        # Std at later times should be larger
        std_t0 = std[0, 0, :].sum().item()
        std_last = std[0, -1, :].sum().item()
        assert std_last > std_t0 - 1e-3  # Allow small tolerance


class TestNeuralSDEGradients:
    def test_gradient_flow(self):
        model = NeuralSDE(state_dim=2, method="euler", dt=0.1)
        z0 = torch.randn(2, 2, requires_grad=True)
        t_span = torch.linspace(0, 0.5, 3)
        out = model(z0, t_span, num_samples=1)
        loss = out.sum()
        loss.backward()
        # Check that at least some parameters got gradients
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        assert has_grad


class TestNeuralSDERegistry:
    def test_registered(self):
        from reactor_twin.utils.registry import NEURAL_DE_REGISTRY

        assert "neural_sde" in NEURAL_DE_REGISTRY

    def test_get_returns_class(self):
        from reactor_twin.utils.registry import NEURAL_DE_REGISTRY

        cls = NEURAL_DE_REGISTRY.get("neural_sde")
        assert cls is NeuralSDE
