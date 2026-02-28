"""Tests for Bayesian Neural ODE."""

from __future__ import annotations

import pytest
import torch

from reactor_twin.core.bayesian_neural_ode import (
    BayesianLinear,
    BayesianMLPODEFunc,
    BayesianNeuralODE,
)
from reactor_twin.utils.registry import NEURAL_DE_REGISTRY

# ── BayesianLinear ───────────────────────────────────────────────────


class TestBayesianLinear:
    def test_output_shape(self):
        layer = BayesianLinear(4, 8)
        x = torch.randn(3, 4)
        y = layer(x)
        assert y.shape == (3, 8)

    def test_stochastic_forward(self):
        """Two forward passes should give different results (stochastic weights)."""
        layer = BayesianLinear(4, 8)
        x = torch.randn(3, 4)
        y1 = layer(x)
        y2 = layer(x)
        # Very unlikely to be exactly equal with random sampling
        assert not torch.allclose(y1, y2)

    def test_kl_divergence_positive(self):
        layer = BayesianLinear(4, 8)
        kl = layer.kl_divergence()
        assert kl.item() >= 0

    def test_kl_divergence_finite(self):
        layer = BayesianLinear(4, 8)
        kl = layer.kl_divergence()
        assert torch.isfinite(kl)

    def test_kl_zero_at_prior(self):
        """KL should be ~0 when posterior matches prior (mu=0, sigma=prior_sigma)."""
        layer = BayesianLinear(4, 8, prior_sigma=1.0)
        # Set posterior to match prior
        with torch.no_grad():
            layer.weight_mu.fill_(0.0)
            layer.bias_mu.fill_(0.0)
            layer.weight_log_sigma.fill_(0.0)  # sigma = 1.0 = prior_sigma
            layer.bias_log_sigma.fill_(0.0)
        kl = layer.kl_divergence()
        assert kl.item() == pytest.approx(0.0, abs=1e-5)

    def test_parameters_require_grad(self):
        layer = BayesianLinear(4, 8)
        for p in layer.parameters():
            assert p.requires_grad

    def test_different_prior_sigma(self):
        l1 = BayesianLinear(4, 8, prior_sigma=0.1)
        l2 = BayesianLinear(4, 8, prior_sigma=10.0)
        kl1 = l1.kl_divergence()
        kl2 = l2.kl_divergence()
        # Both should be finite
        assert torch.isfinite(kl1)
        assert torch.isfinite(kl2)

    def test_single_input(self):
        layer = BayesianLinear(1, 1)
        x = torch.randn(1, 1)
        y = layer(x)
        assert y.shape == (1, 1)

    def test_prior_sigma_zero_raises(self):
        with pytest.raises(ValueError, match="prior_sigma"):
            BayesianLinear(4, 8, prior_sigma=0.0)

    def test_prior_sigma_negative_raises(self):
        with pytest.raises(ValueError, match="prior_sigma"):
            BayesianLinear(4, 8, prior_sigma=-1.0)


# ── BayesianMLPODEFunc ───────────────────────────────────────────────


class TestBayesianMLPODEFunc:
    def test_output_shape(self):
        func = BayesianMLPODEFunc(state_dim=3, hidden_dim=32, num_layers=2)
        t = torch.tensor(0.0)
        z = torch.randn(5, 3)
        dz = func(t, z)
        assert dz.shape == (5, 3)

    def test_stochastic_output(self):
        func = BayesianMLPODEFunc(state_dim=3, hidden_dim=32, num_layers=2)
        t = torch.tensor(0.0)
        z = torch.randn(5, 3)
        dz1 = func(t, z)
        dz2 = func(t, z)
        assert not torch.allclose(dz1, dz2)

    def test_kl_divergence(self):
        func = BayesianMLPODEFunc(state_dim=3, hidden_dim=32, num_layers=2)
        kl = func.kl_divergence()
        assert torch.isfinite(kl)
        assert kl.item() >= 0

    def test_with_input_dim(self):
        func = BayesianMLPODEFunc(state_dim=3, input_dim=2, hidden_dim=32, num_layers=2)
        t = torch.tensor(0.0)
        z = torch.randn(5, 3)
        u = torch.randn(5, 2)
        dz = func(t, z, u)
        assert dz.shape == (5, 3)

    def test_gradients_flow(self):
        func = BayesianMLPODEFunc(state_dim=3, hidden_dim=32, num_layers=2)
        t = torch.tensor(0.0)
        z = torch.randn(5, 3, requires_grad=True)
        dz = func(t, z)
        loss = dz.sum()
        loss.backward()
        assert z.grad is not None
        # Check gradients flow to Bayesian params
        for p in func.parameters():
            assert p.grad is not None

    def test_batched_time(self):
        func = BayesianMLPODEFunc(state_dim=3, hidden_dim=16, num_layers=2)
        t = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
        z = torch.randn(5, 3)
        dz = func(t, z)
        assert dz.shape == (5, 3)

    def test_tanh_activation(self):
        func = BayesianMLPODEFunc(state_dim=3, hidden_dim=16, num_layers=2, activation="tanh")
        t = torch.tensor(0.0)
        z = torch.randn(2, 3)
        dz = func(t, z)
        assert dz.shape == (2, 3)

    def test_relu_activation(self):
        func = BayesianMLPODEFunc(state_dim=3, hidden_dim=16, num_layers=2, activation="relu")
        t = torch.tensor(0.0)
        z = torch.randn(2, 3)
        dz = func(t, z)
        assert dz.shape == (2, 3)

    def test_invalid_activation_raises(self):
        with pytest.raises(ValueError, match="Unknown activation"):
            BayesianMLPODEFunc(state_dim=3, activation="invalid")

    def test_bayesian_layers_detected(self):
        func = BayesianMLPODEFunc(state_dim=3, hidden_dim=16, num_layers=3)
        assert len(func._bayesian_layers) == 3


# ── BayesianNeuralODE ────────────────────────────────────────────────


class TestBayesianNeuralODE:
    def test_instantiation(self):
        model = BayesianNeuralODE(state_dim=2, hidden_dim=16, num_layers=2)
        assert model.state_dim == 2

    def test_forward_single_sample(self):
        model = BayesianNeuralODE(state_dim=2, solver="euler", hidden_dim=16, num_layers=2)
        z0 = torch.randn(3, 2)
        t = torch.linspace(0, 0.1, 5)
        out = model(z0, t)
        assert out.shape == (3, 5, 2)

    def test_forward_multi_sample(self):
        model = BayesianNeuralODE(state_dim=2, solver="euler", hidden_dim=16, num_layers=2)
        z0 = torch.randn(3, 2)
        t = torch.linspace(0, 0.1, 5)
        out = model(z0, t, num_samples=4)
        assert out.shape == (4, 3, 5, 2)

    def test_multi_sample_predictions_differ(self):
        """Multi-sample forward should produce distinct samples."""
        model = BayesianNeuralODE(state_dim=2, solver="euler", hidden_dim=16, num_layers=2)
        z0 = torch.randn(2, 2)
        t = torch.linspace(0, 0.1, 5)
        out = model(z0, t, num_samples=3)
        # At least two of the three samples should differ
        assert not torch.allclose(out[0], out[1]) or not torch.allclose(out[1], out[2])

    def test_forward_finite(self):
        model = BayesianNeuralODE(state_dim=2, solver="euler", hidden_dim=16, num_layers=2)
        z0 = torch.randn(2, 2) * 0.1
        t = torch.linspace(0, 0.05, 3)
        out = model(z0, t)
        assert torch.all(torch.isfinite(out))

    def test_compute_loss_single_sample(self):
        model = BayesianNeuralODE(state_dim=2, solver="euler", hidden_dim=16, num_layers=2)
        z0 = torch.randn(3, 2)
        t = torch.linspace(0, 0.1, 5)
        pred = model(z0, t)
        targets = torch.randn(3, 5, 2)
        losses = model.compute_loss(pred, targets)
        assert "total" in losses
        assert "data" in losses
        assert "kl" in losses
        assert torch.isfinite(losses["total"])

    def test_compute_loss_multi_sample(self):
        model = BayesianNeuralODE(state_dim=2, solver="euler", hidden_dim=16, num_layers=2)
        z0 = torch.randn(3, 2)
        t = torch.linspace(0, 0.1, 5)
        pred = model(z0, t, num_samples=3)
        targets = torch.randn(3, 5, 2)
        losses = model.compute_loss(pred, targets)
        assert torch.isfinite(losses["total"])

    def test_elbo_loss_includes_kl(self):
        """Total loss should be > data loss when beta > 0."""
        model = BayesianNeuralODE(
            state_dim=2, solver="euler", hidden_dim=16, num_layers=2, beta=1.0
        )
        z0 = torch.randn(3, 2)
        t = torch.linspace(0, 0.1, 5)
        pred = model(z0, t)
        targets = torch.randn(3, 5, 2)
        losses = model.compute_loss(pred, targets)
        assert losses["kl"].item() >= 0
        # total = data + beta * kl
        expected = losses["data"] + model.beta * losses["kl"]
        assert losses["total"].item() == pytest.approx(expected.item(), rel=1e-5)

    def test_beta_zero_reduces_to_mse(self):
        model = BayesianNeuralODE(
            state_dim=2, solver="euler", hidden_dim=16, num_layers=2, beta=0.0
        )
        z0 = torch.randn(3, 2)
        t = torch.linspace(0, 0.1, 5)
        pred = model(z0, t)
        targets = torch.randn(3, 5, 2)
        losses = model.compute_loss(pred, targets)
        assert losses["total"].item() == pytest.approx(losses["data"].item(), rel=1e-5)

    def test_predict_with_uncertainty(self):
        model = BayesianNeuralODE(state_dim=2, solver="euler", hidden_dim=16, num_layers=2)
        z0 = torch.randn(3, 2)
        t = torch.linspace(0, 0.1, 5)
        mean, std = model.predict_with_uncertainty(z0, t, num_samples=10)
        assert mean.shape == (3, 5, 2)
        assert std.shape == (3, 5, 2)
        assert torch.all(std >= 0)

    def test_predict_with_uncertainty_restores_mode(self):
        """predict_with_uncertainty should restore training mode."""
        model = BayesianNeuralODE(state_dim=2, solver="euler", hidden_dim=16, num_layers=2)
        model.train()
        z0 = torch.randn(2, 2)
        t = torch.linspace(0, 0.1, 3)
        model.predict_with_uncertainty(z0, t, num_samples=5)
        assert model.training  # Should be restored

    def test_uncertainty_std_nonzero(self):
        """Bayesian model should produce non-zero uncertainty."""
        model = BayesianNeuralODE(state_dim=2, solver="euler", hidden_dim=16, num_layers=2)
        z0 = torch.randn(3, 2)
        t = torch.linspace(0, 0.1, 5)
        _, std = model.predict_with_uncertainty(z0, t, num_samples=20)
        # At least some std should be > 0
        assert std.max().item() > 0

    def test_gradients_flow(self):
        model = BayesianNeuralODE(
            state_dim=2, solver="euler", adjoint=False, hidden_dim=16, num_layers=2
        )
        z0 = torch.randn(2, 2)
        t = torch.linspace(0, 0.05, 3)
        pred = model(z0, t)
        targets = torch.randn(2, 3, 2)
        losses = model.compute_loss(pred, targets)
        losses["total"].backward()
        grad_count = sum(1 for p in model.parameters() if p.grad is not None)
        assert grad_count > 0

    def test_train_step(self):
        model = BayesianNeuralODE(
            state_dim=2, solver="euler", adjoint=False, hidden_dim=16, num_layers=2
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        batch = {
            "z0": torch.randn(3, 2),
            "t_span": torch.linspace(0, 0.1, 5),
            "targets": torch.randn(3, 5, 2),
        }
        losses = model.train_step(batch, optimizer)
        assert "total" in losses
        assert isinstance(losses["total"], float)

    def test_predict_method(self):
        model = BayesianNeuralODE(state_dim=2, solver="euler", hidden_dim=16, num_layers=2)
        z0 = torch.randn(3, 2)
        t = torch.linspace(0, 0.1, 5)
        pred = model.predict(z0, t)
        assert pred.shape == (3, 5, 2)

    def test_registry(self):
        assert "bayesian_neural_ode" in NEURAL_DE_REGISTRY
        cls = NEURAL_DE_REGISTRY.get("bayesian_neural_ode")
        assert cls is BayesianNeuralODE

    def test_different_solvers(self):
        for solver in ["euler", "rk4"]:
            model = BayesianNeuralODE(state_dim=2, solver=solver, hidden_dim=16, num_layers=2)
            z0 = torch.randn(2, 2) * 0.1
            t = torch.linspace(0, 0.05, 3)
            out = model(z0, t)
            assert out.shape == (2, 3, 2)
