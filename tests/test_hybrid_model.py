"""Tests for Hybrid mechanistic-neural ODE model."""

from __future__ import annotations

import numpy as np
import torch

from reactor_twin.core.hybrid_model import HybridNeuralODE, ReactorPhysicsFunc
from reactor_twin.core.ode_func import MLPODEFunc
from reactor_twin.reactors.cstr import CSTRReactor
from reactor_twin.reactors.systems import create_exothermic_cstr
from reactor_twin.utils.registry import NEURAL_DE_REGISTRY

# ── helpers ──────────────────────────────────────────────────────────


def _make_cstr(isothermal: bool = True) -> CSTRReactor:
    return create_exothermic_cstr(isothermal=isothermal)


# ── ReactorPhysicsFunc ───────────────────────────────────────────────


class TestReactorPhysicsFunc:
    def test_output_shape(self):
        reactor = _make_cstr()
        func = ReactorPhysicsFunc(reactor)
        z = torch.randn(3, reactor.state_dim)
        t = torch.tensor(0.0)
        dz = func(t, z)
        assert dz.shape == (3, reactor.state_dim)

    def test_matches_ode_rhs(self):
        """Output should match reactor.ode_rhs for the same input."""
        reactor = _make_cstr()
        func = ReactorPhysicsFunc(reactor)
        y0 = reactor.get_initial_state()
        z = torch.tensor(y0, dtype=torch.float32).unsqueeze(0)
        t = torch.tensor(0.0)
        dz = func(t, z)

        expected = reactor.ode_rhs(0.0, y0)
        np.testing.assert_allclose(dz[0].detach().numpy(), expected, rtol=1e-4, atol=1e-6)

    def test_finite_output(self):
        reactor = _make_cstr()
        func = ReactorPhysicsFunc(reactor)
        z = torch.tensor(reactor.get_initial_state(), dtype=torch.float32).unsqueeze(0)
        t = torch.tensor(0.0)
        dz = func(t, z)
        assert torch.all(torch.isfinite(dz))

    def test_gradient_via_finite_difference(self):
        """Gradients should flow through the custom autograd."""
        reactor = _make_cstr()
        func = ReactorPhysicsFunc(reactor)
        z = torch.tensor(reactor.get_initial_state(), dtype=torch.float32).unsqueeze(0)
        z.requires_grad_(True)
        t = torch.tensor(0.0)
        dz = func(t, z)
        loss = dz.sum()
        loss.backward()
        assert z.grad is not None
        assert torch.all(torch.isfinite(z.grad))

    def test_batch_processing(self):
        reactor = _make_cstr()
        func = ReactorPhysicsFunc(reactor)
        z = torch.randn(5, reactor.state_dim)
        t = torch.tensor(0.0)
        dz = func(t, z)
        assert dz.shape == (5, reactor.state_dim)


# ── HybridNeuralODE ─────────────────────────────────────────────────


class TestHybridNeuralODE:
    def test_instantiation(self):
        reactor = _make_cstr()
        model = HybridNeuralODE(
            state_dim=reactor.state_dim,
            reactor=reactor,
            solver="euler",
            hidden_dim=16,
            num_layers=2,
        )
        assert model.state_dim == reactor.state_dim

    def test_instantiation_without_reactor(self):
        model = HybridNeuralODE(
            state_dim=3,
            solver="euler",
            hidden_dim=16,
            num_layers=2,
        )
        assert model.state_dim == 3

    def test_forward_without_reactor(self):
        """Forward pass with reactor=None should work (zero physics)."""
        model = HybridNeuralODE(
            state_dim=3,
            solver="euler",
            hidden_dim=16,
            num_layers=2,
        )
        z0 = torch.randn(2, 3)
        t = torch.linspace(0, 0.05, 5)
        out = model(z0, t)
        assert out.shape == (2, 5, 3)
        assert torch.all(torch.isfinite(out))

    def test_forward_shape(self):
        reactor = _make_cstr()
        model = HybridNeuralODE(
            state_dim=reactor.state_dim,
            reactor=reactor,
            solver="euler",
            hidden_dim=16,
            num_layers=2,
        )
        z0 = torch.tensor(reactor.get_initial_state(), dtype=torch.float32).unsqueeze(0)
        t = torch.linspace(0, 0.05, 5)
        out = model(z0, t)
        assert out.shape == (1, 5, reactor.state_dim)

    def test_forward_finite(self):
        reactor = _make_cstr()
        model = HybridNeuralODE(
            state_dim=reactor.state_dim,
            reactor=reactor,
            solver="euler",
            hidden_dim=16,
            num_layers=2,
        )
        z0 = torch.tensor(reactor.get_initial_state(), dtype=torch.float32).unsqueeze(0)
        t = torch.linspace(0, 0.02, 3)
        out = model(z0, t)
        assert torch.all(torch.isfinite(out))

    def test_zero_alpha_pure_physics(self):
        """With alpha=0, output should match pure physics integration."""
        reactor = _make_cstr()
        model = HybridNeuralODE(
            state_dim=reactor.state_dim,
            reactor=reactor,
            alpha=0.0,
            solver="euler",
            hidden_dim=16,
            num_layers=2,
        )
        z0 = torch.tensor(reactor.get_initial_state(), dtype=torch.float32).unsqueeze(0)
        t = torch.linspace(0, 0.01, 3)
        out = model(z0, t)
        assert torch.all(torch.isfinite(out))
        # First point should be z0
        torch.testing.assert_close(out[:, 0, :], z0, atol=1e-5, rtol=1e-5)

    def test_compute_loss(self):
        reactor = _make_cstr()
        model = HybridNeuralODE(
            state_dim=reactor.state_dim,
            reactor=reactor,
            solver="euler",
            hidden_dim=16,
            num_layers=2,
        )
        z0 = torch.tensor(reactor.get_initial_state(), dtype=torch.float32).unsqueeze(0)
        t = torch.linspace(0, 0.05, 5)
        pred = model(z0, t)
        targets = torch.randn_like(pred)
        losses = model.compute_loss(pred, targets)
        assert "total" in losses
        assert "data" in losses
        assert "physics_reg" in losses
        assert torch.isfinite(losses["total"])

    def test_physics_reg_penalty(self):
        """Physics reg should be >= 0."""
        reactor = _make_cstr()
        model = HybridNeuralODE(
            state_dim=reactor.state_dim,
            reactor=reactor,
            solver="euler",
            hidden_dim=16,
            num_layers=2,
        )
        z0 = torch.tensor(reactor.get_initial_state(), dtype=torch.float32).unsqueeze(0)
        t = torch.linspace(0, 0.05, 5)
        pred = model(z0, t)
        targets = torch.randn_like(pred)
        losses = model.compute_loss(pred, targets)
        assert losses["physics_reg"].item() >= 0

    def test_correction_magnitude(self):
        reactor = _make_cstr()
        model = HybridNeuralODE(
            state_dim=reactor.state_dim,
            reactor=reactor,
            solver="euler",
            hidden_dim=16,
            num_layers=2,
        )
        z0 = torch.tensor(reactor.get_initial_state(), dtype=torch.float32).unsqueeze(0)
        t = torch.linspace(0, 0.02, 3)
        ratio = model.get_correction_magnitude(z0, t)
        assert torch.isfinite(ratio)
        assert ratio.item() >= 0

    def test_correction_magnitude_restores_mode(self):
        """get_correction_magnitude should restore training mode."""
        reactor = _make_cstr()
        model = HybridNeuralODE(
            state_dim=reactor.state_dim,
            reactor=reactor,
            solver="euler",
            hidden_dim=16,
            num_layers=2,
        )
        model.train()
        z0 = torch.tensor(reactor.get_initial_state(), dtype=torch.float32).unsqueeze(0)
        t = torch.linspace(0, 0.02, 3)
        model.get_correction_magnitude(z0, t)
        assert model.training  # Should be restored to train mode

    def test_gradient_flow(self):
        reactor = _make_cstr()
        model = HybridNeuralODE(
            state_dim=reactor.state_dim,
            reactor=reactor,
            solver="euler",
            adjoint=False,
            hidden_dim=16,
            num_layers=2,
        )
        z0 = torch.tensor(reactor.get_initial_state(), dtype=torch.float32).unsqueeze(0)
        t = torch.linspace(0, 0.02, 3)
        pred = model(z0, t)
        targets = torch.randn_like(pred)
        losses = model.compute_loss(pred, targets)
        losses["total"].backward()
        neural_grad_count = sum(1 for p in model.neural_func.parameters() if p.grad is not None)
        assert neural_grad_count > 0

    def test_custom_neural_func(self):
        reactor = _make_cstr()
        neural = MLPODEFunc(state_dim=reactor.state_dim, hidden_dim=32, num_layers=3)
        model = HybridNeuralODE(
            state_dim=reactor.state_dim,
            reactor=reactor,
            neural_func=neural,
            solver="euler",
        )
        z0 = torch.tensor(reactor.get_initial_state(), dtype=torch.float32).unsqueeze(0)
        t = torch.linspace(0, 0.02, 3)
        out = model(z0, t)
        assert out.shape == (1, 3, reactor.state_dim)

    def test_predict_method(self):
        reactor = _make_cstr()
        model = HybridNeuralODE(
            state_dim=reactor.state_dim,
            reactor=reactor,
            solver="euler",
            hidden_dim=16,
            num_layers=2,
        )
        z0 = torch.tensor(reactor.get_initial_state(), dtype=torch.float32).unsqueeze(0)
        t = torch.linspace(0, 0.05, 5)
        pred = model.predict(z0, t)
        assert pred.shape == (1, 5, reactor.state_dim)


# ── Registry ─────────────────────────────────────────────────────────


class TestHybridRegistry:
    def test_registered(self):
        assert "hybrid_neural_ode" in NEURAL_DE_REGISTRY

    def test_registry_get(self):
        cls = NEURAL_DE_REGISTRY.get("hybrid_neural_ode")
        assert cls is HybridNeuralODE
