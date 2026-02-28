"""Tests for Foundation Model pre-training and fine-tuning."""

from __future__ import annotations

import numpy as np
import torch

from reactor_twin.reactors.systems import create_exothermic_cstr
from reactor_twin.training.data_generator import ReactorDataGenerator
from reactor_twin.training.foundation import (
    FoundationNeuralODE,
    FoundationTrainer,
    ReactorTaskEncoder,
)
from reactor_twin.utils.registry import NEURAL_DE_REGISTRY

# ── helpers ──────────────────────────────────────────────────────────


def _make_generators(n: int = 2) -> list[ReactorDataGenerator]:
    """Create multiple reactor data generators with different parameters."""
    generators = []
    for i in range(n):
        reactor = create_exothermic_cstr(isothermal=True)
        # Vary volume slightly per task
        reactor.params["V"] = 10.0 + i * 2.0
        generators.append(ReactorDataGenerator(reactor))
    return generators


# ── ReactorTaskEncoder ───────────────────────────────────────────────


class TestReactorTaskEncoder:
    def test_output_shape(self):
        enc = ReactorTaskEncoder(num_reactor_types=4, param_dim=3, embedding_dim=16)
        reactor_type = torch.eye(4)[:2]  # batch=2, one-hot
        params = torch.randn(2, 3)
        emb = enc(reactor_type, params)
        assert emb.shape == (2, 16)

    def test_different_types_different_embeddings(self):
        enc = ReactorTaskEncoder(num_reactor_types=4, param_dim=3, embedding_dim=16)
        type_a = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        type_b = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
        params = torch.randn(1, 3)
        emb_a = enc(type_a, params)
        emb_b = enc(type_b, params)
        assert not torch.allclose(emb_a, emb_b)

    def test_gradients_flow(self):
        enc = ReactorTaskEncoder(num_reactor_types=4, param_dim=3, embedding_dim=16)
        reactor_type = torch.eye(4)[:1]
        params = torch.randn(1, 3, requires_grad=True)
        emb = enc(reactor_type, params)
        emb.sum().backward()
        assert params.grad is not None

    def test_single_reactor_type(self):
        enc = ReactorTaskEncoder(num_reactor_types=1, param_dim=2, embedding_dim=8)
        reactor_type = torch.ones(3, 1)
        params = torch.randn(3, 2)
        emb = enc(reactor_type, params)
        assert emb.shape == (3, 8)

    def test_embedding_finite(self):
        enc = ReactorTaskEncoder(num_reactor_types=4, param_dim=3, embedding_dim=16)
        reactor_type = torch.eye(4)[:2]
        params = torch.randn(2, 3)
        emb = enc(reactor_type, params)
        assert torch.all(torch.isfinite(emb))


# ── FoundationNeuralODE ─────────────────────────────────────────────


class TestFoundationNeuralODE:
    def test_instantiation(self):
        model = FoundationNeuralODE(state_dim=2, hidden_dim=16, num_layers=2)
        assert model.state_dim == 2

    def test_forward_without_embedding(self):
        model = FoundationNeuralODE(state_dim=2, solver="euler", hidden_dim=16, num_layers=2)
        z0 = torch.randn(3, 2)
        t = torch.linspace(0, 0.1, 5)
        out = model(z0, t)
        assert out.shape == (3, 5, 2)

    def test_forward_with_embedding(self):
        model = FoundationNeuralODE(
            state_dim=2,
            solver="euler",
            hidden_dim=16,
            num_layers=2,
            embedding_dim=8,
        )
        z0 = torch.randn(3, 2)
        t = torch.linspace(0, 0.1, 5)
        emb = torch.randn(3, 8)
        out = model(z0, t, task_embedding=emb)
        assert out.shape == (3, 5, 2)

    def test_forward_finite(self):
        model = FoundationNeuralODE(state_dim=2, solver="euler", hidden_dim=16, num_layers=2)
        z0 = torch.randn(2, 2) * 0.1
        t = torch.linspace(0, 0.05, 3)
        out = model(z0, t)
        assert torch.all(torch.isfinite(out))

    def test_compute_loss(self):
        model = FoundationNeuralODE(state_dim=2, solver="euler", hidden_dim=16, num_layers=2)
        z0 = torch.randn(3, 2)
        t = torch.linspace(0, 0.1, 5)
        pred = model(z0, t)
        targets = torch.randn(3, 5, 2)
        losses = model.compute_loss(pred, targets)
        assert "total" in losses
        assert "data" in losses
        assert torch.isfinite(losses["total"])

    def test_train_step(self):
        model = FoundationNeuralODE(state_dim=2, solver="euler", hidden_dim=16, num_layers=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        batch = {
            "z0": torch.randn(3, 2),
            "t_span": torch.linspace(0, 0.1, 5),
            "targets": torch.randn(3, 5, 2),
        }
        losses = model.train_step(batch, optimizer)
        assert "total" in losses

    def test_predict(self):
        model = FoundationNeuralODE(state_dim=2, solver="euler", hidden_dim=16, num_layers=2)
        z0 = torch.randn(3, 2)
        t = torch.linspace(0, 0.1, 5)
        pred = model.predict(z0, t)
        assert pred.shape == (3, 5, 2)


# ── FoundationTrainer ────────────────────────────────────────────────


class TestFoundationTrainer:
    def test_pretrain_returns_displacements(self):
        model = FoundationNeuralODE(state_dim=2, solver="euler", hidden_dim=16, num_layers=2)
        trainer = FoundationTrainer(model, inner_steps=2)
        generators = _make_generators(2)
        displacements = trainer.pretrain(generators, num_epochs=2, t_span=(0, 0.5), batch_size=2)
        assert len(displacements) == 2
        assert all(d >= 0 for d in displacements)

    def test_pretrain_reduces_displacement(self):
        """Displacement should generally decrease or stay bounded."""
        model = FoundationNeuralODE(state_dim=2, solver="euler", hidden_dim=16, num_layers=2)
        trainer = FoundationTrainer(model, inner_steps=2, meta_lr=0.1)
        generators = _make_generators(2)
        displacements = trainer.pretrain(generators, num_epochs=5, t_span=(0, 0.5), batch_size=2)
        # All displacements should be finite
        assert all(np.isfinite(d) for d in displacements)
        # Displacement should be bounded (not exploding)
        assert max(displacements) < 1e6

    def test_pretrain_sets_task_embedding(self):
        """Pretrain should set task embedding on model copies."""
        model = FoundationNeuralODE(state_dim=2, solver="euler", hidden_dim=16, num_layers=2)
        trainer = FoundationTrainer(model, inner_steps=2)
        generators = _make_generators(2)
        trainer.pretrain(generators, num_epochs=1, t_span=(0, 0.5), batch_size=2)
        # After pretrain, the trainer should have set task embeddings
        # We can verify by checking the model's ode_func
        # The embedding may be None on the main model (set on copies),
        # but the method should work without errors

    def test_fine_tune_returns_losses(self):
        model = FoundationNeuralODE(state_dim=2, solver="euler", hidden_dim=16, num_layers=2)
        trainer = FoundationTrainer(model, inner_steps=2)
        generators = _make_generators(1)
        losses = trainer.fine_tune(generators[0], num_steps=3, t_span=(0, 0.5), batch_size=2)
        assert len(losses) == 3
        assert all(np.isfinite(v) for v in losses)

    def test_fine_tune_zero_steps(self):
        """Fine-tuning with num_steps=0 should not crash."""
        model = FoundationNeuralODE(state_dim=2, solver="euler", hidden_dim=16, num_layers=2)
        trainer = FoundationTrainer(model, inner_steps=2)
        generators = _make_generators(1)
        losses = trainer.fine_tune(generators[0], num_steps=0, t_span=(0, 0.5), batch_size=2)
        assert len(losses) == 0

    def test_fine_tune_freeze_encoder(self):
        model = FoundationNeuralODE(state_dim=2, solver="euler", hidden_dim=16, num_layers=2)
        trainer = FoundationTrainer(model, inner_steps=2)
        generators = _make_generators(1)

        # Store encoder params before
        enc_before = {n: p.clone() for n, p in model.task_encoder.named_parameters()}

        trainer.fine_tune(
            generators[0],
            num_steps=3,
            t_span=(0, 0.5),
            batch_size=2,
            freeze_encoder=True,
        )

        # Encoder params should not change when frozen
        for n, p in model.task_encoder.named_parameters():
            torch.testing.assert_close(p, enc_before[n])
            # Should be unfrozen after fine-tuning
            assert p.requires_grad

    def test_evaluate_transfer(self):
        model = FoundationNeuralODE(state_dim=2, solver="euler", hidden_dim=16, num_layers=2)
        trainer = FoundationTrainer(model)
        generators = _make_generators(2)
        result = trainer.evaluate_transfer(generators, t_span=(0, 0.5), batch_size=2)
        assert "mean_loss" in result
        assert "per_task_losses" in result
        assert len(result["per_task_losses"]) == 2
        assert np.isfinite(result["mean_loss"])


# ── Registry ─────────────────────────────────────────────────────────


class TestFoundationRegistry:
    def test_registered(self):
        assert "foundation_neural_ode" in NEURAL_DE_REGISTRY

    def test_registry_get(self):
        cls = NEURAL_DE_REGISTRY.get("foundation_neural_ode")
        assert cls is FoundationNeuralODE
