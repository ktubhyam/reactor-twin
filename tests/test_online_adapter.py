"""Tests for reactor_twin.digital_twin.online_adapter."""

from __future__ import annotations

import pytest
import torch

from reactor_twin.core.neural_ode import NeuralODE
from reactor_twin.core.ode_func import MLPODEFunc
from reactor_twin.digital_twin.online_adapter import (
    ElasticWeightConsolidation,
    OnlineAdapter,
    ReplayBuffer,
)


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(42)


@pytest.fixture
def simple_model():
    ode_func = MLPODEFunc(state_dim=2, hidden_dim=16, num_layers=2)
    return NeuralODE(state_dim=2, ode_func=ode_func)


@pytest.fixture
def t_span():
    return torch.linspace(0, 1, 10)


# ── ReplayBuffer ──────────────────────────────────────────────────


class TestReplayBuffer:
    def test_add_and_len(self):
        buf = ReplayBuffer(capacity=100)
        assert len(buf) == 0
        buf.add(torch.randn(2), torch.linspace(0, 1, 5), torch.randn(5, 2))
        assert len(buf) == 1

    def test_capacity_overflow(self):
        buf = ReplayBuffer(capacity=3)
        for _ in range(5):
            buf.add(torch.randn(2), torch.linspace(0, 1, 5), torch.randn(5, 2))
        assert len(buf) == 3

    def test_sample_returns_correct_keys(self):
        buf = ReplayBuffer(capacity=10)
        for _ in range(5):
            buf.add(torch.randn(2), torch.linspace(0, 1, 5), torch.randn(5, 2))
        batch = buf.sample(3)
        assert "z0" in batch
        assert "t_span" in batch
        assert "targets" in batch

    def test_sample_batch_dim(self):
        buf = ReplayBuffer(capacity=10)
        for _ in range(5):
            buf.add(torch.randn(2), torch.linspace(0, 1, 5), torch.randn(5, 2))
        batch = buf.sample(3)
        assert batch["z0"].ndim == 2
        assert batch["z0"].shape[0] == 3
        assert batch["targets"].ndim == 3

    def test_sample_with_batch_z0(self):
        """z0 already has batch dim — should not double-unsqueeze."""
        buf = ReplayBuffer(capacity=10)
        for _ in range(3):
            buf.add(torch.randn(1, 2), torch.linspace(0, 1, 5), torch.randn(1, 5, 2))
        batch = buf.sample(2)
        assert batch["z0"].ndim == 2
        assert batch["z0"].shape[1] == 2

    def test_sample_clamps_to_buffer_size(self):
        buf = ReplayBuffer(capacity=10)
        for _ in range(2):
            buf.add(torch.randn(2), torch.linspace(0, 1, 5), torch.randn(5, 2))
        batch = buf.sample(100)
        assert batch["z0"].shape[0] == 2


# ── ElasticWeightConsolidation ────────────────────────────────────


class TestEWC:
    def test_penalty_before_consolidation(self, simple_model):
        ewc = ElasticWeightConsolidation(simple_model, ewc_lambda=100.0)
        p = ewc.penalty()
        assert p.item() == pytest.approx(0.0)

    def test_consolidate_without_data(self, simple_model):
        ewc = ElasticWeightConsolidation(simple_model, ewc_lambda=100.0)
        ewc.consolidate(data_batches=None)
        assert ewc._consolidated is True
        # Fisher should be all ones
        for f in ewc._fisher_diag.values():
            assert torch.all(f == 1.0)

    def test_penalty_after_consolidation_is_zero(self, simple_model):
        """Right after consolidation (no param change), penalty should be ~0."""
        ewc = ElasticWeightConsolidation(simple_model, ewc_lambda=100.0)
        ewc.consolidate(data_batches=None)
        p = ewc.penalty()
        assert p.item() == pytest.approx(0.0, abs=1e-7)

    def test_penalty_increases_after_param_change(self, simple_model):
        ewc = ElasticWeightConsolidation(simple_model, ewc_lambda=100.0)
        ewc.consolidate(data_batches=None)
        # Change params
        with torch.no_grad():
            for p in simple_model.parameters():
                p.add_(torch.randn_like(p) * 0.1)
        penalty = ewc.penalty()
        assert penalty.item() > 0.0

    def test_consolidate_with_data(self, simple_model, t_span):
        ewc = ElasticWeightConsolidation(simple_model, ewc_lambda=100.0)
        z0 = torch.randn(4, 2)
        targets = torch.randn(4, 10, 2)
        batch = {"z0": z0, "t_span": t_span, "targets": targets}
        ewc.consolidate(data_batches=[batch], num_samples=1)
        assert ewc._consolidated is True
        # Fisher should have been estimated (not all ones)
        # (It could be zeros if grads are zero, but it ran)
        assert len(ewc._fisher_diag) > 0

    def test_penalty_is_differentiable(self, simple_model):
        ewc = ElasticWeightConsolidation(simple_model, ewc_lambda=100.0)
        ewc.consolidate(data_batches=None)
        with torch.no_grad():
            for p in simple_model.parameters():
                p.add_(0.01)
        p = ewc.penalty()
        p.backward()
        grads = [p.grad for p in simple_model.parameters() if p.grad is not None]
        assert len(grads) > 0


# ── OnlineAdapter ─────────────────────────────────────────────────


class TestOnlineAdapter:
    def test_init(self, simple_model):
        adapter = OnlineAdapter(simple_model, lr=1e-4, ewc_lambda=50.0)
        assert adapter.replay_ratio == 0.5
        assert len(adapter.replay_buffer) == 0

    def test_add_experience(self, simple_model, t_span):
        adapter = OnlineAdapter(simple_model)
        adapter.add_experience(torch.randn(2), t_span, torch.randn(10, 2))
        assert len(adapter.replay_buffer) == 1

    def test_adapt_returns_losses(self, simple_model, t_span):
        adapter = OnlineAdapter(simple_model, lr=1e-3)
        z0 = torch.randn(4, 2)
        targets = torch.randn(4, 10, 2)
        new_data = {"z0": z0, "t_span": t_span, "targets": targets}
        losses = adapter.adapt(new_data, num_steps=3)
        assert len(losses) == 3
        assert all(isinstance(v, float) for v in losses)

    def test_adapt_with_replay(self, simple_model, t_span):
        adapter = OnlineAdapter(simple_model, lr=1e-3)
        # Add some experience first
        for _ in range(5):
            adapter.add_experience(torch.randn(2), t_span, torch.randn(10, 2))
        z0 = torch.randn(4, 2)
        targets = torch.randn(4, 10, 2)
        new_data = {"z0": z0, "t_span": t_span, "targets": targets}
        losses = adapter.adapt(new_data, num_steps=3, batch_size=4)
        assert len(losses) == 3

    def test_consolidate(self, simple_model):
        adapter = OnlineAdapter(simple_model)
        adapter.consolidate()
        assert adapter.ewc._consolidated is True

    def test_adapt_with_1d_z0(self, simple_model, t_span):
        """adapt handles 1D z0 (single sample, no batch dim)."""
        adapter = OnlineAdapter(simple_model, lr=1e-3)
        new_data = {"z0": torch.randn(2), "t_span": t_span, "targets": torch.randn(10, 2)}
        losses = adapter.adapt(new_data, num_steps=2)
        assert len(losses) == 2
