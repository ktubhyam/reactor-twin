"""Tests for reactor_twin.training.distributed (single-GPU / CPU fallback)."""

from __future__ import annotations

import pytest
import torch

from reactor_twin.core.neural_ode import NeuralODE
from reactor_twin.core.ode_func import MLPODEFunc
from reactor_twin.training.distributed import (
    DistributedTrainer,
    cleanup_distributed,
    setup_distributed,
)


@pytest.fixture
def simple_model():
    ode_func = MLPODEFunc(state_dim=2, hidden_dim=16, num_layers=2)
    return NeuralODE(state_dim=2, ode_func=ode_func)


class TestSetup:
    def test_single_process_setup(self):
        """Without environment variables, defaults to rank=0, world_size=1."""
        rank, world_size = setup_distributed(backend="gloo")
        assert rank == 0
        assert world_size == 1
        # cleanup is a no-op when not initialized
        cleanup_distributed()


class TestDistributedTrainer:
    def test_init_single_gpu(self, simple_model):
        """DistributedTrainer works with world_size=1 (no DDP wrapping)."""
        from reactor_twin.reactors.systems import create_exothermic_cstr
        from reactor_twin.training.data_generator import ReactorDataGenerator

        reactor = create_exothermic_cstr()
        gen = ReactorDataGenerator(reactor)
        trainer = DistributedTrainer(
            model=simple_model,
            data_generator=gen,
            rank=0,
            world_size=1,
        )
        assert trainer.device.type in ("cpu", "cuda")
        assert trainer._raw_model is simple_model

    def test_shard_data(self, simple_model):
        from reactor_twin.reactors.systems import create_exothermic_cstr
        from reactor_twin.training.data_generator import ReactorDataGenerator

        reactor = create_exothermic_cstr()
        gen = ReactorDataGenerator(reactor)
        trainer = DistributedTrainer(
            model=simple_model,
            data_generator=gen,
            rank=0,
            world_size=2,
        )
        data = [{"a": torch.tensor(i)} for i in range(10)]
        shard = trainer._shard_data(data)
        assert len(shard) == 5  # every other element

    def test_accumulation_steps(self, simple_model):
        from reactor_twin.reactors.systems import create_exothermic_cstr
        from reactor_twin.training.data_generator import ReactorDataGenerator

        reactor = create_exothermic_cstr()
        gen = ReactorDataGenerator(reactor)
        trainer = DistributedTrainer(
            model=simple_model,
            data_generator=gen,
            rank=0,
            world_size=1,
            accumulation_steps=4,
        )
        assert trainer.accumulation_steps == 4

    def test_save_checkpoint_rank0(self, simple_model, tmp_path):
        from reactor_twin.reactors.systems import create_exothermic_cstr
        from reactor_twin.training.data_generator import ReactorDataGenerator

        reactor = create_exothermic_cstr()
        gen = ReactorDataGenerator(reactor)
        trainer = DistributedTrainer(
            model=simple_model,
            data_generator=gen,
            rank=0,
            world_size=1,
        )
        trainer.save_checkpoint(tmp_path, "test.pt")
        assert (tmp_path / "test.pt").exists()

    def test_save_checkpoint_rank1_noop(self, simple_model, tmp_path):
        from reactor_twin.reactors.systems import create_exothermic_cstr
        from reactor_twin.training.data_generator import ReactorDataGenerator

        reactor = create_exothermic_cstr()
        gen = ReactorDataGenerator(reactor)
        trainer = DistributedTrainer(
            model=simple_model,
            data_generator=gen,
            rank=1,
            world_size=2,
        )
        trainer.save_checkpoint(tmp_path, "test.pt")
        assert not (tmp_path / "test.pt").exists()
