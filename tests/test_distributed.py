"""Tests for reactor_twin.training.distributed (single-GPU / CPU fallback)."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
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
    return NeuralODE(state_dim=2, ode_func=ode_func, solver="euler", adjoint=False)


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

        reactor = create_exothermic_cstr(isothermal=True)
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

        reactor = create_exothermic_cstr(isothermal=True)
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

        reactor = create_exothermic_cstr(isothermal=True)
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

        reactor = create_exothermic_cstr(isothermal=True)
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

        reactor = create_exothermic_cstr(isothermal=True)
        gen = ReactorDataGenerator(reactor)
        trainer = DistributedTrainer(
            model=simple_model,
            data_generator=gen,
            rank=1,
            world_size=2,
        )
        trainer.save_checkpoint(tmp_path, "test.pt")
        assert not (tmp_path / "test.pt").exists()

    def test_train_epoch(self, simple_model):
        import numpy as np

        from reactor_twin.reactors.systems import create_exothermic_cstr
        from reactor_twin.training.data_generator import ReactorDataGenerator

        reactor = create_exothermic_cstr(isothermal=True)
        gen = ReactorDataGenerator(reactor)
        trainer = DistributedTrainer(
            model=simple_model,
            data_generator=gen,
            rank=0,
            world_size=1,
        )
        t_span = (0.0, 1.0)
        t_eval = np.linspace(0.0, 1.0, 20)
        train_data = gen.generate_dataset(10, t_span, t_eval, batch_size=8)
        metrics = trainer.train_epoch(train_data, log_interval=100)
        assert isinstance(metrics, dict)
        assert "total" in metrics
        assert metrics["total"] > 0

    def test_validate(self, simple_model):
        import numpy as np

        from reactor_twin.reactors.systems import create_exothermic_cstr
        from reactor_twin.training.data_generator import ReactorDataGenerator

        reactor = create_exothermic_cstr(isothermal=True)
        gen = ReactorDataGenerator(reactor)
        trainer = DistributedTrainer(
            model=simple_model,
            data_generator=gen,
            rank=0,
            world_size=1,
        )
        t_span = (0.0, 1.0)
        t_eval = np.linspace(0.0, 1.0, 20)
        val_data = gen.generate_dataset(5, t_span, t_eval, batch_size=8)
        metrics = trainer.validate(val_data)
        assert isinstance(metrics, dict)
        assert "total" in metrics

    def test_train_full_loop(self, simple_model):
        import numpy as np

        from reactor_twin.reactors.systems import create_exothermic_cstr
        from reactor_twin.training.data_generator import ReactorDataGenerator

        reactor = create_exothermic_cstr(isothermal=True)
        gen = ReactorDataGenerator(reactor)
        trainer = DistributedTrainer(
            model=simple_model,
            data_generator=gen,
            rank=0,
            world_size=1,
        )
        t_eval = np.linspace(0.0, 1.0, 20)
        history = trainer.train(
            num_epochs=1,
            t_span=(0.0, 1.0),
            t_eval=t_eval,
            train_trajectories=10,
            val_trajectories=5,
            batch_size=8,
            log_interval=100,
            val_interval=1,
        )
        assert "train_loss" in history
        assert len(history["train_loss"]) == 1
        assert "val_loss" in history
        assert len(history["val_loss"]) == 1

    def test_shard_data_various_world_sizes(self, simple_model):
        from reactor_twin.reactors.systems import create_exothermic_cstr
        from reactor_twin.training.data_generator import ReactorDataGenerator

        reactor = create_exothermic_cstr()
        gen = ReactorDataGenerator(reactor)
        data = [{"a": torch.tensor(i)} for i in range(12)]

        for ws in [1, 2, 3, 4]:
            trainer = DistributedTrainer(
                model=simple_model, data_generator=gen, rank=0, world_size=ws
            )
            shard = trainer._shard_data(data)
            assert len(shard) == len(data[0::ws])

    def test_gradient_accumulation(self, simple_model):
        import numpy as np

        from reactor_twin.reactors.systems import create_exothermic_cstr
        from reactor_twin.training.data_generator import ReactorDataGenerator

        reactor = create_exothermic_cstr(isothermal=True)
        gen = ReactorDataGenerator(reactor)
        trainer = DistributedTrainer(
            model=simple_model,
            data_generator=gen,
            rank=0,
            world_size=1,
            accumulation_steps=2,
        )
        t_span = (0.0, 1.0)
        t_eval = np.linspace(0.0, 1.0, 20)
        train_data = gen.generate_dataset(10, t_span, t_eval, batch_size=8)
        initial_step = trainer.global_step
        trainer.train_epoch(train_data, log_interval=100)
        # With accumulation_steps=2, global_step should advance less
        assert trainer.global_step > initial_step

    def test_lr_scheduler(self, simple_model):
        import numpy as np

        from reactor_twin.reactors.systems import create_exothermic_cstr
        from reactor_twin.training.data_generator import ReactorDataGenerator

        reactor = create_exothermic_cstr(isothermal=True)
        gen = ReactorDataGenerator(reactor)
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
        trainer = DistributedTrainer(
            model=simple_model,
            data_generator=gen,
            rank=0,
            world_size=1,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        t_eval = np.linspace(0.0, 1.0, 20)
        initial_lr = optimizer.param_groups[0]["lr"]
        trainer.train(
            num_epochs=2,
            t_span=(0.0, 1.0),
            t_eval=t_eval,
            train_trajectories=5,
            val_trajectories=3,
            batch_size=8,
            log_interval=100,
            val_interval=5,
        )
        final_lr = optimizer.param_groups[0]["lr"]
        assert final_lr < initial_lr

    def test_train_epoch_log_interval_triggers(self, simple_model):
        """Cover line 211: log_interval triggers logging output."""
        from reactor_twin.reactors.systems import create_exothermic_cstr
        from reactor_twin.training.data_generator import ReactorDataGenerator

        reactor = create_exothermic_cstr(isothermal=True)
        gen = ReactorDataGenerator(reactor)
        trainer = DistributedTrainer(
            model=simple_model,
            data_generator=gen,
            rank=0,
            world_size=1,
        )
        t_span = (0.0, 1.0)
        t_eval = np.linspace(0.0, 1.0, 20)
        # Generate enough data so that log_interval=1 triggers logging
        train_data = gen.generate_dataset(10, t_span, t_eval, batch_size=4)
        # log_interval=1 => every batch triggers the logger.info on line 211
        metrics = trainer.train_epoch(train_data, log_interval=1)
        assert "total" in metrics
        assert metrics["total"] > 0

    def test_train_epoch_gradient_flush(self, simple_model):
        """Cover lines 218-220: flush remaining gradients when batch count
        is not divisible by accumulation_steps."""
        from reactor_twin.reactors.systems import create_exothermic_cstr
        from reactor_twin.training.data_generator import ReactorDataGenerator

        reactor = create_exothermic_cstr(isothermal=True)
        gen = ReactorDataGenerator(reactor)
        trainer = DistributedTrainer(
            model=simple_model,
            data_generator=gen,
            rank=0,
            world_size=1,
            accumulation_steps=3,  # 3 won't evenly divide typical batch counts
        )
        t_span = (0.0, 1.0)
        t_eval = np.linspace(0.0, 1.0, 20)
        # Generate 5 trajectories with batch_size=2 => 3 batches (sharded to 3 for ws=1)
        # 3 batches with accumulation_steps=3 => 3 % 3 == 0, no flush
        # Instead: 7 trajectories, batch_size=2 => 4 batches, 4 % 3 != 0 => flush
        train_data = gen.generate_dataset(7, t_span, t_eval, batch_size=2)
        initial_step = trainer.global_step
        metrics = trainer.train_epoch(train_data, log_interval=100)
        # global_step should have advanced (1 full accumulation + 1 flush)
        assert trainer.global_step > initial_step
        assert "total" in metrics

    def test_train_with_checkpoints(self, simple_model, tmp_path):
        """Cover lines 335, 341: best checkpoint save and periodic checkpoint save."""
        from reactor_twin.reactors.systems import create_exothermic_cstr
        from reactor_twin.training.data_generator import ReactorDataGenerator

        reactor = create_exothermic_cstr(isothermal=True)
        gen = ReactorDataGenerator(reactor)
        trainer = DistributedTrainer(
            model=simple_model,
            data_generator=gen,
            rank=0,
            world_size=1,
        )
        t_eval = np.linspace(0.0, 1.0, 20)
        # Run 10 epochs with val_interval=1 (triggers best checkpoint every improving epoch)
        # and checkpoint_dir set (triggers periodic save at epoch 10)
        history = trainer.train(
            num_epochs=10,
            t_span=(0.0, 1.0),
            t_eval=t_eval,
            train_trajectories=5,
            val_trajectories=3,
            batch_size=8,
            log_interval=100,
            val_interval=1,
            checkpoint_dir=tmp_path,
        )
        assert "train_loss" in history
        assert len(history["train_loss"]) == 10
        # Best model checkpoint should exist (line 335)
        assert (tmp_path / "best_model.pt").exists()
        # Periodic checkpoint at epoch 9 (index 9, epoch+1=10 which is divisible by 10)
        assert (tmp_path / "checkpoint_epoch_9.pt").exists()


class TestSetupDistributedMultiGPU:
    """Tests for setup_distributed and cleanup_distributed with mocked distributed env."""

    @patch.dict("os.environ", {"RANK": "1", "WORLD_SIZE": "4", "LOCAL_RANK": "1"})
    @patch("reactor_twin.training.distributed.torch.cuda.set_device")
    @patch("reactor_twin.training.distributed.dist.init_process_group")
    def test_setup_distributed_multi_gpu(self, mock_init_pg, mock_set_device):
        """Cover lines 46-50: dist.init_process_group when world_size > 1."""
        rank, world_size = setup_distributed(backend="nccl")
        assert rank == 1
        assert world_size == 4
        mock_init_pg.assert_called_once_with(backend="nccl", world_size=4, rank=1)
        mock_set_device.assert_called_once_with(1)

    @patch.dict("os.environ", {"RANK": "0", "WORLD_SIZE": "2", "LOCAL_RANK": "0"})
    @patch("reactor_twin.training.distributed.torch.cuda.set_device")
    @patch("reactor_twin.training.distributed.dist.init_process_group")
    def test_setup_distributed_with_init_method(self, mock_init_pg, mock_set_device):
        """Cover lines 46-50: init_method is passed through when provided."""
        rank, world_size = setup_distributed(
            backend="gloo", init_method="tcp://localhost:29500"
        )
        assert rank == 0
        assert world_size == 2
        mock_init_pg.assert_called_once_with(
            backend="gloo",
            world_size=2,
            rank=0,
            init_method="tcp://localhost:29500",
        )

    @patch("reactor_twin.training.distributed.dist.destroy_process_group")
    @patch("reactor_twin.training.distributed.dist.is_initialized", return_value=True)
    def test_cleanup_distributed_when_initialized(self, mock_is_init, mock_destroy):
        """Cover line 59: dist.destroy_process_group when is_initialized."""
        cleanup_distributed()
        mock_is_init.assert_called_once()
        mock_destroy.assert_called_once()


class TestDistributedTrainerCUDAPaths:
    """Tests for CUDA device selection paths with mocked torch.cuda."""

    def test_cuda_multi_gpu_device_selection(self):
        """Cover lines 109-110: CUDA device selection with world_size > 1."""
        ode_func = MLPODEFunc(state_dim=2, hidden_dim=16, num_layers=2)
        model = NeuralODE(state_dim=2, ode_func=ode_func, solver="euler", adjoint=False)

        from reactor_twin.reactors.systems import create_exothermic_cstr
        from reactor_twin.training.data_generator import ReactorDataGenerator

        reactor = create_exothermic_cstr(isothermal=True)
        gen = ReactorDataGenerator(reactor)

        with (
            patch("reactor_twin.training.distributed.torch.cuda.is_available", return_value=True),
            patch.dict("os.environ", {"LOCAL_RANK": "1"}),
            patch.object(model, "to"),  # Prevent actual CUDA move
            patch("reactor_twin.training.distributed.nn.parallel.DistributedDataParallel") as mock_ddp,
        ):
            mock_ddp.return_value = model
            mock_ddp.return_value.parameters = model.parameters
            trainer = DistributedTrainer(
                model=model,
                data_generator=gen,
                rank=1,
                world_size=2,
            )
            # line 109-110: device = cuda:1 from LOCAL_RANK
            assert trainer.device == torch.device("cuda:1")
            # line 120: DDP wrapping was called
            mock_ddp.assert_called_once()

    def test_cuda_single_gpu_device_selection(self):
        """Cover line 112: CUDA device selection with world_size = 1."""
        ode_func = MLPODEFunc(state_dim=2, hidden_dim=16, num_layers=2)
        model = NeuralODE(state_dim=2, ode_func=ode_func, solver="euler", adjoint=False)

        from reactor_twin.reactors.systems import create_exothermic_cstr
        from reactor_twin.training.data_generator import ReactorDataGenerator

        reactor = create_exothermic_cstr(isothermal=True)
        gen = ReactorDataGenerator(reactor)

        with (
            patch("reactor_twin.training.distributed.torch.cuda.is_available", return_value=True),
            patch.object(model, "to"),  # Prevent actual CUDA move
        ):
            trainer = DistributedTrainer(
                model=model,
                data_generator=gen,
                rank=0,
                world_size=1,
            )
            # line 112: device = cuda:0 when single GPU
            assert trainer.device == torch.device("cuda:0")


class TestDistributedTrainerAllReduce:
    """Tests for all-reduce in validate when world_size > 1."""

    def test_validate_all_reduce(self):
        """Cover lines 263-265: all-reduce when world_size > 1 and dist.is_initialized."""
        ode_func = MLPODEFunc(state_dim=2, hidden_dim=16, num_layers=2)
        model = NeuralODE(state_dim=2, ode_func=ode_func, solver="euler", adjoint=False)

        from reactor_twin.reactors.systems import create_exothermic_cstr
        from reactor_twin.training.data_generator import ReactorDataGenerator

        reactor = create_exothermic_cstr(isothermal=True)
        gen = ReactorDataGenerator(reactor)

        # Create trainer with world_size=2 but on CPU (no actual DDP)
        trainer = DistributedTrainer(
            model=model,
            data_generator=gen,
            rank=0,
            world_size=2,
        )

        t_span = (0.0, 1.0)
        t_eval = np.linspace(0.0, 1.0, 20)
        val_data = gen.generate_dataset(5, t_span, t_eval, batch_size=8)

        with (
            patch("reactor_twin.training.distributed.dist.is_initialized", return_value=True),
            patch("reactor_twin.training.distributed.dist.all_reduce") as mock_all_reduce,
        ):
            metrics = trainer.validate(val_data)
            # Lines 263-265: all_reduce should have been called
            mock_all_reduce.assert_called_once()
            assert "total" in metrics
