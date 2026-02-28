"""Distributed training for Neural Differential Equations.

Supports multi-GPU data-parallel training via ``torch.nn.parallel.DistributedDataParallel``
with gradient accumulation for effective large batch sizes.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import nn, optim

from reactor_twin.core.base import AbstractNeuralDE
from reactor_twin.training.data_generator import ReactorDataGenerator
from reactor_twin.training.losses import MultiObjectiveLoss

logger = logging.getLogger(__name__)


def setup_distributed(
    backend: str = "nccl",
    init_method: str | None = None,
) -> tuple[int, int]:
    """Initialize the distributed process group.

    Reads ``RANK``, ``WORLD_SIZE``, and ``LOCAL_RANK`` from environment
    variables (set by ``torchrun`` or ``torch.distributed.launch``).

    Args:
        backend: Communication backend (``"nccl"`` for GPU, ``"gloo"`` for CPU).
        init_method: URL for rendezvous (defaults to env://).

    Returns:
        ``(rank, world_size)``
    """
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size > 1:
        kwargs: dict[str, Any] = {"backend": backend, "world_size": world_size, "rank": rank}
        if init_method is not None:
            kwargs["init_method"] = init_method
        dist.init_process_group(**kwargs)
        torch.cuda.set_device(local_rank)

    logger.info(f"Distributed setup: rank={rank}, world_size={world_size}, local_rank={local_rank}")
    return rank, world_size


def cleanup_distributed() -> None:
    """Destroy the distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


class DistributedTrainer:
    """Multi-GPU distributed trainer for Neural DEs.

    Wraps the model in ``DistributedDataParallel`` and handles data
    partitioning across ranks.  Supports gradient accumulation to achieve
    large effective batch sizes without exceeding per-GPU memory.

    Example::

        rank, world_size = setup_distributed()
        trainer = DistributedTrainer(model, data_generator, rank=rank, world_size=world_size)
        history = trainer.train(num_epochs=50, ...)
        cleanup_distributed()
    """

    def __init__(
        self,
        model: AbstractNeuralDE,
        data_generator: ReactorDataGenerator,
        loss_fn: MultiObjectiveLoss | None = None,
        optimizer: optim.Optimizer | None = None,
        scheduler: Any | None = None,
        rank: int = 0,
        world_size: int = 1,
        accumulation_steps: int = 1,
    ) -> None:
        """Initialize distributed trainer.

        Args:
            model: Neural DE model (will be wrapped in DDP).
            data_generator: Reactor data generator.
            loss_fn: Multi-objective loss function.
            optimizer: PyTorch optimizer (created after DDP wrapping if None).
            scheduler: LR scheduler.
            rank: Process rank (0 = master).
            world_size: Total number of processes.
            accumulation_steps: Number of micro-batches before an
                optimiser step.  Effective batch size =
                ``batch_size * accumulation_steps * world_size``.
        """
        self.rank = rank
        self.world_size = world_size
        self.accumulation_steps = accumulation_steps
        self.loss_fn = loss_fn or MultiObjectiveLoss()

        # Device
        if torch.cuda.is_available() and world_size > 1:
            local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
            self.device = torch.device(f"cuda:{local_rank}")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        model.to(self.device)

        # Wrap in DDP
        if world_size > 1 and torch.cuda.is_available():
            self.model: AbstractNeuralDE = nn.parallel.DistributedDataParallel(  # type: ignore[assignment]
                model,
                device_ids=[self.device.index],
                output_device=self.device.index,
            )
        else:
            self.model = model
        self._raw_model = model  # unwrapped reference for saving

        self.data_generator = data_generator

        self.optimizer: torch.optim.Optimizer
        if optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        else:
            self.optimizer = optimizer
        self.scheduler = scheduler

        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

        if rank == 0:
            logger.info(
                f"DistributedTrainer: world_size={world_size}, "
                f"device={self.device}, accumulation_steps={accumulation_steps}"
            )

    def _shard_data(
        self,
        data: list[dict[str, torch.Tensor]],
    ) -> list[dict[str, torch.Tensor]]:
        """Partition data across ranks.

        Each rank gets every ``world_size``-th batch starting at its
        rank index.
        """
        return data[self.rank :: self.world_size]

    def train_epoch(
        self,
        train_data: list[dict[str, torch.Tensor]],
        log_interval: int = 10,
    ) -> dict[str, float]:
        """Train for one epoch with gradient accumulation.

        Args:
            train_data: Full dataset (will be sharded automatically).
            log_interval: Log every N batches.

        Returns:
            Average losses for this rank's partition.
        """
        self.model.train()
        sharded = self._shard_data(train_data)
        epoch_losses: dict[str, list[float]] = {}
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(sharded):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            raw = self._raw_model
            predictions = raw(
                z0=batch["z0"],
                t_span=batch["t_span"],
                controls=batch.get("controls"),
            )

            losses = self.loss_fn(
                predictions=predictions,
                targets=batch["targets"],
                model=raw,
            )

            # Scale loss by accumulation steps
            scaled_loss = losses["total"] / self.accumulation_steps
            scaled_loss.backward()

            # Step optimizer every accumulation_steps micro-batches
            if (batch_idx + 1) % self.accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.global_step += 1

            for loss_name, loss_value in losses.items():
                if loss_name not in epoch_losses:
                    epoch_losses[loss_name] = []
                epoch_losses[loss_name].append(loss_value.item())

            if self.rank == 0 and (batch_idx + 1) % log_interval == 0:
                logger.info(
                    f"Epoch {self.epoch} [{batch_idx + 1}/{len(sharded)}] "
                    f"Loss: {losses['total'].item():.6f}"
                )

        # Flush remaining gradients
        if len(sharded) % self.accumulation_steps != 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.global_step += 1

        avg_losses = {name: sum(vals) / len(vals) for name, vals in epoch_losses.items()}
        return avg_losses

    def validate(
        self,
        val_data: list[dict[str, torch.Tensor]],
    ) -> dict[str, float]:
        """Validate (run on all ranks, average results).

        Args:
            val_data: Validation batches.

        Returns:
            Average validation losses.
        """
        self.model.eval()
        val_losses: dict[str, list[float]] = {}

        raw = self._raw_model
        with torch.no_grad():
            for batch in val_data:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                predictions = raw.predict(
                    z0=batch["z0"],
                    t_span=batch["t_span"],
                    controls=batch.get("controls"),
                )
                losses = self.loss_fn(
                    predictions=predictions,
                    targets=batch["targets"],
                    model=raw,
                )
                for loss_name, loss_value in losses.items():
                    if loss_name not in val_losses:
                        val_losses[loss_name] = []
                    val_losses[loss_name].append(loss_value.item())

        avg_losses = {name: sum(vals) / len(vals) for name, vals in val_losses.items()}

        # Average across ranks
        if self.world_size > 1 and dist.is_initialized():
            total_tensor = torch.tensor(avg_losses.get("total", 0.0), device=self.device)
            dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
            avg_losses["total"] = total_tensor.item() / self.world_size

        return avg_losses

    def train(
        self,
        num_epochs: int,
        t_span: tuple[float, float],
        t_eval: Any,
        train_trajectories: int = 100,
        val_trajectories: int = 20,
        batch_size: int = 32,
        log_interval: int = 10,
        val_interval: int = 5,
        checkpoint_dir: str | Path | None = None,
    ) -> dict[str, list[float]]:
        """Full distributed training loop.

        Args:
            num_epochs: Number of epochs.
            t_span: Time interval (t_start, t_end).
            t_eval: Time evaluation points.
            train_trajectories: Training trajectories per epoch.
            val_trajectories: Validation trajectories.
            batch_size: Per-GPU batch size.
            log_interval: Log every N batches.
            val_interval: Validate every N epochs.
            checkpoint_dir: Checkpoint directory (only rank 0 saves).

        Returns:
            Training history.
        """
        if self.rank == 0:
            logger.info(
                f"Starting distributed training: {num_epochs} epochs, "
                f"{self.world_size} GPUs, "
                f"effective batch = {batch_size} x {self.accumulation_steps} x {self.world_size}"
            )

        val_data = self.data_generator.generate_dataset(
            val_trajectories, t_span, t_eval, batch_size
        )

        for epoch in range(num_epochs):
            self.epoch = epoch

            train_data = self.data_generator.generate_dataset(
                train_trajectories, t_span, t_eval, batch_size
            )

            train_losses = self.train_epoch(train_data, log_interval)
            self.history["train_loss"].append(train_losses["total"])

            if self.rank == 0:
                logger.info(
                    f"Epoch {epoch}/{num_epochs} - Train Loss: {train_losses['total']:.6f}"
                )

            if (epoch + 1) % val_interval == 0:
                val_losses = self.validate(val_data)
                self.history["val_loss"].append(val_losses["total"])

                if self.rank == 0:
                    logger.info(
                        f"Epoch {epoch}/{num_epochs} - Val Loss: {val_losses['total']:.6f}"
                    )

                    if val_losses["total"] < self.best_val_loss:
                        self.best_val_loss = val_losses["total"]
                        if checkpoint_dir is not None:
                            self.save_checkpoint(checkpoint_dir, "best_model.pt")

            if self.scheduler is not None:
                self.scheduler.step()

            if self.rank == 0 and checkpoint_dir is not None and (epoch + 1) % 10 == 0:
                self.save_checkpoint(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")

        if self.rank == 0:
            logger.info("Distributed training complete!")
        return self.history

    def save_checkpoint(self, checkpoint_dir: str | Path, filename: str) -> None:
        """Save checkpoint (rank 0 only)."""
        if self.rank != 0:
            return
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "epoch": self.epoch,
                "global_step": self.global_step,
                "model_state_dict": self._raw_model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_val_loss": self.best_val_loss,
                "history": self.history,
            },
            checkpoint_dir / filename,
        )
        logger.debug(f"Saved checkpoint: {checkpoint_dir / filename}")


__all__ = [
    "setup_distributed",
    "cleanup_distributed",
    "DistributedTrainer",
]
