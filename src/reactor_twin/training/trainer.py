"""Trainer class for Neural Differential Equations."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch import optim

from reactor_twin.core.base import AbstractNeuralDE
from reactor_twin.training.data_generator import ReactorDataGenerator
from reactor_twin.training.losses import MultiObjectiveLoss

logger = logging.getLogger(__name__)


class Trainer:
    """Training engine for Neural Differential Equations.

    Handles:
    1. Training loop with data loading
    2. Multi-objective loss optimization
    3. Validation and checkpointing
    4. Learning rate scheduling
    5. Early stopping

    Attributes:
        model: Neural DE model to train.
        data_generator: Reactor data generator.
        loss_fn: Multi-objective loss function.
        optimizer: PyTorch optimizer.
        scheduler: Learning rate scheduler (optional).
        device: Compute device (CPU or CUDA).
    """

    def __init__(
        self,
        model: AbstractNeuralDE,
        data_generator: ReactorDataGenerator,
        loss_fn: MultiObjectiveLoss | None = None,
        optimizer: optim.Optimizer | None = None,
        scheduler: Any | None = None,
        device: str | torch.device = "cpu",
    ):
        """Initialize trainer.

        Args:
            model: Neural DE model.
            data_generator: Data generator for reactor trajectories.
            loss_fn: Loss function. If None, creates default MultiObjectiveLoss.
            optimizer: Optimizer. If None, creates Adam with lr=1e-3.
            scheduler: Learning rate scheduler (optional).
            device: Compute device ('cpu', 'cuda', or torch.device).
        """
        self.model = model
        self.data_generator = data_generator
        self.loss_fn = loss_fn or MultiObjectiveLoss()
        self.device = torch.device(device)

        # Move model to device
        self.model.to(self.device)

        # Create optimizer if not provided
        if optimizer is None:
            self.optimizer = optim.Adam(model.parameters(), lr=1e-3)
        else:
            self.optimizer = optimizer

        self.scheduler = scheduler

        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
        }

        logger.info(f"Initialized Trainer: model={model.__class__.__name__}, device={self.device}")

    def train_epoch(
        self,
        train_data: list[dict[str, torch.Tensor]],
        log_interval: int = 10,
    ) -> dict[str, float]:
        """Train for one epoch.

        Args:
            train_data: List of batch dictionaries from data generator.
            log_interval: Log every N batches.

        Returns:
            Dictionary of average losses for the epoch.
        """
        self.model.train()
        epoch_losses: dict[str, list[float]] = {}

        for batch_idx, batch in enumerate(train_data):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Forward pass
            predictions = self.model(
                z0=batch["z0"],
                t_span=batch["t_span"],
                controls=batch.get("controls"),
            )

            # Compute loss
            losses = self.loss_fn(
                predictions=predictions,
                targets=batch["targets"],
                model=self.model,
            )

            # Backward pass
            self.optimizer.zero_grad()
            losses["total"].backward()
            self.optimizer.step()

            # Log losses
            for loss_name, loss_value in losses.items():
                if loss_name not in epoch_losses:
                    epoch_losses[loss_name] = []
                epoch_losses[loss_name].append(loss_value.item())

            # Log progress
            if (batch_idx + 1) % log_interval == 0:
                logger.info(
                    f"Epoch {self.epoch} [{batch_idx + 1}/{len(train_data)}] "
                    f"Loss: {losses['total'].item():.6f}"
                )

            self.global_step += 1

        # Compute average losses
        avg_losses = {name: sum(values) / len(values) for name, values in epoch_losses.items()}

        return avg_losses

    def validate(
        self,
        val_data: list[dict[str, torch.Tensor]],
    ) -> dict[str, float]:
        """Validate on validation set.

        Args:
            val_data: List of validation batch dictionaries.

        Returns:
            Dictionary of average validation losses.
        """
        self.model.eval()
        val_losses: dict[str, list[float]] = {}

        with torch.no_grad():
            for batch in val_data:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Forward pass
                predictions = self.model.predict(
                    z0=batch["z0"],
                    t_span=batch["t_span"],
                    controls=batch.get("controls"),
                )

                # Compute loss
                losses = self.loss_fn(
                    predictions=predictions,
                    targets=batch["targets"],
                    model=self.model,
                )

                # Log losses
                for loss_name, loss_value in losses.items():
                    if loss_name not in val_losses:
                        val_losses[loss_name] = []
                    val_losses[loss_name].append(loss_value.item())

        # Compute average losses
        avg_losses = {name: sum(values) / len(values) for name, values in val_losses.items()}

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
        """Full training loop.

        Args:
            num_epochs: Number of training epochs.
            t_span: Time interval (t_start, t_end).
            t_eval: Time points for evaluation.
            train_trajectories: Number of training trajectories per epoch.
            val_trajectories: Number of validation trajectories.
            batch_size: Batch size.
            log_interval: Log every N batches.
            val_interval: Validate every N epochs.
            checkpoint_dir: Directory to save checkpoints (optional).

        Returns:
            Training history dictionary.
        """
        logger.info(
            f"Starting training: {num_epochs} epochs, "
            f"{train_trajectories} train trajectories, "
            f"{val_trajectories} val trajectories"
        )

        # Generate validation data once
        val_data = self.data_generator.generate_dataset(
            val_trajectories, t_span, t_eval, batch_size
        )
        logger.info(f"Generated validation set: {len(val_data)} batches")

        for epoch in range(num_epochs):
            self.epoch = epoch

            # Generate training data for this epoch
            train_data = self.data_generator.generate_dataset(
                train_trajectories, t_span, t_eval, batch_size
            )

            # Train
            train_losses = self.train_epoch(train_data, log_interval)
            self.history["train_loss"].append(train_losses["total"])

            logger.info(f"Epoch {epoch}/{num_epochs} - Train Loss: {train_losses['total']:.6f}")

            # Validate
            if (epoch + 1) % val_interval == 0:
                val_losses = self.validate(val_data)
                self.history["val_loss"].append(val_losses["total"])

                logger.info(f"Epoch {epoch}/{num_epochs} - Val Loss: {val_losses['total']:.6f}")

                # Save best model
                if val_losses["total"] < self.best_val_loss:
                    self.best_val_loss = val_losses["total"]
                    if checkpoint_dir is not None:
                        self.save_checkpoint(checkpoint_dir, "best_model.pt")
                        logger.info(f"Saved best model (val_loss={self.best_val_loss:.6f})")

            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()

            # Save periodic checkpoint
            if checkpoint_dir is not None and (epoch + 1) % 10 == 0:
                self.save_checkpoint(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")

        logger.info("Training complete!")
        return self.history

    def save_checkpoint(self, checkpoint_dir: str | Path, filename: str) -> None:
        """Save training checkpoint.

        Args:
            checkpoint_dir: Directory to save checkpoint.
            filename: Checkpoint filename.
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / filename
        torch.save(
            {
                "epoch": self.epoch,
                "global_step": self.global_step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_val_loss": self.best_val_loss,
                "history": self.history,
            },
            checkpoint_path,
        )
        logger.debug(f"Saved checkpoint: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        """Load training checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.history = checkpoint["history"]
        logger.info(f"Loaded checkpoint from {checkpoint_path} (epoch {self.epoch})")


__all__ = ["Trainer"]
