"""Base abstract class for all Neural Differential Equation models."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch
from torch import nn

logger = logging.getLogger(__name__)


class AbstractNeuralDE(nn.Module, ABC):
    """Abstract base class for Neural Differential Equation models.

    All Neural DE variants (standard, latent, augmented, SDE, CDE, hybrid)
    must inherit from this class and implement the abstract methods.

    Attributes:
        state_dim: Dimension of the state space.
        input_dim: Dimension of external inputs (controls).
        output_dim: Dimension of observations.
    """

    def __init__(
        self,
        state_dim: int,
        input_dim: int = 0,
        output_dim: int | None = None,
    ):
        """Initialize Neural DE model.

        Args:
            state_dim: Dimension of latent state.
            input_dim: Dimension of external inputs/controls. Defaults to 0.
            output_dim: Dimension of observations. Defaults to state_dim.
        """
        super().__init__()
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.output_dim = output_dim or state_dim
        logger.debug(
            f"Initialized {self.__class__.__name__}: "
            f"state_dim={state_dim}, input_dim={input_dim}, "
            f"output_dim={self.output_dim}"
        )

    @abstractmethod
    def forward(
        self,
        z0: torch.Tensor,
        t_span: torch.Tensor,
        controls: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass: integrate ODE from z0 over t_span.

        Args:
            z0: Initial state, shape (batch, state_dim).
            t_span: Time points to evaluate at, shape (num_times,).
            controls: External inputs at each time, shape (batch, num_times, input_dim).
                Defaults to None.

        Returns:
            Trajectory z(t), shape (batch, num_times, state_dim or output_dim).
        """
        raise NotImplementedError("Subclasses must implement forward()")

    @abstractmethod
    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        loss_weights: dict[str, float] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute multi-objective loss.

        Args:
            predictions: Model predictions, shape (batch, num_times, output_dim).
            targets: Ground truth, shape (batch, num_times, output_dim).
            loss_weights: Dictionary of loss component weights.

        Returns:
            Dictionary with keys:
                - 'total': Total weighted loss (scalar).
                - Individual loss components (e.g., 'data', 'physics', 'constraint').
        """
        raise NotImplementedError("Subclasses must implement compute_loss()")

    def train_step(
        self,
        batch: dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
    ) -> dict[str, float]:
        """Single training step.

        Args:
            batch: Dictionary with keys 'z0', 't_span', 'targets', optionally 'controls'.
            optimizer: PyTorch optimizer.

        Returns:
            Dictionary of scalar loss values.
        """
        optimizer.zero_grad()

        # Forward pass
        predictions = self.forward(
            z0=batch["z0"],
            t_span=batch["t_span"],
            controls=batch.get("controls"),
        )

        # Compute loss
        losses = self.compute_loss(
            predictions=predictions,
            targets=batch["targets"],
        )

        # Backward pass
        losses["total"].backward()  # type: ignore[no-untyped-call]
        optimizer.step()

        # Convert to scalars
        return {k: v.item() for k, v in losses.items()}

    def predict(
        self,
        z0: torch.Tensor,
        t_span: torch.Tensor,
        controls: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Inference mode prediction.

        Args:
            z0: Initial state, shape (batch, state_dim).
            t_span: Time points, shape (num_times,).
            controls: External inputs, shape (batch, num_times, input_dim).

        Returns:
            Predictions, shape (batch, num_times, output_dim).
        """
        self.eval()
        with torch.no_grad():
            return self.forward(z0, t_span, controls)

    def save(self, path: str | Path) -> None:
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "state_dim": self.state_dim,
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
            },
            path,
        )
        logger.info(f"Saved model to {path}")

    @classmethod
    def load(cls, path: str | Path, **kwargs: Any) -> AbstractNeuralDE:
        """Load model from checkpoint.

        Args:
            path: Path to checkpoint.
            **kwargs: Additional arguments for model initialization.

        Returns:
            Loaded model instance.
        """
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(
            state_dim=checkpoint["state_dim"],
            input_dim=checkpoint["input_dim"],
            output_dim=checkpoint["output_dim"],
            **kwargs,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded model from {path}")
        return model


__all__ = ["AbstractNeuralDE"]
