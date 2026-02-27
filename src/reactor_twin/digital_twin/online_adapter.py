"""Online continual-learning adapter for Neural ODE models.

Provides experience replay and Elastic Weight Consolidation (EWC)
so that a pre-trained Neural ODE can be fine-tuned on streaming
plant data without catastrophically forgetting prior knowledge.
"""

from __future__ import annotations

import copy
import logging
from collections import deque
from typing import Any

import torch
import torch.nn as nn

from reactor_twin.core.base import AbstractNeuralDE

logger = logging.getLogger(__name__)


# ======================================================================
# Replay Buffer
# ======================================================================

class ReplayBuffer:
    """FIFO experience buffer storing ``(z0, t_span, targets)`` tuples.

    Attributes:
        capacity: Maximum number of stored experiences.
    """

    def __init__(self, capacity: int = 1000) -> None:
        self.capacity = capacity
        self._buffer: deque[dict[str, torch.Tensor]] = deque(maxlen=capacity)

    def add(
        self,
        z0: torch.Tensor,
        t_span: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        """Store a new experience.

        Args:
            z0: Initial state, shape ``(state_dim,)`` or ``(batch, state_dim)``.
            t_span: Time points, shape ``(num_times,)``.
            targets: Ground-truth trajectory, shape matching model output.
        """
        self._buffer.append({
            "z0": z0.detach().cpu(),
            "t_span": t_span.detach().cpu(),
            "targets": targets.detach().cpu(),
        })

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        """Sample a random mini-batch from the buffer.

        Args:
            batch_size: Number of experiences to sample.

        Returns:
            Collated batch dictionary with keys ``z0``, ``t_span``, ``targets``.
        """
        indices = torch.randint(0, len(self._buffer), (min(batch_size, len(self._buffer)),))
        samples = [self._buffer[i] for i in indices]

        # Use the first sample's t_span (all should share the same grid)
        t_span = samples[0]["t_span"]

        z0_list, targets_list = [], []
        for s in samples:
            z0_i = s["z0"]
            tgt_i = s["targets"]
            # Ensure batch dim
            if z0_i.ndim == 1:
                z0_i = z0_i.unsqueeze(0)
            if tgt_i.ndim == 2:
                tgt_i = tgt_i.unsqueeze(0)
            z0_list.append(z0_i)
            targets_list.append(tgt_i)

        return {
            "z0": torch.cat(z0_list, dim=0),
            "t_span": t_span,
            "targets": torch.cat(targets_list, dim=0),
        }

    def __len__(self) -> int:
        return len(self._buffer)


# ======================================================================
# Elastic Weight Consolidation
# ======================================================================

class ElasticWeightConsolidation:
    """Diagonal Fisher-information penalty to mitigate catastrophic forgetting.

    After each consolidation snapshot the diagonal Fisher information
    matrix is estimated from recent data, and a quadratic penalty
    ``lambda/2 * F * (theta - theta_star)^2`` is added to the loss.

    Attributes:
        model: The Neural DE being adapted.
        ewc_lambda: Penalty strength.
    """

    def __init__(
        self,
        model: AbstractNeuralDE,
        ewc_lambda: float = 100.0,
    ) -> None:
        self.model = model
        self.ewc_lambda = ewc_lambda

        self._reference_params: dict[str, torch.Tensor] = {}
        self._fisher_diag: dict[str, torch.Tensor] = {}
        self._consolidated = False

    def consolidate(
        self,
        data_batches: list[dict[str, torch.Tensor]] | None = None,
        num_samples: int = 50,
    ) -> None:
        """Snapshot current parameters and estimate Fisher diagonal.

        If ``data_batches`` is provided the Fisher is estimated from
        gradient samples; otherwise the Fisher is set to ones (uniform
        prior over parameters).

        Args:
            data_batches: Optional list of training batches for Fisher estimation.
            num_samples: Number of gradient samples for Fisher estimate.
        """
        # Snapshot parameters
        self._reference_params = {
            n: p.detach().clone()
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

        # Estimate diagonal Fisher
        if data_batches is not None and len(data_batches) > 0:
            self._estimate_fisher(data_batches, num_samples)
        else:
            self._fisher_diag = {
                n: torch.ones_like(p)
                for n, p in self.model.named_parameters()
                if p.requires_grad
            }

        self._consolidated = True
        logger.info("EWC consolidation complete.")

    def _estimate_fisher(
        self,
        data_batches: list[dict[str, torch.Tensor]],
        num_samples: int,
    ) -> None:
        """Estimate diagonal Fisher via sampled squared gradients."""
        fisher: dict[str, torch.Tensor] = {
            n: torch.zeros_like(p)
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

        self.model.train()
        count = 0
        for batch in data_batches:
            if count >= num_samples:
                break
            self.model.zero_grad()
            preds = self.model.forward(
                z0=batch["z0"],
                t_span=batch["t_span"],
                controls=batch.get("controls"),
            )
            losses = self.model.compute_loss(preds, batch["targets"])
            losses["total"].backward()

            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.detach() ** 2

            count += 1

        # Average
        for n in fisher:
            fisher[n] /= max(count, 1)

        self._fisher_diag = fisher

    def penalty(self) -> torch.Tensor:
        """Compute the EWC quadratic penalty.

        Returns:
            Scalar penalty tensor (differentiable).
        """
        if not self._consolidated:
            return torch.tensor(0.0, device=next(self.model.parameters()).device)

        loss = torch.tensor(0.0, device=next(self.model.parameters()).device)
        for n, p in self.model.named_parameters():
            if n in self._reference_params:
                diff = p - self._reference_params[n]
                loss = loss + (self._fisher_diag[n] * diff ** 2).sum()

        return 0.5 * self.ewc_lambda * loss


# ======================================================================
# Online Adapter
# ======================================================================

class OnlineAdapter:
    """Unified continual-learning engine combining replay + EWC.

    Manages the adaptation loop: add new streaming experiences, perform
    a few gradient steps mixing replay data with new data, and
    periodically consolidate to protect important weights.

    Attributes:
        model: Neural DE being adapted.
        replay_buffer: Experience replay store.
        ewc: Elastic Weight Consolidation module.
        lr: Online learning rate.
        replay_ratio: Fraction of each mini-batch drawn from replay.
    """

    def __init__(
        self,
        model: AbstractNeuralDE,
        lr: float = 1e-4,
        ewc_lambda: float = 100.0,
        buffer_capacity: int = 1000,
        replay_ratio: float = 0.5,
        device: str | torch.device = "cpu",
    ) -> None:
        """Initialize the online adapter.

        Args:
            model: Pre-trained Neural DE.
            lr: Learning rate for online updates.
            ewc_lambda: EWC penalty coefficient.
            buffer_capacity: Replay buffer size.
            replay_ratio: Fraction of training batch from replay.
            device: Torch device.
        """
        self.model = model
        self.device = torch.device(device)
        self.replay_ratio = replay_ratio

        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)
        self.ewc = ElasticWeightConsolidation(model, ewc_lambda=ewc_lambda)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        logger.info(
            f"OnlineAdapter: lr={lr}, ewc_lambda={ewc_lambda}, "
            f"buffer={buffer_capacity}, replay_ratio={replay_ratio}"
        )

    def add_experience(
        self,
        z0: torch.Tensor,
        t_span: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        """Store a new observation in the replay buffer.

        Args:
            z0: Initial state(s).
            t_span: Time grid.
            targets: Ground-truth trajectory.
        """
        self.replay_buffer.add(z0, t_span, targets)

    def adapt(
        self,
        new_data: dict[str, torch.Tensor],
        num_steps: int = 5,
        batch_size: int = 16,
    ) -> list[float]:
        """Run K gradient steps mixing replay and new data with EWC.

        Args:
            new_data: Fresh observation batch (``z0``, ``t_span``, ``targets``).
            num_steps: Number of gradient steps.
            batch_size: Mini-batch size for replay sampling.

        Returns:
            List of total loss values (one per step).
        """
        self.model.train()
        losses: list[float] = []

        for _ in range(num_steps):
            self.optimizer.zero_grad()

            # --- New data loss ---
            new_z0 = new_data["z0"].to(self.device)
            new_t = new_data["t_span"].to(self.device)
            new_tgt = new_data["targets"].to(self.device)

            if new_z0.ndim == 1:
                new_z0 = new_z0.unsqueeze(0)
            if new_tgt.ndim == 2:
                new_tgt = new_tgt.unsqueeze(0)

            preds = self.model.forward(new_z0, new_t)
            loss_dict = self.model.compute_loss(preds, new_tgt)
            loss = loss_dict["total"]

            # --- Replay loss ---
            if len(self.replay_buffer) > 0 and self.replay_ratio > 0:
                replay_batch = self.replay_buffer.sample(batch_size)
                r_z0 = replay_batch["z0"].to(self.device)
                r_t = replay_batch["t_span"].to(self.device)
                r_tgt = replay_batch["targets"].to(self.device)

                r_preds = self.model.forward(r_z0, r_t)
                r_loss = self.model.compute_loss(r_preds, r_tgt)["total"]
                loss = (1 - self.replay_ratio) * loss + self.replay_ratio * r_loss

            # --- EWC penalty ---
            loss = loss + self.ewc.penalty()

            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())

        return losses

    def consolidate(
        self,
        data_batches: list[dict[str, torch.Tensor]] | None = None,
    ) -> None:
        """Snapshot parameters for the next EWC round.

        Args:
            data_batches: Optional data for Fisher estimation.
        """
        self.ewc.consolidate(data_batches)
        logger.info("OnlineAdapter: consolidation complete.")


__all__ = [
    "ReplayBuffer",
    "ElasticWeightConsolidation",
    "OnlineAdapter",
]
