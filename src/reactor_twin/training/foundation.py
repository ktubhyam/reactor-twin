"""Foundation model for cross-reactor pre-training and fine-tuning.

Extends the Reptile meta-learning approach with a task encoder that
conditions the ODE function on reactor type and operating parameters,
enabling knowledge transfer across different reactor configurations.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import torch
from torch import nn
from torchdiffeq import odeint

from reactor_twin.core.base import AbstractNeuralDE
from reactor_twin.training.data_generator import ReactorDataGenerator
from reactor_twin.utils.registry import NEURAL_DE_REGISTRY

logger = logging.getLogger(__name__)


class ReactorTaskEncoder(nn.Module):
    """Encode reactor type and operating parameters into a task embedding.

    Takes a one-hot reactor type vector and normalised operating parameters,
    and produces a fixed-size task embedding used to condition the ODE
    function.

    Attributes:
        num_reactor_types: Number of reactor type categories.
        param_dim: Dimension of operating parameter vector.
        embedding_dim: Output embedding dimension.
    """

    def __init__(
        self,
        num_reactor_types: int,
        param_dim: int = 4,
        embedding_dim: int = 16,
        hidden_dim: int = 32,
    ):
        super().__init__()
        self.num_reactor_types = num_reactor_types
        self.param_dim = param_dim
        self.embedding_dim = embedding_dim

        input_dim = num_reactor_types + param_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(
        self,
        reactor_type: torch.Tensor,
        params: torch.Tensor,
    ) -> torch.Tensor:
        """Compute task embedding.

        Args:
            reactor_type: One-hot vector, shape (batch, num_reactor_types).
            params: Normalized parameters, shape (batch, param_dim).

        Returns:
            Task embedding, shape (batch, embedding_dim).
        """
        x = torch.cat([reactor_type, params], dim=-1)
        return cast(torch.Tensor, self.net(x))


class _ConditionedODEFunc(nn.Module):
    """ODE function conditioned on a task embedding."""

    def __init__(
        self,
        state_dim: int,
        embedding_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.embedding_dim = embedding_dim

        # Input: z + t + task_embedding
        total_input = state_dim + 1 + embedding_dim
        layers: list[nn.Module] = []
        for i in range(num_layers):
            in_f = total_input if i == 0 else hidden_dim
            out_f = state_dim if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_f, out_f))
            if i < num_layers - 1:
                layers.append(nn.Softplus())
        self.net = nn.Sequential(*layers)

        # Task embedding (set before integration)
        self._task_embedding: torch.Tensor | None = None

    def set_task_embedding(self, embedding: torch.Tensor) -> None:
        """Set task embedding for the current integration."""
        self._task_embedding = embedding

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        batch_size = z.shape[0]
        if t.ndim == 0:
            t_expand = t.expand(batch_size, 1)
        else:
            t_expand = t.reshape(batch_size, 1)

        if self._task_embedding is not None:
            emb = self._task_embedding
            if emb.shape[0] != batch_size:
                emb = emb.expand(batch_size, -1)
            x = torch.cat([z, t_expand, emb], dim=-1)
        else:
            # Zero embedding fallback
            zeros = torch.zeros(batch_size, self.embedding_dim, device=z.device)
            x = torch.cat([z, t_expand, zeros], dim=-1)

        return cast(torch.Tensor, self.net(x))


@NEURAL_DE_REGISTRY.register("foundation_neural_ode")
class FoundationNeuralODE(AbstractNeuralDE):
    """Foundation Neural ODE conditioned on task embeddings.

    The ODE function receives a task embedding from the ReactorTaskEncoder,
    enabling a single model to handle multiple reactor types.

    Attributes:
        task_encoder: Encodes reactor metadata into embedding.
        ode_func: Task-conditioned ODE function.
    """

    def __init__(
        self,
        state_dim: int,
        num_reactor_types: int = 6,
        param_dim: int = 4,
        embedding_dim: int = 16,
        hidden_dim: int = 64,
        num_layers: int = 3,
        solver: str = "dopri5",
        atol: float = 1e-6,
        rtol: float = 1e-3,
        input_dim: int = 0,
        output_dim: int | None = None,
    ):
        super().__init__(state_dim, input_dim, output_dim)

        self.task_encoder = ReactorTaskEncoder(
            num_reactor_types=num_reactor_types,
            param_dim=param_dim,
            embedding_dim=embedding_dim,
        )

        self.ode_func = _ConditionedODEFunc(
            state_dim=state_dim,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )

        self.solver = solver
        self.atol = atol
        self.rtol = rtol

        logger.info(
            f"Initialized FoundationNeuralODE: state_dim={state_dim}, "
            f"num_reactor_types={num_reactor_types}, embedding_dim={embedding_dim}"
        )

    def forward(
        self,
        z0: torch.Tensor,
        t_span: torch.Tensor,
        controls: torch.Tensor | None = None,
        task_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward with optional task conditioning.

        Args:
            z0: Initial state, shape (batch, state_dim).
            t_span: Time points, shape (num_times,).
            controls: Not used.
            task_embedding: Pre-computed embedding, shape (batch, embedding_dim).

        Returns:
            Trajectory, shape (batch, num_times, state_dim).
        """
        if task_embedding is not None:
            self.ode_func.set_task_embedding(task_embedding)

        z_traj = odeint(
            self.ode_func,
            z0,
            t_span,
            rtol=self.rtol,
            atol=self.atol,
            method=self.solver,
        )
        return cast(torch.Tensor, z_traj.transpose(0, 1))

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        loss_weights: dict[str, float] | None = None,
    ) -> dict[str, torch.Tensor]:
        if loss_weights is None:
            loss_weights = {"data": 1.0}

        data_loss = torch.mean((predictions - targets) ** 2)
        total = loss_weights.get("data", 1.0) * data_loss

        return {"total": total, "data": data_loss}


class FoundationTrainer:
    """Pre-train and fine-tune FoundationNeuralODE across reactor types.

    Extends the Reptile meta-learning pattern with task-conditioned
    ODE functions for cross-reactor transfer.

    Attributes:
        model: FoundationNeuralODE to train.
        meta_lr: Outer-loop learning rate.
        inner_lr: Inner-loop learning rate.
        inner_steps: Number of gradient steps per task.
    """

    def __init__(
        self,
        model: FoundationNeuralODE,
        meta_lr: float = 1e-3,
        inner_lr: float = 1e-3,
        inner_steps: int = 5,
        device: str | torch.device = "cpu",
    ):
        self.model = model
        self.meta_lr = meta_lr
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.device = torch.device(device)
        self.model.to(self.device)

    def _prepare_task_embedding(
        self,
        generator: ReactorDataGenerator,
        model: FoundationNeuralODE | None = None,
    ) -> None:
        """Create and set task embedding from a data generator's reactor.

        Args:
            generator: Data generator whose reactor provides type/params.
            model: Model to set embedding on (defaults to self.model).
        """
        if model is None:
            model = self.model

        reactor = generator.reactor
        # Build one-hot reactor type vector
        num_types = model.task_encoder.num_reactor_types
        type_vec = torch.zeros(1, num_types, device=self.device)
        # Use a hash of reactor class name to pick an index
        type_idx = hash(type(reactor).__name__) % num_types
        type_vec[0, type_idx] = 1.0

        # Build normalised params vector from reactor params
        param_dim = model.task_encoder.param_dim
        param_vals = []
        for v in list(reactor.params.values())[:param_dim]:
            if isinstance(v, (int, float)):
                param_vals.append(float(v))
            else:
                break
        # Pad or truncate to param_dim
        while len(param_vals) < param_dim:
            param_vals.append(0.0)
        param_vals = param_vals[:param_dim]

        params_tensor = torch.tensor([param_vals], dtype=torch.float32, device=self.device)
        emb = model.task_encoder(type_vec, params_tensor)
        model.ode_func.set_task_embedding(emb.detach())

    def pretrain(
        self,
        task_generators: list[ReactorDataGenerator],
        num_epochs: int = 50,
        t_span: tuple[float, float] = (0.0, 1.0),
        t_eval: npt.NDArray[Any] | None = None,
        batch_size: int = 8,
    ) -> list[float]:
        """Pre-train with Reptile meta-learning across reactor tasks.

        Args:
            task_generators: Data generators for different reactors.
            num_epochs: Number of outer-loop meta-steps.
            t_span: Simulation time interval.
            t_eval: Evaluation time points.
            batch_size: Inner-loop batch size.

        Returns:
            List of displacement values per meta-step.
        """
        if t_eval is None:
            t_eval = np.linspace(t_span[0], t_span[1], 30)

        displacements: list[float] = []

        for epoch in range(num_epochs):
            # Sample all tasks
            all_adapted: list[dict[str, torch.Tensor]] = []

            for tg in task_generators:
                # Clone model for inner loop
                model_copy = copy.deepcopy(self.model)
                model_copy.to(self.device)
                model_copy.train()
                self._prepare_task_embedding(tg, model_copy)
                inner_opt = torch.optim.SGD(model_copy.parameters(), lr=self.inner_lr)

                for _ in range(self.inner_steps):
                    batch = tg.generate_batch(batch_size, t_span, t_eval)
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    model_copy.train_step(batch, inner_opt)

                adapted = {n: p.detach().clone() for n, p in model_copy.named_parameters()}
                all_adapted.append(adapted)

            # Reptile update
            displacement = 0.0
            with torch.no_grad():
                for n, p in self.model.named_parameters():
                    avg = torch.stack([a[n] for a in all_adapted]).mean(dim=0)
                    delta = avg.to(self.device) - p.data
                    p.data.add_(delta, alpha=self.meta_lr)
                    displacement += delta.norm().item()

            displacements.append(displacement)

            if (epoch + 1) % 10 == 0:
                logger.info(f"Pretrain step {epoch + 1}/{num_epochs}: disp={displacement:.6f}")

        return displacements

    def fine_tune(
        self,
        target_generator: ReactorDataGenerator,
        num_steps: int = 20,
        t_span: tuple[float, float] = (0.0, 1.0),
        t_eval: npt.NDArray[Any] | None = None,
        batch_size: int = 8,
        freeze_encoder: bool = True,
    ) -> list[float]:
        """Fine-tune on a specific target reactor.

        Args:
            target_generator: Data generator for the target reactor.
            num_steps: Number of fine-tuning steps.
            t_span: Time interval.
            t_eval: Evaluation time points.
            batch_size: Batch size.
            freeze_encoder: If True, freeze task encoder during fine-tuning.

        Returns:
            List of loss values per step.
        """
        if t_eval is None:
            t_eval = np.linspace(t_span[0], t_span[1], 30)

        if freeze_encoder:
            for p in self.model.task_encoder.parameters():
                p.requires_grad_(False)

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.inner_lr,
        )

        self.model.train()
        self._prepare_task_embedding(target_generator)
        losses: list[float] = []

        for _ in range(num_steps):
            batch = target_generator.generate_batch(batch_size, t_span, t_eval)
            batch = {k: v.to(self.device) for k, v in batch.items()}
            step_losses = self.model.train_step(batch, optimizer)
            losses.append(step_losses["total"])

        # Unfreeze
        if freeze_encoder:
            for p in self.model.task_encoder.parameters():
                p.requires_grad_(True)

        if losses:
            logger.info(f"Fine-tune complete: {num_steps} steps, final loss={losses[-1]:.6f}")
        else:
            logger.info(f"Fine-tune complete: {num_steps} steps (no steps executed)")
        return losses

    def evaluate_transfer(
        self,
        test_generators: list[ReactorDataGenerator],
        t_span: tuple[float, float] = (0.0, 1.0),
        t_eval: npt.NDArray[Any] | None = None,
        batch_size: int = 8,
    ) -> dict[str, float | list[float]]:
        """Evaluate model on test reactors.

        Args:
            test_generators: Data generators for test reactors.
            t_span: Time interval.
            t_eval: Evaluation time points.
            batch_size: Batch size.

        Returns:
            Dict with 'mean_loss' and 'per_task_losses'.
        """
        if t_eval is None:
            t_eval = np.linspace(t_span[0], t_span[1], 30)

        self.model.eval()
        per_task_losses: list[float] = []

        with torch.no_grad():
            for tg in test_generators:
                batch = tg.generate_batch(batch_size, t_span, t_eval)
                batch = {k: v.to(self.device) for k, v in batch.items()}
                pred = self.model(batch["z0"], batch["t_span"])
                losses = self.model.compute_loss(pred, batch["targets"])
                per_task_losses.append(losses["total"].item())

        return {
            "mean_loss": float(np.mean(per_task_losses)),
            "per_task_losses": per_task_losses,
        }


__all__ = ["ReactorTaskEncoder", "FoundationNeuralODE", "FoundationTrainer"]
