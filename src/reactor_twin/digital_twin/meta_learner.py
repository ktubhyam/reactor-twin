"""Reptile meta-learner for cross-reactor transfer learning.

Implements the first-order Reptile meta-learning algorithm so that a
single Neural ODE can quickly adapt to new reactor types or operating
conditions with only a few gradient steps.

Reference:
    Nichol, A., Achiam, J. & Schulman, J. (2018)
    "On First-Order Meta-Learning Algorithms."  arXiv:1803.02999
"""

from __future__ import annotations

import copy
import logging
from typing import Any

import torch

from reactor_twin.core.base import AbstractNeuralDE
from reactor_twin.training.data_generator import ReactorDataGenerator

logger = logging.getLogger(__name__)


class ReptileMetaLearner:
    """Reptile meta-learning for Neural ODE cross-reactor transfer.

    Each *task* is a ``ReactorDataGenerator`` that can produce training
    batches for one reactor configuration.  The meta-learner repeatedly
    samples tasks, runs K inner-loop gradient steps, then performs the
    Reptile outer update:

    .. math::
        \\theta \\leftarrow \\theta + \\varepsilon\\,(\\tilde{\\theta}_{\\text{task}} - \\theta)

    Attributes:
        model: The meta-model being trained.
        meta_lr: Outer-loop (Reptile) step size.
        inner_lr: Inner-loop learning rate.
        inner_steps: Number of inner gradient steps per task.
    """

    def __init__(
        self,
        model: AbstractNeuralDE,
        meta_lr: float = 1e-3,
        inner_lr: float = 1e-3,
        inner_steps: int = 5,
        device: str | torch.device = "cpu",
    ) -> None:
        """Initialize the Reptile meta-learner.

        Args:
            model: Neural DE to meta-train.
            meta_lr: Outer-loop learning rate (epsilon in Reptile).
            inner_lr: Inner-loop SGD learning rate.
            inner_steps: Gradient steps per task in the inner loop.
            device: Torch device.
        """
        self.model = model
        self.meta_lr = meta_lr
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.device = torch.device(device)

        logger.info(
            f"ReptileMetaLearner: meta_lr={meta_lr}, inner_lr={inner_lr}, inner_steps={inner_steps}"
        )

    # ------------------------------------------------------------------
    # Inner loop
    # ------------------------------------------------------------------

    def _inner_loop(
        self,
        task_generator: ReactorDataGenerator,
        t_span: tuple[float, float],
        t_eval: Any,
        batch_size: int = 16,
    ) -> dict[str, torch.Tensor]:
        """Run K gradient steps on a single reactor task.

        The model is temporarily cloned so the meta-parameters are not
        modified in-place.

        Args:
            task_generator: Data generator for this task's reactor.
            t_span: Simulation time interval.
            t_eval: Evaluation time points (numpy array).
            batch_size: Samples per inner-loop batch.

        Returns:
            Dictionary mapping parameter names to the adapted values
            ``theta_task`` after K steps.
        """
        # Clone model for inner-loop adaptation (isolated from meta-model)
        model_copy = copy.deepcopy(self.model)
        model_copy.to(self.device)
        model_copy.train()

        inner_opt = torch.optim.SGD(model_copy.parameters(), lr=self.inner_lr)

        for _ in range(self.inner_steps):
            batch = task_generator.generate_batch(batch_size, t_span, t_eval)
            batch = {k: v.to(self.device) for k, v in batch.items()}
            model_copy.train_step(batch, inner_opt)

        # Collect adapted parameters
        adapted_params = {n: p.detach().clone() for n, p in model_copy.named_parameters()}
        return adapted_params

    # ------------------------------------------------------------------
    # Outer (meta) step
    # ------------------------------------------------------------------

    def meta_step(
        self,
        task_generators: list[ReactorDataGenerator],
        t_span: tuple[float, float],
        t_eval: Any,
        tasks_per_step: int | None = None,
        batch_size: int = 16,
    ) -> float:
        """Perform one Reptile meta-update.

        Samples a subset of tasks, runs the inner loop on each, then
        updates the meta-parameters toward the average adapted parameters.

        Args:
            task_generators: Pool of reactor data generators (one per task).
            t_span: Simulation time interval.
            t_eval: Evaluation time points.
            tasks_per_step: How many tasks to sample. Defaults to all.
            batch_size: Inner-loop batch size.

        Returns:
            Average parameter displacement (diagnostic scalar).
        """
        num_tasks = len(task_generators)
        if tasks_per_step is None or tasks_per_step >= num_tasks:
            selected = task_generators
        else:
            indices = torch.randperm(num_tasks)[:tasks_per_step].tolist()
            selected = [task_generators[i] for i in indices]

        # Collect adapted params from each task
        all_adapted: list[dict[str, torch.Tensor]] = []
        for tg in selected:
            adapted = self._inner_loop(tg, t_span, t_eval, batch_size)
            all_adapted.append(adapted)

        # Reptile update: theta += epsilon * mean(theta_task - theta)
        displacement = 0.0
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                # Average across tasks
                avg_adapted = torch.stack([a[n] for a in all_adapted]).mean(dim=0)
                delta = avg_adapted.to(self.device) - p.data
                p.data.add_(delta, alpha=self.meta_lr)
                displacement += delta.norm().item()

        return displacement

    # ------------------------------------------------------------------
    # Full meta-training loop
    # ------------------------------------------------------------------

    def meta_train(
        self,
        task_generators: list[ReactorDataGenerator],
        num_steps: int = 100,
        t_span: tuple[float, float] = (0.0, 1.0),
        t_eval: Any = None,
        tasks_per_step: int | None = None,
        batch_size: int = 16,
        log_interval: int = 10,
    ) -> list[float]:
        """Full Reptile meta-training loop.

        Args:
            task_generators: Pool of reactor data generators.
            num_steps: Number of outer meta-steps.
            t_span: Simulation time interval.
            t_eval: Evaluation time points. Defaults to 50 uniform points.
            tasks_per_step: Tasks sampled per meta-step.
            batch_size: Inner-loop batch size.
            log_interval: Print progress every N steps.

        Returns:
            List of displacement values (one per meta-step).
        """
        import numpy as np

        if t_eval is None:
            t_eval = np.linspace(t_span[0], t_span[1], 50)

        displacements: list[float] = []

        for step in range(num_steps):
            disp = self.meta_step(
                task_generators,
                t_span,
                t_eval,
                tasks_per_step=tasks_per_step,
                batch_size=batch_size,
            )
            displacements.append(disp)

            if (step + 1) % log_interval == 0:
                logger.info(f"Meta-step {step + 1}/{num_steps}: displacement={disp:.6f}")

        logger.info(f"Meta-training complete: {num_steps} steps.")
        return displacements

    # ------------------------------------------------------------------
    # Few-shot fine-tuning
    # ------------------------------------------------------------------

    def fine_tune(
        self,
        task_generator: ReactorDataGenerator,
        t_span: tuple[float, float] = (0.0, 1.0),
        t_eval: Any = None,
        num_steps: int = 10,
        batch_size: int = 16,
        lr: float | None = None,
    ) -> list[float]:
        """Few-shot adaptation on a new reactor.

        Fine-tunes the meta-learned model *in-place* for a specific
        reactor task.

        Args:
            task_generator: Data generator for the target reactor.
            t_span: Simulation time interval.
            t_eval: Evaluation time points.
            num_steps: Number of fine-tuning gradient steps.
            batch_size: Samples per step.
            lr: Learning rate. Defaults to ``inner_lr``.

        Returns:
            List of loss values per step.
        """
        import numpy as np

        if t_eval is None:
            t_eval = np.linspace(t_span[0], t_span[1], 50)

        lr = lr or self.inner_lr
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.model.train()
        losses: list[float] = []

        for _step in range(num_steps):
            batch = task_generator.generate_batch(batch_size, t_span, t_eval)
            batch = {k: v.to(self.device) for k, v in batch.items()}
            step_losses = self.model.train_step(batch, optimizer)
            losses.append(step_losses["total"])

        logger.info(f"Fine-tuning complete: {num_steps} steps, final loss={losses[-1]:.6f}")
        return losses


__all__ = ["ReptileMetaLearner"]
