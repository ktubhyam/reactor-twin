"""Evaluation metrics for reactor models and Neural DEs."""

from __future__ import annotations

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


def rmse(
    y_pred: np.ndarray | torch.Tensor,
    y_true: np.ndarray | torch.Tensor,
) -> float:
    """Root mean squared error.

    Args:
        y_pred: Predicted values, shape (n_samples, n_features)
        y_true: True values, shape (n_samples, n_features)

    Returns:
        RMSE value

    Raises:
        NotImplementedError: TODO
    """
    raise NotImplementedError("TODO")


def relative_rmse(
    y_pred: np.ndarray | torch.Tensor,
    y_true: np.ndarray | torch.Tensor,
) -> float:
    """Relative root mean squared error (normalized by mean).

    Args:
        y_pred: Predicted values, shape (n_samples, n_features)
        y_true: True values, shape (n_samples, n_features)

    Returns:
        Relative RMSE value (percentage)

    Raises:
        NotImplementedError: TODO
    """
    raise NotImplementedError("TODO")


def mass_balance_error(
    state: torch.Tensor,
    stoichiometry: torch.Tensor,
    initial_state: torch.Tensor,
) -> torch.Tensor:
    """Compute mass balance violation.

    Args:
        state: Current state, shape (batch, n_species)
        stoichiometry: Stoichiometric matrix, shape (n_reactions, n_species)
        initial_state: Initial conditions, shape (batch, n_species)

    Returns:
        Mass balance error, shape (batch,)

    Raises:
        NotImplementedError: TODO
    """
    raise NotImplementedError("TODO")


def energy_balance_error(
    temperature: torch.Tensor,
    concentrations: torch.Tensor,
    heat_of_reactions: torch.Tensor,
) -> torch.Tensor:
    """Compute energy balance violation.

    Args:
        temperature: Temperature values, shape (batch, n_steps)
        concentrations: Concentration profiles, shape (batch, n_steps, n_species)
        heat_of_reactions: Reaction enthalpies, shape (n_reactions,)

    Returns:
        Energy balance error, shape (batch,)

    Raises:
        NotImplementedError: TODO
    """
    raise NotImplementedError("TODO")


def positivity_violations(
    state: torch.Tensor,
    threshold: float = 0.0,
) -> tuple[int, float]:
    """Count and measure positivity constraint violations.

    Args:
        state: State tensor, shape (batch, n_steps, n_features)
        threshold: Minimum allowed value

    Returns:
        Tuple of (number of violations, maximum violation magnitude)

    Raises:
        NotImplementedError: TODO
    """
    raise NotImplementedError("TODO")


def stoichiometric_error(
    concentrations: torch.Tensor,
    stoichiometry: torch.Tensor,
) -> torch.Tensor:
    """Compute stoichiometric consistency error.

    Args:
        concentrations: Concentration profiles, shape (batch, n_steps, n_species)
        stoichiometry: Stoichiometric matrix, shape (n_reactions, n_species)

    Returns:
        Stoichiometric error per timestep, shape (batch, n_steps)

    Raises:
        NotImplementedError: TODO
    """
    raise NotImplementedError("TODO")


def gibbs_monotonicity_score(
    gibbs_energy: torch.Tensor,
) -> float:
    """Compute percentage of timesteps where Gibbs energy decreases.

    Args:
        gibbs_energy: Gibbs free energy trajectory, shape (n_steps,)

    Returns:
        Percentage of monotonic decreasing steps (0-100)

    Raises:
        NotImplementedError: TODO
    """
    raise NotImplementedError("TODO")


def rollout_divergence(
    short_rollout: torch.Tensor,
    long_rollout: torch.Tensor,
    horizon_ratio: float,
) -> float:
    """Measure trajectory divergence over extended rollouts.

    Args:
        short_rollout: Prediction at training horizon, shape (n_steps, n_features)
        long_rollout: Prediction at extended horizon, shape (n_steps_long, n_features)
        horizon_ratio: Ratio of long/short horizon

    Returns:
        RMSE degradation factor

    Raises:
        NotImplementedError: TODO
    """
    raise NotImplementedError("TODO")
