"""Evaluation metrics for reactor models and Neural DEs."""

from __future__ import annotations

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _to_numpy(x: np.ndarray | torch.Tensor) -> np.ndarray:
    """Convert tensor to numpy array."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


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
    """
    pred = _to_numpy(y_pred)
    true = _to_numpy(y_true)
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def relative_rmse(
    y_pred: np.ndarray | torch.Tensor,
    y_true: np.ndarray | torch.Tensor,
) -> float:
    """Relative root mean squared error (normalized by mean of |y_true|).

    Args:
        y_pred: Predicted values, shape (n_samples, n_features)
        y_true: True values, shape (n_samples, n_features)

    Returns:
        Relative RMSE value (percentage)
    """
    true = _to_numpy(y_true)
    mean_abs = np.mean(np.abs(true))
    if mean_abs < 1e-12:
        logger.warning("Near-zero mean in y_true; relative RMSE may be unreliable")
        mean_abs = 1e-12
    return float(rmse(y_pred, y_true) / mean_abs * 100.0)


def mass_balance_error(
    state: torch.Tensor,
    stoichiometry: torch.Tensor,
    initial_state: torch.Tensor,
) -> torch.Tensor:
    """Compute mass balance violation.

    For a closed system the change in species concentrations must lie in
    the column space of the stoichiometric matrix transposed.  The error
    is the norm of the component orthogonal to that subspace.

    Args:
        state: Current state, shape (batch, n_species)
        stoichiometry: Stoichiometric matrix, shape (n_reactions, n_species)
        initial_state: Initial conditions, shape (batch, n_species)

    Returns:
        Mass balance error, shape (batch,)
    """
    delta = state - initial_state  # (batch, n_species)
    # Project delta onto the row space of stoichiometry (= column space of S^T)
    S = stoichiometry.float()  # (n_reactions, n_species)
    # Pseudo-inverse: S^T (S S^T)^{-1} S
    SST = S @ S.T
    SST_inv = torch.linalg.pinv(SST)
    proj_matrix = S.T @ SST_inv @ S  # (n_species, n_species)
    projected = delta @ proj_matrix.T  # (batch, n_species)
    residual = delta - projected  # (batch, n_species)
    return torch.norm(residual, dim=-1)


def energy_balance_error(
    temperature: torch.Tensor,
    concentrations: torch.Tensor,
    heat_of_reactions: torch.Tensor,
) -> torch.Tensor:
    """Compute energy balance violation.

    Measures the discrepancy between temperature change and the heat
    released/absorbed by concentration changes across the trajectory.

    Args:
        temperature: Temperature values, shape (batch, n_steps)
        concentrations: Concentration profiles, shape (batch, n_steps, n_species)
        heat_of_reactions: Reaction enthalpies, shape (n_reactions,)

    Returns:
        Energy balance error, shape (batch,)
    """
    # Approximate: sum of (dH * dC) should be proportional to dT
    dT = temperature[:, -1] - temperature[:, 0]  # (batch,)
    dC = concentrations[:, -1, :] - concentrations[:, 0, :]  # (batch, n_species)
    # If n_reactions != n_species, use the available reactions
    n_rxn = heat_of_reactions.shape[0]
    n_species = dC.shape[-1]
    n_use = min(n_rxn, n_species)
    heat_contribution = (dC[:, :n_use] * heat_of_reactions[:n_use]).sum(dim=-1)
    # Error: mismatch between heat contribution and temperature change (units differ,
    # so we report the L2 norm of the standardised difference)
    dT_std = dT / (dT.abs().max().clamp(min=1e-12))
    heat_std = heat_contribution / (heat_contribution.abs().max().clamp(min=1e-12))
    return (dT_std - heat_std).abs()


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
    """
    violations = threshold - state  # positive where state < threshold
    mask = violations > 0
    count = int(mask.sum().item())
    max_violation = float(violations[mask].max().item()) if count > 0 else 0.0
    return count, max_violation


def stoichiometric_error(
    concentrations: torch.Tensor,
    stoichiometry: torch.Tensor,
) -> torch.Tensor:
    """Compute stoichiometric consistency error.

    At each timestep, the concentration change from the previous step
    should lie in the row space of the stoichiometric matrix.

    Args:
        concentrations: Concentration profiles, shape (batch, n_steps, n_species)
        stoichiometry: Stoichiometric matrix, shape (n_reactions, n_species)

    Returns:
        Stoichiometric error per timestep, shape (batch, n_steps)
    """
    S = stoichiometry.float()
    SST = S @ S.T
    SST_inv = torch.linalg.pinv(SST)
    proj_matrix = S.T @ SST_inv @ S  # (n_species, n_species)

    # Compute per-step concentration changes
    dC = torch.diff(concentrations, dim=1)  # (batch, n_steps-1, n_species)
    projected = dC @ proj_matrix.T
    residual = dC - projected
    errors = torch.norm(residual, dim=-1)  # (batch, n_steps-1)

    # Pad first step with zero error
    batch = concentrations.shape[0]
    zero_col = torch.zeros(batch, 1, device=errors.device, dtype=errors.dtype)
    return torch.cat([zero_col, errors], dim=1)


def gibbs_monotonicity_score(
    gibbs_energy: torch.Tensor,
) -> float:
    """Compute percentage of timesteps where Gibbs energy decreases.

    Args:
        gibbs_energy: Gibbs free energy trajectory, shape (n_steps,)

    Returns:
        Percentage of monotonic decreasing steps (0-100)
    """
    if gibbs_energy.numel() < 2:
        return 100.0
    dG = torch.diff(gibbs_energy)
    n_decreasing = int((dG <= 0).sum().item())
    return float(n_decreasing / len(dG) * 100.0)


def rollout_divergence(
    short_rollout: torch.Tensor,
    long_rollout: torch.Tensor,
    horizon_ratio: float,
) -> float:
    """Measure trajectory divergence over extended rollouts.

    Computes the ratio of RMSE at the end of the long rollout vs the
    end of the short rollout.  A ratio of 1.0 means no divergence;
    values > 1.0 indicate the model degrades at longer horizons.

    Args:
        short_rollout: Prediction at training horizon, shape (n_steps, n_features)
        long_rollout: Prediction at extended horizon, shape (n_steps_long, n_features)
        horizon_ratio: Ratio of long/short horizon

    Returns:
        RMSE degradation factor
    """
    short_np = _to_numpy(short_rollout)
    long_np = _to_numpy(long_rollout)

    n_short = short_np.shape[0]
    # RMSE of short rollout (last quarter as proxy for accumulated error)
    quarter = max(1, n_short // 4)
    short_tail_rmse = float(np.sqrt(np.mean(short_np[-quarter:] ** 2)))

    # RMSE of long rollout (last quarter)
    n_long = long_np.shape[0]
    quarter_long = max(1, n_long // 4)
    long_tail_rmse = float(np.sqrt(np.mean(long_np[-quarter_long:] ** 2)))

    if short_tail_rmse < 1e-12:
        return horizon_ratio  # degenerate case: return nominal ratio
    return float(long_tail_rmse / short_tail_rmse)
