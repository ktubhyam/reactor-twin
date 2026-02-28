"""Visualization utilities for reactor dynamics and Neural DE analysis."""

from __future__ import annotations

import logging
from typing import Any, cast

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
import torch

logger = logging.getLogger(__name__)


def _to_numpy(x: npt.NDArray[Any] | torch.Tensor) -> npt.NDArray[Any]:
    """Convert tensor to numpy array."""
    if isinstance(x, torch.Tensor):
        return cast(npt.NDArray[Any], x.detach().cpu().numpy())
    return np.asarray(x)


def plot_trajectory(
    t: npt.NDArray[Any] | torch.Tensor,
    y: npt.NDArray[Any] | torch.Tensor,
    labels: list[str] | None = None,
    title: str = "Reactor Trajectory",
    backend: str = "matplotlib",
) -> Any:
    """Plot reactor state trajectory over time.

    Args:
        t: Time points, shape (n_steps,)
        y: State values, shape (n_steps, n_states)
        labels: State variable labels
        title: Plot title
        backend: "matplotlib" or "plotly"

    Returns:
        Figure object (matplotlib.Figure or plotly.graph_objects.Figure)
    """
    t_np = _to_numpy(t)
    y_np = _to_numpy(y)

    if y_np.ndim == 1:
        y_np = y_np.reshape(-1, 1)

    n_states = y_np.shape[1]
    if labels is None:
        labels = [f"State {i}" for i in range(n_states)]

    if backend == "plotly":
        fig = go.Figure()
        for i in range(n_states):
            fig.add_trace(go.Scatter(x=t_np, y=y_np[:, i], mode="lines", name=labels[i]))
        fig.update_layout(title=title, xaxis_title="Time", yaxis_title="Value")
        return fig

    # matplotlib
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(n_states):
        ax.plot(t_np, y_np[:, i], label=labels[i])
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    return fig


def plot_phase_portrait(
    x: npt.NDArray[Any] | torch.Tensor,
    y: npt.NDArray[Any] | torch.Tensor,
    xlabel: str = "x",
    ylabel: str = "y",
    title: str = "Phase Portrait",
) -> go.Figure:
    """Plot 2D phase portrait of reactor dynamics.

    Args:
        x: First state variable, shape (n_steps,)
        y: Second state variable, shape (n_steps,)
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        title: Plot title

    Returns:
        Plotly Figure object
    """
    x_np = _to_numpy(x)
    y_np = _to_numpy(y)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_np, y=y_np, mode="lines",
        line={"color": "steelblue", "width": 2},
        name="Trajectory",
    ))
    # Mark start and end
    fig.add_trace(go.Scatter(
        x=[x_np[0]], y=[y_np[0]], mode="markers",
        marker={"size": 10, "color": "green", "symbol": "circle"},
        name="Start",
    ))
    fig.add_trace(go.Scatter(
        x=[x_np[-1]], y=[y_np[-1]], mode="markers",
        marker={"size": 10, "color": "red", "symbol": "x"},
        name="End",
    ))
    fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title=ylabel)
    return fig


def plot_bifurcation_diagram(
    param_values: npt.NDArray[Any],
    steady_states: npt.NDArray[Any],
    param_name: str = "Parameter",
    state_name: str = "Steady State",
) -> go.Figure:
    """Plot bifurcation diagram.

    Args:
        param_values: Parameter sweep values, shape (n_params,)
        steady_states: Steady state values, shape (n_params, n_branches)
        param_name: Parameter label
        state_name: State variable label

    Returns:
        Plotly Figure object
    """
    param_np = _to_numpy(param_values)
    ss_np = _to_numpy(steady_states)

    if ss_np.ndim == 1:
        ss_np = ss_np.reshape(-1, 1)

    fig = go.Figure()
    n_branches = ss_np.shape[1]
    for b in range(n_branches):
        fig.add_trace(go.Scatter(
            x=param_np, y=ss_np[:, b], mode="lines+markers",
            marker={"size": 4},
            name=f"Branch {b + 1}",
        ))
    fig.update_layout(title="Bifurcation Diagram", xaxis_title=param_name, yaxis_title=state_name)
    return fig


def plot_residual_time_distribution(
    rtd: npt.NDArray[Any],
    time_bins: npt.NDArray[Any],
    theoretical: npt.NDArray[Any] | None = None,
) -> matplotlib.figure.Figure:
    """Plot residence time distribution (RTD).

    Args:
        rtd: Measured RTD, shape (n_bins,)
        time_bins: Time bin centers, shape (n_bins,)
        theoretical: Theoretical RTD for comparison, shape (n_bins,)

    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(time_bins, rtd, width=np.diff(time_bins, append=time_bins[-1] * 1.1).mean() * 0.8,
           alpha=0.7, label="Measured RTD", color="steelblue")
    if theoretical is not None:
        ax.plot(time_bins, theoretical, "r-", linewidth=2, label="Theoretical")
    ax.set_xlabel("Time")
    ax.set_ylabel("E(t)")
    ax.set_title("Residence Time Distribution")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_sensitivity_heatmap(
    sensitivity_matrix: npt.NDArray[Any],
    param_names: list[str],
    state_names: list[str],
) -> go.Figure:
    """Plot sensitivity analysis heatmap.

    Args:
        sensitivity_matrix: Sensitivity values, shape (n_params, n_states)
        param_names: Parameter labels
        state_names: State variable labels

    Returns:
        Plotly Figure object
    """
    sm = _to_numpy(sensitivity_matrix)

    fig = go.Figure(data=go.Heatmap(
        z=sm,
        x=state_names,
        y=param_names,
        colorscale="RdBu_r",
        zmid=0,
        text=np.round(sm, 3).astype(str),
        texttemplate="%{text}",
    ))
    fig.update_layout(
        title="Sensitivity Analysis",
        xaxis_title="State Variables",
        yaxis_title="Parameters",
    )
    return fig


def plot_pareto_front(
    objective_1: npt.NDArray[Any],
    objective_2: npt.NDArray[Any],
    labels: list[str] | None = None,
) -> go.Figure:
    """Plot Pareto front for multi-objective optimization.

    Args:
        objective_1: First objective values, shape (n_points,)
        objective_2: Second objective values, shape (n_points,)
        labels: Point labels for hover

    Returns:
        Plotly Figure object
    """
    o1 = _to_numpy(objective_1)
    o2 = _to_numpy(objective_2)

    # Identify Pareto-optimal points
    is_pareto = np.ones(len(o1), dtype=bool)
    for i in range(len(o1)):
        if is_pareto[i]:
            # A point is dominated if another point is <= on both and < on at least one
            is_pareto[i] = not np.any(
                (o1[:i][is_pareto[:i]] <= o1[i]) & (o2[:i][is_pareto[:i]] <= o2[i])
                & ((o1[:i][is_pareto[:i]] < o1[i]) | (o2[:i][is_pareto[:i]] < o2[i]))
            ) and not np.any(
                (o1[i + 1:] <= o1[i]) & (o2[i + 1:] <= o2[i])
                & ((o1[i + 1:] < o1[i]) | (o2[i + 1:] < o2[i]))
            )

    fig = go.Figure()
    # All points
    fig.add_trace(go.Scatter(
        x=o1, y=o2, mode="markers",
        marker={"size": 8, "color": "lightgray"},
        text=labels,
        name="All Solutions",
    ))
    # Pareto front
    pareto_idx = np.where(is_pareto)[0]
    sorted_idx = pareto_idx[np.argsort(o1[pareto_idx])]
    fig.add_trace(go.Scatter(
        x=o1[sorted_idx], y=o2[sorted_idx], mode="lines+markers",
        marker={"size": 10, "color": "red"},
        line={"color": "red", "width": 2},
        name="Pareto Front",
    ))
    fig.update_layout(
        title="Pareto Front",
        xaxis_title="Objective 1",
        yaxis_title="Objective 2",
    )
    return fig


def plot_latent_space(
    z: torch.Tensor,
    labels: torch.Tensor | None = None,
    method: str = "pca",
) -> go.Figure:
    """Visualize latent space of Latent Neural ODE.

    Args:
        z: Latent embeddings, shape (n_samples, latent_dim)
        labels: Optional labels for coloring, shape (n_samples,)
        method: Dimensionality reduction method ("pca", "tsne", "umap")

    Returns:
        Plotly Figure object
    """
    z_np = _to_numpy(z)
    n_samples, latent_dim = z_np.shape

    if latent_dim <= 2:
        z_2d = z_np[:, :2]
    elif method == "pca":
        # Simple PCA via SVD (no sklearn dependency)
        z_centered = z_np - z_np.mean(axis=0)
        U, S, Vt = np.linalg.svd(z_centered, full_matrices=False)
        z_2d = z_centered @ Vt[:2].T
    elif method == "tsne":
        try:
            from sklearn.manifold import TSNE
        except ImportError:
            logger.warning("sklearn not available; falling back to PCA")
            return plot_latent_space(z, labels, method="pca")
        z_2d = TSNE(n_components=2, perplexity=min(30, n_samples - 1)).fit_transform(z_np)
    elif method == "umap":
        try:
            import umap
        except ImportError:
            logger.warning("umap not available; falling back to PCA")
            return plot_latent_space(z, labels, method="pca")
        z_2d = umap.UMAP(n_components=2).fit_transform(z_np)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'pca', 'tsne', or 'umap'.")

    color = None
    if labels is not None:
        color = _to_numpy(labels)

    fig = go.Figure(data=go.Scatter(
        x=z_2d[:, 0],
        y=z_2d[:, 1] if z_2d.shape[1] > 1 else np.zeros(n_samples),
        mode="markers",
        marker={"size": 6, "color": color, "colorscale": "Viridis", "showscale": color is not None},
    ))
    fig.update_layout(
        title=f"Latent Space ({method.upper()})",
        xaxis_title="Component 1",
        yaxis_title="Component 2",
    )
    return fig
