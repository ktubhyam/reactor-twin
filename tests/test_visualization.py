"""Tests for reactor_twin.utils.visualization."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest
import torch

matplotlib.use("Agg")  # non-interactive backend for testing

import matplotlib.pyplot as plt
import plotly.graph_objects as go

from reactor_twin.utils.visualization import (
    plot_bifurcation_diagram,
    plot_latent_space,
    plot_pareto_front,
    plot_phase_portrait,
    plot_residual_time_distribution,
    plot_sensitivity_heatmap,
    plot_trajectory,
)


class TestPlotTrajectory:
    def test_matplotlib_backend(self):
        t = np.linspace(0, 10, 50)
        y = np.column_stack([np.sin(t), np.cos(t)])
        fig = plot_trajectory(t, y, labels=["sin", "cos"], backend="matplotlib")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plotly_backend(self):
        t = np.linspace(0, 10, 50)
        y = np.column_stack([np.sin(t), np.cos(t)])
        fig = plot_trajectory(t, y, backend="plotly")
        assert isinstance(fig, go.Figure)

    def test_torch_input(self):
        t = torch.linspace(0, 5, 20)
        y = torch.randn(20, 3)
        fig = plot_trajectory(t, y, backend="matplotlib")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_1d_state(self):
        t = np.linspace(0, 5, 20)
        y = np.sin(t)
        fig = plot_trajectory(t, y, backend="matplotlib")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotPhasePortrait:
    def test_basic(self):
        x = np.sin(np.linspace(0, 2 * np.pi, 100))
        y = np.cos(np.linspace(0, 2 * np.pi, 100))
        fig = plot_phase_portrait(x, y)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 3  # trajectory + start + end


class TestPlotBifurcationDiagram:
    def test_single_branch(self):
        params = np.linspace(0, 10, 50)
        ss = np.sin(params)
        fig = plot_bifurcation_diagram(params, ss)
        assert isinstance(fig, go.Figure)

    def test_multi_branch(self):
        params = np.linspace(0, 10, 50)
        ss = np.column_stack([params, params**2])
        fig = plot_bifurcation_diagram(params, ss)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 2


class TestPlotRTD:
    def test_basic(self):
        bins = np.linspace(0, 10, 20)
        rtd = np.exp(-bins)
        fig = plot_residual_time_distribution(rtd, bins)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_with_theoretical(self):
        bins = np.linspace(0, 10, 20)
        rtd = np.exp(-bins)
        theoretical = np.exp(-bins) * 0.9
        fig = plot_residual_time_distribution(rtd, bins, theoretical=theoretical)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotSensitivityHeatmap:
    def test_basic(self):
        sm = np.random.randn(3, 4)
        params = ["k1", "k2", "k3"]
        states = ["C_A", "C_B", "T", "V"]
        fig = plot_sensitivity_heatmap(sm, params, states)
        assert isinstance(fig, go.Figure)


class TestPlotParetoFront:
    def test_basic(self):
        o1 = np.random.rand(20)
        o2 = np.random.rand(20)
        fig = plot_pareto_front(o1, o2)
        assert isinstance(fig, go.Figure)

    def test_with_labels(self):
        o1 = np.array([1, 2, 3, 4])
        o2 = np.array([4, 3, 2, 1])
        labels = ["A", "B", "C", "D"]
        fig = plot_pareto_front(o1, o2, labels=labels)
        assert isinstance(fig, go.Figure)


class TestPlotLatentSpace:
    def test_pca(self):
        z = torch.randn(50, 10)
        fig = plot_latent_space(z, method="pca")
        assert isinstance(fig, go.Figure)

    def test_with_labels(self):
        z = torch.randn(50, 5)
        labels = torch.randint(0, 3, (50,))
        fig = plot_latent_space(z, labels=labels)
        assert isinstance(fig, go.Figure)

    def test_2d_input(self):
        z = torch.randn(20, 2)
        fig = plot_latent_space(z)
        assert isinstance(fig, go.Figure)

    def test_tsne_with_sklearn(self):
        pytest.importorskip("sklearn")
        z = torch.randn(30, 8)
        fig = plot_latent_space(z, method="tsne")
        assert isinstance(fig, go.Figure)

    def test_tsne_fallback_without_sklearn(self):
        from unittest import mock

        z = torch.randn(30, 8)
        with mock.patch.dict("sys.modules", {"sklearn": None, "sklearn.manifold": None}):
            # Should fall back to PCA
            fig = plot_latent_space(z, method="tsne")
            assert isinstance(fig, go.Figure)

    def test_umap_fallback_without_umap(self):
        from unittest import mock

        z = torch.randn(30, 8)
        with mock.patch.dict("sys.modules", {"umap": None}):
            fig = plot_latent_space(z, method="umap")
            assert isinstance(fig, go.Figure)

    def test_unknown_method_raises(self):
        z = torch.randn(20, 5)
        with pytest.raises(ValueError, match="Unknown method"):
            plot_latent_space(z, method="invalid_method")
