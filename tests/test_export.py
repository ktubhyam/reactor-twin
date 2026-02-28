"""Tests for the ONNX export module."""

from __future__ import annotations

import builtins
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from reactor_twin.core.augmented_neural_ode import AugmentedNeuralODE
from reactor_twin.core.latent_neural_ode import LatentNeuralODE
from reactor_twin.core.neural_ode import NeuralODE
from reactor_twin.exceptions import ExportError
from reactor_twin.export.onnx_export import (
    ONNXExporter,
    ONNXInferenceRunner,
    benchmark_inference,
)


@pytest.fixture
def tmp_export_dir(tmp_path: Path) -> Path:
    d = tmp_path / "onnx_export"
    d.mkdir()
    return d


@pytest.fixture
def neural_ode():
    return NeuralODE(state_dim=3, hidden_dim=16, num_layers=2, adjoint=False)


@pytest.fixture
def augmented_ode():
    return AugmentedNeuralODE(
        state_dim=3, augment_dim=2, hidden_dim=16, num_layers=2, adjoint=False
    )


@pytest.fixture
def latent_ode():
    return LatentNeuralODE(
        state_dim=5,
        latent_dim=3,
        encoder_hidden_dim=16,
        decoder_hidden_dim=16,
        encoder_type="mlp",
        adjoint=False,
    )


# ── ONNXExporter.export ──────────────────────────────────────────────


class TestONNXExport:
    def test_export_neural_ode(self, neural_ode, tmp_export_dir):
        paths = ONNXExporter.export(neural_ode, tmp_export_dir)
        assert "ode_func" in paths
        assert paths["ode_func"].exists()

    def test_export_augmented_ode(self, augmented_ode, tmp_export_dir):
        paths = ONNXExporter.export(augmented_ode, tmp_export_dir)
        assert "ode_func" in paths
        assert paths["ode_func"].exists()

    def test_export_latent_ode(self, latent_ode, tmp_export_dir):
        paths = ONNXExporter.export(latent_ode, tmp_export_dir)
        assert "ode_func" in paths
        assert "encoder" in paths
        assert "decoder" in paths
        for p in paths.values():
            assert p.exists()

    def test_export_sde_partial(self, tmp_export_dir):
        pytest.importorskip("torchsde")
        from reactor_twin.core.neural_sde import NeuralSDE

        model = NeuralSDE(state_dim=3)
        paths = ONNXExporter.export(model, tmp_export_dir)
        assert "drift_fn" in paths
        assert paths["drift_fn"].exists()

    def test_export_cde_partial(self, tmp_path):
        pytest.importorskip("torchcde")
        from reactor_twin.core.neural_cde import NeuralCDE

        model = NeuralCDE(state_dim=4, input_dim=2)
        out_dir = tmp_path / "cde_export"
        out_dir.mkdir()
        paths = ONNXExporter.export(model, out_dir)
        assert "cde_func" in paths
        assert paths["cde_func"].exists()

    def test_export_creates_directory(self, neural_ode, tmp_path):
        new_dir = tmp_path / "new_subdir" / "export"
        paths = ONNXExporter.export(neural_ode, new_dir)
        assert new_dir.exists()
        assert paths["ode_func"].exists()

    def test_exported_onnx_valid(self, neural_ode, tmp_export_dir):
        paths = ONNXExporter.export(neural_ode, tmp_export_dir, validate=True)
        model = onnx.load(str(paths["ode_func"]))
        onnx.checker.check_model(model)


# ── ONNXExporter.validate_export ─────────────────────────────────────


class TestONNXValidation:
    def test_validate_neural_ode(self, neural_ode, tmp_export_dir):
        paths = ONNXExporter.export(neural_ode, tmp_export_dir)
        errors = ONNXExporter.validate_export(neural_ode, paths)
        assert "ode_func" in errors
        assert errors["ode_func"] < 1e-4

    def test_validate_augmented_ode(self, augmented_ode, tmp_export_dir):
        paths = ONNXExporter.export(augmented_ode, tmp_export_dir)
        errors = ONNXExporter.validate_export(augmented_ode, paths)
        assert errors["ode_func"] < 1e-4


# ── ONNXInferenceRunner ──────────────────────────────────────────────


class TestONNXInferenceRunner:
    def test_predict_shape(self, neural_ode, tmp_export_dir):
        paths = ONNXExporter.export(neural_ode, tmp_export_dir)
        runner = ONNXInferenceRunner(paths, solver="rk4")
        z0 = np.random.randn(2, 3).astype(np.float32)
        t_span = np.linspace(0, 1, 5).astype(np.float32)
        result = runner.predict(z0, t_span)
        assert result.shape == (2, 5, 3)

    def test_predict_euler(self, neural_ode, tmp_export_dir):
        paths = ONNXExporter.export(neural_ode, tmp_export_dir)
        runner = ONNXInferenceRunner(paths, solver="euler")
        z0 = np.random.randn(1, 3).astype(np.float32)
        t_span = np.linspace(0, 0.5, 10).astype(np.float32)
        result = runner.predict(z0, t_span)
        assert result.shape == (1, 10, 3)
        assert np.all(np.isfinite(result))

    def test_predict_rk4_finite(self, neural_ode, tmp_export_dir):
        paths = ONNXExporter.export(neural_ode, tmp_export_dir)
        runner = ONNXInferenceRunner(paths, solver="rk4")
        z0 = np.random.randn(2, 3).astype(np.float32)
        t_span = np.linspace(0, 0.5, 5).astype(np.float32)
        result = runner.predict(z0, t_span)
        assert np.all(np.isfinite(result))

    def test_initial_condition_preserved(self, neural_ode, tmp_export_dir):
        paths = ONNXExporter.export(neural_ode, tmp_export_dir)
        runner = ONNXInferenceRunner(paths, solver="rk4")
        z0 = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        t_span = np.linspace(0, 1, 5).astype(np.float32)
        result = runner.predict(z0, t_span)
        np.testing.assert_allclose(result[0, 0, :], z0[0], atol=1e-6)


# ── benchmark_inference ──────────────────────────────────────────────


class TestBenchmark:
    def test_benchmark_returns_dict(self, neural_ode, tmp_export_dir):
        paths = ONNXExporter.export(neural_ode, tmp_export_dir)
        runner = ONNXInferenceRunner(paths, solver="rk4")
        z0 = np.random.randn(1, 3).astype(np.float32)
        t_span = np.linspace(0, 1, 10).astype(np.float32)
        result = benchmark_inference(neural_ode, runner, z0, t_span, n_repeats=3)
        assert "pytorch_ms" in result
        assert "onnx_ms" in result
        assert "speedup" in result
        assert result["pytorch_ms"] > 0
        assert result["onnx_ms"] > 0


# ── Error handling paths ─────────────────────────────────────────────


class TestONNXExportErrorPaths:
    """Tests covering error-handling code paths in onnx_export.py."""

    def test_export_raises_when_onnx_not_available(self, tmp_path):
        """Lines 62-63: onnx ImportError when onnx package is missing."""
        model = NeuralODE(state_dim=3, hidden_dim=16, num_layers=2, adjoint=False)
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "onnx":
                raise ImportError("No module named 'onnx'")
            return real_import(name, *args, **kwargs)

        with (
            patch("builtins.__import__", side_effect=fake_import),
            pytest.raises(ExportError, match="onnx package is required"),
        ):
            ONNXExporter.export(model, tmp_path / "out")

    def test_export_raises_when_model_has_no_ode_func(self, tmp_path):
        """Line 81: Model without ode_func attribute."""
        # Create a simple nn.Module without ode_func
        model = torch.nn.Linear(3, 3)
        # The model class name must not be NeuralSDE or NeuralCDE
        with pytest.raises(ExportError, match="does not have an 'ode_func' attribute"):
            ONNXExporter.export(model, tmp_path / "out")

    def test_export_raises_on_ode_func_export_failure(self, tmp_path):
        """Lines 110-111: Exception during ODE func export."""
        model = NeuralODE(state_dim=3, hidden_dim=16, num_layers=2, adjoint=False)
        with (
            patch("torch.onnx.export", side_effect=RuntimeError("export failed")),
            pytest.raises(ExportError, match="Failed to export ode_func"),
        ):
            ONNXExporter.export(model, tmp_path / "out", validate=False)

    def test_export_raises_on_encoder_export_failure(self, tmp_path):
        """Lines 133-134: Exception during encoder export."""
        latent_model = LatentNeuralODE(
            state_dim=5,
            latent_dim=3,
            encoder_hidden_dim=16,
            decoder_hidden_dim=16,
            encoder_type="mlp",
            adjoint=False,
        )
        call_count = 0
        real_export = torch.onnx.export

        def fail_on_second_call(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("encoder export failed")
            return real_export(*args, **kwargs)

        with (
            patch("torch.onnx.export", side_effect=fail_on_second_call),
            pytest.raises(ExportError, match="Failed to export encoder"),
        ):
            ONNXExporter.export(latent_model, tmp_path / "out", validate=False)

    def test_export_raises_on_decoder_export_failure(self, tmp_path):
        """Lines 151-152: Exception during decoder export."""
        latent_model = LatentNeuralODE(
            state_dim=5,
            latent_dim=3,
            encoder_hidden_dim=16,
            decoder_hidden_dim=16,
            encoder_type="mlp",
            adjoint=False,
        )
        call_count = 0
        real_export = torch.onnx.export

        def fail_on_third_call(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 3:
                raise RuntimeError("decoder export failed")
            return real_export(*args, **kwargs)

        with (
            patch("torch.onnx.export", side_effect=fail_on_third_call),
            pytest.raises(ExportError, match="Failed to export decoder"),
        ):
            ONNXExporter.export(latent_model, tmp_path / "out", validate=False)

    def test_validate_export_raises_when_onnxruntime_not_available(self, tmp_path):
        """Lines 179-180: onnxruntime ImportError for validation."""
        model = NeuralODE(state_dim=3, hidden_dim=16, num_layers=2, adjoint=False)
        paths = ONNXExporter.export(model, tmp_path / "out")

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "onnxruntime":
                raise ImportError("No module named 'onnxruntime'")
            return real_import(name, *args, **kwargs)

        with (
            patch("builtins.__import__", side_effect=fake_import),
            pytest.raises(ExportError, match="onnxruntime is required"),
        ):
            ONNXExporter.validate_export(model, paths)

    def test_inference_runner_raises_when_onnxruntime_not_available(self):
        """Lines 233-234: onnxruntime ImportError for ONNXInferenceRunner."""
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "onnxruntime":
                raise ImportError("No module named 'onnxruntime'")
            return real_import(name, *args, **kwargs)

        with (
            patch("builtins.__import__", side_effect=fake_import),
            pytest.raises(ExportError, match="onnxruntime is required"),
        ):
            ONNXInferenceRunner({"ode_func": "/fake/path.onnx"})
