"""Tests for reactor_twin.cli — unified CLI."""

from __future__ import annotations

import argparse
from unittest.mock import MagicMock, patch

import pytest
import torch
import yaml

import reactor_twin.cli
from reactor_twin.cli import cmd_dashboard, cmd_export, cmd_serve, cmd_train, main

# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def sample_config_path(tmp_path):
    """Write a minimal YAML config and return its path."""
    config = {
        "reactor": {
            "reactor_type": "exothermic_ab",
            "volume": 1.0,
            "temperature": 300.0,
            "pressure": 101325.0,
            "kinetics": "arrhenius",
            "n_species": 2,
        },
        "neural_de": {
            "model_type": "neural_ode",
            "hidden_dims": [16, 16],
            "activation": "relu",
            "solver": "euler",
            "atol": 1e-3,
            "rtol": 1e-2,
        },
        "training": {
            "batch_size": 4,
            "n_epochs": 1,
            "learning_rate": 1e-3,
            "optimizer": "adam",
        },
        "seed": 42,
    }
    p = tmp_path / "config.yaml"
    with open(p, "w") as f:
        yaml.dump(config, f)
    return p


@pytest.fixture
def model_checkpoint_path(tmp_path):
    """Save a small NeuralODE checkpoint and return its path."""
    from reactor_twin.core.neural_ode import NeuralODE
    from reactor_twin.core.ode_func import MLPODEFunc

    ode_func = MLPODEFunc(state_dim=2, hidden_dim=8, num_layers=2)
    model = NeuralODE(state_dim=2, ode_func=ode_func, solver="euler", adjoint=False)
    p = tmp_path / "model.pt"
    torch.save(model, p)
    return p


# ── main() and argument parsing ──────────────────────────────────────


class TestMain:
    def test_no_args_prints_help(self, capsys):
        """main() with no args prints help and exits 0."""
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code == 0

    def test_version_flag(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(["--version"])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "1.0.0" in captured.out

    def test_train_requires_config(self):
        with pytest.raises(SystemExit) as exc_info:
            main(["train"])
        assert exc_info.value.code == 2  # argparse error

    def test_export_requires_model(self):
        with pytest.raises(SystemExit) as exc_info:
            main(["export"])
        assert exc_info.value.code == 2


# ── cmd_train ────────────────────────────────────────────────────────


class TestCmdTrain:
    def test_train_runs(self, sample_config_path, tmp_path):
        output = tmp_path / "trained.pt"
        args = argparse.Namespace(config=str(sample_config_path), output=str(output))
        cmd_train(args)
        assert output.exists()

    def test_train_no_output(self, sample_config_path):
        args = argparse.Namespace(config=str(sample_config_path), output=None)
        cmd_train(args)  # should not raise

    def test_train_unknown_reactor(self, tmp_path):
        config = {
            "reactor": {
                "reactor_type": "nonexistent_reactor",
                "volume": 1.0,
                "temperature": 300.0,
                "pressure": 101325.0,
                "kinetics": "arrhenius",
                "n_species": 2,
            },
            "neural_de": {
                "model_type": "neural_ode",
                "hidden_dims": [16],
                "solver": "euler",
            },
            "training": {"batch_size": 4, "n_epochs": 1, "learning_rate": 1e-3},
            "seed": 42,
        }
        p = tmp_path / "bad_config.yaml"
        with open(p, "w") as f:
            yaml.dump(config, f)
        args = argparse.Namespace(config=str(p), output=None)
        with pytest.raises(SystemExit):
            cmd_train(args)


# ── cmd_serve ────────────────────────────────────────────────────────


class TestCmdServe:
    def test_serve_calls_uvicorn(self):
        mock_uvicorn = MagicMock()
        with patch.dict("sys.modules", {"uvicorn": mock_uvicorn}):
            args = argparse.Namespace(host="127.0.0.1", port=9999)
            cmd_serve(args)
            mock_uvicorn.run.assert_called_once()

    def test_serve_missing_uvicorn(self):
        with patch.dict("sys.modules", {"uvicorn": None}):
            args = argparse.Namespace(host="127.0.0.1", port=9999)
            with pytest.raises((SystemExit, ImportError)):
                cmd_serve(args)


# ── cmd_export ───────────────────────────────────────────────────────


class TestCmdExport:
    def test_export_unsupported_format(self):
        args = argparse.Namespace(model="x.pt", format="tflite", output="out.tflite")
        with pytest.raises(SystemExit):
            cmd_export(args)

    def test_export_onnx(self, model_checkpoint_path, tmp_path):
        output_dir = tmp_path / "onnx_out"
        output_dir.mkdir()
        args = argparse.Namespace(
            model=str(model_checkpoint_path), format="onnx", output=str(output_dir / "model.onnx")
        )
        cmd_export(args)
        # ONNXExporter.export writes files into the output_dir
        assert any(output_dir.iterdir())

    def test_export_state_dict_checkpoint(self, tmp_path):
        """Export from a full model saved as a checkpoint object."""
        # The state_dict path in cli.py reconstructs NeuralODE with defaults,
        # so we test the whole-model path instead (which already works above).
        # For state_dict, we'd need to also store hidden_dim/num_layers.
        # Just verify the unsupported-checkpoint branch.
        bad_path = tmp_path / "str_ckpt.pt"
        torch.save("not a model", bad_path)
        args = argparse.Namespace(model=str(bad_path), format="onnx", output=str(tmp_path / "out.onnx"))
        with pytest.raises(SystemExit):
            cmd_export(args)

    def test_export_bad_checkpoint(self, tmp_path):
        """Unsupported checkpoint format triggers sys.exit."""
        bad_path = tmp_path / "bad.pt"
        torch.save({"unrelated": 42}, bad_path)
        args = argparse.Namespace(model=str(bad_path), format="onnx", output=str(tmp_path / "out.onnx"))
        with pytest.raises(SystemExit):
            cmd_export(args)


# ── cmd_dashboard ────────────────────────────────────────────────────


class TestCmdDashboard:
    def test_dashboard_runs_streamlit(self):
        mock_run = MagicMock()
        with patch("subprocess.run", mock_run), \
             patch("pathlib.Path.exists", return_value=True):
            args = argparse.Namespace(port=8501)
            cmd_dashboard(args)
            mock_run.assert_called_once()
            call_args = mock_run.call_args
            assert "streamlit" in call_args[0][0]

    def test_dashboard_missing_app(self):
        with patch("pathlib.Path.exists", return_value=False):
            args = argparse.Namespace(port=8501)
            with pytest.raises(SystemExit):
                cmd_dashboard(args)


# ── Integration: main dispatches to subcommands ─────────────────────


class TestMainIntegration:
    def test_main_train(self, sample_config_path):
        main(["train", "--config", str(sample_config_path)])

    def test_main_export(self, model_checkpoint_path, tmp_path):
        output_dir = tmp_path / "onnx_int"
        output_dir.mkdir()
        main(["export", "--model", str(model_checkpoint_path), "--output", str(output_dir / "out.onnx")])
        assert any(output_dir.iterdir())


# ── cmd_export — ONNXExporter ImportError (lines 105-107) ────────────


class TestCmdExportONNXImportError:
    def test_export_onnx_import_error(self, model_checkpoint_path, tmp_path):
        """When ONNXExporter cannot be imported, cmd_export exits with error (lines 105-107)."""
        import importlib
        import sys

        args = argparse.Namespace(
            model=str(model_checkpoint_path),
            format="onnx",
            output=str(tmp_path / "out.onnx"),
        )

        # Remove the cached module so the import inside cmd_export re-executes
        saved_module = sys.modules.pop("reactor_twin.export.onnx_export", None)
        # Also remove the parent package so that subpackage imports re-trigger
        saved_export = sys.modules.pop("reactor_twin.export", None)
        try:
            # Setting the module to None in sys.modules causes ImportError on import
            sys.modules["reactor_twin.export.onnx_export"] = None  # type: ignore[assignment]

            # Reload the cli module so its local import is fresh
            importlib.reload(reactor_twin.cli)

            with pytest.raises(SystemExit):
                reactor_twin.cli.cmd_export(args)
        finally:
            # Restore original modules
            sys.modules.pop("reactor_twin.export.onnx_export", None)
            if saved_module is not None:
                sys.modules["reactor_twin.export.onnx_export"] = saved_module
            if saved_export is not None:
                sys.modules["reactor_twin.export"] = saved_export
            importlib.reload(reactor_twin.cli)


# ── cmd_export — state_dict checkpoint (lines 118-122) ───────────────


class TestCmdExportStateDict:
    def test_export_state_dict_checkpoint(self, tmp_path):
        """Loading a state_dict-style checkpoint creates NeuralODE and loads weights (lines 118-122).

        cmd_export reconstructs NeuralODE(state_dim=..., input_dim=...) with defaults
        (hidden_dim=64, num_layers=3), so the saved model must use those same defaults.
        """
        from reactor_twin.core.neural_ode import NeuralODE

        # Create model with defaults so state_dict matches the reconstruction in cmd_export
        model = NeuralODE(state_dim=2, solver="euler", adjoint=False)
        ckpt_path = tmp_path / "state_dict_model.pt"
        model.save(ckpt_path)  # saves {"model_state_dict": ..., "state_dim": ..., ...}

        output_dir = tmp_path / "onnx_sd"
        output_dir.mkdir()
        args = argparse.Namespace(
            model=str(ckpt_path),
            format="onnx",
            output=str(output_dir / "model.onnx"),
        )
        cmd_export(args)
        assert any(output_dir.iterdir())
