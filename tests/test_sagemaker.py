"""Tests for the AWS SageMaker inference module."""

from __future__ import annotations

import json
import tarfile
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from reactor_twin.deploy.sagemaker import (
    input_fn,
    model_fn,
    output_fn,
    pack_model_tar,
    predict_fn,
)
from reactor_twin.utils.config import ReactorConfig


class _FakeModel(nn.Module):
    """Module-level fake model so torch.save can pickle it."""

    def forward(
        self,
        z0: torch.Tensor,
        t_span: torch.Tensor,
        controls: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch = z0.shape[0]
        n_times = t_span.shape[0]
        state_dim = z0.shape[1]
        return torch.zeros(batch, n_times, state_dim)


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def reactor_config() -> ReactorConfig:
    return ReactorConfig(
        reactor_type="cstr",
        volume=1.0,
        kinetics="simple_ab",
        n_species=2,
    )


@pytest.fixture
def dummy_model() -> nn.Module:
    return _FakeModel()


@pytest.fixture
def model_dir(tmp_path: Path, dummy_model: nn.Module, reactor_config: ReactorConfig) -> Path:
    """Create a temporary model directory with weights and config."""
    import yaml

    torch.save(dummy_model, str(tmp_path / "model_weights.pt"))
    (tmp_path / "config.yaml").write_text(yaml.dump(reactor_config.model_dump()))
    return tmp_path


# ── model_fn ────────────────────────────────────────────────────────


class TestModelFn:
    def test_returns_bundle_keys(self, model_dir: Path) -> None:
        bundle = model_fn(str(model_dir))
        assert "model" in bundle
        assert "config" in bundle

    def test_config_is_reactor_config(self, model_dir: Path) -> None:
        bundle = model_fn(str(model_dir))
        assert isinstance(bundle["config"], ReactorConfig)

    def test_model_loaded(self, model_dir: Path) -> None:
        bundle = model_fn(str(model_dir))
        assert bundle["model"] is not None

    def test_raises_if_weights_missing(self, tmp_path: Path, reactor_config: ReactorConfig) -> None:
        import yaml

        (tmp_path / "config.yaml").write_text(yaml.dump(reactor_config.model_dump()))
        with pytest.raises(FileNotFoundError, match="model_weights.pt"):
            model_fn(str(tmp_path))

    def test_raises_if_config_missing(self, tmp_path: Path, dummy_model: nn.Module) -> None:
        torch.save(dummy_model, str(tmp_path / "model_weights.pt"))
        with pytest.raises(FileNotFoundError, match="config.yaml"):
            model_fn(str(tmp_path))


# ── input_fn ────────────────────────────────────────────────────────


class TestInputFn:
    def test_parses_json_bytes(self) -> None:
        body = json.dumps({"z0": [1.0, 2.0], "t_span": [0.0, 1.0]}).encode()
        data = input_fn(body, "application/json")
        assert data["z0"] == [1.0, 2.0]
        assert data["t_span"] == [0.0, 1.0]

    def test_parses_json_string(self) -> None:
        body = json.dumps({"z0": [0.5], "t_span": [0.0, 0.5, 1.0]})
        data = input_fn(body, "application/json")
        assert len(data["t_span"]) == 3

    def test_includes_controls_if_present(self) -> None:
        body = json.dumps({"z0": [1.0], "t_span": [0.0, 1.0], "controls": [[0.1], [0.2]]})
        data = input_fn(body, "application/json")
        assert "controls" in data

    def test_raises_on_unsupported_content_type(self) -> None:
        with pytest.raises(ValueError, match="Unsupported content type"):
            input_fn(b"data", "text/plain")

    def test_raises_if_z0_missing(self) -> None:
        body = json.dumps({"t_span": [0.0, 1.0]})
        with pytest.raises(ValueError, match="z0"):
            input_fn(body, "application/json")

    def test_raises_if_t_span_missing(self) -> None:
        body = json.dumps({"z0": [1.0]})
        with pytest.raises(ValueError, match="t_span"):
            input_fn(body, "application/json")


# ── predict_fn ──────────────────────────────────────────────────────


class TestPredictFn:
    def test_returns_trajectory(self, dummy_model: nn.Module) -> None:
        bundle = {"model": dummy_model, "config": None}
        input_data = {"z0": [[1.0, 2.0]], "t_span": [0.0, 0.5, 1.0]}
        result = predict_fn(input_data, bundle)
        assert "trajectory" in result
        assert result["success"] is True

    def test_trajectory_is_nested_list(self, dummy_model: nn.Module) -> None:
        bundle = {"model": dummy_model, "config": None}
        input_data = {"z0": [[1.0, 2.0]], "t_span": [0.0, 0.5, 1.0]}
        result = predict_fn(input_data, bundle)
        traj = result["trajectory"]
        assert isinstance(traj, list)
        assert isinstance(traj[0], list)

    def test_auto_unsqueeze_z0(self, dummy_model: nn.Module) -> None:
        bundle = {"model": dummy_model, "config": None}
        input_data = {"z0": [1.0, 2.0], "t_span": [0.0, 1.0]}
        result = predict_fn(input_data, bundle)
        assert result["success"] is True

    def test_raises_if_model_not_callable(self) -> None:
        bundle = {"model": "not_a_model", "config": None}
        input_data = {"z0": [[1.0]], "t_span": [0.0, 1.0]}
        with pytest.raises(ValueError, match="callable"):
            predict_fn(input_data, bundle)


# ── output_fn ───────────────────────────────────────────────────────


class TestOutputFn:
    def test_returns_json_string(self) -> None:
        prediction = {"trajectory": [[0.0, 0.0]], "success": True}
        body, content_type = output_fn(prediction, "application/json")
        assert content_type == "application/json"
        assert json.loads(body) == prediction

    def test_accepts_wildcard(self) -> None:
        prediction = {"trajectory": [], "success": True}
        body, content_type = output_fn(prediction, "*/*")
        assert content_type == "application/json"

    def test_raises_on_unsupported_accept(self) -> None:
        with pytest.raises(ValueError, match="Unsupported accept type"):
            output_fn({}, "text/xml")


# ── pack_model_tar ──────────────────────────────────────────────────


class TestPackModelTar:
    def test_creates_tar_gz(
        self, tmp_path: Path, dummy_model: nn.Module, reactor_config: ReactorConfig
    ) -> None:
        out = tmp_path / "model.tar.gz"
        result = pack_model_tar(dummy_model, reactor_config, out)
        assert result == out
        assert out.exists()

    def test_tar_contains_weights(
        self, tmp_path: Path, dummy_model: nn.Module, reactor_config: ReactorConfig
    ) -> None:
        out = tmp_path / "model.tar.gz"
        pack_model_tar(dummy_model, reactor_config, out)
        with tarfile.open(str(out), "r:gz") as tar:
            names = tar.getnames()
        assert "model_weights.pt" in names

    def test_tar_contains_config(
        self, tmp_path: Path, dummy_model: nn.Module, reactor_config: ReactorConfig
    ) -> None:
        out = tmp_path / "model.tar.gz"
        pack_model_tar(dummy_model, reactor_config, out)
        with tarfile.open(str(out), "r:gz") as tar:
            names = tar.getnames()
        assert "config.yaml" in names

    def test_tar_contains_inference_script(
        self, tmp_path: Path, dummy_model: nn.Module, reactor_config: ReactorConfig
    ) -> None:
        out = tmp_path / "model.tar.gz"
        pack_model_tar(dummy_model, reactor_config, out)
        with tarfile.open(str(out), "r:gz") as tar:
            names = tar.getnames()
        assert any("inference.py" in n for n in names)

    def test_creates_parent_dirs(
        self, tmp_path: Path, dummy_model: nn.Module, reactor_config: ReactorConfig
    ) -> None:
        out = tmp_path / "nested" / "dir" / "model.tar.gz"
        pack_model_tar(dummy_model, reactor_config, out)
        assert out.exists()
