"""Tests for reactor_twin.utils.model_registry."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from reactor_twin.utils.model_registry import ModelRegistry


@pytest.fixture
def tmp_registry(tmp_path):
    return ModelRegistry(registry_dir=tmp_path / "models")


@pytest.fixture
def dummy_model():
    return nn.Linear(3, 2)


class TestModelRegistry:
    def test_save_model(self, tmp_registry, dummy_model):
        meta = tmp_registry.save_model(dummy_model, name="test_model")
        assert meta.name == "test_model"
        assert meta.version == "1.0.0"
        assert meta.checkpoint_path != ""

    def test_auto_version_increment(self, tmp_registry, dummy_model):
        m1 = tmp_registry.save_model(dummy_model, name="m")
        m2 = tmp_registry.save_model(dummy_model, name="m")
        assert m1.version == "1.0.0"
        assert m2.version == "1.0.1"

    def test_explicit_version(self, tmp_registry, dummy_model):
        meta = tmp_registry.save_model(dummy_model, name="m", version="2.0.0")
        assert meta.version == "2.0.0"

    def test_load_model_latest(self, tmp_registry, dummy_model):
        tmp_registry.save_model(dummy_model, name="m", version="1.0.0")
        tmp_registry.save_model(dummy_model, name="m", version="1.0.1")
        result = tmp_registry.load_model("m")
        assert result["metadata"].version == "1.0.1"

    def test_load_model_specific_version(self, tmp_registry, dummy_model):
        tmp_registry.save_model(dummy_model, name="m", version="1.0.0")
        tmp_registry.save_model(dummy_model, name="m", version="2.0.0")
        result = tmp_registry.load_model("m", version="1.0.0")
        assert result["metadata"].version == "1.0.0"

    def test_load_into_model(self, tmp_registry, dummy_model):
        tmp_registry.save_model(dummy_model, name="m")
        fresh_model = nn.Linear(3, 2)
        result = tmp_registry.load_model("m", model=fresh_model)
        assert "model" in result
        # Weights should match
        orig_w = dummy_model.weight.data
        loaded_w = result["model"].weight.data
        torch.testing.assert_close(orig_w, loaded_w)

    def test_load_nonexistent_raises(self, tmp_registry):
        with pytest.raises(KeyError, match="No model registered"):
            tmp_registry.load_model("nonexistent")

    def test_load_wrong_version_raises(self, tmp_registry, dummy_model):
        tmp_registry.save_model(dummy_model, name="m")
        with pytest.raises(KeyError, match="Version"):
            tmp_registry.load_model("m", version="9.9.9")

    def test_list_models(self, tmp_registry, dummy_model):
        tmp_registry.save_model(dummy_model, name="a")
        tmp_registry.save_model(dummy_model, name="b")
        all_models = tmp_registry.list_models()
        assert len(all_models) == 2

    def test_list_by_name(self, tmp_registry, dummy_model):
        tmp_registry.save_model(dummy_model, name="a")
        tmp_registry.save_model(dummy_model, name="b")
        a_models = tmp_registry.list_models(name="a")
        assert len(a_models) == 1
        assert a_models[0].name == "a"

    def test_list_by_tag(self, tmp_registry, dummy_model):
        tmp_registry.save_model(dummy_model, name="m", tags=["cstr", "neural-ode"])
        tmp_registry.save_model(dummy_model, name="m2", tags=["pfr"])
        cstr = tmp_registry.list_models(tag="cstr")
        assert len(cstr) == 1

    def test_compare_models(self, tmp_registry, dummy_model):
        tmp_registry.save_model(dummy_model, name="m", version="1.0.0", metrics={"rmse": 0.1})
        tmp_registry.save_model(dummy_model, name="m", version="1.0.1", metrics={"rmse": 0.05})
        comparison = tmp_registry.compare_models("m")
        assert len(comparison) == 2
        assert comparison[0]["rmse"] == 0.1
        assert comparison[1]["rmse"] == 0.05

    def test_delete_specific_version(self, tmp_registry, dummy_model):
        tmp_registry.save_model(dummy_model, name="m", version="1.0.0")
        tmp_registry.save_model(dummy_model, name="m", version="1.0.1")
        tmp_registry.delete_model("m", version="1.0.0")
        remaining = tmp_registry.list_models(name="m")
        assert len(remaining) == 1
        assert remaining[0].version == "1.0.1"

    def test_delete_all_versions(self, tmp_registry, dummy_model):
        tmp_registry.save_model(dummy_model, name="m")
        tmp_registry.delete_model("m")
        assert len(tmp_registry.list_models(name="m")) == 0

    def test_delete_nonexistent_raises(self, tmp_registry):
        with pytest.raises(KeyError):
            tmp_registry.delete_model("nonexistent")

    def test_metrics_saved(self, tmp_registry, dummy_model):
        tmp_registry.save_model(
            dummy_model, name="m",
            metrics={"rmse": 0.03, "coverage": 95.0},
        )
        result = tmp_registry.load_model("m")
        assert result["metadata"].metrics["rmse"] == 0.03

    def test_training_config_saved(self, tmp_registry, dummy_model):
        tmp_registry.save_model(
            dummy_model, name="m",
            training_config={"lr": 1e-3, "epochs": 100},
        )
        result = tmp_registry.load_model("m")
        assert result["metadata"].training_config["lr"] == 1e-3
