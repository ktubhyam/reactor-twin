"""Tests for configuration system."""

from __future__ import annotations

import textwrap

import pytest

from reactor_twin.exceptions import ConfigurationError
from reactor_twin.utils.config import (
    DigitalTwinConfig,
    ExperimentConfig,
    LoggingConfig,
    NeuralDEConfig,
    ReactorConfig,
    TrainingConfig,
    load_config,
    merge_configs,
    save_config,
)

# ── helpers ──────────────────────────────────────────────────────────

def _sample_config() -> ExperimentConfig:
    return ExperimentConfig(
        reactor=ReactorConfig(
            reactor_type="cstr",
            volume=10.0,
            temperature=350.0,
            kinetics="arrhenius",
            n_species=2,
        ),
        neural_de=NeuralDEConfig(
            model_type="neural_ode",
            hidden_dims=[64, 64],
        ),
        training=TrainingConfig(
            batch_size=32,
            n_epochs=100,
            learning_rate=1e-3,
        ),
        seed=42,
    )


def _write_yaml(tmp_path, content: str) -> str:
    p = tmp_path / "config.yaml"
    p.write_text(textwrap.dedent(content))
    return str(p)


# ── Pydantic models ─────────────────────────────────────────────────

class TestPydanticModels:
    def test_reactor_config(self):
        cfg = ReactorConfig(
            reactor_type="cstr", volume=10.0, kinetics="arrhenius", n_species=2
        )
        assert cfg.reactor_type == "cstr"
        assert cfg.temperature is None

    def test_neural_de_config_defaults(self):
        cfg = NeuralDEConfig(model_type="neural_ode", hidden_dims=[64])
        assert cfg.activation == "relu"
        assert cfg.solver == "dopri5"

    def test_training_config_defaults(self):
        cfg = TrainingConfig()
        assert cfg.batch_size == 32
        assert cfg.n_epochs == 1000

    def test_experiment_config(self):
        cfg = _sample_config()
        assert cfg.seed == 42
        assert cfg.reactor.reactor_type == "cstr"

    def test_digital_twin_config(self):
        cfg = DigitalTwinConfig()
        assert cfg.fault_detector is False
        assert cfg.update_interval == 1.0

    def test_logging_config(self):
        cfg = LoggingConfig()
        assert cfg.level == "INFO"
        assert cfg.format == "text"

    def test_logging_config_custom(self):
        cfg = LoggingConfig(
            level="DEBUG",
            format="json",
            log_file="/tmp/test.log",
            module_levels={"reactor_twin.core": "DEBUG"},
        )
        assert cfg.log_file == "/tmp/test.log"


# ── save_config / load_config round-trip ─────────────────────────────

class TestSaveLoadConfig:
    def test_save_and_load_roundtrip(self, tmp_path):
        cfg = _sample_config()
        path = tmp_path / "test_config.yaml"
        save_config(cfg, path)
        loaded = load_config(path)
        assert loaded.reactor.reactor_type == cfg.reactor.reactor_type
        assert loaded.reactor.volume == cfg.reactor.volume
        assert loaded.neural_de.model_type == cfg.neural_de.model_type
        assert loaded.training.batch_size == cfg.training.batch_size
        assert loaded.seed == cfg.seed

    def test_save_creates_parent_dirs(self, tmp_path):
        cfg = _sample_config()
        path = tmp_path / "sub" / "dir" / "config.yaml"
        save_config(cfg, path)
        assert path.exists()

    def test_load_missing_file_raises(self):
        with pytest.raises(ConfigurationError, match="not found"):
            load_config("/nonexistent/path/config.yaml")

    def test_load_invalid_yaml_raises(self, tmp_path):
        path = _write_yaml(tmp_path, "{{invalid yaml: [")
        with pytest.raises(ConfigurationError, match="Invalid YAML"):
            load_config(path)

    def test_load_invalid_structure_raises(self, tmp_path):
        path = _write_yaml(tmp_path, """
            reactor:
              reactor_type: cstr
            # Missing required fields
        """)
        with pytest.raises(ConfigurationError, match="Invalid config"):
            load_config(path)


# ── Environment variable interpolation ───────────────────────────────

class TestEnvInterpolation:
    def test_env_var_substitution(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RT_VOLUME", "25.0")
        content = """
            reactor:
              reactor_type: cstr
              volume: ${RT_VOLUME}
              kinetics: arrhenius
              n_species: 2
            neural_de:
              model_type: neural_ode
              hidden_dims: [64]
            training:
              batch_size: 32
        """
        path = _write_yaml(tmp_path, content)
        cfg = load_config(path)
        assert cfg.reactor.volume == 25.0

    def test_env_var_with_default(self, tmp_path, monkeypatch):
        monkeypatch.delenv("RT_MISSING_VAR", raising=False)
        content = """
            reactor:
              reactor_type: cstr
              volume: ${RT_MISSING_VAR:10.0}
              kinetics: arrhenius
              n_species: 2
            neural_de:
              model_type: neural_ode
              hidden_dims: [64]
            training:
              batch_size: 32
        """
        path = _write_yaml(tmp_path, content)
        cfg = load_config(path)
        assert cfg.reactor.volume == 10.0

    def test_env_var_override_default(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RT_VOL", "99.0")
        content = """
            reactor:
              reactor_type: cstr
              volume: ${RT_VOL:10.0}
              kinetics: arrhenius
              n_species: 2
            neural_de:
              model_type: neural_ode
              hidden_dims: [64]
            training:
              batch_size: 32
        """
        path = _write_yaml(tmp_path, content)
        cfg = load_config(path)
        assert cfg.reactor.volume == 99.0


# ── merge_configs ────────────────────────────────────────────────────

class TestMergeConfigs:
    def test_simple_merge(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = merge_configs(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_deep_merge(self):
        base = {"a": {"x": 1, "y": 2}, "b": 3}
        override = {"a": {"y": 99, "z": 100}}
        result = merge_configs(base, override)
        assert result == {"a": {"x": 1, "y": 99, "z": 100}, "b": 3}

    def test_override_replaces_non_dict(self):
        base = {"a": [1, 2, 3]}
        override = {"a": [4, 5]}
        result = merge_configs(base, override)
        assert result == {"a": [4, 5]}

    def test_empty_override(self):
        base = {"a": 1, "b": 2}
        result = merge_configs(base, {})
        assert result == base

    def test_empty_base(self):
        override = {"a": 1}
        result = merge_configs({}, override)
        assert result == {"a": 1}

    def test_base_unchanged(self):
        base = {"a": {"x": 1}}
        override = {"a": {"y": 2}}
        merge_configs(base, override)
        # Original base should not be mutated
        assert "y" not in base["a"]

    def test_nested_three_levels(self):
        base = {"a": {"b": {"c": 1, "d": 2}}}
        override = {"a": {"b": {"d": 99}}}
        result = merge_configs(base, override)
        assert result["a"]["b"]["c"] == 1
        assert result["a"]["b"]["d"] == 99
