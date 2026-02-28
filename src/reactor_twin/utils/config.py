"""Configuration management for ReactorTwin."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, cast

import yaml
from pydantic import BaseModel, Field

from reactor_twin.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


class ReactorConfig(BaseModel):
    """Configuration for reactor parameters."""

    reactor_type: str = Field(..., description="Reactor type (e.g., 'cstr', 'batch')")
    volume: float = Field(..., description="Reactor volume (m^3)")
    temperature: float | None = Field(None, description="Operating temperature (K)")
    pressure: float | None = Field(None, description="Operating pressure (Pa)")
    kinetics: str = Field(..., description="Kinetics model name")
    n_species: int = Field(..., description="Number of chemical species")


class NeuralDEConfig(BaseModel):
    """Configuration for Neural DE model."""

    model_type: str = Field(..., description="Neural DE variant (e.g., 'neural_ode')")
    hidden_dims: list[int] = Field(..., description="Hidden layer dimensions")
    activation: str = Field(default="relu", description="Activation function")
    solver: str = Field(default="dopri5", description="ODE solver")
    atol: float = Field(default=1e-6, description="Absolute tolerance")
    rtol: float = Field(default=1e-5, description="Relative tolerance")


class TrainingConfig(BaseModel):
    """Configuration for training loop."""

    batch_size: int = Field(default=32, description="Training batch size")
    n_epochs: int = Field(default=1000, description="Number of epochs")
    learning_rate: float = Field(default=1e-3, description="Learning rate")
    optimizer: str = Field(default="adam", description="Optimizer name")
    scheduler: str | None = Field(None, description="Learning rate scheduler")
    loss_weights: dict[str, float] = Field(
        default_factory=dict,
        description="Weights for multi-objective loss",
    )


class ExperimentConfig(BaseModel):
    """Top-level experiment configuration."""

    reactor: ReactorConfig
    neural_de: NeuralDEConfig
    training: TrainingConfig
    constraints: list[str] = Field(default_factory=list, description="Constraint names")
    seed: int = Field(default=42, description="Random seed")


class DigitalTwinConfig(BaseModel):
    """Configuration for digital twin components."""

    state_estimator: str | None = Field(None, description="State estimator type")
    fault_detector: bool = Field(default=False, description="Enable fault detection")
    controller: str | None = Field(None, description="Controller type")
    update_interval: float = Field(default=1.0, description="Update interval (s)")


class LoggingConfig(BaseModel):
    """Configuration for structured logging."""

    level: str = Field(default="INFO", description="Log level")
    format: str = Field(default="text", description="Log format ('text' or 'json')")
    log_file: str | None = Field(None, description="Log file path")
    module_levels: dict[str, str] = Field(
        default_factory=dict,
        description="Per-module log levels",
    )


def _interpolate_env_vars(text: str) -> str:
    """Replace ${VAR:default} patterns with environment variable values."""

    def _replace(match: re.Match[str]) -> str:
        var_name = match.group(1)
        default = match.group(3)  # group 3 is after the optional colon
        value = os.environ.get(var_name)
        if value is not None:
            return value
        if default is not None:
            return cast(str, default)
        return match.group(0)  # Leave unchanged if no env var and no default

    return re.sub(r"\$\{(\w+)(:([^}]*))?\}", _replace, text)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load configuration from YAML file with env var interpolation.

    Supports ``${VAR}`` and ``${VAR:default}`` syntax for environment
    variable substitution in string values.

    Args:
        path: Path to YAML config file.

    Returns:
        Parsed configuration object.

    Raises:
        ConfigurationError: If the file cannot be read or parsed.
    """
    path = Path(path)
    if not path.exists():
        raise ConfigurationError(f"Config file not found: {path}")

    try:
        raw_text = path.read_text()
        interpolated = _interpolate_env_vars(raw_text)
        data = yaml.safe_load(interpolated)
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Invalid YAML in {path}: {exc}") from exc

    try:
        config = ExperimentConfig(**data)
    except Exception as exc:
        raise ConfigurationError(f"Invalid config structure: {exc}") from exc

    logger.info(f"Loaded config from {path}")
    return config


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    """Save configuration to YAML file.

    Args:
        config: Configuration object.
        path: Output path.

    Raises:
        ConfigurationError: If the file cannot be written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        data = config.model_dump()
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    except Exception as exc:
        raise ConfigurationError(f"Failed to save config to {path}: {exc}") from exc

    logger.info(f"Saved config to {path}")


def merge_configs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two configuration dictionaries.

    For nested dicts, recursively merges rather than replacing.
    For all other types, the override value wins.

    Args:
        base: Base configuration.
        override: Override values (takes precedence).

    Returns:
        New merged configuration dictionary.
    """
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    return merged
