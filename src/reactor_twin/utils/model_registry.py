"""Local model versioning and registry for trained Neural DEs.

Tracks trained models with metadata (reactor type, training config,
performance metrics) in a simple JSON-based registry.  Supports
optional W&B / MLflow adapters for remote tracking.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

_DEFAULT_REGISTRY_DIR = Path.home() / ".reactor_twin" / "models"


@dataclass
class ModelMetadata:
    """Metadata for a registered model.

    Attributes:
        name: Human-readable model name.
        version: Semantic version string (e.g. "1.0.0").
        model_class: Fully qualified class name of the model.
        reactor_type: Reactor type used for training.
        state_dim: State dimension.
        input_dim: Control input dimension.
        training_config: Dict of hyper-parameters (epochs, lr, etc.).
        metrics: Dict of evaluation metrics (RMSE, coverage, etc.).
        tags: Free-form tags for filtering.
        created_at: Unix timestamp of registration.
        checkpoint_path: Path to the saved checkpoint file.
    """

    name: str
    version: str = "1.0.0"
    model_class: str = ""
    reactor_type: str = ""
    state_dim: int = 0
    input_dim: int = 0
    training_config: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    created_at: float = 0.0
    checkpoint_path: str = ""


class ModelRegistry:
    """File-based model registry.

    Models are stored as checkpoint files under ``registry_dir/<name>/<version>/``,
    with a JSON manifest file tracking all metadata.

    Example::

        registry = ModelRegistry()
        registry.save_model(model, name="cstr_node", metrics={"rmse": 0.03})
        loaded = registry.load_model("cstr_node", version="1.0.0")
        comparison = registry.compare_models("cstr_node")
    """

    def __init__(self, registry_dir: str | Path | None = None) -> None:
        """Initialize the registry.

        Args:
            registry_dir: Root directory for model storage.
                Defaults to ``~/.reactor_twin/models``.
        """
        self.registry_dir = Path(registry_dir) if registry_dir else _DEFAULT_REGISTRY_DIR
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self.registry_dir / "manifest.json"
        self._manifest = self._load_manifest()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_manifest(self) -> dict[str, list[dict[str, Any]]]:
        """Load or create the manifest file."""
        if self._manifest_path.exists():
            with open(self._manifest_path) as f:
                return json.load(f)
        return {}

    def _save_manifest(self) -> None:
        """Persist manifest to disk."""
        with open(self._manifest_path, "w") as f:
            json.dump(self._manifest, f, indent=2, default=str)

    def _next_version(self, name: str) -> str:
        """Auto-increment version for a model name."""
        entries = self._manifest.get(name, [])
        if not entries:
            return "1.0.0"
        versions = [e["version"] for e in entries]
        # Parse major.minor.patch and bump patch
        latest = sorted(versions, key=lambda v: list(map(int, v.split("."))))[-1]
        parts = list(map(int, latest.split(".")))
        parts[2] += 1
        return ".".join(map(str, parts))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save_model(
        self,
        model: torch.nn.Module,
        name: str,
        version: str | None = None,
        reactor_type: str = "",
        training_config: dict[str, Any] | None = None,
        metrics: dict[str, float] | None = None,
        tags: list[str] | None = None,
    ) -> ModelMetadata:
        """Save a model to the registry.

        Args:
            model: PyTorch model to save.
            name: Model name.
            version: Explicit version (auto-incremented if None).
            reactor_type: Reactor type used for training.
            training_config: Training hyper-parameters.
            metrics: Evaluation metrics.
            tags: Free-form tags.

        Returns:
            The registered model metadata.
        """
        if version is None:
            version = self._next_version(name)

        model_dir = self.registry_dir / name / version
        model_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = model_dir / "model.pt"

        torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)

        state_dim = getattr(model, "state_dim", 0)
        input_dim = getattr(model, "input_dim", 0)

        meta = ModelMetadata(
            name=name,
            version=version,
            model_class=f"{model.__class__.__module__}.{model.__class__.__name__}",
            reactor_type=reactor_type,
            state_dim=state_dim,
            input_dim=input_dim,
            training_config=training_config or {},
            metrics=metrics or {},
            tags=tags or [],
            created_at=time.time(),
            checkpoint_path=str(checkpoint_path),
        )

        if name not in self._manifest:
            self._manifest[name] = []
        self._manifest[name].append(asdict(meta))
        self._save_manifest()

        logger.info(f"Registered model '{name}' v{version} at {checkpoint_path}")
        return meta

    def load_model(
        self,
        name: str,
        version: str | None = None,
        model: torch.nn.Module | None = None,
        device: str | torch.device = "cpu",
    ) -> dict[str, Any]:
        """Load a model from the registry.

        Args:
            name: Model name.
            version: Specific version (latest if None).
            model: Pre-constructed model to load weights into.
                If None, returns the raw state_dict.
            device: Device to load onto.

        Returns:
            Dict with ``state_dict``, ``metadata``, and optionally
            the populated ``model``.
        """
        entries = self._manifest.get(name, [])
        if not entries:
            raise KeyError(f"No model registered with name '{name}'")

        if version is None:
            entry = entries[-1]  # latest
        else:
            matching = [e for e in entries if e["version"] == version]
            if not matching:
                available = [e["version"] for e in entries]
                raise KeyError(
                    f"Version '{version}' not found for '{name}'. Available: {available}"
                )
            entry = matching[-1]

        checkpoint = torch.load(
            entry["checkpoint_path"], map_location=device, weights_only=False
        )

        result: dict[str, Any] = {
            "state_dict": checkpoint["model_state_dict"],
            "metadata": ModelMetadata(**{k: v for k, v in entry.items()}),
        }

        if model is not None:
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(device)
            result["model"] = model

        logger.info(f"Loaded model '{name}' v{entry['version']}")
        return result

    def list_models(
        self,
        name: str | None = None,
        tag: str | None = None,
    ) -> list[ModelMetadata]:
        """List registered models.

        Args:
            name: Filter by model name (all if None).
            tag: Filter by tag.

        Returns:
            List of matching model metadata.
        """
        results: list[ModelMetadata] = []
        for model_name, entries in self._manifest.items():
            if name is not None and model_name != name:
                continue
            for entry in entries:
                meta = ModelMetadata(**{k: v for k, v in entry.items()})
                if tag is not None and tag not in meta.tags:
                    continue
                results.append(meta)
        return results

    def compare_models(
        self,
        name: str,
        metric_keys: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Compare all versions of a model by metrics.

        Args:
            name: Model name.
            metric_keys: Specific metric keys to compare.
                All available metrics if None.

        Returns:
            List of dicts with ``version`` and metric values, sorted
            by version.
        """
        entries = self._manifest.get(name, [])
        if not entries:
            raise KeyError(f"No model registered with name '{name}'")

        comparisons = []
        for entry in entries:
            row: dict[str, Any] = {"version": entry["version"]}
            metrics = entry.get("metrics", {})
            keys = metric_keys if metric_keys else list(metrics.keys())
            for k in keys:
                row[k] = metrics.get(k)
            comparisons.append(row)

        return sorted(comparisons, key=lambda x: x["version"])

    def delete_model(
        self,
        name: str,
        version: str | None = None,
    ) -> None:
        """Delete a model (or specific version) from the registry.

        Args:
            name: Model name.
            version: Specific version to delete.  Deletes all
                versions if None.
        """
        if name not in self._manifest:
            raise KeyError(f"No model registered with name '{name}'")

        if version is None:
            # Delete entire model
            model_dir = self.registry_dir / name
            if model_dir.exists():
                shutil.rmtree(model_dir)
            del self._manifest[name]
            logger.info(f"Deleted all versions of model '{name}'")
        else:
            entries = self._manifest[name]
            remaining = [e for e in entries if e["version"] != version]
            if len(remaining) == len(entries):
                raise KeyError(f"Version '{version}' not found for '{name}'")

            version_dir = self.registry_dir / name / version
            if version_dir.exists():
                shutil.rmtree(version_dir)

            if remaining:
                self._manifest[name] = remaining
            else:
                del self._manifest[name]
            logger.info(f"Deleted model '{name}' v{version}")

        self._save_manifest()


__all__ = [
    "ModelMetadata",
    "ModelRegistry",
]
