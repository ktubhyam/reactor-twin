"""AWS SageMaker inference module for ReactorTwin models.

Implements the four SageMaker inference contract functions plus a utility
for packaging model artifacts into a model.tar.gz for upload.

Usage (inside a SageMaker container)::

    from reactor_twin.deploy.sagemaker import model_fn, input_fn, predict_fn, output_fn

Packaging a model for upload::

    from reactor_twin.deploy.sagemaker import pack_model_tar
    tar_path = pack_model_tar(model, config, output_path="model.tar.gz")
"""

from __future__ import annotations

import io
import json
import logging
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn

from reactor_twin.utils.config import ReactorConfig

logger = logging.getLogger(__name__)

# ── SageMaker inference contract ─────────────────────────────────────


def model_fn(model_dir: str) -> dict[str, Any]:
    """Load model from SageMaker model directory.

    SageMaker calls this once at container startup to load artifacts.

    Args:
        model_dir: Directory containing model_weights.pt and config.yaml.

    Returns:
        Bundle with keys 'model' (nn.Module) and 'config' (ReactorConfig).
    """
    model_dir_path = Path(model_dir)
    weights_path = model_dir_path / "model_weights.pt"
    config_path = model_dir_path / "config.yaml"

    if not weights_path.exists():
        raise FileNotFoundError(f"model_weights.pt not found in {model_dir}")
    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml not found in {model_dir}")

    config_data = yaml.safe_load(config_path.read_text())
    config = ReactorConfig(**config_data)

    checkpoint = torch.load(str(weights_path), map_location="cpu", weights_only=False)
    logger.info(f"Loaded model from {weights_path}")

    return {"model": checkpoint, "config": config}


def input_fn(request_body: str | bytes, content_type: str) -> dict[str, Any]:
    """Deserialize an inference request.

    Args:
        request_body: Raw request body.
        content_type: MIME type of the request (must be application/json).

    Returns:
        Parsed input dict with keys 'z0', 't_span', and optionally 'controls'.

    Raises:
        ValueError: If content_type is not application/json or body is malformed.
    """
    if content_type != "application/json":
        raise ValueError(
            f"Unsupported content type: {content_type}. Use application/json."
        )

    if isinstance(request_body, bytes):
        request_body = request_body.decode("utf-8")

    data: dict[str, Any] = json.loads(request_body)

    if "z0" not in data or "t_span" not in data:
        raise ValueError("Request must contain 'z0' and 't_span' keys.")

    return data


def predict_fn(
    input_data: dict[str, Any],
    model: dict[str, Any],
) -> dict[str, Any]:
    """Run reactor trajectory prediction.

    Args:
        input_data: Parsed input from input_fn. Must have 'z0' and 't_span'.
        model: Model bundle from model_fn. Must have 'model' key with callable model.

    Returns:
        Dict with 'trajectory' (list of lists) and 'success' (bool).
    """
    callable_model: nn.Module = model["model"]

    if not callable(callable_model):
        raise ValueError("model bundle 'model' key must be a callable nn.Module")

    z0 = torch.tensor(input_data["z0"], dtype=torch.float32)
    t_span = torch.tensor(input_data["t_span"], dtype=torch.float32)

    if z0.dim() == 1:
        z0 = z0.unsqueeze(0)

    controls = None
    if "controls" in input_data and input_data["controls"] is not None:
        controls = torch.tensor(input_data["controls"], dtype=torch.float32)
        if controls.dim() == 2:
            controls = controls.unsqueeze(0)

    callable_model.eval()
    with torch.no_grad():
        result = callable_model(z0=z0, t_span=t_span, controls=controls)

    trajectory: list[list[float]] = result.squeeze(0).tolist()
    return {"trajectory": trajectory, "success": True}


def output_fn(
    prediction: dict[str, Any],
    accept: str,
) -> tuple[str, str]:
    """Serialize prediction for the response.

    Args:
        prediction: Output from predict_fn.
        accept: Desired response MIME type (must be application/json).

    Returns:
        Tuple of (serialized_body, content_type).

    Raises:
        ValueError: If accept type is not supported.
    """
    if accept not in ("application/json", "*/*"):
        raise ValueError(f"Unsupported accept type: {accept}. Use application/json.")

    return json.dumps(prediction), "application/json"


# ── Model packaging ───────────────────────────────────────────────────

_INFERENCE_SCRIPT = '''\
"""SageMaker inference entry point for ReactorTwin."""
from reactor_twin.deploy.sagemaker import model_fn, input_fn, predict_fn, output_fn

__all__ = ["model_fn", "input_fn", "predict_fn", "output_fn"]
'''


def pack_model_tar(
    model: nn.Module,
    config: ReactorConfig,
    output_path: str | Path,
) -> Path:
    """Bundle model weights and config into a model.tar.gz for SageMaker.

    The archive structure matches the SageMaker model artifact convention::

        model.tar.gz
        ├── model_weights.pt
        ├── config.yaml
        └── code/
            └── inference.py

    Args:
        model: PyTorch model to save.
        config: ReactorConfig instance to serialise as YAML.
        output_path: Destination path for the .tar.gz file.

    Returns:
        Path to the created archive.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # Save model weights
        weights_path = tmp / "model_weights.pt"
        torch.save(model, str(weights_path))

        # Save config
        config_path = tmp / "config.yaml"
        config_path.write_text(yaml.dump(config.model_dump()))

        # Write inference entry point
        code_dir = tmp / "code"
        code_dir.mkdir()
        (code_dir / "inference.py").write_text(_INFERENCE_SCRIPT)

        # Pack into tar.gz
        with tarfile.open(str(output_path), "w:gz") as tar:
            tar.add(str(weights_path), arcname="model_weights.pt")
            tar.add(str(config_path), arcname="config.yaml")
            tar.add(str(code_dir), arcname="code")

    logger.info(f"Model archive written to {output_path}")
    return output_path


__all__ = [
    "input_fn",
    "model_fn",
    "output_fn",
    "pack_model_tar",
    "predict_fn",
]
