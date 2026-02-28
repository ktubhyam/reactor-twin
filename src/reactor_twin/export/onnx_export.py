"""ONNX export, validation, and inference for Neural DE models."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from reactor_twin.exceptions import ExportError

logger = logging.getLogger(__name__)


def _onnx_export_kwargs() -> dict[str, Any]:
    """Return extra kwargs for torch.onnx.export to force legacy exporter on PyTorch 2.6+."""
    major, minor = (int(x) for x in torch.__version__.split(".")[:2])
    if major > 2 or (major == 2 and minor >= 6):
        return {"dynamo": False}
    return {}


class _ODEFuncWrapper(nn.Module):
    """Wraps an ODE function for ONNX tracing: forward(t, z) -> dz_dt."""

    def __init__(self, ode_func: nn.Module):
        super().__init__()
        self.ode_func = ode_func

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.ode_func(t, z))


def _export_sde_drift(
    model: nn.Module,
    output_dir: Path,
    opset_version: int,
    validate: bool,
) -> dict[str, Path]:
    """Export NeuralSDE drift function only (deterministic approximation).

    The diffusion term cannot be traced to ONNX. This exports a deterministic
    mean-field approximation suitable for deployment where uncertainty is not needed.

    Args:
        model: NeuralSDE instance.
        output_dir: Directory to write ONNX files.
        opset_version: ONNX opset version.
        validate: If True, validate the exported model.

    Returns:
        Dict mapping 'drift_fn' to the exported ONNX path.
    """
    import onnx

    logger.warning(
        "NeuralSDE: exporting drift function only. "
        "Diffusion (stochastic) term cannot be traced to ONNX. "
        "This is a deterministic approximation."
    )

    drift_func = model.sde_func.drift_func
    state_dim = drift_func.state_dim
    wrapper = _ODEFuncWrapper(drift_func)
    wrapper.eval()

    drift_path = output_dir / "drift_fn.onnx"
    dummy_t = torch.tensor(0.0)
    dummy_z = torch.randn(1, state_dim)

    try:
        torch.onnx.export(
            wrapper,
            (dummy_t, dummy_z),
            str(drift_path),
            opset_version=opset_version,
            input_names=["t", "z"],
            output_names=["dz_dt"],
            dynamic_axes={"z": {0: "batch"}, "dz_dt": {0: "batch"}},
            **_onnx_export_kwargs(),
        )
    except Exception as exc:
        raise ExportError(f"Failed to export NeuralSDE drift_fn: {exc}") from exc

    if validate:
        onnx.checker.check_model(onnx.load(str(drift_path)))

    paths = {"drift_fn": drift_path}
    logger.info(f"Exported NeuralSDE drift_fn to {drift_path}")
    return paths


def _export_cde_func(
    model: nn.Module,
    output_dir: Path,
    opset_version: int,
    validate: bool,
) -> dict[str, Path]:
    """Export NeuralCDE vector field function only.

    Control path interpolation (torchcde) cannot be traced to ONNX. This exports
    the neural network f_theta(z) that computes the vector field matrix.

    Args:
        model: NeuralCDE instance.
        output_dir: Directory to write ONNX files.
        opset_version: ONNX opset version.
        validate: If True, validate the exported model.

    Returns:
        Dict mapping 'cde_func' to the exported ONNX path.
    """
    import onnx

    logger.warning(
        "NeuralCDE: exporting cde_func (vector field) only. "
        "Control path interpolation stays in Python. "
        "The exported network computes f_theta(z) -> (state_dim, input_dim) matrix."
    )

    cde_func = model.cde_func
    state_dim: int = model.state_dim
    cde_path = output_dir / "cde_func.onnx"

    dummy_t = torch.tensor(0.0)
    dummy_z = torch.randn(1, state_dim)

    try:
        torch.onnx.export(
            cde_func,
            (dummy_t, dummy_z),
            str(cde_path),
            opset_version=opset_version,
            input_names=["t", "z"],
            output_names=["f_theta"],
            dynamic_axes={"z": {0: "batch"}, "f_theta": {0: "batch"}},
            **_onnx_export_kwargs(),
        )
    except Exception as exc:
        raise ExportError(f"Failed to export NeuralCDE cde_func: {exc}") from exc

    if validate:
        onnx.checker.check_model(onnx.load(str(cde_path)))

    paths = {"cde_func": cde_path}
    logger.info(f"Exported NeuralCDE cde_func to {cde_path}")
    return paths


class ONNXExporter:
    """Static methods for exporting Neural DE models to ONNX format."""

    @staticmethod
    def export(
        model: nn.Module,
        output_dir: str | Path,
        opset_version: int = 17,
        validate: bool = True,
    ) -> dict[str, Path]:
        """Export a Neural DE model to ONNX.

        Supported models:
            - NeuralODE -> exports ode_func (1 file)
            - LatentNeuralODE -> exports ode_func + encoder + decoder (3 files)
            - AugmentedNeuralODE -> exports ode_func (1 file, augmented state)
            - NeuralSDE -> partial export: drift_fn only (deterministic approximation,
              diffusion/stochastic term cannot be traced). Returns {'drift_fn': path}.
            - NeuralCDE -> partial export: cde_func only (control path interpolation
              stays in Python). Returns {'cde_func': path}.

        Args:
            model: A Neural DE model instance.
            output_dir: Directory to write ONNX files.
            opset_version: ONNX opset version.
            validate: If True, run validation after export.

        Returns:
            Dictionary mapping component name to file path.

        Raises:
            ExportError: If the model type is not supported.
        """
        try:
            import onnx
        except ImportError as exc:
            raise ExportError(
                "onnx package is required for export. Install with: pip install onnx"
            ) from exc

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        model.eval()
        class_name = type(model).__name__
        paths: dict[str, Path] = {}

        if class_name == "NeuralSDE":
            return _export_sde_drift(model, output_dir, opset_version, validate)

        if class_name == "NeuralCDE":
            return _export_cde_func(model, output_dir, opset_version, validate)

        if not hasattr(model, "ode_func"):
            raise ExportError(f"Model {class_name} does not have an 'ode_func' attribute.")

        # Export ODE function
        ode_path = output_dir / "ode_func.onnx"
        wrapper = _ODEFuncWrapper(model.ode_func)
        wrapper.eval()

        state_dim = (
            model.ode_func.state_dim if hasattr(model.ode_func, "state_dim") else model.state_dim
        )
        if class_name == "AugmentedNeuralODE":
            state_dim = model.full_dim

        dummy_t = torch.tensor(0.0)
        dummy_z = torch.randn(1, state_dim)

        try:
            torch.onnx.export(
                wrapper,
                (dummy_t, dummy_z),
                str(ode_path),
                opset_version=opset_version,
                input_names=["t", "z"],
                output_names=["dz_dt"],
                dynamic_axes={
                    "z": {0: "batch"},
                    "dz_dt": {0: "batch"},
                },
                **_onnx_export_kwargs(),
            )
        except Exception as exc:
            raise ExportError(f"Failed to export ode_func: {exc}") from exc

        paths["ode_func"] = ode_path
        logger.info(f"Exported ode_func to {ode_path}")

        # LatentNeuralODE: also export encoder + decoder
        if class_name == "LatentNeuralODE":
            # Encoder
            encoder_path = output_dir / "encoder.onnx"
            try:
                dummy_x = torch.randn(1, model.state_dim)
                torch.onnx.export(
                    model.encoder,
                    (dummy_x,),
                    str(encoder_path),
                    opset_version=opset_version,
                    input_names=["x"],
                    output_names=["z_mean", "z_logvar"],
                    dynamic_axes={"x": {0: "batch"}},
                    **_onnx_export_kwargs(),
                )
                paths["encoder"] = encoder_path
                logger.info(f"Exported encoder to {encoder_path}")
            except Exception as exc:
                raise ExportError(f"Failed to export encoder: {exc}") from exc

            # Decoder
            decoder_path = output_dir / "decoder.onnx"
            try:
                dummy_z_dec = torch.randn(1, model.latent_dim)
                torch.onnx.export(
                    model.decoder,
                    (dummy_z_dec,),
                    str(decoder_path),
                    opset_version=opset_version,
                    input_names=["z"],
                    output_names=["x_hat"],
                    dynamic_axes={"z": {0: "batch"}},
                    **_onnx_export_kwargs(),
                )
                paths["decoder"] = decoder_path
                logger.info(f"Exported decoder to {decoder_path}")
            except Exception as exc:
                raise ExportError(f"Failed to export decoder: {exc}") from exc

        # Validate
        if validate:
            for name, path in paths.items():
                onnx_model = onnx.load(str(path))
                onnx.checker.check_model(onnx_model)
                logger.info(f"Validated {name}: {path}")

        return paths

    @staticmethod
    def validate_export(
        pytorch_model: nn.Module,
        onnx_paths: dict[str, Path],
    ) -> dict[str, float]:
        """Compare PyTorch and ONNX outputs numerically.

        Args:
            pytorch_model: Original PyTorch model.
            onnx_paths: Dictionary from :meth:`export`.

        Returns:
            Dictionary mapping component name to max absolute error.
        """
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ExportError(
                "onnxruntime is required for validation. Install with: pip install onnxruntime"
            ) from exc

        pytorch_model.eval()
        errors: dict[str, float] = {}

        if "ode_func" in onnx_paths:
            session = ort.InferenceSession(str(onnx_paths["ode_func"]))
            state_dim = (
                pytorch_model.ode_func.state_dim
                if hasattr(pytorch_model.ode_func, "state_dim")
                else pytorch_model.state_dim
            )
            class_name = type(pytorch_model).__name__
            if class_name == "AugmentedNeuralODE":
                state_dim = pytorch_model.full_dim

            t_np = np.array(0.0, dtype=np.float32)
            z_np = np.random.randn(1, state_dim).astype(np.float32)

            # PyTorch output
            with torch.no_grad():
                wrapper = _ODEFuncWrapper(pytorch_model.ode_func)
                pt_out = wrapper(
                    torch.tensor(t_np),
                    torch.tensor(z_np),
                ).numpy()

            # ONNX output
            ort_out = session.run(None, {"t": t_np, "z": z_np})[0]
            errors["ode_func"] = float(np.max(np.abs(pt_out - ort_out)))

        return errors


class ONNXInferenceRunner:
    """Run inference using exported ONNX models with a fixed-step solver."""

    def __init__(
        self,
        onnx_paths: dict[str, str | Path],
        solver: str = "rk4",
    ):
        """Initialize ONNX inference runner.

        Args:
            onnx_paths: Dict mapping component name to ONNX file path.
                        Must contain at least ``"ode_func"``.
            solver: Fixed-step solver: ``"euler"`` or ``"rk4"``.
        """
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ExportError(
                "onnxruntime is required. Install with: pip install onnxruntime"
            ) from exc

        self.solver = solver
        self._sessions: dict[str, Any] = {}
        for name, path in onnx_paths.items():
            self._sessions[name] = ort.InferenceSession(str(path))

    def _ode_eval(self, t: float, z: npt.NDArray[Any]) -> npt.NDArray[Any]:
        """Evaluate the ONNX ODE function."""
        t_np = np.array(t, dtype=np.float32)
        z_np = z.astype(np.float32)
        return cast(npt.NDArray[Any], self._sessions["ode_func"].run(None, {"t": t_np, "z": z_np})[0])

    @staticmethod
    def _euler_step(
        f: Any,
        t: float,
        z: npt.NDArray[Any],
        dt: float,
    ) -> npt.NDArray[Any]:
        return cast(npt.NDArray[Any], z + dt * f(t, z))

    @staticmethod
    def _rk4_step(
        f: Any,
        t: float,
        z: npt.NDArray[Any],
        dt: float,
    ) -> npt.NDArray[Any]:
        k1 = f(t, z)
        k2 = f(t + dt / 2, z + dt / 2 * k1)
        k3 = f(t + dt / 2, z + dt / 2 * k2)
        k4 = f(t + dt, z + dt * k3)
        return cast(npt.NDArray[Any], z + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4))

    def predict(
        self,
        z0: npt.NDArray[Any],
        t_span: npt.NDArray[Any],
    ) -> npt.NDArray[Any]:
        """Integrate ODE using fixed-step solver + ONNX Runtime.

        Args:
            z0: Initial state, shape ``(batch, state_dim)``.
            t_span: Time points, shape ``(num_times,)``.

        Returns:
            Trajectory, shape ``(batch, num_times, state_dim)``.
        """
        step_fn = self._rk4_step if self.solver == "rk4" else self._euler_step

        batch_size = z0.shape[0]
        num_times = len(t_span)
        state_dim = z0.shape[1]

        trajectory = np.zeros((batch_size, num_times, state_dim), dtype=np.float32)
        trajectory[:, 0, :] = z0

        z_current = z0.astype(np.float32).copy()

        for i in range(1, num_times):
            dt = float(t_span[i] - t_span[i - 1])
            t = float(t_span[i - 1])
            z_current = step_fn(self._ode_eval, t, z_current, dt)
            trajectory[:, i, :] = z_current

        return trajectory


def benchmark_inference(
    pytorch_model: nn.Module,
    onnx_runner: ONNXInferenceRunner,
    z0: npt.NDArray[Any],
    t_span: npt.NDArray[Any],
    n_repeats: int = 10,
) -> dict[str, float]:
    """Compare PyTorch vs ONNX inference speed.

    Args:
        pytorch_model: Original model.
        onnx_runner: ONNX runner.
        z0: Initial state (numpy).
        t_span: Time span (numpy).
        n_repeats: Number of timing repeats.

    Returns:
        Dict with ``pytorch_ms``, ``onnx_ms``, ``speedup``.
    """
    pytorch_model.eval()

    # PyTorch timing
    z0_torch = torch.tensor(z0, dtype=torch.float32)
    t_torch = torch.tensor(t_span, dtype=torch.float32)

    times_pt = []
    for _ in range(n_repeats):
        start = time.perf_counter()
        with torch.no_grad():
            pytorch_model(z0_torch, t_torch)
        times_pt.append((time.perf_counter() - start) * 1000)

    # ONNX timing
    times_onnx = []
    for _ in range(n_repeats):
        start = time.perf_counter()
        onnx_runner.predict(z0, t_span)
        times_onnx.append((time.perf_counter() - start) * 1000)

    pt_ms = float(np.median(times_pt))
    onnx_ms = float(np.median(times_onnx))

    return {
        "pytorch_ms": pt_ms,
        "onnx_ms": onnx_ms,
        "speedup": pt_ms / max(onnx_ms, 1e-6),
    }


__all__ = ["ONNXExporter", "ONNXInferenceRunner", "benchmark_inference"]
