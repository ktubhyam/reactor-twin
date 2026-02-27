"""Multi-level fault detection and diagnosis for reactor digital twins.

Provides four detection levels:
- L1: Statistical process control (EWMA / CUSUM) on raw sensor data
- L2: Residual-based detection via one-step-ahead Neural ODE predictions
- L3: Per-variable contribution analysis for fault isolation
- L4: Machine-learning fault classification (SVM / Random Forest)

A unified ``FaultDetector`` class orchestrates all levels.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np
import torch

from reactor_twin.core.base import AbstractNeuralDE

logger = logging.getLogger(__name__)


# ======================================================================
# Alarm severity
# ======================================================================


class AlarmLevel(Enum):
    """Severity levels for fault alarms."""

    NORMAL = auto()
    WARNING = auto()
    ALARM = auto()
    CRITICAL = auto()


@dataclass
class Alarm:
    """Single alarm event."""

    level: AlarmLevel
    source: str
    message: str
    timestamp: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)


# ======================================================================
# Level 1: Statistical Process Control
# ======================================================================


class SPCChart:
    """EWMA and CUSUM control charts on raw sensor signals.

    Detects small sustained shifts (EWMA) and cumulative drift (CUSUM)
    in individual measured variables.

    Attributes:
        num_vars: Number of monitored variables.
        ewma_lambda: EWMA smoothing factor (0 < lambda <= 1).
        ewma_L: Control limit width in sigma units.
        cusum_k: CUSUM allowance (slack) parameter.
        cusum_h: CUSUM decision interval.
    """

    def __init__(
        self,
        num_vars: int,
        ewma_lambda: float = 0.2,
        ewma_L: float = 3.0,
        cusum_k: float = 0.5,
        cusum_h: float = 5.0,
    ) -> None:
        self.num_vars = num_vars
        self.ewma_lambda = ewma_lambda
        self.ewma_L = ewma_L
        self.cusum_k = cusum_k
        self.cusum_h = cusum_h

        # Statistics learned from baseline
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

        # Running state
        self._ewma: np.ndarray | None = None
        self._cusum_pos: np.ndarray | None = None
        self._cusum_neg: np.ndarray | None = None

    def set_baseline(self, data: np.ndarray) -> None:
        """Learn normal-operation statistics.

        Args:
            data: Baseline data, shape ``(num_samples, num_vars)``.
        """
        self.mean = data.mean(axis=0)
        self.std = data.std(axis=0) + 1e-10
        self.reset()

    def reset(self) -> None:
        """Reset running accumulators to baseline means."""
        if self.mean is None:
            return
        self._ewma = self.mean.copy()
        self._cusum_pos = np.zeros(self.num_vars)
        self._cusum_neg = np.zeros(self.num_vars)

    def update(self, x: np.ndarray) -> dict[str, Any]:
        """Process one observation and return SPC alarms.

        Args:
            x: Observation vector, shape ``(num_vars,)``.

        Returns:
            Dictionary with keys ``ewma_alarm``, ``cusum_alarm``
            (boolean arrays) and ``ewma_values``, ``cusum_pos``,
            ``cusum_neg`` (current chart values).
        """
        if self.mean is None:
            raise RuntimeError("Call set_baseline() before update().")

        z = (x - self.mean) / self.std  # standardized

        # EWMA update: ewma_new = lambda * x_new + (1 - lambda) * ewma_prev
        lam = self.ewma_lambda
        ewma_prev = self._ewma
        self._ewma = lam * x + (1 - lam) * ewma_prev  # type: ignore[operator]
        ewma_z = (self._ewma - self.mean) / self.std  # type: ignore[operator]
        ewma_limit = self.ewma_L * np.sqrt(lam / (2 - lam))
        ewma_alarm = np.abs(ewma_z) > ewma_limit

        # CUSUM update
        self._cusum_pos = np.maximum(0, self._cusum_pos + z - self.cusum_k)  # type: ignore[operator]
        self._cusum_neg = np.maximum(0, self._cusum_neg - z - self.cusum_k)  # type: ignore[operator]
        cusum_alarm = (self._cusum_pos > self.cusum_h) | (self._cusum_neg > self.cusum_h)

        return {
            "ewma_alarm": ewma_alarm,
            "cusum_alarm": cusum_alarm,
            "ewma_values": self._ewma.copy(),
            "cusum_pos": self._cusum_pos.copy(),
            "cusum_neg": self._cusum_neg.copy(),
        }


# ======================================================================
# Level 2: Residual-based Detection
# ======================================================================


class ResidualDetector:
    """One-step-ahead prediction error monitoring.

    Uses a Neural ODE to predict the next state from the current state,
    then checks the residual against adaptive thresholds learned from
    normal operation.

    Attributes:
        model: Neural DE providing ``predict()``.
        state_dim: Dimension of state vector.
        threshold_sigma: Number of standard deviations for alarm.
    """

    def __init__(
        self,
        model: AbstractNeuralDE,
        state_dim: int,
        dt: float = 0.01,
        threshold_sigma: float = 3.0,
        device: str | torch.device = "cpu",
    ) -> None:
        self.model = model
        self.state_dim = state_dim
        self.dt = dt
        self.threshold_sigma = threshold_sigma
        self.device = torch.device(device)

        self.residual_mean: np.ndarray | None = None
        self.residual_std: np.ndarray | None = None

    def set_baseline(self, residuals: np.ndarray) -> None:
        """Learn residual statistics from normal operation.

        Args:
            residuals: Residual vectors, shape ``(num_samples, state_dim)``.
        """
        self.residual_mean = residuals.mean(axis=0)
        self.residual_std = residuals.std(axis=0) + 1e-10

    def compute_residual(
        self,
        z_current: torch.Tensor,
        z_next_measured: torch.Tensor,
    ) -> np.ndarray:
        """Compute prediction residual.

        Args:
            z_current: Current state, shape ``(state_dim,)``.
            z_next_measured: Measured next state, shape ``(state_dim,)``.

        Returns:
            Residual vector as numpy array, shape ``(state_dim,)``.
        """
        z0 = z_current.unsqueeze(0).to(self.device)
        t_span = torch.tensor([0.0, self.dt], device=self.device)
        with torch.no_grad():
            z_pred = self.model.predict(z0, t_span)  # (1, 2, state_dim)
        z_pred_next = z_pred[0, -1].cpu().numpy()
        z_meas = z_next_measured.cpu().numpy()
        return z_meas - z_pred_next

    def update(
        self,
        z_current: torch.Tensor,
        z_next_measured: torch.Tensor,
    ) -> dict[str, Any]:
        """Process one timestep and return residual alarm info.

        Args:
            z_current: Current state, shape ``(state_dim,)``.
            z_next_measured: Measured next state, shape ``(state_dim,)``.

        Returns:
            Dictionary with ``residual``, ``alarm`` (bool array),
            and ``z_score`` arrays.
        """
        residual = self.compute_residual(z_current, z_next_measured)

        if self.residual_mean is not None:
            z_score = np.abs(residual - self.residual_mean) / self.residual_std  # type: ignore[operator]
            alarm = z_score > self.threshold_sigma
        else:
            z_score = np.zeros(self.state_dim)
            alarm = np.zeros(self.state_dim, dtype=bool)

        return {
            "residual": residual,
            "alarm": alarm,
            "z_score": z_score,
        }


# ======================================================================
# Level 3: Fault Isolation
# ======================================================================


class FaultIsolator:
    """Per-variable contribution analysis for fault isolation.

    When a multivariate alarm triggers, this class decomposes the
    aggregated statistic into per-variable contributions to identify
    which variables are responsible.

    Attributes:
        state_dim: Number of state variables.
    """

    def __init__(self, state_dim: int) -> None:
        self.state_dim = state_dim
        self.baseline_residual_cov: np.ndarray | None = None

    def set_baseline(self, residuals: np.ndarray) -> None:
        """Learn residual covariance from normal data.

        Args:
            residuals: Residual data, shape ``(num_samples, state_dim)``.
        """
        self.baseline_residual_cov = (
            np.cov(residuals, rowvar=False) + np.eye(self.state_dim) * 1e-10
        )

    def isolate(self, residual: np.ndarray) -> dict[str, Any]:
        """Compute per-variable contribution to squared prediction error.

        Args:
            residual: Current residual vector, shape ``(state_dim,)``.

        Returns:
            Dictionary with ``contributions`` (per-variable float array)
            and ``ranking`` (variable indices sorted by contribution).
        """
        if self.baseline_residual_cov is not None:
            cov_inv = np.linalg.inv(self.baseline_residual_cov)
        else:
            cov_inv = np.eye(self.state_dim)

        # Contribution of each variable to the squared Mahalanobis distance
        contributions = np.array(
            [residual[i] * (cov_inv[i, :] @ residual) for i in range(self.state_dim)]
        )
        contributions = np.abs(contributions)

        ranking = np.argsort(contributions)[::-1].tolist()

        return {
            "contributions": contributions,
            "ranking": ranking,
            "spe": float(residual @ cov_inv @ residual),
        }


# ======================================================================
# Level 4: Fault Classification
# ======================================================================


class FaultClassifier:
    """Machine-learning fault classification on residual features.

    Wraps scikit-learn classifiers (SVM / Random Forest) to label the
    type of fault from residual feature vectors.  ``scikit-learn`` is an
    *optional* dependency.

    Attributes:
        classifier: Fitted sklearn classifier or ``None``.
    """

    def __init__(self, method: str = "rf", **kwargs: Any) -> None:
        """Initialize classifier.

        Args:
            method: ``"svm"`` or ``"rf"`` (Random Forest).
            **kwargs: Forwarded to the sklearn estimator.
        """
        self.method = method
        self._kwargs = kwargs
        self.classifier: Any | None = None
        self.classes: list[str] = []

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray | list[str],
    ) -> None:
        """Train the classifier.

        Args:
            features: Feature matrix, shape ``(n_samples, n_features)``.
            labels: Fault class labels, shape ``(n_samples,)``.
        """
        try:
            if self.method == "svm":
                from sklearn.svm import SVC

                self.classifier = SVC(probability=True, **self._kwargs)
            else:
                from sklearn.ensemble import RandomForestClassifier

                self.classifier = RandomForestClassifier(**self._kwargs)
        except ImportError as exc:
            raise ImportError(
                "scikit-learn is required for FaultClassifier. "
                "Install with: pip install reactor-twin[digital_twin]"
            ) from exc

        self.classifier.fit(features, labels)
        self.classes = list(self.classifier.classes_)
        logger.info(f"FaultClassifier trained ({self.method}): classes={self.classes}")

    def predict(self, features: np.ndarray) -> dict[str, Any]:
        """Classify fault type from features.

        Args:
            features: Feature vector(s), shape ``(n_features,)`` or
                ``(n_samples, n_features)``.

        Returns:
            Dictionary with ``label`` (predicted class) and
            ``probabilities`` (per-class probabilities).
        """
        if self.classifier is None:
            return {"label": "unknown", "probabilities": {}}

        if features.ndim == 1:
            features = features.reshape(1, -1)

        label = self.classifier.predict(features)[0]
        probs = self.classifier.predict_proba(features)[0]
        prob_dict = dict(zip(self.classes, probs.tolist(), strict=False))

        return {"label": label, "probabilities": prob_dict}


# ======================================================================
# Unified Fault Detector
# ======================================================================


class FaultDetector:
    """Unified multi-level fault detection orchestrator.

    Coordinates L1-L4 detection and returns per-level alarms in a
    single ``update()`` call.

    Args:
        model: Neural DE for residual computation (L2).
        state_dim: Number of state variables.
        obs_dim: Number of observed variables (for SPC).
        dt: Prediction time step for L2.
        device: Torch device.
    """

    def __init__(
        self,
        model: AbstractNeuralDE,
        state_dim: int,
        obs_dim: int | None = None,
        dt: float = 0.01,
        device: str | torch.device = "cpu",
    ) -> None:
        obs_dim = obs_dim or state_dim
        self.state_dim = state_dim
        self.obs_dim = obs_dim

        self.spc = SPCChart(num_vars=obs_dim)
        self.residual_detector = ResidualDetector(
            model=model,
            state_dim=state_dim,
            dt=dt,
            device=device,
        )
        self.isolator = FaultIsolator(state_dim=state_dim)
        self.classifier = FaultClassifier()

        self._has_baseline = False

    def set_baseline(self, normal_data: dict[str, np.ndarray]) -> None:
        """Learn normal-operation statistics for all levels.

        Args:
            normal_data: Dictionary with keys:

                - ``observations``: shape ``(N, obs_dim)`` for L1
                - ``residuals``: shape ``(N, state_dim)`` for L2-L3
                - ``features`` and ``labels``: for L4 (optional)
        """
        if "observations" in normal_data:
            self.spc.set_baseline(normal_data["observations"])

        if "residuals" in normal_data:
            self.residual_detector.set_baseline(normal_data["residuals"])
            self.isolator.set_baseline(normal_data["residuals"])

        if "features" in normal_data and "labels" in normal_data:
            self.classifier.fit(normal_data["features"], normal_data["labels"])

        self._has_baseline = True
        logger.info("FaultDetector baseline set for all levels.")

    def update(
        self,
        z_current: torch.Tensor,
        z_next_measured: torch.Tensor,
        t: float = 0.0,
        observation: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Process one timestep across all detection levels.

        Args:
            z_current: Current full state, shape ``(state_dim,)``.
            z_next_measured: Next measured state, shape ``(state_dim,)``.
            t: Current time (for alarm timestamps).
            observation: Raw sensor reading, shape ``(obs_dim,)``.
                If ``None``, derived from ``z_next_measured``.

        Returns:
            Dictionary with keys ``L1``, ``L2``, ``L3``, ``L4``, and
            ``alarms`` (list of ``Alarm`` objects for triggered events).
        """
        alarms: list[Alarm] = []
        result: dict[str, Any] = {"alarms": alarms, "time": t}

        # L1: SPC on raw observation
        if observation is None:
            observation = z_next_measured.cpu().numpy()
            if len(observation) > self.obs_dim:
                observation = observation[: self.obs_dim]

        if self.spc.mean is not None:
            l1 = self.spc.update(observation)
            result["L1"] = l1
            if np.any(l1["ewma_alarm"]) or np.any(l1["cusum_alarm"]):
                alarms.append(
                    Alarm(
                        level=AlarmLevel.WARNING,
                        source="L1_SPC",
                        message="SPC control limit exceeded",
                        timestamp=t,
                        details={
                            "ewma": l1["ewma_alarm"].tolist(),
                            "cusum": l1["cusum_alarm"].tolist(),
                        },
                    )
                )

        # L2: Residual detection
        l2 = self.residual_detector.update(z_current, z_next_measured)
        result["L2"] = l2
        if np.any(l2["alarm"]):
            alarms.append(
                Alarm(
                    level=AlarmLevel.ALARM,
                    source="L2_Residual",
                    message="Prediction residual exceeds threshold",
                    timestamp=t,
                    details={"z_score": l2["z_score"].tolist()},
                )
            )

        # L3: Fault isolation (only if L2 alarmed)
        if np.any(l2["alarm"]):
            l3 = self.isolator.isolate(l2["residual"])
            result["L3"] = l3

        # L4: Classification (if classifier is fitted)
        if self.classifier.classifier is not None:
            features = l2["residual"]
            l4 = self.classifier.predict(features)
            result["L4"] = l4
            if l4["label"] != "normal":
                alarms.append(
                    Alarm(
                        level=AlarmLevel.CRITICAL,
                        source="L4_Classifier",
                        message=f"Fault classified: {l4['label']}",
                        timestamp=t,
                        details=l4,
                    )
                )

        return result


__all__ = [
    "AlarmLevel",
    "Alarm",
    "SPCChart",
    "ResidualDetector",
    "FaultIsolator",
    "FaultClassifier",
    "FaultDetector",
]
