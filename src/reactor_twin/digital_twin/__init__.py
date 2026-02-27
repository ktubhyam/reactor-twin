"""Digital twin layer: state estimation, fault detection, control, and adaptation."""

from __future__ import annotations

from reactor_twin.digital_twin.fault_detector import (
    Alarm,
    AlarmLevel,
    FaultClassifier,
    FaultDetector,
    FaultIsolator,
    ResidualDetector,
    SPCChart,
)
from reactor_twin.digital_twin.mpc_controller import (
    ControlConstraints,
    MPCController,
    MPCObjective,
)
from reactor_twin.digital_twin.meta_learner import ReptileMetaLearner
from reactor_twin.digital_twin.online_adapter import (
    ElasticWeightConsolidation,
    OnlineAdapter,
    ReplayBuffer,
)
from reactor_twin.digital_twin.state_estimator import EKFStateEstimator

__all__ = [
    # State estimation
    "EKFStateEstimator",
    # Fault detection
    "AlarmLevel",
    "Alarm",
    "SPCChart",
    "ResidualDetector",
    "FaultIsolator",
    "FaultClassifier",
    "FaultDetector",
    # Model predictive control
    "MPCObjective",
    "ControlConstraints",
    "MPCController",
    # Online adaptation
    "ReplayBuffer",
    "ElasticWeightConsolidation",
    "OnlineAdapter",
    # Meta-learning
    "ReptileMetaLearner",
]
