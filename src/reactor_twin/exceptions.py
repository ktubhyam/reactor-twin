"""ReactorTwin exception hierarchy.

All library-specific exceptions inherit from :class:`ReactorTwinError`,
enabling callers to catch the broad base class or narrow subtypes.
"""

from __future__ import annotations


class ReactorTwinError(Exception):
    """Base exception for all ReactorTwin errors."""


class SolverError(ReactorTwinError):
    """ODE/SDE solver failures (divergence, non-convergence, etc.)."""


class ValidationError(ReactorTwinError):
    """Invalid inputs, shapes, types, or parameter values."""


class ExportError(ReactorTwinError):
    """ONNX or other model-export failures."""


class ConstraintViolationError(ReactorTwinError):
    """Physics constraint violations that cannot be resolved."""


class RegistryError(ReactorTwinError):
    """Registry lookup or registration failures."""


class ConfigurationError(ReactorTwinError):
    """Reactor or model configuration errors (missing/invalid parameters)."""


__all__ = [
    "ReactorTwinError",
    "SolverError",
    "ValidationError",
    "ExportError",
    "ConstraintViolationError",
    "RegistryError",
    "ConfigurationError",
]
