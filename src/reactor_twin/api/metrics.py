"""Prometheus metrics for the ReactorTwin API."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from prometheus_client import Counter, Gauge, Histogram

try:
    from prometheus_client import Counter, Gauge, Histogram, make_asgi_app

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus-client not installed. /metrics endpoint will not be available.")

REQUEST_COUNT: Any = None
REQUEST_LATENCY: Any = None
ODE_SOLVE_TIME: Any = None
ACTIVE_WS_SESSIONS: Any = None

if PROMETHEUS_AVAILABLE:
    REQUEST_COUNT = Counter(
        "reactor_twin_requests_total",
        "Total HTTP requests",
        ["method", "endpoint", "status"],
    )
    REQUEST_LATENCY = Histogram(
        "reactor_twin_request_duration_seconds",
        "HTTP request latency",
        ["endpoint"],
    )
    ODE_SOLVE_TIME = Histogram(
        "reactor_twin_ode_solve_seconds",
        "ODE solve duration",
        ["reactor_type"],
    )
    ACTIVE_WS_SESSIONS = Gauge(
        "reactor_twin_active_ws_sessions",
        "Active WebSocket sessions",
    )


def make_metrics_app() -> Any:
    """Return ASGI app exposing Prometheus metrics.

    Returns:
        Prometheus ASGI app, or None if prometheus_client is not installed.
    """
    if not PROMETHEUS_AVAILABLE:
        return None
    return make_asgi_app()


__all__ = [
    "ACTIVE_WS_SESSIONS",
    "ODE_SOLVE_TIME",
    "PROMETHEUS_AVAILABLE",
    "REQUEST_COUNT",
    "REQUEST_LATENCY",
    "make_metrics_app",
]
