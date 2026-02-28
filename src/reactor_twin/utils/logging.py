"""Structured logging for ReactorTwin."""

from __future__ import annotations

import json
import logging
import sys
import uuid
from contextvars import ContextVar
from typing import Any

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)


class JSONFormatter(logging.Formatter):
    """Format log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add request_id if present
        req_id = _request_id.get()
        if req_id is not None:
            log_entry["request_id"] = req_id

        # Add exception info if present
        if record.exc_info and record.exc_info[1] is not None:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        for key in ("request_id", "extra"):
            if hasattr(record, key) and key not in log_entry:
                log_entry[key] = getattr(record, key)

        return json.dumps(log_entry, default=str)


class RequestTracer:
    """Context manager that attaches a request_id to all log records.

    Usage::

        with RequestTracer() as tracer:
            logger.info("Processing request")  # includes request_id
        # request_id cleared after exit
    """

    def __init__(self, request_id: str | None = None):
        self.request_id = request_id or uuid.uuid4().hex[:12]
        self._token = None

    def __enter__(self) -> RequestTracer:
        self._token = _request_id.set(self.request_id)
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._token is not None:
            _request_id.reset(self._token)


class _RequestIDFilter(logging.Filter):
    """Injects request_id from context var into log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        req_id = _request_id.get()
        if req_id is not None:
            record.request_id = req_id  # type: ignore[attr-defined]
        return True


def setup_logging(
    level: str = "INFO",
    log_format: str = "text",
    log_file: str | None = None,
    module_levels: dict[str, str] | None = None,
) -> None:
    """Configure ReactorTwin logging.

    Args:
        level: Root log level (e.g. 'DEBUG', 'INFO', 'WARNING').
        log_format: 'text' for human-readable or 'json' for structured output.
        log_file: Optional file path to write logs to.
        module_levels: Per-module log levels (e.g. {'reactor_twin.core': 'DEBUG'}).
    """
    root_logger = logging.getLogger("reactor_twin")
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()

    # Create formatter
    if log_format == "json":
        formatter: logging.Formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    # Console handler
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(formatter)
    console.addFilter(_RequestIDFilter())
    root_logger.addHandler(console)

    # File handler
    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(_RequestIDFilter())
        root_logger.addHandler(file_handler)

    # Per-module levels
    if module_levels:
        for module, mod_level in module_levels.items():
            logging.getLogger(module).setLevel(getattr(logging, mod_level.upper(), logging.INFO))

    root_logger.debug(f"Logging configured: level={level}, format={log_format}")


__all__ = ["JSONFormatter", "RequestTracer", "setup_logging"]
