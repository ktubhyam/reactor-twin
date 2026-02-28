"""Tests for structured logging."""

from __future__ import annotations

import json
import logging

from reactor_twin.utils.logging import JSONFormatter, RequestTracer, setup_logging


class TestJSONFormatter:
    def test_format_basic(self):
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="hello",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        data = json.loads(output)
        assert data["level"] == "INFO"
        assert data["message"] == "hello"
        assert data["logger"] == "test"
        assert "timestamp" in data

    def test_format_with_args(self):
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.WARNING,
            pathname="",
            lineno=0,
            msg="value=%d",
            args=(42,),
            exc_info=None,
        )
        output = formatter.format(record)
        data = json.loads(output)
        assert data["message"] == "value=42"

    def test_format_with_exception(self):
        formatter = JSONFormatter()
        try:
            raise ValueError("test error")
        except ValueError:
            import sys

            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg="failed",
            args=(),
            exc_info=exc_info,
        )
        output = formatter.format(record)
        data = json.loads(output)
        assert "exception" in data
        assert "ValueError" in data["exception"]

    def test_output_is_single_line(self):
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test",
            args=(),
            exc_info=None,
        )
        output = formatter.format(record)
        # JSON should be a single line (no pretty printing)
        assert "\n" not in output


class TestRequestTracer:
    def test_request_id_default(self):
        with RequestTracer() as tracer:
            assert tracer.request_id is not None
            assert len(tracer.request_id) == 12

    def test_custom_request_id(self):
        with RequestTracer(request_id="abc123") as tracer:
            assert tracer.request_id == "abc123"

    def test_context_cleared_after_exit(self):
        from reactor_twin.utils.logging import _request_id

        assert _request_id.get() is None
        with RequestTracer(request_id="test"):
            assert _request_id.get() == "test"
        assert _request_id.get() is None

    def test_nested_tracers(self):
        from reactor_twin.utils.logging import _request_id

        with RequestTracer(request_id="outer"):
            assert _request_id.get() == "outer"
            with RequestTracer(request_id="inner"):
                assert _request_id.get() == "inner"
            assert _request_id.get() == "outer"
        assert _request_id.get() is None

    def test_json_formatter_includes_request_id(self):
        formatter = JSONFormatter()
        from reactor_twin.utils.logging import _request_id

        token = _request_id.set("req-42")
        try:
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg="traced",
                args=(),
                exc_info=None,
            )
            output = formatter.format(record)
            data = json.loads(output)
            assert data.get("request_id") == "req-42"
        finally:
            _request_id.reset(token)


class TestSetupLogging:
    def test_setup_text_format(self):
        setup_logging(level="DEBUG", log_format="text")
        logger = logging.getLogger("reactor_twin")
        assert logger.level == logging.DEBUG
        assert len(logger.handlers) >= 1

    def test_setup_json_format(self):
        setup_logging(level="INFO", log_format="json")
        logger = logging.getLogger("reactor_twin")
        handler = logger.handlers[0]
        assert isinstance(handler.formatter, JSONFormatter)

    def test_setup_file_output(self, tmp_path):
        log_file = str(tmp_path / "test.log")
        setup_logging(level="INFO", log_format="text", log_file=log_file)
        logger = logging.getLogger("reactor_twin")
        logger.info("file test message")
        # File handler should exist
        file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) == 1

    def test_module_levels(self):
        setup_logging(
            level="WARNING",
            log_format="text",
            module_levels={"reactor_twin.core": "DEBUG"},
        )
        core_logger = logging.getLogger("reactor_twin.core")
        assert core_logger.level == logging.DEBUG
