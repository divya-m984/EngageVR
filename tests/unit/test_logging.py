"""Tests for structured logging."""

from __future__ import annotations

import json
import logging

from engagevr._logging import get_logger, setup_logging


class TestSetupLogging:
    def test_json_format(self, capfd):
        logger = setup_logging(level="DEBUG", fmt="json")
        logger.info("test message")
        # JSON goes to stderr
        captured = capfd.readouterr()
        record = json.loads(captured.err.strip())
        assert record["level"] == "INFO"
        assert record["message"] == "test message"
        assert "timestamp" in record

    def test_text_format(self, capfd):
        logger = setup_logging(level="DEBUG", fmt="text")
        logger.info("hello")
        captured = capfd.readouterr()
        assert "hello" in captured.err
        assert "INFO" in captured.err

    def test_level_filtering(self, capfd):
        logger = setup_logging(level="WARNING", fmt="json")
        logger.debug("should not appear")
        logger.warning("should appear")
        captured = capfd.readouterr()
        lines = [line for line in captured.err.strip().split("\n") if line]
        assert len(lines) == 1
        assert "should appear" in lines[0]

    def test_file_logging(self, tmp_path):
        log_file = tmp_path / "test.log"
        logger = setup_logging(level="INFO", fmt="json", log_file=str(log_file))
        logger.info("file test")
        content = log_file.read_text()
        record = json.loads(content.strip())
        assert record["message"] == "file test"


class TestGetLogger:
    def test_child_logger(self):
        child = get_logger("test_module")
        assert child.name == "engagevr.test_module"

    def test_base_logger(self):
        base = get_logger()
        assert base.name == "engagevr"

    def teardown_method(self):
        # Clean up handlers to avoid cross-test pollution
        root = logging.getLogger("engagevr")
        root.handlers.clear()
