"""Logging configuration for RAGMark.

This module provides standardized logging utilities that enforce
the project's logging policy. It supports both human-readable (standard)
and machine-readable (JSON) formats, with different verbosity levels.

Environment Variables:
    RAGMARK_LOG_LEVEL: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
                       Default: INFO
    RAGMARK_LOG_FORMAT: Output format ("standard" or "json").
                        Default: standard
"""

import json
import logging
import os
import sys
from typing import Any

LOG_FORMAT_STANDARD = "%(asctime)s.%(msecs)03d - %(levelname)s - %(message)s"

LOG_FORMAT_DEBUG = (
    "%(asctime)s.%(msecs)03d - %(levelname)s - "
    "%(filename)s:%(lineno)d - %(name)s:%(funcName)s - %(message)s"
)

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _resolve_log_level(level: int | str | None) -> int | str:
    """Resolve a log level, falling back to environment defaults."""
    if level is None:
        level_str = os.getenv("RAGMARK_LOG_LEVEL", "INFO").upper()
        return getattr(logging, level_str, logging.INFO)
    if isinstance(level, str):
        return getattr(logging, level.upper(), logging.INFO)
    return level


class JSONFormatter(logging.Formatter):
    """Format logs as newline-delimited JSON for production systems.

    Output structure includes timestamp, level, message, logger name,
    module, function, line number, and optional exception stack trace.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Convert log record to single-line JSON string."""
        log_data: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data, ensure_ascii=False)


def configure_logger(
    name: str,
    level: int | str | None = None,
    format_type: str | None = None,
    handler: logging.Handler | None = None,
) -> logging.Logger:
    """Configure a logger with RAGMark standard formatting.

    Respect environment variables RAGMARK_LOG_LEVEL and RAGMARK_LOG_FORMAT
    for runtime configuration without code changes.

    Args:
        name: Typically __name__ of the calling module.
        level: Override default level from environment.
        format_type: Either "standard" or "json".
        handler: Custom handler; defaults to StreamHandler on stderr.

    Returns:
        Configured logger instance ready for use.
    """
    logger = logging.getLogger(name)

    # Skip reconfiguration to avoid duplicate handlers
    if logger.handlers:
        return logger

    # Determine log level from parameter or environment
    level = _resolve_log_level(level)

    logger.setLevel(level)

    # Determine format type from parameter or environment
    if format_type is None:
        format_type = os.getenv("RAGMARK_LOG_FORMAT", "standard").lower()

    if handler is None:
        handler = logging.StreamHandler(sys.stderr)

    handler.setLevel(level)

    # DEBUG mode requires extended format with file location
    if format_type == "json":
        formatter: logging.Formatter = JSONFormatter(datefmt=DATE_FORMAT)
    elif level == logging.DEBUG:
        formatter = logging.Formatter(LOG_FORMAT_DEBUG, datefmt=DATE_FORMAT)
    else:
        formatter = logging.Formatter(LOG_FORMAT_STANDARD, datefmt=DATE_FORMAT)

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Prevent propagation to avoid duplicate logs in root logger
    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a configured logger using default environment settings.

    Args:
        name: Typically __name__ of the calling module.

    Returns:
        Configured logger instance.
    """
    return configure_logger(name)


def set_log_level(level: int | str) -> None:
    """Update log level for all RAGMark loggers dynamically.

    Args:
        level: New log level as int constant or string name.
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    ragmark_logger = logging.getLogger("ragmark")
    ragmark_logger.setLevel(level)

    for handler in ragmark_logger.handlers:
        handler.setLevel(level)

        # DEBUG mode requires extended format with context
        if level == logging.DEBUG:
            formatter = logging.Formatter(LOG_FORMAT_DEBUG, datefmt=DATE_FORMAT)
        else:
            formatter = logging.Formatter(LOG_FORMAT_STANDARD, datefmt=DATE_FORMAT)
        handler.setFormatter(formatter)


def mask_sensitive(value: str, prefix_len: int = 4, suffix_len: int = 4) -> str:
    """Mask sensitive data for safe logging of credentials.

    Args:
        value: Sensitive string to mask.
        prefix_len: Characters preserved at start.
        suffix_len: Characters preserved at end.

    Returns:
        Masked string with middle replaced by asterisks.
    """
    if len(value) <= prefix_len + suffix_len:
        return "***"
    return f"{value[:prefix_len]}***{value[-suffix_len:]}"
