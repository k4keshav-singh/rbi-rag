"""Structured JSON logging.

Built on the standard library rather than structlog: the only requirement is
one JSON object per line, and that is a formatter, not a dependency.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import sys
from typing import Any

#: Attributes LogRecord always carries. Anything else came from `extra` and is
#: the interesting part, so it is emitted alongside the message.
_RESERVED = frozenset(
    [
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
        "taskName",
    ]
)


class JsonFormatter(logging.Formatter):
    """Renders each record as a single JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": dt.datetime.fromtimestamp(record.created, tz=dt.UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = _safe(value)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def _safe(value: Any) -> Any:
    """Keep JSON-native values as they are; stringify everything else."""
    if isinstance(value, str | int | float | bool | type(None)):
        return value
    if isinstance(value, list | tuple):
        return [_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    return str(value)


def configure_logging(level: str = "INFO") -> None:
    """Send JSON lines to stdout. Safe to call more than once."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
