"""Structured logging.

Three user-visible levels (INFO / WARNING / ERROR) drive the dashboard log
stream; full DEBUG output goes to rotating files for download.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import sys
import time
from pathlib import Path

_CONFIGURED = False

_RESERVED = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "taskName", "message", "asctime",
}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                try:
                    json.dumps(value)
                    payload[key] = value
                except (TypeError, ValueError):
                    payload[key] = repr(value)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class HumanFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[38;5;244m",
        "INFO": "\033[38;5;39m",
        "WARNING": "\033[38;5;214m",
        "ERROR": "\033[38;5;203m",
        "CRITICAL": "\033[48;5;203m\033[38;5;231m",
    }
    RESET = "\033[0m"

    def __init__(self, color: bool = True) -> None:
        super().__init__("%(asctime)s %(levelname)-8s %(name)s :: %(message)s", "%H:%M:%S")
        self.color = color and sys.stderr.isatty()

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        if self.color:
            return f"{self.COLORS.get(record.levelname, '')}{text}{self.RESET}"
        return text


def configure_logging(level: str = "INFO", json_output: bool = True,
                      log_dir: Path | None = None) -> None:
    """Idempotently configure root logging."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    stream = logging.StreamHandler(sys.stderr)
    stream.setLevel(getattr(logging, level.upper(), logging.INFO))
    stream.setFormatter(JsonFormatter() if json_output else HumanFormatter())
    root.addHandler(stream)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / "neurotribe.log", maxBytes=32 * 1024 * 1024, backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(JsonFormatter())
        root.addHandler(file_handler)

    # Third-party noise control.
    for noisy in ("urllib3", "botocore", "asyncio", "matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.LoggerAdapter:
    """Logger that automatically tags records with the current job id, if any."""
    base = logging.getLogger(name)
    return logging.LoggerAdapter(base, {"job_id": os.environ.get("NEUROTRIBE_JOB_ID")})
