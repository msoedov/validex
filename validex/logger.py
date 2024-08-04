import logging
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any


class ColorFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": 95,  # gray
        "INFO": 92,  # blue
        "WARNING": 91,  # yellow
        "ERROR": 31,  # red
        "CRITICAL": 31,  # red
    }

    LEVEL_MAP = {
        "DEBUG": "DEBG",
        "INFO": "INFO",
        "WARNING": "WARN",
        "ERROR": "ERRO",
        "CRITICAL": "PANI",
    }

    def __init__(
        self,
        force_colors: bool = False,
        disable_colors: bool = False,
        force_quote: bool = False,
        disable_quote: bool = False,
        environment_override_colors: bool = False,
        disable_timestamp: bool = False,
        full_timestamp: bool = False,
        timestamp_format: str | None = None,
        disable_sorting: bool = False,
        sorting_func: Callable[[list[str]], None] | None = None,
        disable_level_truncation: bool = False,
        pad_level_text: bool = False,
        quote_empty_fields: bool = False,
        field_map: dict[str, str] | None = None,
        caller_prettyfier: Callable[[str, str, int], tuple[str, str]] | None = None,
    ):
        super().__init__()
        self.force_colors = force_colors
        self.disable_colors = disable_colors
        self.force_quote = force_quote
        self.disable_quote = disable_quote
        self.environment_override_colors = environment_override_colors
        self.disable_timestamp = disable_timestamp
        self.full_timestamp = full_timestamp
        self.timestamp_format = timestamp_format or "%Y-%m-%d %H:%M:%S"
        self.disable_sorting = disable_sorting
        self.sorting_func = sorting_func
        self.disable_level_truncation = disable_level_truncation
        self.pad_level_text = pad_level_text
        self.quote_empty_fields = quote_empty_fields
        self.field_map = field_map or {}
        self.caller_prettyfier = caller_prettyfier
        self.level_text_max_length = max(
            len(level) for level in logging._levelToName.values()
        )

        self.base_timestamp = time.time()

    def format(self, record: logging.LogRecord) -> str:
        log_entry = self.formatMessage(record)

        if self.disable_colors:
            return log_entry

        levelname = record.levelname
        color = self.COLORS.get(levelname, 37)
        level_text = self.LEVEL_MAP.get(levelname, levelname[:4])
        if self.pad_level_text:
            level_text = level_text.ljust(self.level_text_max_length)

        message = f"\x1b[{color}m{level_text}\x1b[0m[{int(record.created - self.base_timestamp):04d}] {log_entry}"

        if self.caller_prettyfier and record.exc_info:
            func_name, file_name = self.caller_prettyfier(
                record.funcName, record.pathname, record.lineno
            )
            message += (
                f" \x1b[{color}m{func_name}\x1b[0m \x1b[{color}m{file_name}\x1b[0m"
            )

        return message

    def formatMessage(self, record: logging.LogRecord) -> str:
        log_message = record.getMessage()

        # Extract all dynamic parameters
        standard_attrs = {
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
        }
        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in standard_attrs
        }
        if self.field_map:
            extras = {self.field_map.get(k, k): v for k, v in extras.items()}

        levelname = record.levelname
        color = self.COLORS.get(levelname, 37)

        extra_parts = [
            f"\x1b[{color}m{key}={self._format_value(value)}\x1b[0m"
            for key, value in extras.items()
            if value
        ]

        if not self.disable_sorting:
            extra_parts.sort()

        return f"{log_message} {' '.join(extra_parts)}"

    def _format_value(self, value: Any) -> str:
        if isinstance(value, str):
            if (
                self.force_quote
                or (self.quote_empty_fields and value == "")
                or self._needs_quoting(value)
            ):
                return f'"{value}"'
        return str(value)

    def _needs_quoting(self, text: str) -> bool:
        if self.force_quote:
            return True
        if self.disable_quote:
            return False
        for ch in text:
            if not (ch.isalnum() or ch in "-._/@^+"):
                return True
        return False

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        if self.disable_timestamp:
            return ""
        ct = datetime.fromtimestamp(record.created)
        if self.full_timestamp:
            return ct.strftime(datefmt or self.default_time_format)
        else:
            return f"{int(record.created - self.base_timestamp)}"


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    handler = logging.StreamHandler()
    handler.setFormatter(ColorFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger


logger = get_logger(__name__)
log = logger
