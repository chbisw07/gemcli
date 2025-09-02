# logging_setup.py
from __future__ import annotations

import os
import sys
import re
from pathlib import Path
from typing import Optional, Union
from loguru import logger

# -----------------------------
# Globals
# -----------------------------
_SINK_IDS: list[int] = []
_LAST_CFG = {
    "console": True,
    "log_file": None,
    "rotation": "5 MB",
    "retention": 10,  # keep last 10 files by default
    "enqueue": True,
    "backtrace": False,
    "diagnose": False,
}
# Show file:line at each log site (your ask), compact & readable timestamps
_DEFAULT_FMT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "{level:<7} | "
    "{name}:{line} | "
    "{message}"
)

_VALID_LEVELS = {"TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"}


# -----------------------------
# Helpers
# -----------------------------
def _resolve_level(level: Optional[str]) -> str:
    """Normalize level: prefer explicit arg, else env GEMCLI_LOG_LEVEL, else INFO."""
    val = (level or os.getenv("GEMCLI_LOG_LEVEL") or "INFO").strip().upper()
    if val not in _VALID_LEVELS:
        # tolerate common aliases
        aliases = {"WARN": "WARNING"}
        val = aliases.get(val, val)
    return val if val in _VALID_LEVELS else "INFO"


def _normalize_retention(value: Union[int, str]) -> Union[int, str]:
    """
    Accept:
      - int  → number of files (recommended for rolling)
      - "10 files" → coerced to 10
      - duration strings (e.g., "7 days", "1 week") → passed through
    """
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        s = value.strip().lower()
        m = re.match(r"^(\d+)\s*files?$", s)
        if m:
            return int(m.group(1))
        # otherwise let Loguru parse it as a duration
        return value
    return value


def _coerce_log_file(path_like: Optional[Union[str, Path]]) -> Optional[str]:
    """
    If None, default to a local project log under ./.logs/app.log.
    Ensures parent directory exists.
    """
    if path_like is None:
        log_path = Path.cwd() / ".logs" / "app.log"
    else:
        log_path = Path(path_like)
        # if they passed a directory, drop a default filename inside
        if log_path.suffix == "":
            log_path = log_path / "app.log"

    log_path.parent.mkdir(parents=True, exist_ok=True)
    return str(log_path)


def _remove_existing_sinks():
    global _SINK_IDS
    try:
        for sid in _SINK_IDS:
            logger.remove(sid)
    finally:
        _SINK_IDS = []


def _reconfigure(level: str):
    """(Re)create console & file sinks based on _LAST_CFG."""
    global _SINK_IDS
    _remove_existing_sinks()

    # Console sink
    if _LAST_CFG.get("console", True):
        _SINK_IDS.append(
            logger.add(
                sys.stderr,
                level=level,
                format=_DEFAULT_FMT,
                enqueue=_LAST_CFG.get("enqueue", True),
                backtrace=_LAST_CFG.get("backtrace", False),
                diagnose=_LAST_CFG.get("diagnose", False),
            )
        )

    # File sink
    log_file = _coerce_log_file(_LAST_CFG.get("log_file"))
    rotation = _LAST_CFG.get("rotation", "5 MB")
    retention = _normalize_retention(_LAST_CFG.get("retention", 10))

    _SINK_IDS.append(
        logger.add(
            log_file,
            level=level,
            format=_DEFAULT_FMT,
            rotation=rotation,         # e.g., "5 MB" or "1 day"
            retention=retention,       # int (files) or duration string
            encoding="utf-8",
            enqueue=_LAST_CFG.get("enqueue", True),
            backtrace=_LAST_CFG.get("backtrace", False),
            diagnose=_LAST_CFG.get("diagnose", False),
        )
    )


# -----------------------------
# Public API
# -----------------------------
def configure_logging(
    level: Optional[str] = None,
    *,
    console: bool = True,
    log_file: Optional[Union[str, Path]] = None,
    rotation: str = "5 MB",
    retention: Union[int, str] = 10,   # ✅ default fixed: keep last 10 files
    enqueue: bool = True,
    backtrace: bool = False,
    diagnose: bool = False,
) -> None:
    """
    Configure Loguru once at app start.

    Args:
        level: "DEBUG"/"INFO"/"WARNING"/... (env fallback: GEMCLI_LOG_LEVEL)
        console: also log to stderr
        log_file: file path or directory (directory -> writes 'app.log' inside)
        rotation: Loguru rotation policy (e.g., "5 MB", "1 day")
        retention: number of files (int) or duration string (e.g., "7 days")
        enqueue: use multiprocessing-safe queue
        backtrace/diagnose: enable Loguru's rich tracebacks (dev only)
    """
    _LAST_CFG.update(
        dict(
            console=console,
            log_file=log_file,
            rotation=rotation,
            retention=retention,
            enqueue=enqueue,
            backtrace=backtrace,
            diagnose=diagnose,
        )
    )
    _reconfigure(_resolve_level(level))


def set_level(level: str) -> None:
    """Change level at runtime."""
    _reconfigure(_resolve_level(level))


def enable_debug() -> None:
    """Shortcut to flip to DEBUG."""
    set_level("DEBUG")


def disable_debug() -> None:
    """Shortcut to flip to INFO."""
    set_level("INFO")


def current_config() -> dict:
    """Peek at the active base config (without dynamic sink IDs)."""
    return {
        **_LAST_CFG,
        "level": _resolve_level(None),
        "sinks": len(_SINK_IDS),
        "log_file": _coerce_log_file(_LAST_CFG.get("log_file")),
    }
