"""
news_settings.py
----------------
Shared in-memory news subscription settings used by both the Dash app
(callback thread) and the APScheduler background thread.

Why: Railway's ephemeral filesystem wipes daily_news_sub.json on every
redeploy. By keeping an authoritative in-memory copy that gets synced from
the browser localStorage on every page load, the scheduler always has the
correct settings regardless of filesystem state.

Public API:
    get()        → copy of current settings dict
    put(d)       → merge d into current settings, persist to file
    DEFAULTS     → default settings dict
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Canonical path for file-based persistence (same location used by app.py)
_FILE: Path = Path(__file__).parent.parent / "daily_news_sub.json"

DEFAULTS: dict = {
    "enabled":  False,
    "hour_utc": 23,       # default 08:00 KST = 23:00 UTC previous day
    "timezone": "KST",
    "topics":   [],
}

# Authoritative in-memory state — updated by Dash callbacks and read by scheduler
_current: dict = dict(DEFAULTS)

# Try to pre-populate from file (survives app restarts when file exists)
try:
    if _FILE.exists():
        _from_file: dict = json.loads(_FILE.read_text())
        _current.update(_from_file)
        logger.info(
            "[news_settings] Loaded from file — enabled=%s, hour_utc=%s",
            _current.get("enabled"), _current.get("hour_utc"),
        )
except Exception as _exc:
    logger.debug("[news_settings] Could not load from file: %s", _exc)


def get() -> dict:
    """Return a copy of the current subscription settings."""
    return dict(_current)


def put(settings: dict, *, skip_if_disabled_override: bool = False) -> None:
    """Merge `settings` into in-memory state and persist to file.

    Args:
        settings: dict with any subset of DEFAULTS keys.
        skip_if_disabled_override: when True, don't overwrite an already-enabled
            subscription with disabled defaults (prevents initial-render race).
    """
    if skip_if_disabled_override:
        # Don't stomp an existing enabled subscription with defaults
        if _current.get("enabled") and not settings.get("enabled", False):
            # Only update non-enabled fields if the incoming dict looks like defaults
            fields_to_skip = {"enabled"}
            safe_update = {k: v for k, v in settings.items() if k not in fields_to_skip}
            _current.update(safe_update)
            logger.debug("[news_settings] skip_if_disabled_override — kept enabled=True")
            _persist()
            return

    _current.update(settings)
    logger.debug("[news_settings] Updated — enabled=%s, hour_utc=%s",
                 _current.get("enabled"), _current.get("hour_utc"))
    _persist()


def _persist() -> None:
    try:
        _FILE.write_text(json.dumps(_current))
    except Exception as exc:
        logger.warning("[news_settings] File write failed: %s", exc)
