"""
alert_daemon.py — WhaleTracker AI | Standalone Alert Daemon
------------------------------------------------------------
Runs the full scheduler (Form 4 real-time + whale refresh + daily digest)
INDEPENDENTLY of the Dash web application.

Usage:
    python alert_daemon.py

    # Background (keep running after terminal closes):
    nohup python alert_daemon.py > logs/daemon.log 2>&1 &

    # Check if running:
    ps aux | grep alert_daemon

    # Stop:
    kill $(cat logs/daemon.pid)   # if PID file written
    # OR: kill the PID from 'ps aux'

Requires:
    SLACK_BOT_TOKEN      — in .env or environment
    FORM4_WATCH_TICKERS  — comma-separated tickers to monitor (or ALERT_WATCHLIST)
    DATA_MODE=live       — recommended for real signal detection

This process can run on any server (VPS, Raspberry Pi, cloud instance)
completely separately from the web dashboard.
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import time
from pathlib import Path

# ── Bootstrap ──────────────────────────────────────────────────────────────────
# Load .env before importing project modules (they read env at import time)
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("alert_daemon")

# ── Validate required config ───────────────────────────────────────────────────
SLACK_TOKEN = os.getenv("SLACK_BOT_TOKEN", "")
if not SLACK_TOKEN:
    logger.error(
        "SLACK_BOT_TOKEN is not set — no alerts can be sent.\n"
        "Set it in your .env file and restart the daemon."
    )
    sys.exit(1)

F4_TICKERS = os.getenv("FORM4_WATCH_TICKERS", "") or os.getenv("ALERT_WATCHLIST", "")
if not F4_TICKERS:
    logger.warning(
        "Neither FORM4_WATCH_TICKERS nor ALERT_WATCHLIST is set.\n"
        "Real-time Form 4 alerts will not fire. "
        "Set at least one in .env (e.g. FORM4_WATCH_TICKERS=AAPL,NVDA,MSFT)."
    )

# ── PID file (optional, helps with process management) ────────────────────────
_PID_PATH = Path("logs/daemon.pid")
try:
    _PID_PATH.parent.mkdir(parents=True, exist_ok=True)
    _PID_PATH.write_text(str(os.getpid()))
except OSError:
    pass  # non-critical


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    from src.scheduler import start, stop, is_running  # noqa: PLC0415

    logger.info("=== WhaleTracker Alert Daemon starting ===")
    logger.info("Form 4 tickers : %s", F4_TICKERS or "(none configured)")
    logger.info("Refresh interval: %sh", os.getenv("REFRESH_INTERVAL_HOURS", "4"))
    logger.info("Form 4 interval : %smin", os.getenv("FORM4_REFRESH_MINUTES", "30"))
    logger.info("Daily digest    : %s:00 UTC", os.getenv("DAILY_DIGEST_HOUR", "8"))

    # Pass an empty app_data dict — the scheduler fetches its own fresh data on each cycle
    start({})

    # Graceful shutdown on SIGINT (Ctrl+C) or SIGTERM (kill / systemd stop)
    def _shutdown(sig: int, _frame: object) -> None:
        logger.info("Received signal %d — shutting down daemon …", sig)
        stop()
        try:
            _PID_PATH.unlink(missing_ok=True)
        except OSError:
            pass
        logger.info("=== Alert Daemon stopped ===")
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    logger.info("=== Alert Daemon running — press Ctrl+C to stop ===")
    while is_running():
        time.sleep(30)

    logger.info("Scheduler stopped unexpectedly — exiting.")


if __name__ == "__main__":
    main()
