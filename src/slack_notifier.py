"""
slack_notifier.py
-----------------
Sends Whale alert messages to a Slack channel using the Bolt SDK.

Set these in .env:
  SLACK_BOT_TOKEN    - xoxb-...
  SLACK_APP_TOKEN    - xapp-... (for Socket Mode)
  SLACK_ALERT_CHANNEL - e.g. #whaletracker-alerts
"""

import logging
import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

SLACK_BOT_TOKEN: str = os.getenv("SLACK_BOT_TOKEN", "")
SLACK_ALERT_CHANNEL: str = os.getenv("SLACK_ALERT_CHANNEL", "#whaletracker-alerts")

# Lazy-init the Slack WebClient so the module is importable even when
# the SDK is not installed (e.g., during unit tests with mocks).
_client = None


def _get_client():
    global _client
    if _client is None:
        from slack_sdk import WebClient  # noqa: PLC0415
        if not SLACK_BOT_TOKEN:
            raise EnvironmentError(
                "SLACK_BOT_TOKEN is not set. Add it to your .env file."
            )
        _client = WebClient(token=SLACK_BOT_TOKEN)
    return _client


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def send_whale_alert(recommendation: dict[str, Any], channel: str | None = None) -> bool:
    """Post a formatted Whale signal alert to Slack.

    Args:
        recommendation: A single item from analysis_engine.build_recommendations().
        channel:        Override destination channel.

    Returns:
        True if message was sent successfully, False otherwise.
    """
    target_channel = channel or SLACK_ALERT_CHANNEL
    text = _format_recommendation(recommendation)

    try:
        _get_client().chat_postMessage(channel=target_channel, text=text, mrkdwn=True)
        logger.info("Alert sent for %s to %s", recommendation["ticker"], target_channel)
        return True
    except Exception as exc:
        logger.error("Slack send failed: %s", exc)
        return False


def send_rebalancing_digest(
    suggestions: list[dict[str, Any]],
    channel: str | None = None,
) -> bool:
    """Post a daily portfolio rebalancing digest.

    Args:
        suggestions: Output of portfolio_manager.suggest_rebalancing().
        channel:     Override destination channel.

    Returns:
        True if message was sent successfully.
    """
    if not suggestions:
        logger.info("No rebalancing suggestions to send.")
        return False

    target_channel = channel or SLACK_ALERT_CHANNEL
    lines = [":bar_chart: *WhaleTracker — Portfolio Rebalancing Digest*\n"]

    for s in suggestions:
        icon = ":arrow_up:" if s["action"] == "INCREASE" else ":arrow_down:"
        lines.append(
            f"{icon} *{s['sector']}*  "
            f"{s['current_weight']:.0%} → {s['target_weight']:.0%}  "
            f"| {s['rationale']}"
        )

    text = "\n".join(lines)
    try:
        _get_client().chat_postMessage(channel=target_channel, text=text, mrkdwn=True)
        return True
    except Exception as exc:
        logger.error("Slack digest send failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

RECOMMENDATION_ICONS = {
    "STRONG BUY": ":rocket:",
    "BUY":        ":green_heart:",
    "HOLD":       ":yellow_heart:",
    "SELL":       ":red_circle:",
}


def _format_recommendation(rec: dict[str, Any]) -> str:
    icon = RECOMMENDATION_ICONS.get(rec["recommendation"], ":white_circle:")
    whales = ", ".join(rec.get("supporting_whales", []))
    macro = f"\n> {rec['macro_note']}" if rec.get("macro_note") else ""

    return (
        f"{icon} *{rec['ticker']} — {rec['recommendation']}*\n"
        f"*Company:* {rec.get('company', 'N/A')}\n"
        f"*Signals:* {rec.get('signal_summary', 'N/A')}\n"
        f"*Whales:* {whales or 'N/A'}  "
        f"*Conviction:* {rec.get('conviction_score', 0)}"
        f"{macro}"
    )
