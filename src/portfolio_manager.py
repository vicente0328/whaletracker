"""
portfolio_manager.py
--------------------
Loads your personal holdings from my_portfolio.json, calculates current
sector weights, and suggests rebalancing actions based on Whale trends.
"""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PORTFOLIO_PATH = Path(__file__).parent.parent / "my_portfolio.json"

# Minimum drift before a rebalance suggestion is surfaced (5 pp)
REBALANCE_DRIFT_THRESHOLD = 0.05


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_portfolio() -> dict[str, Any]:
    """Load and return the contents of my_portfolio.json."""
    if not PORTFOLIO_PATH.exists():
        raise FileNotFoundError(
            f"Portfolio file not found at {PORTFOLIO_PATH}. "
            "Create it from the template or run the initialiser."
        )
    return json.loads(PORTFOLIO_PATH.read_text())


def get_current_sector_weights(portfolio: dict[str, Any]) -> dict[str, float]:
    """Calculate current sector weights by market value (uses avg_cost * quantity).

    Returns:
        {sector: weight_fraction}
    """
    holdings = portfolio.get("holdings", [])
    sector_value: dict[str, float] = {}

    for h in holdings:
        sector = h.get("sector", "Unknown")
        value = h.get("quantity", 0) * h.get("avg_cost", 0.0)
        sector_value[sector] = sector_value.get(sector, 0.0) + value

    total = sum(sector_value.values())
    if total == 0:
        return {}

    return {sector: val / total for sector, val in sector_value.items()}


def suggest_rebalancing(
    portfolio: dict[str, Any],
    sector_rotation_signals: dict[str, float],
) -> list[dict[str, Any]]:
    """Generate rebalancing suggestions by comparing current weights to
    target weights adjusted for Whale sector-rotation signals.

    Args:
        portfolio:               Output of load_portfolio().
        sector_rotation_signals: Output of analysis_engine.get_sector_rotation_signals().

    Returns:
        List of action dicts:
          {sector, current_weight, target_weight, action, rationale}
    """
    current_weights = get_current_sector_weights(portfolio)
    target_weights: dict[str, float] = portfolio.get("target_sector_weights", {})

    # Nudge target weights toward sectors with strong Whale inflows
    adjusted_targets = _adjust_targets_for_rotation(target_weights, sector_rotation_signals)

    suggestions = []
    all_sectors = set(current_weights) | set(adjusted_targets)

    for sector in all_sectors:
        current = current_weights.get(sector, 0.0)
        target = adjusted_targets.get(sector, 0.0)
        drift = target - current

        if abs(drift) < REBALANCE_DRIFT_THRESHOLD:
            continue  # Within tolerance — no action needed

        action = "INCREASE" if drift > 0 else "DECREASE"
        whale_flow = sector_rotation_signals.get(sector, 0)
        rationale = _build_rationale(sector, current, target, drift, whale_flow)

        suggestions.append({
            "sector": sector,
            "current_weight": round(current, 4),
            "target_weight": round(target, 4),
            "drift": round(drift, 4),
            "action": action,
            "rationale": rationale,
        })

    suggestions.sort(key=lambda s: abs(s["drift"]), reverse=True)
    return suggestions


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _adjust_targets_for_rotation(
    targets: dict[str, float],
    rotation_signals: dict[str, float],
) -> dict[str, float]:
    """Shift target weights by up to 3 pp based on Whale flow scores."""
    max_signal = max(rotation_signals.values(), default=1) or 1
    adjusted = dict(targets)

    for sector, score in rotation_signals.items():
        if sector in adjusted:
            nudge = (score / max_signal) * 0.03  # cap at ±3 pp
            adjusted[sector] = min(1.0, adjusted[sector] + nudge)

    # Re-normalise so weights still sum to 1.0
    total = sum(adjusted.values())
    if total > 0:
        adjusted = {s: w / total for s, w in adjusted.items()}

    return adjusted


def _build_rationale(
    sector: str,
    current: float,
    target: float,
    drift: float,
    whale_flow: float,
) -> str:
    direction = "increasing" if drift > 0 else "reducing"
    whale_note = ""
    if whale_flow > 3:
        whale_note = f" Whales show strong inflows into {sector}."
    elif whale_flow < 0:
        whale_note = f" Whales are rotating out of {sector}."

    return (
        f"Consider {direction} {sector} exposure from "
        f"{current:.0%} → {target:.0%}.{whale_note}"
    )
