"""
market_events.py
----------------
Tracks high-impact scheduled market events and fires Slack alerts a
configurable number of days before each event.

Supported event types:
  FED_MEETING      — FOMC rate decision (8 per year)
  CPI_RELEASE      — US CPI inflation data
  JOBS_REPORT      — US Non-Farm Payrolls
  EARNINGS_SEASON  — Start of major earnings season
  ELECTION         — US mid-term / presidential election
  DEBT_CEILING     — US debt ceiling deadline
  CUSTOM           — Any user-defined date

Configuration (.env):
  EVENT_ALERT_DAYS_BEFORE   — comma-separated lead times, e.g. "7,1"
                              Alerts fire 7 days before AND 1 day before.
                              Default: "7,1"
  EVENT_CHECK_HOUR          — UTC hour to run the daily check (default: 7)
"""

import logging
import os
from datetime import date, datetime, timedelta, timezone
from typing import Any

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Days-before values at which to send an alert (e.g. 7 days out + 1 day out)
_RAW_DAYS = os.getenv("EVENT_ALERT_DAYS_BEFORE", "7,1")
ALERT_DAYS_BEFORE: list[int] = sorted(
    {int(d.strip()) for d in _RAW_DAYS.split(",") if d.strip().isdigit()},
    reverse=True,
)

EVENT_CHECK_HOUR: int = int(os.getenv("EVENT_CHECK_HOUR", "7"))

# ---------------------------------------------------------------------------
# Event calendar
# ---------------------------------------------------------------------------
# Each event: {date, type, title, description, impact}
# impact: "HIGH" | "MEDIUM"
# Dates are updated quarterly — add new events as they are announced.
# ---------------------------------------------------------------------------

MARKET_EVENTS: list[dict[str, Any]] = [
    # ── 2025 FOMC meetings ─────────────────────────────────────────────────
    {
        "date": date(2025, 3, 19),
        "type": "FED_MEETING",
        "title": "FOMC Rate Decision — March 2025",
        "description": (
            "연방준비제도(Fed)가 기준금리를 발표합니다. "
            "금리 결정은 주식·채권·달러 등 전 자산군에 즉각적인 영향을 미칩니다. "
            "현재 시장은 동결 또는 25bp 인하를 주목하고 있습니다."
        ),
        "impact": "HIGH",
    },
    {
        "date": date(2025, 5, 7),
        "type": "FED_MEETING",
        "title": "FOMC Rate Decision — May 2025",
        "description": (
            "5월 FOMC 회의 결과가 발표됩니다. "
            "PCE 인플레이션 추이와 고용 데이터가 핵심 변수입니다."
        ),
        "impact": "HIGH",
    },
    {
        "date": date(2025, 6, 18),
        "type": "FED_MEETING",
        "title": "FOMC Rate Decision — June 2025",
        "description": (
            "6월 FOMC는 분기별 경제전망(SEP)과 점도표(Dot Plot)도 함께 발표합니다. "
            "연간 통화정책 방향을 가늠할 수 있는 중요한 회의입니다."
        ),
        "impact": "HIGH",
    },
    {
        "date": date(2025, 7, 30),
        "type": "FED_MEETING",
        "title": "FOMC Rate Decision — July 2025",
        "description": (
            "7월 FOMC 결과 발표. SEP는 없으며 성명서·파월 의장 기자회견이 핵심입니다."
        ),
        "impact": "HIGH",
    },
    {
        "date": date(2025, 9, 17),
        "type": "FED_MEETING",
        "title": "FOMC Rate Decision — September 2025",
        "description": (
            "9월 FOMC는 분기별 경제전망(SEP) 및 점도표를 포함합니다. "
            "연말 정책 경로의 핵심 지표가 공개됩니다."
        ),
        "impact": "HIGH",
    },
    {
        "date": date(2025, 10, 29),
        "type": "FED_MEETING",
        "title": "FOMC Rate Decision — October 2025",
        "description": "10월 FOMC 결과 발표. 4분기 경제 지표와 연말 정책 방향이 주목됩니다.",
        "impact": "HIGH",
    },
    {
        "date": date(2025, 12, 10),
        "type": "FED_MEETING",
        "title": "FOMC Rate Decision — December 2025",
        "description": (
            "연말 FOMC. 2025년 마지막 금리 결정이자 2026년 통화정책 방향을 제시합니다. "
            "SEP 및 점도표 포함."
        ),
        "impact": "HIGH",
    },
    # ── 2026 FOMC meetings ─────────────────────────────────────────────────
    {
        "date": date(2026, 1, 28),
        "type": "FED_MEETING",
        "title": "FOMC Rate Decision — January 2026",
        "description": "2026년 첫 FOMC. 신년 경제 진단 및 금리 방향이 결정됩니다.",
        "impact": "HIGH",
    },
    {
        "date": date(2026, 3, 18),
        "type": "FED_MEETING",
        "title": "FOMC Rate Decision — March 2026",
        "description": "3월 FOMC. SEP·점도표 포함, 2026년 통화정책 경로를 갱신합니다.",
        "impact": "HIGH",
    },
    {
        "date": date(2026, 5, 6),
        "type": "FED_MEETING",
        "title": "FOMC Rate Decision — May 2026",
        "description": "5월 FOMC 결과 발표.",
        "impact": "HIGH",
    },
    # ── US CPI releases (approximate — 2nd Tuesday of each month) ──────────
    {
        "date": date(2025, 3, 12),
        "type": "CPI_RELEASE",
        "title": "US CPI — February 2025",
        "description": (
            "미국 2월 소비자물가지수(CPI)가 발표됩니다. "
            "예상치 대비 서프라이즈 여부에 따라 채권·주식시장이 크게 반응할 수 있습니다."
        ),
        "impact": "HIGH",
    },
    {
        "date": date(2025, 4, 10),
        "type": "CPI_RELEASE",
        "title": "US CPI — March 2025",
        "description": "미국 3월 CPI 발표. 1분기 인플레이션 추이를 확정합니다.",
        "impact": "HIGH",
    },
    {
        "date": date(2025, 5, 13),
        "type": "CPI_RELEASE",
        "title": "US CPI — April 2025",
        "description": "미국 4월 CPI 발표. 5월 FOMC 이후 첫 물가 확인입니다.",
        "impact": "HIGH",
    },
    {
        "date": date(2025, 6, 11),
        "type": "CPI_RELEASE",
        "title": "US CPI — May 2025",
        "description": "미국 5월 CPI 발표. 6월 FOMC 전 핵심 데이터입니다.",
        "impact": "HIGH",
    },
    {
        "date": date(2025, 7, 10),
        "type": "CPI_RELEASE",
        "title": "US CPI — June 2025 (반기 합산)",
        "description": "미국 6월 CPI 발표. 상반기 인플레이션 추이를 마무리합니다.",
        "impact": "HIGH",
    },
    {
        "date": date(2025, 9, 10),
        "type": "CPI_RELEASE",
        "title": "US CPI — August 2025",
        "description": "미국 8월 CPI 발표. 9월 FOMC 전 마지막 주요 물가 지표입니다.",
        "impact": "HIGH",
    },
    {
        "date": date(2025, 12, 10),
        "type": "CPI_RELEASE",
        "title": "US CPI — November 2025",
        "description": "미국 11월 CPI. 12월 FOMC 직전 발표되는 핵심 인플레이션 지표입니다.",
        "impact": "HIGH",
    },
    # ── Non-Farm Payrolls (first Friday of each month) ─────────────────────
    {
        "date": date(2025, 3, 7),
        "type": "JOBS_REPORT",
        "title": "Non-Farm Payrolls — February 2025",
        "description": (
            "미국 2월 비농업 고용 지표가 발표됩니다. "
            "고용 시장 냉각 여부가 Fed의 금리 인하 타이밍을 좌우합니다."
        ),
        "impact": "HIGH",
    },
    {
        "date": date(2025, 4, 4),
        "type": "JOBS_REPORT",
        "title": "Non-Farm Payrolls — March 2025",
        "description": "미국 3월 비농업 고용 지표 발표.",
        "impact": "HIGH",
    },
    {
        "date": date(2025, 5, 2),
        "type": "JOBS_REPORT",
        "title": "Non-Farm Payrolls — April 2025",
        "description": "미국 4월 비농업 고용 지표 발표. 5월 FOMC 직전 발표입니다.",
        "impact": "HIGH",
    },
    {
        "date": date(2025, 9, 5),
        "type": "JOBS_REPORT",
        "title": "Non-Farm Payrolls — August 2025",
        "description": "미국 8월 비농업 고용 지표. 9월 FOMC 전 핵심 고용 데이터입니다.",
        "impact": "HIGH",
    },
    # ── Earnings seasons ───────────────────────────────────────────────────
    {
        "date": date(2025, 4, 11),
        "type": "EARNINGS_SEASON",
        "title": "Q1 2025 Earnings Season 시작",
        "description": (
            "S&P 500 기업들의 2025년 1분기 실적 발표 시즌이 시작됩니다. "
            "JPMorgan, Wells Fargo 등 대형 금융주가 선두를 끊습니다. "
            "EPS 서프라이즈와 가이던스가 섹터 로테이션을 촉발할 수 있습니다."
        ),
        "impact": "MEDIUM",
    },
    {
        "date": date(2025, 7, 11),
        "type": "EARNINGS_SEASON",
        "title": "Q2 2025 Earnings Season 시작",
        "description": (
            "S&P 500 기업들의 2분기 실적 시즌. "
            "AI 투자 ROI, 소비자 지출 동향, 마진 추이가 핵심 관전 포인트입니다."
        ),
        "impact": "MEDIUM",
    },
    {
        "date": date(2025, 10, 10),
        "type": "EARNINGS_SEASON",
        "title": "Q3 2025 Earnings Season 시작",
        "description": (
            "3분기 실적 시즌. 연말 소비 시즌을 앞둔 소매·기술주 가이던스가 주목됩니다."
        ),
        "impact": "MEDIUM",
    },
    {
        "date": date(2026, 1, 9),
        "type": "EARNINGS_SEASON",
        "title": "Q4 2025 Earnings Season 시작",
        "description": (
            "2025년 연간 실적 및 2026년 전망을 확인하는 핵심 실적 시즌입니다."
        ),
        "impact": "MEDIUM",
    },
    # ── US Elections ───────────────────────────────────────────────────────
    {
        "date": date(2026, 11, 3),
        "type": "ELECTION",
        "title": "미국 중간선거 (Midterm Elections) 2026",
        "description": (
            "미국 하원 전체 435석, 상원 1/3 의석을 결정하는 중간선거입니다. "
            "선거 결과에 따라 재정·규제 정책이 바뀌며 방산·에너지·헬스케어 등 "
            "정책 민감 섹터에 큰 영향을 미칩니다. "
            "역사적으로 선거 후 12개월 S&P 500은 평균 +17% 상승했습니다."
        ),
        "impact": "HIGH",
    },
    # ── Jackson Hole Symposium ─────────────────────────────────────────────
    {
        "date": date(2025, 8, 21),
        "type": "FED_MEETING",
        "title": "Jackson Hole 경제 심포지엄 2025",
        "description": (
            "미국 와이오밍주 잭슨홀에서 열리는 연준 연례 심포지엄입니다. "
            "파월 의장의 연설은 금리 정책 방향에 대한 강력한 시그널을 제공합니다. "
            "2022년 잭슨홀에서 공격적 금리인상을 예고하며 시장이 급락한 전례가 있습니다."
        ),
        "impact": "HIGH",
    },
    # ── US Debt Ceiling ────────────────────────────────────────────────────
    {
        "date": date(2025, 7, 31),
        "type": "DEBT_CEILING",
        "title": "미국 부채 한도 협상 데드라인 (예상)",
        "description": (
            "미국 연방 부채 한도 유예 기간 만료 예정일입니다 (의회 협상에 따라 변동). "
            "X-date 근접 시 단기 국채 금리 급등, 신용등급 우려, 위험자산 회피가 나타날 수 있습니다. "
            "2023년 S&P 신용등급 강등 사례를 참고하세요."
        ),
        "impact": "HIGH",
    },
]


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def get_upcoming_events(days_ahead: int = 30) -> list[dict[str, Any]]:
    """Return events occurring within the next `days_ahead` days, sorted by date."""
    today = date.today()
    cutoff = today + timedelta(days=days_ahead)
    return sorted(
        [e for e in MARKET_EVENTS if today <= e["date"] <= cutoff],
        key=lambda e: e["date"],
    )


def get_events_due_for_alert() -> list[tuple[dict[str, Any], int]]:
    """Return (event, days_before) pairs for events that should fire an alert today.

    An alert fires when today is exactly ALERT_DAYS_BEFORE[n] days before the event date.
    Returns a list of (event_dict, days_remaining) tuples.
    """
    today = date.today()
    due: list[tuple[dict[str, Any], int]] = []
    for event in MARKET_EVENTS:
        delta = (event["date"] - today).days
        if delta in ALERT_DAYS_BEFORE:
            due.append((event, delta))
    return due
