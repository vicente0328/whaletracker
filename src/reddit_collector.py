"""
reddit_collector.py
-------------------
Reddit DD (Due Diligence) 수집기.

업보트 300+ · 본문 500자+ 조건을 충족하는 심층 분석글만 필터링하여
Claude claude-sonnet-4-6로 티커 추출 · 감성 분석 · 한국어 요약을 수행합니다.

환경변수:
  REDDIT_CLIENT_ID      Reddit OAuth 클라이언트 ID
  REDDIT_CLIENT_SECRET  Reddit OAuth 클라이언트 시크릿
  REDDIT_USER_AGENT     User-Agent 문자열 (예: "WhaleTracker/1.0 by /u/username")

미설정 시 MOCK_POSTS를 자동으로 반환합니다 (앱 크래시 없음).
Reddit REST API를 requests로 직접 호출 — praw 라이브러리 불필요.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ── 환경변수 ─────────────────────────────────────────────────────────────────
REDDIT_CLIENT_ID     = os.getenv("REDDIT_CLIENT_ID", "").strip()
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "").strip()
REDDIT_USER_AGENT    = os.getenv(
    "REDDIT_USER_AGENT",
    "WhaleTracker/1.0 by /u/whaletracker_ai",
)

# ── 상수 ─────────────────────────────────────────────────────────────────────
MIN_UPVOTES = 300       # DD 품질 필터: 최소 업보트
MIN_CHARS   = 500       # DD 품질 필터: 본문 최소 글자 수
MAX_POSTS   = 15        # 새로고침당 최대 포스트 수
CACHE_TTL   = 3600      # in-memory 캐시 TTL (초)

# (subreddit, DD_flair_required)
# WSB는 "DD" flair 필수 / 나머지는 길이 필터만 적용
SUBREDDITS: list[tuple[str, bool]] = [
    ("wallstreetbets",   True),
    ("stocks",           False),
    ("SecurityAnalysis", False),
]

# Reddit OAuth endpoints
_TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
_API_BASE  = "https://oauth.reddit.com"

# ── in-memory 캐시 ────────────────────────────────────────────────────────────
_cache:    list[dict[str, Any]] = []
_cache_ts: float                = 0.0

# ── Mock 데이터 ───────────────────────────────────────────────────────────────
#  5개 포스트: aligned_bull / hype_trap / divergence / hype_trap / aligned_bear
#  각각 mock sentiment 및 summary_ko 사전 설정 (Claude 엔리치먼트 불필요)
_now = int(time.time())

MOCK_POSTS: list[dict[str, Any]] = [
    {
        "id":          "mock_001",
        "title":       "NVDA DD: Why the AI infrastructure supercycle has 3+ years left — H100 backlog deep dive",
        "subreddit":   "wallstreetbets",
        "flair":       "DD",
        "upvotes":     12_400,
        "upvote_ratio": 0.94,
        "num_comments": 2847,
        "url":         "https://reddit.com/r/wallstreetbets/mock_001",
        "author":      "ape_alpha_dd",
        "created_utc": _now - 7_200,
        "body_excerpt": (
            "Everyone's talking about NVDA plateauing but the H100 allocation backlog "
            "through 2026 tells a different story. Data center capex from hyperscalers "
            "is accelerating, not decelerating. I've gone through every earnings call "
            "transcript and channel check available..."
        ),
        "tickers":    ["NVDA"],
        "sentiment":   0.82,
        "summary_ko": (
            "NVDA의 H100 수요가 2026년까지 지속될 것으로 분석되며, "
            "하이퍼스케일러의 데이터센터 투자 가속화가 핵심 동인으로 제시됩니다. "
            "단기 조정에도 불구하고 AI 인프라 사이클은 아직 초기 단계라는 주장입니다."
        ),
    },
    {
        "id":          "mock_002",
        "title":       "META is the most undervalued big tech stock right now — DCF + Threads growth analysis",
        "subreddit":   "stocks",
        "flair":       None,
        "upvotes":     4_800,
        "upvote_ratio": 0.88,
        "num_comments": 632,
        "url":         "https://reddit.com/r/stocks/mock_002",
        "author":      "fundamentals_ape",
        "created_utc": _now - 14_400,
        "body_excerpt": (
            "Running a DCF at 18x FCF for META feels wrong when you look at the Threads "
            "growth trajectory and Reality Labs finally approaching breakeven. "
            "The ad business alone justifies current valuation, with everything else "
            "essentially free..."
        ),
        "tickers":    ["META"],
        "sentiment":   0.68,
        "summary_ko": (
            "META의 현재 FCF 배수가 성장성 대비 저평가되어 있다는 DCF 분석이 제시됩니다. "
            "Threads 성장과 Reality Labs 손실 축소가 주요 촉매로 언급됩니다. "
            "광고 비즈니스 단독으로도 현재 시가총액을 정당화한다는 결론입니다."
        ),
    },
    {
        "id":          "mock_003",
        "title":       "TSLA bear case: margins won't recover — the China EV war changes everything permanently",
        "subreddit":   "wallstreetbets",
        "flair":       "DD",
        "upvotes":     7_200,
        "upvote_ratio": 0.72,
        "num_comments": 3841,
        "url":         "https://reddit.com/r/wallstreetbets/mock_003",
        "author":      "bears_are_right_",
        "created_utc": _now - 3_600,
        "body_excerpt": (
            "BYD's latest pricing move in Europe is not just a regional story. "
            "It's a structural margin compression event for TSLA globally. "
            "Auto gross margins falling from 25% to sub-16% and the Robotaxi "
            "thesis is years away from materializing..."
        ),
        "tickers":    ["TSLA"],
        "sentiment":  -0.71,
        "summary_ko": (
            "BYD의 유럽 가격 전략이 테슬라의 글로벌 마진 구조를 위협하고 있다는 분석입니다. "
            "자동차 부문 총이익률이 16% 이하로 하락할 가능성이 제시됩니다. "
            "중국 EV 업체들의 공세가 단기 이슈가 아닌 구조적 문제라는 결론입니다."
        ),
    },
    {
        "id":          "mock_004",
        "title":       "PLTR finally has the moat everyone doubted — AIP crosses enterprise adoption threshold",
        "subreddit":   "SecurityAnalysis",
        "flair":       None,
        "upvotes":     1_900,
        "upvote_ratio": 0.91,
        "num_comments": 218,
        "url":         "https://reddit.com/r/SecurityAnalysis/mock_004",
        "author":      "ai_investment_thesis",
        "created_utc": _now - 28_800,
        "body_excerpt": (
            "Palantir's AIP platform has crossed the enterprise adoption threshold. "
            "Net dollar retention above 120% and government contract win rate exceeding "
            "65% makes this a compounding machine. The bear case relied on no moat "
            "that no longer holds up..."
        ),
        "tickers":    ["PLTR"],
        "sentiment":   0.61,
        "summary_ko": (
            "팔란티어 AIP 플랫폼이 기업 도입 임계점을 넘어서며 모트를 구축했다는 분석입니다. "
            "순 달러 유지율 120% 이상과 정부 계약 수주율 65%가 핵심 지표로 제시됩니다. "
            "복리 성장 모델로서의 가치가 재평가받을 시점이라는 결론입니다."
        ),
    },
    {
        "id":          "mock_005",
        "title":       "INTC is a value trap dressed as a turnaround — why the foundry pivot will fail",
        "subreddit":   "stocks",
        "flair":       None,
        "upvotes":     2_300,
        "upvote_ratio": 0.79,
        "num_comments": 445,
        "url":         "https://reddit.com/r/stocks/mock_005",
        "author":      "chip_skeptic_99",
        "created_utc": _now - 21_600,
        "body_excerpt": (
            "Intel's foundry pivot requires $100B+ capex over 5 years with uncertain "
            "yield rates. Meanwhile TSMC continues to pull ahead on 2nm process nodes. "
            "The dividend cut already signaled management's own confidence level in the "
            "turnaround thesis..."
        ),
        "tickers":    ["INTC"],
        "sentiment":  -0.58,
        "summary_ko": (
            "인텔의 파운드리 전환에 5년간 1000억 달러 이상 투자가 필요하나 "
            "수율 개선이 불확실하다는 비관적 분석입니다. "
            "배당 삭감이 경영진의 자신감 부재를 시사한다는 결론입니다."
        ),
    },
]


# ── Reddit OAuth ──────────────────────────────────────────────────────────────

def _fetch_reddit_token() -> str:
    """client_credentials 방식으로 Reddit OAuth 토큰을 발급합니다."""
    resp = requests.post(
        _TOKEN_URL,
        data={"grant_type": "client_credentials"},
        auth=(REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET),
        headers={"User-Agent": REDDIT_USER_AGENT},
        timeout=10,
    )
    resp.raise_for_status()
    token = resp.json().get("access_token", "")
    if not token:
        raise RuntimeError("Reddit OAuth: access_token 없음")
    return token


def _fetch_raw_posts(token: str) -> list[dict[str, Any]]:
    """각 서브레딧에서 hot 게시글을 가져와 DD 필터를 적용합니다."""
    headers = {
        "Authorization": f"Bearer {token}",
        "User-Agent":    REDDIT_USER_AGENT,
    }
    posts: list[dict[str, Any]] = []

    for sub_name, require_flair in SUBREDDITS:
        try:
            url  = f"{_API_BASE}/r/{sub_name}/hot.json?limit=50"
            resp = requests.get(url, headers=headers, timeout=12)
            resp.raise_for_status()
            items = resp.json().get("data", {}).get("children", [])

            for item in items:
                d     = item.get("data", {})
                flair = (d.get("link_flair_text") or "").strip()
                score = int(d.get("score", 0))
                body  = d.get("selftext", "") or ""

                # DD 품질 필터
                if score < MIN_UPVOTES:
                    continue
                if require_flair and flair.upper() not in {"DD", "DUE DILIGENCE"}:
                    continue
                if len(body) < MIN_CHARS:
                    continue

                posts.append({
                    "id":           d.get("id", ""),
                    "title":        d.get("title", ""),
                    "subreddit":    sub_name,
                    "flair":        flair or None,
                    "upvotes":      score,
                    "upvote_ratio": float(d.get("upvote_ratio", 0)),
                    "num_comments": int(d.get("num_comments", 0)),
                    "url":          "https://reddit.com" + d.get("permalink", ""),
                    "author":       d.get("author", ""),
                    "created_utc":  int(d.get("created_utc", time.time())),
                    "body_excerpt": body[:600],
                    "tickers":      [],
                    "sentiment":    0.0,
                    "summary_ko":   "",
                })

                if len(posts) >= MAX_POSTS:
                    break

        except Exception as exc:
            logger.warning("[reddit] r/%s 패치 실패: %s", sub_name, exc)

    posts.sort(key=lambda p: p["upvotes"], reverse=True)
    return posts[:MAX_POSTS]


# ── Claude 엔리치먼트 ─────────────────────────────────────────────────────────

def enrich_with_claude(posts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Claude claude-sonnet-4-6 배치 호출로 모든 포스트에 티커/감성/한국어요약을 추가합니다.
    실패 시 원본 posts 그대로 반환 (non-blocking).
    """
    if not posts:
        return posts

    try:
        import anthropic  # noqa: PLC0415

        items = [
            {
                "idx":     i,
                "title":   p["title"],
                "excerpt": p.get("body_excerpt", "")[:500],
            }
            for i, p in enumerate(posts)
        ]

        prompt = (
            "다음 Reddit 주식 투자 분석 게시물들을 분석하세요.\n\n"
            "각 항목에 대해 JSON 배열로 반환:\n"
            '[{"idx":0,"tickers":["NVDA"],"sentiment":0.8,'
            '"summary_ko":"핵심 투자 논리 2~3문장 한국어 요약"}]\n\n'
            "규칙:\n"
            "- sentiment: -1.0(매우 부정적) ~ +1.0(매우 긍정적)\n"
            "- tickers: $기호 없이 대문자, 최대 3개, 언급 없으면 []\n"
            "- summary_ko: 투자 논리의 핵심만 2~3문장, 반드시 한국어\n\n"
            f"게시물:\n{json.dumps(items, ensure_ascii=False)}"
        )

        client = anthropic.Anthropic()
        msg    = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2_000,
            system=(
                "당신은 월스트리트 전문 애널리스트입니다. "
                "주식 시장 게시물을 분석해 핵심 투자 논리를 한국어로 요약합니다."
            ),
            messages=[{"role": "user", "content": prompt}],
        )

        raw = msg.content[0].text.strip()
        # 마크다운 펜스 제거
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$",           "", raw)

        results = json.loads(raw)
        lookup  = {r["idx"]: r for r in results}

        for i, post in enumerate(posts):
            if i in lookup:
                r = lookup[i]
                post["tickers"]    = r.get("tickers", [])
                post["sentiment"]  = float(r.get("sentiment", 0.0))
                post["summary_ko"] = r.get("summary_ko", "")

    except Exception as exc:
        logger.error("[reddit] Claude 엔리치먼트 실패 (포스트는 표시됨): %s", exc)

    return posts


# ── 공개 API ──────────────────────────────────────────────────────────────────

def fetch_dd_posts(force: bool = False) -> list[dict[str, Any]]:
    """
    DD 게시글 목록을 반환합니다 (1시간 캐시).

    - REDDIT_CLIENT_ID 미설정 → MOCK_POSTS 즉시 반환
    - Reddit API 오류 → MOCK_POSTS 폴백
    - force=True → 캐시 무시하고 재패치

    Args:
        force: True면 캐시를 무시하고 강제로 재패치합니다.

    Returns:
        엔리치먼트된 DD 게시글 목록 (tickers, sentiment, summary_ko 포함).
    """
    global _cache, _cache_ts

    # Mock 모드
    if not REDDIT_CLIENT_ID or not REDDIT_CLIENT_SECRET:
        logger.info("[reddit] 자격증명 없음 — Mock 데이터 반환")
        return MOCK_POSTS

    # 캐시 히트
    if not force and _cache and (time.time() - _cache_ts < CACHE_TTL):
        logger.debug("[reddit] 캐시 히트 (%d개)", len(_cache))
        return _cache

    try:
        token = _fetch_reddit_token()
        raw   = _fetch_raw_posts(token)
        posts = enrich_with_claude(raw) if raw else []
    except Exception as exc:
        logger.error("[reddit] 패치 실패 — Mock 폴백: %s", exc)
        return MOCK_POSTS

    _cache    = posts
    _cache_ts = time.time()
    logger.info("[reddit] %d개 DD 포스트 패치 완료", len(posts))
    return _cache
