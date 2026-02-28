"""
app.py — WhaleTracker AI | Dash Dashboard
------------------------------------------
Run locally:  python app.py
Production:   gunicorn app:server --bind 0.0.0.0:$PORT
"""

import os
from datetime import datetime

from dash import Dash, html, dcc, Input, Output, State, ctx
import plotly.graph_objects as go
from dotenv import load_dotenv

from src.data_collector import (
    fetch_all_whale_filings,
    fetch_13dg_filings,
    fetch_form4_filings,
    fetch_nport_filings,
    WHALE_TIERS,
)
from src.analysis_engine import (
    build_recommendations,
    get_sector_rotation_signals,
    get_insider_sentiment,
)
from src.portfolio_manager import load_portfolio, suggest_rebalancing, get_current_sector_weights
from src.macro_collector import fetch_macro_indicators
from src.news_collector import fetch_market_news

load_dotenv()
DATA_MODE = os.getenv("DATA_MODE", "mock")

# ── DATA (loaded once at startup) ──────────────────────────────────────────────
filings          = fetch_all_whale_filings()

# In live mode, query Form 4 for every ticker found in the 13F holdings
_live_tickers = list({h["ticker"] for holds in filings.values() for h in holds})

activist         = fetch_13dg_filings()
insiders         = fetch_form4_filings(_live_tickers)
nport            = fetch_nport_filings()

recommendations  = build_recommendations(
    filings,
    activist_filings=activist,
    insider_filings=insiders,
    nport_filings=nport,
)
rotation         = get_sector_rotation_signals(filings)
insider_summary  = get_insider_sentiment(insiders)
portfolio        = load_portfolio()
rebalancing      = suggest_rebalancing(portfolio, rotation)
current_weights  = get_current_sector_weights(portfolio)
macro_data       = fetch_macro_indicators()
market_news      = fetch_market_news(5)

# ── DESIGN TOKENS ──────────────────────────────────────────────────────────────
C = {
    "bg":     "0D0F14",  "card":   "161922",  "card2":  "1E2130",
    "text":   "E8ECF0",  "muted":  "8892A4",  "border": "ffffff12",
    "green":  "00D09C",  "red":    "FF4757",  "blue":   "4B7BE5",
    "amber":  "FFB800",  "purple": "A78BFA",
}

SIG = {
    # 13F signals
    "NEW_ENTRY":           {"color": f"#{C['blue']}",   "label": "NEW ENTRY"},
    "AGGRESSIVE_BUY":      {"color": f"#{C['green']}",  "label": "AGG. BUY"},
    "HIGH_CONCENTRATION":  {"color": f"#{C['amber']}",  "label": "HIGH CONC"},
    "HOLD":                {"color": "#4A5568",           "label": "HOLD"},
    # SC 13D/G signals
    "ACTIVIST_STAKE":      {"color": f"#{C['red']}",    "label": "ACTIVIST"},
    "LARGE_PASSIVE_STAKE": {"color": f"#{C['purple']}", "label": "13G STAKE"},
    # Form 4 signals
    "INSIDER_BUY":         {"color": f"#{C['green']}",  "label": "INSIDER BUY"},
    "INSIDER_SELL":        {"color": f"#{C['red']}",    "label": "INSIDER SELL"},
    "PLANNED_SELL":        {"color": f"#{C['muted']}",  "label": "10b5-1 SELL"},
    # N-PORT signals
    "FUND_ACCUMULATION":   {"color": "#20B2AA",           "label": "FUND ACCUM"},
    "FUND_LIQUIDATION":    {"color": "#FF8C00",            "label": "FUND SELL"},
}

REC = {
    "STRONG BUY": {"color": f"#{C['green']}", "icon": "🚀"},
    "BUY":        {"color": "#1DB954",         "icon": "↑"},
    "HOLD":       {"color": f"#{C['amber']}", "icon": "→"},
    "SELL":       {"color": f"#{C['red']}",   "icon": "↓"},
}

PALETTE = [f"#{C['blue']}", f"#{C['green']}", f"#{C['amber']}",
           f"#{C['purple']}", f"#{C['red']}", "#20B2AA", "#FF8C00", "#9B59B6"]

# ── DERIVED METRICS ────────────────────────────────────────────────────────────
_POSITIVE_SIGNALS = {"NEW_ENTRY", "AGGRESSIVE_BUY", "HIGH_CONCENTRATION",
                     "ACTIVIST_STAKE", "LARGE_PASSIVE_STAKE",
                     "INSIDER_BUY", "FUND_ACCUMULATION"}
active_signals = (
    sum(1 for holds in filings.values()
        for h in holds if h.get("signal", "HOLD") in _POSITIVE_SIGNALS)
    + sum(1 for f in activist.values() if f.get("signal") in _POSITIVE_SIGNALS)
    + sum(1 for txs in insiders.values()
          for t in txs if t.get("signal") in _POSITIVE_SIGNALS)
    + sum(1 for holds in nport.values()
          for h in holds if h.get("signal") in _POSITIVE_SIGNALS)
)
live_whales = len([w for w, h in filings.items() if h])
port_value  = sum(
    h.get("quantity", 0) * h.get("avg_cost", 0.0)
    for h in portfolio.get("holdings", [])
)
top_rec    = recommendations[0] if recommendations else {}
mode_color = C["amber"] if DATA_MODE == "mock" else C["green"]
mode_label = "MOCK DATA" if DATA_MODE == "mock" else "● LIVE"
timestamp  = datetime.now().strftime("%b %d, %Y · %H:%M")


# ── PLOTLY HELPERS ─────────────────────────────────────────────────────────────
def plotly_base(**kwargs) -> dict:
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=f"#{C['text']}"),
        margin=dict(l=0, r=0, t=36, b=0),
        **kwargs,
    )


# ── COMPONENT BUILDERS ─────────────────────────────────────────────────────────
def kpi_card(label, value, sub, accent):
    return html.Div([
        html.Div("◈", style={
            "position": "absolute", "right": "12px", "top": "50%",
            "transform": "translateY(-50%)", "fontSize": "2.8rem",
            "opacity": "0.04", "color": f"#{accent}", "fontWeight": "900",
        }),
        html.Div(label, className="kpi-label"),
        html.Div(value, className="kpi-value"),
        html.Div(sub,   className="kpi-sub"),
    ], className="kpi-card", style={"borderLeft": f"3px solid #{accent}"})


def signal_badge(signal: str):
    si = SIG.get(signal, SIG["HOLD"])
    return html.Span(si["label"], style={
        "background": f"{si['color']}18", "color": si["color"],
        "border": f"1px solid {si['color']}44", "borderRadius": "4px",
        "padding": "1px 7px", "fontSize": "0.6rem", "fontWeight": "700",
        "letterSpacing": "0.3px", "marginRight": "4px", "display": "inline-block",
    })


def holding_card(h: dict):
    sig  = h.get("signal", "HOLD")
    info = SIG.get(sig, SIG["HOLD"])
    val  = h.get("value_usd", 0)
    pct  = h.get("portfolio_pct", 0) * 100
    val_str = f"${val/1e9:.1f}B" if val >= 1e9 else f"${val/1e6:.0f}M"

    return html.Div([
        html.Div([
            html.Span(h["ticker"], className="holding-ticker"),
            html.Span(info["label"], style={
                "background": f"{info['color']}18", "color": info["color"],
                "borderRadius": "4px", "padding": "1px 7px",
                "fontSize": "0.6rem", "fontWeight": "700",
            }),
        ], className="holding-header"),
        html.Div(h.get("company", ""), className="holding-company"),
        html.Div([
            html.Div([
                html.Div("Value",  className="stat-label"),
                html.Div(val_str,  className="stat-value"),
            ]),
            html.Div([
                html.Div("Portfolio", className="stat-label"),
                html.Div(f"{pct:.1f}%", className="stat-value",
                         style={"color": info["color"]}),
            ], style={"textAlign": "right"}),
        ], className="holding-stats"),
    ], className="holding-card",
       style={"borderTop": f"2px solid {info['color']}"})


def rec_card(r: dict):
    rec_key  = r["recommendation"]
    ri       = REC.get(rec_key, {"color": "#4A5568", "icon": "?"})
    score    = r.get("conviction_score", 0)
    bar_pct  = min(100, int(score / 12 * 100))
    whales   = " · ".join(r.get("supporting_whales", []))
    sig_list = [s.strip() for s in (r.get("signal_summary") or "").split(",") if s.strip()]
    macro    = r.get("macro_note", "")

    return html.Div([
        # Ticker + rec badge
        html.Div([
            html.Div([
                html.Div(r["ticker"], className="rec-ticker"),
                html.Div(r.get("company", ""), className="rec-company"),
            ]),
            html.Div([
                html.Div(ri["icon"], style={"fontSize": "1.1rem", "lineHeight": "1.2"}),
                html.Div(rec_key, style={
                    "fontSize": "0.6rem", "fontWeight": "700",
                    "color": ri["color"], "marginTop": "1px", "whiteSpace": "nowrap",
                }),
            ], style={
                "background": f"{ri['color']}1A", "border": f"1px solid {ri['color']}55",
                "borderRadius": "9px", "padding": "5px 11px",
                "textAlign": "center", "minWidth": "80px",
            }),
        ], className="rec-header"),

        # Conviction bar
        html.Div([
            html.Div([
                html.Span("Conviction", className="stat-label"),
                html.Span(f"{score} / 12",
                          style={"fontSize": "0.65rem", "fontWeight": "700", "color": ri["color"]}),
            ], className="conviction-row"),
            html.Div(
                html.Div(style={
                    "width": f"{bar_pct}%", "height": "100%",
                    "background": f"linear-gradient(90deg,{ri['color']}88,{ri['color']})",
                    "borderRadius": "3px",
                }),
                className="conviction-track",
            ),
        ], style={"marginBottom": "0.7rem"}),

        # Signal badges
        html.Div([signal_badge(s) for s in sig_list], style={"marginBottom": "0.6rem"}),

        # Whale footer
        html.Div(f"🐋 {whales or '—'}", className="rec-footer"),

        # Macro note
        html.Div(f"⚡ {macro}",
                 style={"fontSize": "0.65rem", "color": f"#{C['amber']}", "marginTop": "0.45rem"}
                 ) if macro else None,
    ], className="rec-card")


def rebalancing_card(s: dict):
    is_up  = s["action"] == "INCREASE"
    ac     = f"#{C['green']}" if is_up else f"#{C['red']}"
    arrow  = "↑" if is_up else "↓"
    drift  = abs(s["drift"] * 100)

    return html.Div([
        html.Div([
            html.Span(s["sector"], className="reb-sector"),
            html.Span(f"{arrow} {s['action']}", style={
                "background": f"{ac}18", "color": ac,
                "borderRadius": "5px", "padding": "2px 9px",
                "fontSize": "0.68rem", "fontWeight": "700",
            }),
        ], className="reb-header"),
        html.Div([
            html.Div([html.Div("Current", className="stat-label"),
                      html.Div(f"{s['current_weight']*100:.1f}%", className="stat-value")]),
            html.Div("→", style={"color": f"#{C['muted']}", "paddingBottom": "2px"}),
            html.Div([html.Div("Target", className="stat-label"),
                      html.Div(f"{s['target_weight']*100:.1f}%", className="stat-value",
                               style={"color": ac})]),
            html.Div([html.Div("Drift", className="stat-label", style={"textAlign": "right"}),
                      html.Div(f"{drift:.1f}pp", className="stat-value",
                               style={"color": ac, "textAlign": "right"})],
                     style={"marginLeft": "auto"}),
        ], className="reb-stats"),
        html.Div(s["rationale"], className="reb-rationale"),
    ], className="reb-card", style={"borderLeft": f"3px solid {ac}"})


# ── TAB CONTENT BUILDERS ───────────────────────────────────────────────────────
def build_whale_tab():
    sections = []

    # Sector rotation chart
    if rotation:
        sectors = list(rotation.keys())
        scores  = [rotation[s] for s in sectors]
        colors  = [f"#{C['green']}" if s > 0 else f"#{C['red']}" for s in scores]

        fig = go.Figure(go.Bar(
            x=scores, y=sectors, orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=[f" +{int(s)}" if s > 0 else f" {int(s)}" for s in scores],
            textposition="outside",
            textfont=dict(size=11, color=f"#{C['text']}"),
            hovertemplate="<b>%{y}</b><br>Net Score: %{x}<extra></extra>",
        ))
        fig.update_layout(**plotly_base(
            height=175,
            title=dict(text="Sector Rotation — Net Whale Flow",
                       font=dict(size=12, color=f"#{C['muted']}"), x=0, xanchor="left"),
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=True,
                       zerolinecolor="rgba(255,255,255,0.07)", zerolinewidth=1),
            yaxis=dict(showgrid=False, tickfont=dict(size=11), autorange="reversed"),
            bargap=0.4,
        ))
        sections.append(dcc.Graph(figure=fig, config={"displayModeBar": False},
                                  style={"marginBottom": "0.5rem"}))

    # Per-whale sections
    sig_priority = {"AGGRESSIVE_BUY": 4, "NEW_ENTRY": 3, "HIGH_CONCENTRATION": 2, "HOLD": 0}
    for whale, holdings in filings.items():
        if not holdings:
            continue

        non_hold = sum(1 for h in holdings if h.get("signal", "HOLD") != "HOLD")
        top_sig  = max((h.get("signal", "HOLD") for h in holdings),
                       key=lambda s: sig_priority.get(s, 0))
        si = SIG.get(top_sig, SIG["HOLD"])

        # Whale section header
        tier_info  = WHALE_TIERS.get(whale, {"label": "", "tier": 3})
        tier_label = tier_info.get("label", "")
        tier_color = (f"#{C['green']}" if tier_info.get("tier") == 1
                      else f"#{C['amber']}" if tier_info.get("tier") == 2
                      else f"#{C['muted']}")
        sections.append(html.Div([
            html.Div([
                html.Span(whale, className="whale-name"),
                html.Span(si["label"], style={
                    "background": f"{si['color']}18", "color": si["color"],
                    "border": f"1px solid {si['color']}44", "borderRadius": "5px",
                    "padding": "2px 9px", "fontSize": "0.62rem", "fontWeight": "700",
                }),
                *(
                    [html.Span(tier_label, style={
                        "background": f"{tier_color}14", "color": tier_color,
                        "border": f"1px solid {tier_color}33", "borderRadius": "5px",
                        "padding": "2px 8px", "fontSize": "0.57rem", "fontWeight": "600",
                        "letterSpacing": "0.3px",
                    })]
                    if tier_label else []
                ),
            ], style={"display": "flex", "alignItems": "center", "gap": "8px"}),
            html.Span(
                [f"{len(holdings)} positions · ", html.Span(str(non_hold), style={"color": f"#{C['green']}"}), " active"],
                className="whale-meta",
            ),
        ], className="whale-header"))

        # Holdings grid — 4 per row
        N = 4
        for i in range(0, len(holdings), N):
            chunk = holdings[i:i + N]
            cards = [holding_card(h) for h in chunk]
            while len(cards) < N:
                cards.append(html.Div())
            sections.append(html.Div(cards, className="grid-4"))

    # ── SC 13D/G activist / passive stake filings ────────────────────────────
    if activist:
        sections.append(html.Div([
            html.Div([
                html.Span("📋  SC 13D / 13G Filings", className="whale-name"),
                html.Span("≥5% ownership disclosures · 5-10 day lag",
                          className="whale-meta"),
            ], style={"display": "flex", "alignItems": "center", "gap": "10px"}),
        ], className="whale-header"))

        act_cards = []
        for ticker, f in sorted(activist.items(),
                                 key=lambda x: x[1].get("pct_outstanding", 0),
                                 reverse=True):
            sig  = f.get("signal", "LARGE_PASSIVE_STAKE")
            si   = SIG.get(sig, SIG["HOLD"])
            pct  = f.get("pct_outstanding", 0)
            form = f.get("form_type", "SC 13G")
            act_cards.append(html.Div([
                html.Div([
                    html.Span(ticker, className="holding-ticker"),
                    html.Span(si["label"], style={
                        "background": f"{si['color']}18", "color": si["color"],
                        "borderRadius": "4px", "padding": "1px 7px",
                        "fontSize": "0.6rem", "fontWeight": "700",
                    }),
                ], className="holding-header"),
                html.Div(f.get("filer", ""), className="holding-company"),
                html.Div([
                    html.Div([html.Div("Form",    className="stat-label"),
                              html.Div(form,      className="stat-value", style={"fontSize": "0.75rem"})]),
                    html.Div([html.Div("% Owned", className="stat-label"),
                              html.Div(f"{pct:.1%}", className="stat-value",
                                       style={"color": si["color"]})],
                             style={"textAlign": "right"}),
                ], className="holding-stats"),
            ], className="holding-card",
               style={"borderTop": f"2px solid {si['color']}"}))

        N = 4
        for i in range(0, len(act_cards), N):
            chunk = act_cards[i:i + N]
            while len(chunk) < N:
                chunk.append(html.Div())
            sections.append(html.Div(chunk, className="grid-4"))

    # ── Form 4 insider transactions ───────────────────────────────────────────
    if insiders:
        insider_rows = []
        for ticker, txs in insiders.items():
            for tx in txs:
                sig = tx.get("signal", "INSIDER_BUY")
                si  = SIG.get(sig, SIG["HOLD"])
                is_buy = sig == "INSIDER_BUY"
                val_str = f"${tx.get('value_usd', 0) / 1e6:.2f}M"
                insider_rows.append(html.Div([
                    html.Div([
                        html.Span(ticker, className="holding-ticker",
                                  style={"fontSize": "1rem"}),
                        html.Span(si["label"], style={
                            "background": f"{si['color']}18", "color": si["color"],
                            "borderRadius": "4px", "padding": "1px 7px",
                            "fontSize": "0.6rem", "fontWeight": "700",
                        }),
                    ], className="holding-header"),
                    html.Div(f"{tx.get('insider','')} · {tx.get('role','')}",
                             className="holding-company"),
                    html.Div([
                        html.Div([html.Div("Shares", className="stat-label"),
                                  html.Div(f"{tx.get('shares',0):,}", className="stat-value",
                                           style={"fontSize": "0.8rem"})]),
                        html.Div([html.Div("Value", className="stat-label"),
                                  html.Div(val_str, className="stat-value",
                                           style={"color": si["color"], "fontSize": "0.8rem"})],
                                 style={"textAlign": "right"}),
                    ], className="holding-stats"),
                ], className="holding-card",
                   style={"borderTop": f"2px solid {si['color']}"}))

        sections.append(html.Div([
            html.Div([
                html.Span("👤  Form 4 — Insider Transactions", className="whale-name"),
                html.Span("Officers & directors · 2-day filing lag",
                          className="whale-meta"),
            ], style={"display": "flex", "alignItems": "center", "gap": "10px"}),
        ], className="whale-header"))

        N = 4
        for i in range(0, len(insider_rows), N):
            chunk = insider_rows[i:i + N]
            while len(chunk) < N:
                chunk.append(html.Div())
            sections.append(html.Div(chunk, className="grid-4"))

    # ── N-PORT fund holdings ──────────────────────────────────────────────────
    if nport:
        sections.append(html.Div([
            html.Div([
                html.Span("📦  N-PORT — Monthly Fund Holdings", className="whale-name"),
                html.Span("Registered funds · 60-day lag · month-over-month change",
                          className="whale-meta"),
            ], style={"display": "flex", "alignItems": "center", "gap": "10px"}),
        ], className="whale-header"))

        for fund_name, holdings in nport.items():
            non_hold = sum(1 for h in holdings
                           if h.get("signal", "HOLD") not in ("HOLD",))
            sections.append(html.Div([
                html.Div([
                    html.Span(fund_name, style={"fontWeight": "600",
                              "fontSize": "0.85rem", "color": "#E8ECF0"}),
                ]),
                html.Span(f"{len(holdings)} positions · {non_hold} changes",
                          className="whale-meta"),
            ], className="whale-header",
               style={"marginTop": "0.8rem", "marginBottom": "0.5rem",
                      "borderBottomColor": "rgba(255,255,255,0.04)"}))

            N = 4
            for i in range(0, len(holdings), N):
                chunk = holdings[i:i + N]
                cards = []
                for h in chunk:
                    sig = h.get("signal", "HOLD")
                    si  = SIG.get(sig, SIG["HOLD"])
                    chg = h.get("change_pct", 0)
                    chg_str = f"{'+' if chg >= 0 else ''}{chg:.0%} MoM"
                    cards.append(html.Div([
                        html.Div([
                            html.Span(h["ticker"], className="holding-ticker"),
                            html.Span(si["label"], style={
                                "background": f"{si['color']}18", "color": si["color"],
                                "borderRadius": "4px", "padding": "1px 7px",
                                "fontSize": "0.6rem", "fontWeight": "700",
                            }),
                        ], className="holding-header"),
                        html.Div(h.get("company", ""), className="holding-company"),
                        html.Div([
                            html.Div([html.Div("MoM Chg", className="stat-label"),
                                      html.Div(chg_str, className="stat-value",
                                               style={"color": si["color"],
                                                      "fontSize": "0.8rem"})]),
                            html.Div([html.Div("Portfolio", className="stat-label"),
                                      html.Div(f"{h.get('portfolio_pct',0):.1%}",
                                               className="stat-value",
                                               style={"fontSize": "0.8rem"})],
                                     style={"textAlign": "right"}),
                        ], className="holding-stats"),
                    ], className="holding-card",
                       style={"borderTop": f"2px solid {si['color']}"}))
                while len(cards) < N:
                    cards.append(html.Div())
                sections.append(html.Div(cards, className="grid-4"))

    return html.Div(sections)


def build_rec_cards(filter_val: str = "ALL", watchlist: list | None = None):
    if filter_val == "📌 WATCHLIST":
        wl = {t.strip().upper() for t in (watchlist or [])}
        filtered = [r for r in recommendations if r["ticker"] in wl] if wl else []
    elif filter_val == "ALL":
        filtered = recommendations
    else:
        filtered = [r for r in recommendations if r["recommendation"] == filter_val]
    if not filtered:
        msg = ("No tickers in watchlist — use the ＋ Add input above to add tickers."
               if filter_val == "📌 WATCHLIST"
               else f"No {filter_val} recommendations in the current dataset.")
        return html.Div(msg, className="empty-state")
    N = 3
    rows = []
    for i in range(0, len(filtered), N):
        chunk = filtered[i:i + N]
        cards = [rec_card(r) for r in chunk]
        while len(cards) < N:
            cards.append(html.Div())
        rows.append(html.Div(cards, className="grid-3"))
    return html.Div(rows)


def build_macro_tab():
    """📈 Macro Dashboard — FRED economic indicators."""
    # ── KPI cards row ──────────────────────────────────────────────────────────
    kpi_order = ["fed_rate", "yield_10y", "cpi", "unemployment", "gdp_growth"]
    kpi_cards = []
    for key in kpi_order:
        m = macro_data.get(key, {})
        if not m:
            continue
        cur  = m["current"]
        chg  = m["change_1y"]
        col  = m["color"]
        arrow = ("↑" if chg > 0 else "↓") if chg != 0 else "→"
        chg_color = (f"#{C['red']}" if chg > 0 and key in ("fed_rate", "cpi", "yield_10y", "unemployment")
                     else f"#{C['green']}" if chg > 0 else f"#{C['red']}")
        kpi_cards.append(html.Div([
            html.Div("◈", style={
                "position": "absolute", "right": "12px", "top": "50%",
                "transform": "translateY(-50%)", "fontSize": "2.8rem",
                "opacity": "0.04", "color": col, "fontWeight": "900",
            }),
            html.Div(m["name"], className="kpi-label"),
            html.Div(f"{cur:.2f}{m['unit']}", className="kpi-value"),
            html.Div([
                html.Span(f"{arrow} {abs(chg):.2f}pp vs 1Y ago",
                          style={"color": chg_color, "fontWeight": "600",
                                 "fontSize": "0.72rem"}),
            ], className="kpi-sub"),
        ], className="kpi-card", style={"borderLeft": f"3px solid {col}"}))

    kpi_row = html.Div(kpi_cards, className="kpi-strip", style={"marginBottom": "1rem"})

    # ── Line charts (2 per row) ─────────────────────────────────────────────────
    chart_order_pairs = [
        ("fed_rate", "yield_10y"),
        ("cpi", "unemployment"),
    ]
    gdp_solo = "gdp_growth"

    chart_rows = []
    for left_key, right_key in chart_order_pairs:
        row_charts = []
        for key in (left_key, right_key):
            m = macro_data.get(key, {})
            if not m:
                continue
            obs = list(reversed(m.get("observations", [])))  # chronological
            dates  = [o["date"] for o in obs]
            values = [o["value"] for o in obs]
            col = m["color"]
            fig = go.Figure(go.Scatter(
                x=dates, y=values,
                mode="lines",
                line=dict(color=col, width=2),
                fill="tozeroy", fillcolor=f"{col}12",
                hovertemplate="<b>%{x}</b><br>%{y:.2f}" + m["unit"] + "<extra></extra>",
            ))
            fig.update_layout(**plotly_base(
                height=200,
                title=dict(text=m["name"],
                           font=dict(size=11, color=f"#{C['muted']}"),
                           x=0, xanchor="left"),
                xaxis=dict(showgrid=False, tickfont=dict(size=9),
                           tickangle=-30, nticks=8),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                           ticksuffix=m["unit"], tickfont=dict(size=10),
                           zeroline=False),
                margin=dict(l=0, r=0, t=36, b=40),
            ))
            row_charts.append(dcc.Graph(figure=fig, config={"displayModeBar": False},
                                        style={"flex": "1"}))
        chart_rows.append(html.Div(row_charts, style={
            "display": "flex", "gap": "1rem", "marginBottom": "1rem",
        }))

    # GDP solo (quarterly data — wider chart)
    m_gdp = macro_data.get(gdp_solo, {})
    if m_gdp:
        obs  = list(reversed(m_gdp.get("observations", [])))
        dates  = [o["date"] for o in obs]
        values = [o["value"] for o in obs]
        bar_colors = [f"#{C['green']}" if v >= 0 else f"#{C['red']}" for v in values]
        fig_gdp = go.Figure(go.Bar(
            x=dates, y=values,
            marker=dict(color=bar_colors, line=dict(width=0)),
            hovertemplate="<b>%{x}</b><br>%{y:.1f}%<extra></extra>",
        ))
        fig_gdp.update_layout(**plotly_base(
            height=200,
            title=dict(text=m_gdp["name"],
                       font=dict(size=11, color=f"#{C['muted']}"),
                       x=0, xanchor="left"),
            xaxis=dict(showgrid=False, tickfont=dict(size=9), tickangle=-30),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                       ticksuffix="%", tickfont=dict(size=10), zeroline=True,
                       zerolinecolor="rgba(255,255,255,0.12)"),
            margin=dict(l=0, r=0, t=36, b=40),
            bargap=0.3,
        ))
        chart_rows.append(dcc.Graph(figure=fig_gdp, config={"displayModeBar": False},
                                    style={"marginBottom": "1rem"}))

    # ── Whale context note ──────────────────────────────────────────────────────
    fed  = macro_data.get("fed_rate", {}).get("current", 0)
    y10  = macro_data.get("yield_10y", {}).get("current", 0)
    cpi  = macro_data.get("cpi", {}).get("current", 0)
    spread = round(y10 - fed, 2)
    context_lines = []
    if fed >= 5.0:
        context_lines.append("⚠️  Rates are elevated — Whales often rotate into Value/Financials in high-rate environments.")
    elif fed <= 2.0:
        context_lines.append("✅  Low rate environment — Growth/Tech stocks typically benefit from cheap capital.")
    if cpi >= 4.0:
        context_lines.append("⚠️  Inflation above 4% — watch for defensive rotation into Energy, Materials, Consumer Staples.")
    elif cpi <= 2.5:
        context_lines.append("✅  Inflation near Fed target — historically positive for broad equity markets.")
    if spread < 0:
        context_lines.append("⚠️  Inverted yield curve (10Y < Fed rate) — historically precedes economic slowdowns.")
    elif spread > 1.5:
        context_lines.append("✅  Positive yield curve spread — credit markets signalling expansion expectations.")
    if not context_lines:
        context_lines.append("📊  Macro conditions are neutral — monitor for shifts in key indicators.")

    context_card = html.Div([
        html.Div("🔍  Whale Context",
                 style={"fontSize": "0.72rem", "fontWeight": "700",
                        "color": f"#{C['blue']}", "letterSpacing": "0.6px",
                        "textTransform": "uppercase", "marginBottom": "0.6rem"}),
        *[html.P(line, className="grow-desc", style={"marginBottom": "4px"})
          for line in context_lines],
    ], style={
        "background": f"#{C['card2']}", "borderRadius": "12px",
        "padding": "14px 16px", "border": f"1px solid #{C['border']}",
        "borderLeft": f"3px solid #{C['blue']}",
    })

    return html.Div([kpi_row] + chart_rows + [context_card])


def build_news_banner(news_list: list) -> html.Div:
    """Compact horizontal news headline cards."""
    if not news_list:
        return html.Div()

    cards = []
    for item in news_list:
        headline = item.get("headline", "")
        source   = item.get("source", "")
        url      = item.get("url", "")
        pub      = item.get("published_at", "")
        if not headline:
            continue

        inner = html.Div([
            html.Div([
                html.Span("📰", style={"marginRight": "5px", "fontSize": "0.75rem"}),
                html.Span(source, style={
                    "fontSize": "0.58rem", "fontWeight": "700",
                    "color": f"#{C['blue']}", "textTransform": "uppercase",
                    "letterSpacing": "0.4px",
                }),
                *([ html.Span(f"  {pub}", style={
                        "fontSize": "0.58rem", "color": f"#{C['muted']}",
                    })] if pub else []),
            ], style={"marginBottom": "3px"}),
            html.Div(
                headline,
                style={"fontSize": "0.73rem", "color": f"#{C['text']}",
                       "lineHeight": "1.4", "fontWeight": "500"},
            ),
        ], style={
            "background": f"#{C['card']}", "borderRadius": "8px",
            "padding": "8px 12px", "flex": "1", "minWidth": "200px",
            "maxWidth": "280px", "border": f"1px solid #{C['border']}",
            "borderTop": f"2px solid #{C['blue']}",
        })

        if url:
            cards.append(html.A(inner, href=url, target="_blank", style={
                "textDecoration": "none", "flex": "1",
                "minWidth": "200px", "maxWidth": "280px",
            }))
        else:
            cards.append(inner)

    if not cards:
        return html.Div()

    return html.Div([
        html.Div([
            html.Span("📰", style={"fontSize": "0.7rem", "marginRight": "5px"}),
            html.Span("Market Headlines", style={
                "fontSize": "0.62rem", "fontWeight": "700",
                "color": f"#{C['muted']}", "letterSpacing": "0.5px",
                "textTransform": "uppercase",
            }),
        ], style={"marginBottom": "0.5rem"}),
        html.Div(cards, style={
            "display": "flex", "gap": "0.7rem",
            "flexWrap": "nowrap", "overflowX": "auto",
        }),
    ], style={
        "padding": "0.7rem 0", "marginBottom": "0.3rem",
        "borderBottom": f"1px solid #{C['border']}40",
    })


def build_portfolio_tab():
    holdings_list  = portfolio.get("holdings", [])
    target_weights = portfolio.get("target_sector_weights", {})
    top_sector     = max(current_weights, key=current_weights.get) if current_weights else "—"

    # Mini KPIs
    mini_kpis = html.Div([
        kpi_card("TOTAL VALUE",      f"${port_value:,.0f}",
                 f"{len(holdings_list)} positions",                           C["blue"]),
        kpi_card("SECTORS",          str(len(current_weights)),
                 "GICS sectors covered",                                      C["purple"]),
        kpi_card("DOMINANT SECTOR",  top_sector,
                 f"{current_weights.get(top_sector, 0):.0%} of portfolio",   C["amber"]),
    ], className="grid-3", style={"marginBottom": "1rem"})

    # Donut chart
    if current_weights:
        labels = list(current_weights.keys())
        values = [v * 100 for v in current_weights.values()]
        fig_donut = go.Figure(go.Pie(
            labels=labels, values=values, hole=0.62,
            marker=dict(colors=PALETTE[:len(labels)],
                        line=dict(color=f"#{C['bg']}", width=2)),
            textinfo="label+percent",
            textfont=dict(size=11, color=f"#{C['text']}"),
            hovertemplate="<b>%{label}</b><br>%{value:.1f}%<extra></extra>",
        ))
        fig_donut.update_layout(**plotly_base(
            height=260, showlegend=False,
            title=dict(text="Current Allocation",
                       font=dict(size=12, color=f"#{C['muted']}"),
                       x=0.5, xanchor="center"),
            annotations=[dict(
                text=f"<b>${port_value/1000:.1f}K</b>",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=15, color=f"#{C['text']}", family="Inter"),
            )],
        ))
    else:
        fig_donut = go.Figure()

    # Current vs Target grouped bar
    all_sec   = sorted(set(list(current_weights) + list(target_weights)))
    curr_vals = [current_weights.get(s, 0) * 100 for s in all_sec]
    targ_vals = [target_weights.get(s, 0) * 100  for s in all_sec]

    fig_bar = go.Figure([
        go.Bar(name="Current", x=all_sec, y=curr_vals,
               marker=dict(color=f"#{C['blue']}", line=dict(width=0)), opacity=0.85),
        go.Bar(name="Target",  x=all_sec, y=targ_vals,
               marker=dict(color=f"#{C['green']}", line=dict(width=0)), opacity=0.85),
    ])
    fig_bar.update_layout(**plotly_base(
        height=260, barmode="group",
        title=dict(text="Current vs Target Sector Weights (%)",
                   font=dict(size=12, color=f"#{C['muted']}"), x=0, xanchor="left"),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1,
                    font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(showgrid=False, tickfont=dict(size=10)),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.07)",
                   ticksuffix="%", tickfont=dict(size=10)),
        bargap=0.25, bargroupgap=0.06,
    ))

    charts_row = html.Div([
        dcc.Graph(figure=fig_donut, config={"displayModeBar": False}),
        dcc.Graph(figure=fig_bar,   config={"displayModeBar": False}),
    ], className="charts-row")

    # Rebalancing cards
    reb_header = html.Div([
        html.Span("Rebalancing Actions", className="section-title"),
        html.Span("Whale-adjusted targets · ±5pp drift threshold",
                  style={"fontSize": "0.7rem", "color": f"#{C['muted']}", "marginLeft": "8px"}),
    ], className="section-header")

    if not rebalancing:
        reb_content = html.Div(
            "✓  Portfolio is within target weights — no rebalancing needed.",
            className="success-banner",
        )
    else:
        N = 3
        rows = []
        for i in range(0, len(rebalancing), N):
            chunk = rebalancing[i:i + N]
            cards = [rebalancing_card(s) for s in chunk]
            while len(cards) < N:
                cards.append(html.Div())
            rows.append(html.Div(cards, className="grid-3"))
        reb_content = html.Div(rows)

    # Raw holdings table
    th = lambda txt, right=False: html.Th(txt, className="tbl-th",
                                           style={"textAlign": "right" if right else "left"})
    def td(txt, right=False, green=False, bold=False):
        return html.Td(txt, style={
            "padding": "7px 10px", "fontSize": "0.82rem",
            "textAlign": "right" if right else "left",
            "fontWeight": "700" if bold else "400",
            "color": f"#{C['green']}" if green else (f"#{C['text']}" if bold else f"#{C['muted']}"),
        })

    raw_holdings = html.Details([
        html.Summary("📋  Raw Holdings", className="expander-summary"),
        html.Div(
            html.Table([
                html.Thead(html.Tr([
                    th("Ticker"), th("Sector"),
                    th("Qty", right=True), th("Avg Cost", right=True),
                    th("Market Value", right=True),
                ])),
                html.Tbody([
                    html.Tr([
                        td(h["ticker"],                            bold=True),
                        td(h.get("sector", "—")),
                        td(f"{h['quantity']:,}",                   right=True),
                        td(f"${h['avg_cost']:,.2f}",               right=True),
                        td(f"${h['quantity']*h['avg_cost']:,.0f}", right=True, green=True, bold=True),
                    ], style={"borderBottom": f"1px solid #{C['border']}40"})
                    for h in holdings_list
                ]),
            ], className="raw-table"),
            className="raw-table-wrapper",
        ),
    ])

    return html.Div([mini_kpis, charts_row, reb_header, reb_content, raw_holdings])


# ── GUIDE CONTENT ─────────────────────────────────────────────────────────────
def _gsec(title: str, *children):
    """Guide section wrapper."""
    return html.Div([
        html.Div(title, className="gsec-title"),
        *children,
    ], className="gsec")


def _grow(badge_label: str, badge_color: str, score: str, desc: str):
    """Signal/recommendation guide row."""
    return html.Div([
        html.Div([
            html.Span(badge_label, style={
                "background": f"{badge_color}18", "color": badge_color,
                "border": f"1px solid {badge_color}44", "borderRadius": "5px",
                "padding": "2px 9px", "fontSize": "0.7rem", "fontWeight": "700",
                "marginRight": "8px", "whiteSpace": "nowrap",
            }),
            html.Span(score, style={
                "fontSize": "0.68rem", "color": f"#{C['muted']}",
                "background": f"#{C['card2']}", "borderRadius": "4px",
                "padding": "1px 6px", "fontWeight": "600",
            }),
        ], style={"display": "flex", "alignItems": "center", "marginBottom": "4px"}),
        html.Div(desc, className="grow-desc"),
    ], className="grow")


def _whale_row(name: str, manager: str):
    return html.Div([
        html.Span("🐋", style={"marginRight": "8px"}),
        html.Div([
            html.Div(name,    className="gwhale-name"),
            html.Div(manager, className="gwhale-mgr"),
        ]),
    ], className="gwhale-row")


def _gtab(icon: str, name: str, desc: str):
    return html.Div([
        html.Div([html.Span(icon, style={"marginRight": "8px"}), name],
                 className="gtab-title"),
        html.Div(desc, className="grow-desc"),
    ], className="gtab-row")


def _datasource_row(icon: str, name: str, lag: str, lag_color: str,
                    desc_en: str, desc_ko: str, en: bool):
    """Data-source pipeline row with coloured lag badge."""
    return html.Div([
        html.Div([
            html.Span(icon, style={"marginRight": "6px", "fontSize": "0.95rem"}),
            html.Span(name, style={
                "fontWeight": "600", "fontSize": "0.79rem",
                "color": f"#{C['text']}",
            }),
            html.Span(lag, style={
                "background": f"{lag_color}18", "color": lag_color,
                "border": f"1px solid {lag_color}44", "borderRadius": "4px",
                "padding": "1px 7px", "fontSize": "0.58rem", "fontWeight": "700",
                "marginLeft": "8px", "whiteSpace": "nowrap",
            }),
        ], style={
            "display": "flex", "alignItems": "center",
            "flexWrap": "wrap", "gap": "4px", "marginBottom": "4px",
        }),
        html.Div(desc_en if en else desc_ko, className="grow-desc"),
    ], className="grow")


def _strategy_example(title: str, accent: str,
                       badges: list, desc: str):
    """Combined-signal scenario card."""
    badge_els = []
    for i, (src, label, col) in enumerate(badges):
        badge_els.append(html.Span([
            html.Span(f"{src} ", style={
                "fontSize": "0.58rem", "color": f"#{C['muted']}",
            }),
            html.Span(label, style={
                "background": f"{col}18", "color": col,
                "border": f"1px solid {col}44", "borderRadius": "4px",
                "padding": "1px 6px", "fontSize": "0.58rem", "fontWeight": "700",
            }),
        ], style={"display": "inline-flex", "alignItems": "center", "gap": "2px"}))
        if i < len(badges) - 1:
            badge_els.append(
                html.Span(" → ", style={"color": f"#{C['muted']}", "fontSize": "0.68rem"})
            )
    return html.Div([
        html.Div(title, style={
            "fontSize": "0.77rem", "fontWeight": "700",
            "color": accent, "marginBottom": "6px",
        }),
        html.Div(badge_els, style={
            "display": "flex", "alignItems": "center",
            "flexWrap": "wrap", "gap": "3px", "marginBottom": "7px",
        }),
        html.Div(desc, className="grow-desc"),
    ], style={
        "background": f"#{C['card2']}", "borderRadius": "8px",
        "padding": "10px 14px", "marginBottom": "8px",
        "borderLeft": f"3px solid {accent}",
    })


def build_guide(lang: str):
    en = lang == "en"

    # ── Overview ──────────────────────────────────────────────────────────────
    overview = _gsec(
        "Overview" if en else "서비스 개요",
        html.P(
            "WhaleTracker AI compensates for the 13F quarterly lag by stitching together "
            "four complementary SEC filing types — from 2-day insider trades to 45-day "
            "quarterly holdings — so you always have the most current picture of "
            "where Smart Money is moving."
            if en else
            "WhaleTracker AI는 13F의 45일 보고 지연을 보완하기 위해 "
            "내부자 거래(2일)부터 분기 보고(45일)까지 4가지 SEC 공시를 결합합니다. "
            "스마트머니의 실시간 흐름을 가장 빠르게 포착하는 것이 목표입니다.",
            className="grow-desc",
        ),
    )

    # ── Data Sources & Lag Pipeline ───────────────────────────────────────────
    data_pipeline = _gsec(
        "Data Sources & Lag Compensation" if en else "데이터 소스 & 시차 보완 전략",
        html.P(
            "Each SEC filing type has a different reporting deadline. "
            "Sorted fastest → slowest — WhaleTracker tracks all four in parallel "
            "so no move goes unnoticed."
            if en else
            "각 SEC 공시 유형은 보고 기한이 다릅니다. "
            "가장 빠른 것부터 느린 순서로 정렬했으며, "
            "WhaleTracker는 4가지를 동시에 추적해 어떤 움직임도 놓치지 않습니다.",
            className="grow-desc",
            style={"marginBottom": "0.7rem"},
        ),
        _datasource_row(
            "👤", "Form 4 — Insider Transactions" if en else "Form 4 — 내부자 거래",
            "⚡ 2-day lag" if en else "⚡ 2영업일 이내",
            f"#{C['green']}",
            "Officers (CEO, CFO, etc.) and shareholders owning >10% must report every "
            "open-market trade within 2 business days. The fastest data in the system. "
            "An early warning of management conviction — insiders only buy with their "
            "own money when they expect the stock to rise.",
            "임원(CEO, CFO 등)과 10% 이상 대주주는 모든 주식 거래를 2영업일 이내에 공시해야 합니다. "
            "시스템에서 가장 빠른 데이터로, 경영진의 확신도를 가장 먼저 포착합니다. "
            "내부자는 주가 상승을 확신할 때만 자신의 돈으로 매수합니다.",
            en,
        ),
        _datasource_row(
            "📋", "SC 13D / 13G — Activist & Passive Stakes" if en else "SC 13D/G — 행동주의 / 대규모 지분",
            "5–10 day lag" if en else "5–10영업일 이내",
            f"#{C['red']}",
            "Any entity that acquires ≥5% of a listed company must file within 5–10 days. "
            "13D = intent to influence management (activist, legally binding). "
            "13G = passive investment only. "
            "You learn about activist involvement long before the quarterly 13F is published — "
            "letting you distinguish whether the institution wants board seats or just returns.",
            "상장사 지분의 5% 이상 취득 시 5~10영업일 이내에 공시해야 합니다. "
            "13D는 경영 간섭 의지(행동주의, 법적 구속력 있음), "
            "13G는 단순 수동적 투자 목적입니다. "
            "13F가 나오기 훨씬 전에 해당 기관이 이사회에 개입하려는지, "
            "단순 수익 목적인지를 파악할 수 있습니다.",
            en,
        ),
        _datasource_row(
            "📦", "N-PORT — Monthly Fund Holdings" if en else "N-PORT — 월간 펀드 보유현황",
            "Monthly · 60-day lag" if en else "월 단위 · 60일 이내",
            "#20B2AA",
            "Mutual funds (Vanguard, BlackRock, etc.) report their full portfolios every month. "
            "3× more frequent than 13F. "
            "By the time the quarterly report arrives, you've already tracked 70%+ "
            "of their moves through N-PORT — turning a 45-day lag into a ~2 month rolling view.",
            "뮤추얼 펀드(뱅가드, 블랙록 등)는 전체 포트폴리오를 매월 단위로 보고합니다. "
            "13F보다 3배 빠른 업데이트 주기입니다. "
            "13F가 공개될 즈음에는 이미 N-PORT를 통해 펀드 움직임의 70% 이상을 파악한 상태입니다. "
            "45일 지연을 사실상 월 단위 롤링 뷰로 전환합니다.",
            en,
        ),
        _datasource_row(
            "🐋", "13F-HR — Quarterly Whale Holdings" if en else "13F-HR — 분기별 Whale 보유현황",
            "Quarterly · 45-day lag" if en else "분기 단위 · 45일 이내",
            f"#{C['blue']}",
            "Institutions managing >$100M must disclose all equity positions 45 days after "
            "each quarter. The foundation signal with the highest legal significance — "
            "use the three faster sources above to anticipate what the 13F will confirm.",
            "운용자산 1억 달러 이상의 기관은 분기 종료 후 45일 이내에 주식 보유현황을 공시합니다. "
            "법적 의미가 가장 높은 기반 신호입니다. "
            "위의 3가지 빠른 데이터로 13F가 확인할 내용을 미리 예측하는 것이 핵심 전략입니다.",
            en,
        ),
    )

    # ── Combined Signal Strategy ──────────────────────────────────────────────
    strategy = _gsec(
        "Combined Signal Strategy" if en else "복합 신호 전략",
        html.P(
            "The real edge is signal convergence. When multiple independent sources "
            "point in the same direction, conviction rises sharply. "
            "The conviction score aggregates points from all four filing types (max 12)."
            if en else
            "핵심 우위는 신호 수렴입니다. 독립적인 여러 소스가 같은 방향을 가리킬 때 "
            "확신도가 급격히 높아집니다. "
            "컨빅션 점수는 4가지 공시 유형의 점수를 모두 합산합니다(최대 12점).",
            className="grow-desc",
            style={"marginBottom": "0.8rem"},
        ),
        _strategy_example(
            "🚀 Ultra-Strong Buy" if en else "🚀 초강력 매수 신호",
            f"#{C['green']}",
            [
                ("🐋 13F",    "NEW ENTRY",    f"#{C['blue']}"),
                ("📋 13D",    "ACTIVIST",     f"#{C['red']}"),
                ("👤 Form 4", "INSIDER BUY",  f"#{C['green']}"),
            ],
            "A Whale takes a new position (13F: +3 pts) → the same entity files a 13D "
            "showing intent to influence management (+5 pts) → company insiders are also "
            "buying on the open market (Form 4: +3 pts). Three independent sources agree. "
            "Conviction score: 11/12 — this is the pattern to act on."
            if en else
            "Whale이 신규 포지션 진입(13F: +3점) → 같은 기관이 13D 제출, "
            "경영 간섭 의지 표명(+5점) → 회사 내부자들도 공개 시장에서 매수(Form 4: +3점). "
            "세 개의 독립적인 소스가 동일한 방향을 가리킵니다. "
            "컨빅션 점수 11/12 — 이 패턴이 나타날 때 적극 대응하세요.",
        ),
        _strategy_example(
            "📦 N-PORT Pre-Signal" if en else "📦 N-PORT 선행 신호",
            "#20B2AA",
            [
                ("📦 N-PORT",  "FUND ACCUM",  "#20B2AA"),
                ("🐋 13F",    "PENDING →",   "#4A5568"),
            ],
            "N-PORT shows a major fund accumulating shares this month. "
            "The quarterly 13F confirmation is weeks away — but you already know "
            "the direction and can position ahead of the public filing."
            if en else
            "N-PORT에서 대형 펀드가 이번 달 특정 주식을 대규모 매수 중. "
            "분기별 13F 확인은 몇 주 후이지만, 방향성을 먼저 파악해 "
            "공시 이전에 포지션을 선점할 수 있습니다.",
        ),
        _strategy_example(
            "⚠️ Divergence Warning" if en else "⚠️ 신호 괴리 경고",
            f"#{C['amber']}",
            [
                ("🐋 13F",    "AGG. BUY",     f"#{C['green']}"),
                ("👤 Form 4", "INSIDER SELL", f"#{C['red']}"),
            ],
            "A Whale is aggressively buying (13F) but company insiders are quietly "
            "selling on the open market (Form 4). Conflicting signals suggest caution — "
            "hold off until the divergence resolves."
            if en else
            "Whale은 공격적으로 매수(13F) 중이지만, 회사 내부자들은 공개 시장에서 "
            "조용히 매도(Form 4) 중. 신호가 상충되어 주의가 필요합니다 — "
            "괴리가 해소될 때까지 관망을 권고합니다.",
        ),
    )

    # ── Signal Definitions (all 9 signals, grouped by source) ─────────────────
    def _sig_group_label(txt: str):
        return html.Div(txt, style={
            "fontSize": "0.6rem", "fontWeight": "700",
            "color": f"#{C['muted']}", "letterSpacing": "0.7px",
            "textTransform": "uppercase", "marginTop": "0.65rem",
            "marginBottom": "0.25rem", "paddingBottom": "4px",
            "borderBottom": f"1px solid #{C['border']}",
        })

    signals = _gsec(
        "Signal Definitions" if en else "신호 정의",
        # ── 13F ──
        _sig_group_label("🐋 13F Whale Signals — Quarterly" if en else "🐋 13F Whale 신호 — 분기"),
        _grow("AGG. BUY",    f"#{C['green']}", "+4 pts",
              ("Share count increased >20% QoQ — the strongest 13F conviction signal."
               if en else "전 분기 대비 보유 주식 수 20% 초과 증가 — 가장 강한 13F 매수 신호.")),
        _grow("NEW ENTRY",   f"#{C['blue']}",  "+3 pts",
              ("Ticker absent from the prior quarter's 13F — fresh institutional position."
               if en else "이전 분기 공시에 없던 종목 — 기관의 신규 진입 포지션.")),
        _grow("HIGH CONC",   f"#{C['amber']}", "+2 pts",
              ("Position exceeds 5% of the Whale's total portfolio value."
               if en else "해당 종목이 Whale 포트폴리오의 5% 이상을 차지.")),
        _grow("HOLD",        "#4A5568",        "+0 pts",
              ("No significant change from the prior quarter."
               if en else "전 분기 대비 유의미한 변화 없음.")),
        # ── 13D/G ──
        _sig_group_label(
            "📋 SC 13D/G Signals — 5–10 Day" if en else "📋 SC 13D/G 신호 — 5–10영업일"
        ),
        _grow("ACTIVIST",    f"#{C['red']}",   "+5 pts",
              ("SC 13D — filer intends to actively influence management. "
               "Legally binding. Often precedes board changes, M&A, or spin-offs."
               if en else
               "SC 13D — 제출자가 경영에 적극 개입할 의도를 가짐. "
               "법적 구속력 있음. 이사회 교체, M&A, 분사 등 대형 이벤트를 선행하는 경우 多.")),
        _grow("13G STAKE",   f"#{C['purple']}", "+2 pts",
              ("SC 13G — passive ≥5% ownership with no intent to influence management. "
               "Signals large-scale institutional accumulation even without activist intent."
               if en else
               "SC 13G — 경영 개입 의도 없는 5% 이상 수동적 보유. "
               "행동주의 의도 없이도 대규모 기관 매집의 유의미한 신호입니다.")),
        # ── Form 4 ──
        _sig_group_label(
            "👤 Form 4 Signals — 2-Day" if en else "👤 Form 4 신호 — 2영업일"
        ),
        _grow("INSIDER BUY", f"#{C['green']}", "+3 pts",
              ("Open-market purchase by an officer or director using personal funds. "
               "Insiders only buy with their own money when conviction is high."
               if en else
               "임원 또는 이사의 공개 시장 자사주 매수(개인 자금). "
               "내부자는 확신이 클 때만 자신의 돈으로 매수합니다.")),
        _grow("INSIDER SELL", f"#{C['red']}",  "−2 pts",
              ("Open-market sale by an officer or director. "
               "Note: insiders sell for many reasons (tax, diversification). "
               "Most bearish when multiple insiders sell simultaneously."
               if en else
               "임원 또는 이사의 공개 시장 자사주 매도. "
               "참고: 세금, 분산투자 등 비하락 이유도 많습니다. "
               "여러 내부자가 동시에 매도할 때 가장 하락 신호로 해석됩니다.")),
        # ── N-PORT ──
        _sig_group_label(
            "📦 N-PORT Signals — Monthly" if en else "📦 N-PORT 신호 — 월 단위"
        ),
        _grow("FUND ACCUM",  "#20B2AA",        "+2 pts",
              ("Fund increased its position ≥5% month-over-month. "
               "Early indicator of fund-level conviction ahead of the quarterly 13F."
               if en else
               "펀드가 전월 대비 5% 이상 보유량 증가. "
               "분기별 13F보다 먼저 펀드 확신도를 보여주는 조기 지표입니다.")),
        _grow("FUND SELL",   "#FF8C00",        "−1 pt",
              ("Fund reduced its position ≥5% month-over-month. "
               "Persistent multi-month liquidation is a stronger signal than a single month."
               if en else
               "펀드가 전월 대비 5% 이상 보유량 감소. "
               "여러 달에 걸친 지속적인 청산이 단일 월 감소보다 훨씬 강력한 신호입니다.")),
    )

    # ── Recommendation Levels ─────────────────────────────────────────────────
    recs = _gsec(
        "Recommendation Levels" if en else "추천 등급",
        html.P(
            "Conviction score aggregates signal points across ALL four filing types. "
            "Maximum possible: 12 pts."
            if en else
            "컨빅션 점수는 4가지 공시 유형의 신호 점수를 모두 합산합니다. "
            "최대 12점.",
            className="grow-desc",
            style={"marginBottom": "0.6rem"},
        ),
        _grow("🚀 STRONG BUY", f"#{C['green']}", "score ≥ 6  or  ≥ 4 with 2+ Whales",
              ("Highest cross-source conviction — multiple Whales or filing types agree."
               if en else "최고 교차 소스 확신도 — 복수 Whale 또는 복수 공시 유형이 동시에 일치.")),
        _grow("↑ BUY",         "#1DB954",        "score ≥ 3",
              ("Strong single-source signal worth a close look."
               if en else "단일 소스의 강한 신호 — 주목할 만한 종목.")),
        _grow("→ HOLD",        f"#{C['amber']}", "score ≥ 1",
              ("Mild interest detected — monitor but don't rush."
               if en else "낮은 관심도 감지 — 모니터링 유지.")),
        _grow("↓ SELL",        f"#{C['red']}",   "score = 0",
              ("No institutional backing detected across any filing type this cycle."
               if en else "이번 주기에 어떤 공시 유형에서도 기관 지지 없음.")),
    )

    # ── Tracked Institutions ──────────────────────────────────────────────────
    whales = _gsec(
        "Tracked Institutions" if en else "추적 기관",
        _whale_row("Berkshire Hathaway",    "Warren Buffett"),
        _whale_row("Bridgewater Associates","Ray Dalio"),
        _whale_row("Appaloosa Management",  "David Tepper"),
        _whale_row("Pershing Square",       "Bill Ackman"),
        _whale_row("Tiger Global",          "Chase Coleman"),
    )

    # ── How to Use Each Tab ───────────────────────────────────────────────────
    tabs_guide = _gsec(
        "How to Use Each Tab" if en else "탭별 사용법",
        _gtab("🌊", "Whale Heatmap",
              ("① Sector Rotation chart — net institutional inflows by sector (13F). "
               "② Per-Whale holding cards sorted by signal strength. "
               "③ SC 13D/G activist/passive cards (5–10 day lag). "
               "④ Form 4 insider transaction cards (2-day lag). "
               "⑤ N-PORT monthly fund-flow cards (60-day lag). "
               "Read bottom-up (Form 4 → 13D/G → N-PORT → 13F) for a chronological signal chain."
               if en else
               "① 섹터 로테이션 차트: 섹터별 기관 순유입량(13F 기반). "
               "② Whale별 보유 카드: 13F 신호 강도순 정렬. "
               "③ SC 13D/G 행동주의/대규모 지분 카드(5–10영업일 지연). "
               "④ Form 4 내부자 거래 카드(2영업일 지연). "
               "⑤ N-PORT 월간 펀드 유입 카드(60일 지연). "
               "아래→위(Form 4 → 13D/G → N-PORT → 13F) 순서로 읽으면 시간순 신호 체인이 됩니다.")),
        _gtab("💡", "Recommendations",
              ("Filter by ALL / STRONG BUY / BUY / HOLD / SELL. "
               "Conviction bar shows aggregated score from all four filing types (max 12). "
               "Signal badges show exactly which filing types triggered the score. "
               "⚡ Macro note flags significant cross-source divergences or sector themes."
               if en else
               "ALL / STRONG BUY / BUY / HOLD / SELL로 필터링합니다. "
               "컨빅션 바는 4가지 공시 유형을 합산한 점수(최대 12점)를 나타냅니다. "
               "신호 배지는 어떤 공시 유형이 점수를 발생시켰는지 명시합니다. "
               "⚡ 매크로 노트는 중요한 교차 소스 괴리 또는 섹터 테마를 강조합니다.")),
        _gtab("📊", "My Portfolio",
              ("Compare your sector weights against Whale-adjusted targets. "
               "Sectors drifting >5pp trigger a rebalancing card. "
               "Rationale reflects active Whale signals from the latest 13F — "
               "e.g. DECREASE Technology if Whales are trimming tech exposure."
               if en else
               "현재 섹터 비중을 Whale 신호가 반영된 목표 비중과 비교합니다. "
               "5%p 이상 이탈한 섹터는 리밸런싱 카드가 표시됩니다. "
               "근거(Rationale)는 최신 13F의 활성 Whale 신호를 반영합니다 — "
               "예: Whale들이 Tech 비중을 줄이고 있다면 Technology DECREASE로 표시.")),
    )

    # ── Important Notes ───────────────────────────────────────────────────────
    notes = _gsec(
        "Important Notes" if en else "주요 참고사항",
        html.Ul([
            html.Li(
                "13F has a ~45-day lag. Use Form 4 (2-day) and 13D/G (5-10 day) "
                "to anticipate 13F moves before they're public."
                if en else
                "13F는 약 45일 지연됩니다. Form 4(2일)와 13D/G(5-10일)로 "
                "13F 내용을 공개 전에 미리 예측하세요."
            ),
            html.Li(
                "Insider SELL has many non-bearish explanations (tax, diversification). "
                "Only treat as bearish when multiple insiders sell simultaneously."
                if en else
                "내부자 매도(INSIDER SELL)는 세금, 분산투자 등 비하락 이유가 많습니다. "
                "여러 내부자가 동시에 매도할 때만 하락 신호로 해석하세요."
            ),
            html.Li(
                "MOCK MODE shows sample data. Set DATA_MODE=live and FMP_API_KEY in .env for real filings."
                if en else
                "MOCK 모드는 샘플 데이터를 표시합니다. "
                "실시간 데이터는 .env에서 DATA_MODE=live 및 FMP_API_KEY를 설정하세요."
            ),
            html.Li(
                "Edit my_portfolio.json to reflect your actual holdings for accurate rebalancing."
                if en else
                "정확한 리밸런싱을 위해 my_portfolio.json을 실제 보유 종목으로 편집하세요."
            ),
            html.Li(
                "Conviction score max = 12 "
                "(e.g. AGGRESSIVE_BUY +4 · ACTIVIST_STAKE +5 · INSIDER_BUY +3)."
                if en else
                "컨빅션 최대점수 = 12점 "
                "(예: AGGRESSIVE_BUY +4 · ACTIVIST_STAKE +5 · INSIDER_BUY +3 조합)."
            ),
        ], className="guide-notes"),
    )

    return html.Div(
        [overview, data_pipeline, strategy, signals, recs, whales, tabs_guide, notes],
        className="guide-body",
    )


def _bsec(title: str, *children):
    """Beginner guide section — same layout as _gsec but teal title."""
    return html.Div([
        html.Div(title, className="gsec-title", style={"color": "#20B2AA"}),
        *children,
    ], className="gsec")


def build_beginner_guide(lang: str) -> html.Div:
    """Jargon-free guide for stock market newcomers. EN + KO."""
    en = lang == "en"
    T = "#20B2AA"   # teal accent for beginner guide

    # ── 1. What is WhaleTracker? ──────────────────────────────────────────────
    intro = _bsec(
        "What is WhaleTracker AI?" if en else "월트래커 AI가 뭐예요?",
        html.P(
            "Imagine you could see exactly what the world's smartest, best-funded investors "
            "are buying and selling — before most people even notice. That's WhaleTracker. "
            "In the US, large investment funds are legally required to report their stock trades "
            "to the government. WhaleTracker reads those reports automatically and shows you "
            "the key signals in plain English."
            if en else
            "세계에서 가장 똑똑하고 자금력 있는 투자자들이 무엇을 사고 파는지 "
            "미리 알 수 있다면 어떨까요? 그게 바로 월트래커입니다. "
            "미국에서는 대형 투자펀드가 자신의 주식 거래 내역을 정부에 의무적으로 보고해야 합니다. "
            "월트래커는 그 보고서를 자동으로 읽어 핵심 신호를 쉽게 보여줍니다.",
            className="grow-desc",
        ),
        html.Div([
            html.Span("🐳", style={"fontSize": "1.5rem", "marginRight": "10px"}),
            html.Div([
                html.Div(
                    "Think of it like this:" if en else "이렇게 생각해보세요:",
                    style={"fontWeight": "700", "fontSize": "0.8rem",
                           "color": f"#{C['text']}", "marginBottom": "3px"},
                ),
                html.Div(
                    "When the world's top chefs all order the same ingredient, "
                    "you know something delicious is coming. When the world's top investors "
                    "all pile into the same stock — that's a signal worth paying attention to."
                    if en else
                    "세계 최고의 셰프들이 모두 같은 재료를 주문하기 시작하면 "
                    "뭔가 맛있는 게 나올 거라는 걸 알 수 있죠. "
                    "최고의 투자자들이 같은 주식을 사들일 때 — 그게 바로 주목할 신호입니다.",
                    className="grow-desc",
                ),
            ]),
        ], style={
            "background": f"{T}0D", "borderRadius": "8px",
            "padding": "10px 14px", "marginTop": "0.7rem",
            "border": f"1px solid {T}33", "display": "flex", "alignItems": "flex-start",
        }),
    )

    # ── 2. Who are the Whales? ────────────────────────────────────────────────
    whale_descs = {
        "en": {
            "Berkshire Hathaway":    "Warren Buffett · The most famous investor alive. Focuses on great companies at fair prices.",
            "Bridgewater Associates":"Ray Dalio · World's largest hedge fund. Specialises in global macro trends.",
            "Pershing Square":       "Bill Ackman · Known for high-conviction bets and activist campaigns.",
            "Appaloosa Management":  "David Tepper · A master at buying distressed assets when others panic.",
            "Tiger Global":          "Chase Coleman · One of the best tech-focused growth investors.",
        },
        "ko": {
            "Berkshire Hathaway":    "워런 버핏 · 살아있는 전설의 투자자. 좋은 기업을 적정 가격에 사는 방식.",
            "Bridgewater Associates":"레이 달리오 · 세계 최대 헤지펀드. 글로벌 거시경제 트렌드 전문.",
            "Pershing Square":       "빌 애크먼 · 강한 확신 베팅과 행동주의 캠페인으로 유명.",
            "Appaloosa Management":  "데이비드 테퍼 · 공황 상태에서 부실자산을 매수하는 달인.",
            "Tiger Global":          "체이스 콜먼 · 최고의 기술주 성장 투자자 중 한 명.",
        },
    }
    wl = whale_descs["en" if en else "ko"]

    whales = _bsec(
        "Who are the Whales?" if en else "고래(Whale)란 누구인가요?",
        html.P(
            "A 'Whale' is Wall Street slang for a huge institutional investor — someone whose "
            "trades are so large they make waves in the market. WhaleTracker follows 5 of the "
            "most influential ones:"
            if en else
            "'고래(Whale)'는 월스트리트 용어로 시장을 움직일 만큼 거대한 기관 투자자를 말합니다. "
            "월트래커는 가장 영향력 있는 5곳을 추적합니다:",
            className="grow-desc", style={"marginBottom": "0.6rem"},
        ),
        *[html.Div([
            html.Div([
                html.Span("🐋", style={"marginRight": "8px"}),
                html.Div([
                    html.Div(name, className="gwhale-name"),
                    html.Div(desc, className="gwhale-mgr"),
                ]),
            ], style={"display": "flex", "alignItems": "flex-start"}),
        ], className="gwhale-row") for name, desc in wl.items()],
    )

    # ── 3. How does the government help? (Filing types) ───────────────────────
    def _filing_row(icon, name, lag_label, lag_color, analogy):
        return html.Div([
            html.Div([
                html.Span(icon, style={"marginRight": "6px", "fontSize": "1rem"}),
                html.Span(name, style={"fontWeight": "700", "fontSize": "0.78rem",
                                        "color": f"#{C['text']}"}),
                html.Span(lag_label, style={
                    "background": f"{lag_color}18", "color": lag_color,
                    "border": f"1px solid {lag_color}44", "borderRadius": "4px",
                    "padding": "1px 7px", "fontSize": "0.58rem", "fontWeight": "700",
                    "marginLeft": "8px",
                }),
            ], style={"display": "flex", "alignItems": "center",
                      "flexWrap": "wrap", "gap": "4px", "marginBottom": "3px"}),
            html.Div(analogy, className="grow-desc"),
        ], className="grow")

    filings_intro = (
        "In the US, investment funds must file public reports with the SEC "
        "(Securities and Exchange Commission — the government's financial watchdog). "
        "Think of these filings like required homework: funds must show exactly what they own. "
        "There are 4 types, each with a different speed:"
        if en else
        "미국에서 투자펀드는 SEC(증권거래위원회 — 정부의 금융 감시기관)에 공개 보고서를 제출해야 합니다. "
        "이 공시는 일종의 '의무 숙제'입니다: 펀드가 무엇을 보유하고 있는지 공개해야 하죠. "
        "4가지 종류가 있으며, 각각 속도가 다릅니다:"
    )

    filings = _bsec(
        "How Does the Government Help?" if en else "정부가 어떻게 도움이 되나요?",
        html.P(filings_intro, className="grow-desc", style={"marginBottom": "0.7rem"}),
        _filing_row("👤", "Form 4",
                    ("⚡ 2 days" if en else "⚡ 2영업일"),
                    f"#{C['green']}",
                    ("A company executive (CEO, CFO…) buys or sells their own company's stock. "
                     "They MUST report it to the government within 2 business days. "
                     "It's like a receipt you're forced to make public."
                     if en else
                     "회사 임원(CEO, CFO 등)이 자사주를 매수·매도하면 "
                     "2영업일 이내에 정부에 반드시 보고해야 합니다. "
                     "강제로 공개해야 하는 영수증 같은 것입니다.")),
        _filing_row("📋", "SC 13D / 13G",
                    ("5–10 days" if en else "5–10영업일"),
                    f"#{C['red']}",
                    ("When any investor buys 5% or more of a company, they must disclose it within "
                     "5–10 days. 13D = they want to influence management (activist). "
                     "13G = passive, just a big investment."
                     if en else
                     "어떤 투자자든 회사 지분의 5% 이상을 매수하면 "
                     "5~10영업일 이내에 공시해야 합니다. "
                     "13D = 경영에 개입할 의도(행동주의). 13G = 수동적 대규모 투자.")),
        _filing_row("📦", ("N-PORT" if en else "N-PORT"),
                    ("Monthly" if en else "월간"),
                    "#20B2AA",
                    ("Mutual funds (like Vanguard, BlackRock) report their entire portfolio "
                     "every month. It's like a monthly inventory — you see what they added "
                     "or reduced before the quarterly report comes out."
                     if en else
                     "뮤추얼 펀드(뱅가드, 블랙록 등)는 매달 전체 포트폴리오를 보고합니다. "
                     "마치 월간 재고 목록 같아서, 분기 보고서가 나오기 전에 "
                     "무엇을 추가하거나 줄였는지 미리 볼 수 있습니다.")),
        _filing_row("🐋", "13F-HR",
                    ("Quarterly · 45 days" if en else "분기 · 45일"),
                    f"#{C['blue']}",
                    ("Every big fund (over $100M) must publish ALL their stock holdings "
                     "45 days after each quarter ends. This is the main report — but it's slow. "
                     "The other 3 above help you see moves BEFORE this comes out."
                     if en else
                     "1억 달러 이상 대형 펀드는 분기 종료 후 45일 이내에 모든 주식 보유 내역을 "
                     "공개해야 합니다. 이게 핵심 보고서지만 느립니다. "
                     "위의 3가지를 활용해 이 보고서가 나오기 전에 움직임을 파악하세요.")),
    )

    # ── 4. What do the signals mean? ─────────────────────────────────────────
    def _sig_grp(txt):
        return html.Div(txt, style={
            "fontSize": "0.58rem", "fontWeight": "700", "color": T,
            "letterSpacing": "0.6px", "textTransform": "uppercase",
            "marginTop": "0.6rem", "marginBottom": "0.2rem",
            "paddingBottom": "3px", "borderBottom": f"1px solid {T}33",
        })

    signals = _bsec(
        "What Do the Signals Mean?" if en else "신호가 무슨 의미인가요?",
        html.P(
            "Each signal is a one-line summary of what a fund or insider did. "
            "WhaleTracker detects them automatically from the filing data."
            if en else
            "각 신호는 펀드나 내부자가 무엇을 했는지 한 줄로 요약한 것입니다. "
            "월트래커가 공시 데이터에서 자동으로 감지합니다.",
            className="grow-desc", style={"marginBottom": "0.5rem"},
        ),
        # 13F group
        _sig_grp("🐋 13F signals — Quarterly whale moves" if en else "🐋 13F 신호 — 분기 Whale 움직임"),
        _grow("NEW ENTRY",   f"#{C['blue']}",  "+3 pts",
              ("The fund bought this stock for the very first time this quarter. "
               "Like a pro chef suddenly ordering an ingredient they've never used — worth noticing."
               if en else
               "이 펀드가 이번 분기에 처음으로 이 주식을 매수했습니다. "
               "프로 셰프가 전혀 쓰지 않던 재료를 갑자기 주문하는 것처럼 — 주목할 만합니다.")),
        _grow("AGG. BUY",   f"#{C['green']}", "+4 pts",
              ("The fund already owned this stock and just bought 20%+ MORE. "
               "They're doubling down because they're very confident."
               if en else
               "이미 보유 중인 주식을 이번 분기에 20% 이상 추가 매수했습니다. "
               "자신감이 매우 높아 베팅을 늘리는 것입니다.")),
        _grow("HIGH CONC",  f"#{C['amber']}", "+2 pts",
              ("This stock makes up more than 5% of the entire fund's portfolio. "
               "They've put a big chunk of their chips on this one."
               if en else
               "이 주식이 펀드 전체 포트폴리오의 5% 이상을 차지합니다. "
               "이 종목에 큰 비중을 걸고 있다는 뜻입니다.")),
        # 13D/G group
        _sig_grp("📋 13D/G signals — Ownership disclosures" if en else "📋 13D/G 신호 — 지분 공시"),
        _grow("ACTIVIST",   f"#{C['red']}",   "+5 pts",
              ("An investor bought 5%+ AND filed a 13D saying they want to change how "
               "the company is run — new management, sell off divisions, etc. "
               "The strongest signal in the system. Big changes often follow."
               if en else
               "투자자가 5% 이상 취득하고 경영에 개입할 의도를 13D로 공시했습니다 — "
               "새 경영진, 사업부 매각 등. 시스템에서 가장 강력한 신호입니다. 큰 변화가 따르는 경우가 많습니다.")),
        _grow("13G STAKE",  f"#{C['purple']}", "+2 pts",
              ("An investor quietly owns 5%+ but is NOT trying to interfere — "
               "they just see it as a great investment. Still a meaningful signal of institutional interest."
               if en else
               "투자자가 조용히 5% 이상 보유하고 있지만 경영 간섭 의도는 없습니다 — "
               "단순히 좋은 투자처로 보는 것입니다. 그래도 기관의 관심을 보여주는 의미 있는 신호입니다.")),
        # Form 4 group
        _sig_grp("👤 Form 4 signals — Insider trades" if en else "👤 Form 4 신호 — 내부자 거래"),
        _grow("INSIDER BUY", f"#{C['green']}", "+3 pts",
              ("The company's own CEO, CFO, or director bought stock with their PERSONAL money. "
               "Insiders know their company better than anyone — "
               "they only risk their own cash when they're genuinely confident."
               if en else
               "회사의 CEO, CFO, 이사가 자신의 개인 돈으로 자사주를 매수했습니다. "
               "내부자는 회사를 누구보다 잘 압니다 — "
               "진짜 확신이 있을 때만 자기 돈을 걸죠.")),
        _grow("INSIDER SELL", f"#{C['red']}",  "−2 pts",
              ("An insider sold shares. BUT — this can happen for many normal reasons "
               "(paying taxes, buying a house, portfolio diversification). "
               "Only treat it as a warning if MULTIPLE insiders sell at the same time."
               if en else
               "내부자가 주식을 매도했습니다. 하지만 — 세금 납부, 집 구입, "
               "포트폴리오 분산 등 일반적인 이유로 매도하는 경우도 많습니다. "
               "여러 내부자가 동시에 매도할 때만 경고 신호로 해석하세요.")),
        _grow("10b5-1 SELL", f"#{C['muted']}", "−0.5 pts",
              ("A pre-planned sale that was scheduled months ago — NOT a reaction to current news. "
               "Executives often set these plans in advance for tax reasons. "
               "Usually NOT a bearish signal."
               if en else
               "수개월 전에 미리 계획·확정된 매도 — 현재 뉴스에 반응한 것이 아닙니다. "
               "임원들은 세금 이유로 사전에 이런 계획을 세워두는 경우가 많습니다. "
               "보통 하락 신호가 아닙니다.")),
        # N-PORT group
        _sig_grp("📦 N-PORT signals — Monthly fund moves" if en else "📦 N-PORT 신호 — 월간 펀드 움직임"),
        _grow("FUND ACCUM",  "#20B2AA",        "+2 pts",
              ("A mutual fund increased its holdings by 5%+ this month. "
               "Shows growing fund-level confidence — and you're seeing it weeks before the quarterly 13F."
               if en else
               "뮤추얼 펀드가 이번 달 보유량을 5% 이상 늘렸습니다. "
               "펀드 수준의 확신이 높아지고 있음을 보여줍니다 — "
               "분기 13F보다 몇 주 먼저 확인할 수 있습니다.")),
        _grow("FUND SELL",   "#FF8C00",        "−1 pt",
              ("A mutual fund reduced its holdings by 5%+ this month. "
               "One month isn't alarming — but if it happens 2-3 months in a row, pay attention."
               if en else
               "뮤추얼 펀드가 이번 달 보유량을 5% 이상 줄였습니다. "
               "한 달은 큰 문제 아니지만 — 2~3개월 연속이면 주목해야 합니다.")),
    )

    # ── 5. What is the Conviction Score? ─────────────────────────────────────
    def _score_row(label, range_txt, desc, color):
        return html.Div([
            html.Div([
                html.Span(label, style={
                    "background": f"{color}18", "color": color,
                    "border": f"1px solid {color}44", "borderRadius": "5px",
                    "padding": "2px 9px", "fontSize": "0.68rem", "fontWeight": "700",
                    "marginRight": "8px", "whiteSpace": "nowrap",
                }),
                html.Span(range_txt, style={
                    "fontSize": "0.65rem", "color": f"#{C['muted']}",
                    "background": f"#{C['card2']}", "borderRadius": "4px",
                    "padding": "1px 6px", "fontWeight": "600",
                }),
            ], style={"display": "flex", "alignItems": "center", "marginBottom": "3px"}),
            html.Div(desc, className="grow-desc"),
        ], className="grow")

    score = _bsec(
        "What is the Conviction Score?" if en else "컨빅션 점수가 뭐예요?",
        html.P(
            "The Conviction Score (0–12) is like a confidence thermometer. "
            "It adds up all the positive signals across all 4 filing types. "
            "The more signals that agree, the higher the score."
            if en else
            "컨빅션 점수(0~12)는 신뢰도 온도계 같은 것입니다. "
            "4가지 공시 유형에서 발견된 긍정적 신호를 모두 합산합니다. "
            "더 많은 신호가 일치할수록 점수가 높아집니다.",
            className="grow-desc", style={"marginBottom": "0.5rem"},
        ),
        _score_row(
            "0 – 3" if en else "0 – 3점",
            "Mild interest" if en else "낮은 관심",
            ("One small signal detected. Worth knowing about, but don't rush."
             if en else "작은 신호 하나 감지. 알아두면 좋지만 서두를 필요는 없습니다."),
            f"#{C['muted']}",
        ),
        _score_row(
            "4 – 6" if en else "4 – 6점",
            "Worth watching" if en else "주목할 만함",
            ("Multiple signals or a strong single source. Add to your watchlist."
             if en else "여러 신호 또는 강력한 단일 소스. 워치리스트에 추가해 보세요."),
            f"#{C['amber']}",
        ),
        _score_row(
            "7 – 9" if en else "7 – 9점",
            "Strong signal" if en else "강한 신호",
            ("Multiple independent sources agree. High institutional conviction."
             if en else "여러 독립 소스가 동의합니다. 기관의 확신도가 높습니다."),
            "#1DB954",
        ),
        _score_row(
            "10 – 12" if en else "10 – 12점",
            "Exceptional" if en else "매우 강함",
            ("Rare. Multiple Whales + insider buys + activist filing all align. "
             "The strongest possible institutional signal."
             if en else "드문 경우입니다. 복수 Whale + 내부자 매수 + 행동주의 공시 모두 일치. "
             "가능한 가장 강력한 기관 신호입니다."),
            f"#{C['green']}",
        ),
    )

    # ── 6. 3 Steps to get started ─────────────────────────────────────────────
    def _step(n, icon, tab, action_en, action_ko):
        return html.Div([
            html.Div([
                html.Div(str(n), style={
                    "background": f"{T}22", "color": T,
                    "border": f"1px solid {T}55", "borderRadius": "50%",
                    "width": "26px", "height": "26px", "flexShrink": "0",
                    "display": "flex", "alignItems": "center", "justifyContent": "center",
                    "fontSize": "0.72rem", "fontWeight": "800",
                }),
                html.Div([
                    html.Div([
                        html.Span(icon, style={"marginRight": "5px"}),
                        html.Span(tab, style={"fontWeight": "700", "fontSize": "0.8rem",
                                               "color": f"#{C['text']}"}),
                    ], style={"marginBottom": "3px"}),
                    html.Div(action_en if en else action_ko, className="grow-desc"),
                ]),
            ], style={"display": "flex", "gap": "10px", "alignItems": "flex-start"}),
        ], className="grow")

    steps = _bsec(
        "3 Steps to Get Started" if en else "시작하는 3단계",
        html.P(
            "You don't need to understand everything at once. "
            "Follow these 3 steps to get your first useful insight:"
            if en else
            "처음부터 모든 걸 이해할 필요는 없습니다. "
            "3단계를 따라하면 첫 번째 유용한 인사이트를 얻을 수 있습니다:",
            className="grow-desc", style={"marginBottom": "0.5rem"},
        ),
        _step(1, "🌊", "Whale Heatmap",
              "Look at the Sector Rotation chart at the top. "
              "Green bars = sectors where Whales are buying. "
              "Focus on the sector with the biggest green bar — that's where smart money is flowing.",
              "상단의 섹터 로테이션 차트를 보세요. "
              "초록색 막대 = 고래들이 매수하는 섹터. "
              "가장 큰 초록 막대를 가진 섹터에 집중하세요 — 스마트머니가 흘러들어가는 곳입니다."),
        _step(2, "💡", "Recommendations",
              "Click '💡 Recommendations' and filter for 'STRONG BUY'. "
              "These are stocks where multiple Whales or signals agree. "
              "Check the conviction score — higher = more sources backing it.",
              "'💡 추천' 탭을 클릭하고 'STRONG BUY'로 필터링하세요. "
              "여러 고래나 신호가 동의하는 종목들입니다. "
              "컨빅션 점수를 확인하세요 — 높을수록 더 많은 소스가 뒷받침합니다."),
        _step(3, "📊", "My Portfolio",
              "Go to '📊 My Portfolio'. "
              "If Whales are heavily buying Tech but your portfolio is light on Tech, "
              "consider whether to rebalance. The rebalancing cards do this math for you.",
              "'📊 내 포트폴리오' 탭으로 이동하세요. "
              "고래들이 기술주를 대거 매수하는데 내 포트폴리오에 기술주 비중이 낮다면, "
              "리밸런싱을 고려해 보세요. 리밸런싱 카드가 이 계산을 대신해 줍니다."),
    )

    # ── 7. Glossary ───────────────────────────────────────────────────────────
    def _gterm(term, defn):
        return html.Div([
            html.Span(term + ": ", style={
                "fontWeight": "700", "fontSize": "0.78rem", "color": T,
            }),
            html.Span(defn, className="grow-desc",
                      style={"fontSize": "0.77rem"}),
        ], style={"marginBottom": "0.45rem", "lineHeight": "1.5"})

    glossary = _bsec(
        "Glossary — Key Terms Explained" if en else "용어 사전 — 주요 용어 설명",
        *([
            _gterm("Whale",
                   "A large institutional investor (hedge fund, pension fund) managing billions."),
            _gterm("SEC",
                   "Securities and Exchange Commission — the US government body that regulates "
                   "investment funds and requires public filings."),
            _gterm("13F",
                   "A quarterly report that large funds must file with the SEC, "
                   "showing all their stock holdings."),
            _gterm("Institutional Investor",
                   "A professional firm (not an individual) that manages money on behalf of others."),
            _gterm("Activist Investor",
                   "An investor who buys a large stake in a company and then tries to change "
                   "how it's run (new CEO, sell divisions, etc.)."),
            _gterm("Insider",
                   "Anyone with non-public information about a company — typically officers "
                   "and directors (CEO, CFO, board members)."),
            _gterm("Conviction Score",
                   "WhaleTracker's 0–12 score that aggregates all positive signals from all "
                   "4 filing types for a given stock."),
            _gterm("Sector",
                   "A category of the economy (Technology, Healthcare, Energy, Financials, etc.). "
                   "Stocks in the same sector tend to move together."),
            _gterm("Rebalancing",
                   "Adjusting your portfolio weights so they match your target allocation — "
                   "selling what's grown too big, buying what's fallen behind."),
            _gterm("Signal",
                   "An automated pattern detected in SEC filings that suggests institutional "
                   "buying or selling activity."),
        ] if en else [
            _gterm("Whale (고래)",
                   "수십억 달러를 운용하는 대형 기관 투자자(헤지펀드, 연기금 등)."),
            _gterm("SEC",
                   "미국 증권거래위원회 — 투자펀드를 규제하고 공개 보고서 제출을 요구하는 정부 기관."),
            _gterm("13F",
                   "대형 펀드가 분기마다 SEC에 제출해야 하는 보고서. 모든 주식 보유 내역이 담겨 있습니다."),
            _gterm("기관 투자자",
                   "개인이 아닌 타인의 자금을 운용하는 전문 투자 회사."),
            _gterm("행동주의 투자자",
                   "회사 지분을 대량 취득한 후 경영진 교체, 사업부 매각 등 경영 변화를 요구하는 투자자."),
            _gterm("내부자 (Insider)",
                   "비공개 정보에 접근할 수 있는 사람 — 주로 임원 및 이사(CEO, CFO, 이사회 멤버)."),
            _gterm("컨빅션 점수",
                   "4가지 공시 유형의 모든 긍정 신호를 합산한 월트래커의 0~12점 신뢰도 지수."),
            _gterm("섹터",
                   "경제의 카테고리(기술, 헬스케어, 에너지, 금융 등). 같은 섹터 주식은 함께 움직이는 경향."),
            _gterm("리밸런싱",
                   "목표 비중에 맞게 포트폴리오를 조정하는 것 — 너무 커진 것은 팔고, 줄어든 것은 삽니다."),
            _gterm("신호 (Signal)",
                   "SEC 공시에서 자동으로 감지된 기관의 매수 또는 매도 패턴을 나타내는 지표."),
        ]),
    )

    # ── 8. Disclaimer ─────────────────────────────────────────────────────────
    disclaimer = _bsec(
        "Important Disclaimer" if en else "중요 유의사항",
        html.Div([
            html.Div("⚠️", style={"fontSize": "1.4rem", "marginBottom": "6px"}),
            html.P(
                "WhaleTracker is a research and information tool — NOT financial advice. "
                "Institutional investors are brilliant but they are not always right. "
                "Always do your own research and consider your personal financial situation "
                "before making any investment decision. Past signals do not guarantee future results."
                if en else
                "월트래커는 리서치·정보 제공 도구입니다 — 금융 투자 자문이 아닙니다. "
                "기관 투자자들은 뛰어나지만 항상 옳지는 않습니다. "
                "투자 결정을 내리기 전에 반드시 본인만의 조사를 하고 "
                "개인 재무 상황을 고려하세요. 과거 신호가 미래 수익을 보장하지 않습니다.",
                className="grow-desc",
            ),
        ], style={"textAlign": "center", "padding": "0.5rem 0"}),
    )

    return html.Div(
        [intro, whales, filings, signals, score, steps, glossary, disclaimer],
        className="guide-body",
    )


def build_modal():
    return html.Div([
        html.Div([
            # Modal header
            html.Div([
                html.Div([
                    html.Span("📖", style={"marginRight": "8px"}),
                    "User Guide",
                ], className="modal-title"),
                html.Button("✕", id="modal-close", className="modal-close",
                            n_clicks=0),
            ], className="modal-header"),

            # Guide mode toggle (Standard / Beginner)
            dcc.Tabs(id="guide-mode", value="standard", className="lang-tabs", children=[
                dcc.Tab(label="📖 Standard", value="standard",
                        className="lang-tab", selected_className="lang-tab-active"),
                dcc.Tab(label="🔰 Beginner", value="beginner",
                        className="lang-tab", selected_className="lang-tab-active"),
            ]),

            # Language toggle
            dcc.Tabs(id="guide-lang", value="en", className="lang-tabs", children=[
                dcc.Tab(label="English", value="en",
                        className="lang-tab", selected_className="lang-tab-active"),
                dcc.Tab(label="한국어",   value="ko",
                        className="lang-tab", selected_className="lang-tab-active"),
            ]),

            # Guide content (rendered by callback)
            html.Div(id="guide-content", className="guide-scroll"),

        ], className="modal-box"),
    ], id="guide-modal", className="modal-overlay", style={"display": "none"})


# ── APP ────────────────────────────────────────────────────────────────────────
app = Dash(
    __name__,
    external_stylesheets=[
        "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap"
    ],
    suppress_callback_exceptions=True,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server  # Gunicorn entry point

# ── LAYOUT ─────────────────────────────────────────────────────────────────────
app.layout = html.Div([

    # Watchlist persistent store (localStorage)
    dcc.Store(id="watchlist-store", storage_type="local", data=[]),

    # Header
    html.Div([
        html.Div([
            html.Div("🐋", className="logo-emoji"),
            html.Div([
                html.Div(["WhaleTracker ", html.Span("AI", className="logo-ai")],
                         className="logo-title"),
                html.Div("Institutional 13F Intelligence Platform", className="logo-sub"),
            ]),
        ], className="header-left"),
        html.Div([
            html.Span(mode_label, className="mode-badge",
                      style={"background": f"#{mode_color}18", "color": f"#{mode_color}",
                             "border": f"1px solid #{mode_color}44"}),
            html.Span(timestamp, className="timestamp"),
            html.Button("📖 Guide", id="guide-btn", className="guide-btn", n_clicks=0),
        ], className="header-right"),
    ], className="header"),

    # News banner (between header and KPI strip)
    build_news_banner(market_news),

    # KPI strip
    html.Div([
        kpi_card("WHALES TRACKED",   str(live_whales),          "active institutions",    C["blue"]),
        kpi_card("ACTIVE SIGNALS",   str(active_signals),        "13F · 13D/G · Form 4 · N-PORT",  C["green"]),
        kpi_card("PORTFOLIO VALUE",  f"${port_value:,.0f}",      "at avg cost basis",      C["purple"]),
        kpi_card("TOP CONVICTION",   top_rec.get("ticker", "—"), top_rec.get("recommendation", "—"), C["amber"]),
    ], className="kpi-strip"),

    # Tabs
    dcc.Tabs(id="main-tabs", value="tab-whales", className="main-tabs", children=[
        dcc.Tab(label="🌊  Whale Heatmap",   value="tab-whales",
                className="tab", selected_className="tab-active"),
        dcc.Tab(label="💡  Recommendations", value="tab-recs",
                className="tab", selected_className="tab-active"),
        dcc.Tab(label="📊  My Portfolio",    value="tab-port",
                className="tab", selected_className="tab-active"),
        dcc.Tab(label="📈  Macro",           value="tab-macro",
                className="tab", selected_className="tab-active"),
    ]),

    html.Div(id="tab-content", style={"paddingTop": "1.2rem"}),

    # Guide Modal
    build_modal(),

], className="app-shell")


# ── CALLBACKS ──────────────────────────────────────────────────────────────────
@app.callback(Output("tab-content", "children"), Input("main-tabs", "value"))
def render_tab(tab: str):
    if tab == "tab-whales":
        return build_whale_tab()
    if tab == "tab-recs":
        return html.Div([
            # Filter bar
            html.Div([
                dcc.RadioItems(
                    id="rec-filter",
                    options=[{"label": v, "value": v}
                             for v in ["ALL", "STRONG BUY", "BUY", "HOLD", "SELL",
                                       "📌 WATCHLIST"]],
                    value="ALL",
                    inline=True,
                    className="rec-filter",
                    inputStyle={"display": "none"},
                ),
                html.Div(
                    f"{len(recommendations)} tickers · {live_whales} whales",
                    className="rec-count",
                ),
            ], className="rec-filter-row"),
            # Watchlist input (shown only when WATCHLIST filter is active)
            html.Div([
                html.Div("📌 Watchlist", style={
                    "fontSize": "0.68rem", "fontWeight": "700",
                    "color": f"#{C['amber']}", "marginRight": "8px",
                }),
                dcc.Input(
                    id="watchlist-input",
                    type="text",
                    placeholder="Add ticker (e.g. AAPL)…",
                    debounce=False,
                    className="watchlist-input",
                    style={
                        "background": f"#{C['card2']}", "border": f"1px solid #{C['border']}",
                        "borderRadius": "6px", "color": f"#{C['text']}",
                        "padding": "4px 10px", "fontSize": "0.78rem",
                        "outline": "none", "width": "180px", "marginRight": "6px",
                    },
                ),
                html.Button("＋ Add", id="watchlist-add", n_clicks=0, style={
                    "background": f"#{C['amber']}22", "color": f"#{C['amber']}",
                    "border": f"1px solid #{C['amber']}44", "borderRadius": "6px",
                    "padding": "4px 12px", "fontSize": "0.72rem", "fontWeight": "700",
                    "cursor": "pointer",
                }),
                html.Div(id="watchlist-chips", style={
                    "display": "flex", "flexWrap": "wrap",
                    "gap": "5px", "marginLeft": "10px",
                }),
            ], id="watchlist-bar", style={
                "display": "none",   # toggled by callback
                "alignItems": "center", "flexWrap": "wrap", "gap": "6px",
                "background": f"#{C['card']}", "borderRadius": "8px",
                "padding": "8px 14px", "marginBottom": "1rem",
                "border": f"1px solid #{C['amber']}33",
            }),
            html.Div(id="rec-cards", children=build_rec_cards("ALL")),
        ])
    if tab == "tab-port":
        return build_portfolio_tab()
    if tab == "tab-macro":
        return build_macro_tab()


@app.callback(
    Output("rec-cards", "children"),
    Input("rec-filter",       "value"),
    Input("watchlist-store",  "data"),
)
def update_rec_cards(filter_val: str, watchlist):
    return build_rec_cards(filter_val, watchlist or [])


@app.callback(
    Output("watchlist-bar", "style"),
    Input("rec-filter", "value"),
)
def toggle_watchlist_bar(filter_val):
    base = {
        "alignItems": "center", "flexWrap": "wrap", "gap": "6px",
        "background": f"#{C['card']}", "borderRadius": "8px",
        "padding": "8px 14px", "marginBottom": "1rem",
        "border": f"1px solid #{C['amber']}33",
    }
    base["display"] = "flex" if filter_val == "📌 WATCHLIST" else "none"
    return base


@app.callback(
    Output("watchlist-store", "data"),
    Output("watchlist-input", "value"),
    Input("watchlist-add",  "n_clicks"),
    State("watchlist-input", "value"),
    State("watchlist-store", "data"),
    prevent_initial_call=True,
)
def add_to_watchlist(_clicks, ticker_raw, current_list):
    current = list(current_list or [])
    if ticker_raw:
        ticker = ticker_raw.strip().upper()
        if ticker and ticker not in current:
            current.append(ticker)
    return current, ""


@app.callback(
    Output("watchlist-chips", "children"),
    Input("watchlist-store", "data"),
)
def render_watchlist_chips(watchlist):
    chips = []
    for ticker in (watchlist or []):
        chips.append(html.Span([
            ticker,
            html.Span(" ×", style={"cursor": "pointer", "marginLeft": "4px",
                                    "opacity": "0.7"}),
        ], style={
            "background": f"#{C['amber']}1A", "color": f"#{C['amber']}",
            "border": f"1px solid #{C['amber']}44", "borderRadius": "5px",
            "padding": "2px 8px", "fontSize": "0.7rem", "fontWeight": "700",
        }))
    return chips


@app.callback(
    Output("guide-modal", "style"),
    Input("guide-btn",   "n_clicks"),
    Input("modal-close", "n_clicks"),
    prevent_initial_call=True,
)
def toggle_modal(_open, _close):
    return {"display": "flex"} if ctx.triggered_id == "guide-btn" else {"display": "none"}


@app.callback(
    Output("guide-content", "children"),
    Input("guide-lang",  "value"),
    Input("guide-mode",  "value"),
)
def render_guide(lang: str, mode: str):
    if mode == "beginner":
        return build_beginner_guide(lang)
    return build_guide(lang)


# ── SCHEDULER ──────────────────────────────────────────────────────────────────
# Start background Slack-alert job only when a token is configured.
# In mock/dev mode this is a no-op if SLACK_BOT_TOKEN is not set.
if os.getenv("SLACK_BOT_TOKEN"):
    try:
        from src.scheduler import start as _start_scheduler  # noqa: PLC0415
        _start_scheduler({
            "filings":         filings,
            "activist":        activist,
            "insiders":        insiders,
            "nport":           nport,
            "recommendations": recommendations,
            "rebalancing":     rebalancing,
            "portfolio":       portfolio,
        })
    except Exception as _sched_err:
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "Scheduler could not start: %s", _sched_err
        )


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.getenv("PORT", 8050)))
