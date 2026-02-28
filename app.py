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


@app.callback(Output("guide-content", "children"), Input("guide-lang", "value"))
def render_guide(lang: str):
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
