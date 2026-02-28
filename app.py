"""
app.py — WhaleTracker AI | Dash Dashboard
------------------------------------------
Run locally:  python app.py
Production:   gunicorn app:server --bind 0.0.0.0:$PORT
"""

import os
from datetime import datetime

from dash import Dash, html, dcc, Input, Output, ctx
import plotly.graph_objects as go
from dotenv import load_dotenv

from src.data_collector import (
    fetch_all_whale_filings,
    fetch_13dg_filings,
    fetch_form4_filings,
    fetch_nport_filings,
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
        sections.append(html.Div([
            html.Div([
                html.Span(whale, className="whale-name"),
                html.Span(si["label"], style={
                    "background": f"{si['color']}18", "color": si["color"],
                    "border": f"1px solid {si['color']}44", "borderRadius": "5px",
                    "padding": "2px 9px", "fontSize": "0.62rem", "fontWeight": "700",
                }),
            ], style={"display": "flex", "alignItems": "center", "gap": "10px"}),
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


def build_rec_cards(filter_val: str = "ALL"):
    filtered = (
        recommendations if filter_val == "ALL"
        else [r for r in recommendations if r["recommendation"] == filter_val]
    )
    if not filtered:
        return html.Div(
            f"No {filter_val} recommendations in the current dataset.",
            className="empty-state",
        )
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


def build_guide(lang: str):
    en = lang == "en"

    # ── Overview ──────────────────────────────────────────────────────────────
    overview = _gsec(
        "Overview" if en else "서비스 개요",
        html.P(
            "WhaleTracker AI monitors 13F filings from top institutional investors "
            "(Whales) and translates Smart Money movements into actionable signals "
            "and portfolio recommendations."
            if en else
            "WhaleTracker AI는 대형 기관 투자자(Whale)의 SEC 13F 공시를 분석해 "
            "스마트머니의 움직임을 신호와 포트폴리오 추천으로 변환합니다.",
            className="grow-desc",
        ),
    )

    # ── Tracked Whales ────────────────────────────────────────────────────────
    whales = _gsec(
        "Tracked Institutions" if en else "추적 기관",
        _whale_row("Berkshire Hathaway",    "Warren Buffett"),
        _whale_row("Bridgewater Associates","Ray Dalio"),
        _whale_row("Appaloosa Management",  "David Tepper"),
        _whale_row("Pershing Square",       "Bill Ackman"),
        _whale_row("Tiger Global",          "Chase Coleman"),
    )

    # ── Signal Definitions ────────────────────────────────────────────────────
    signals = _gsec(
        "Signal Definitions" if en else "신호 정의",
        _grow("AGG. BUY",    f"#{C['green']}", "+4 pts",
              ("Share count increased >20% QoQ — the strongest conviction signal."
               if en else "전 분기 대비 보유 주식 수 20% 초과 증가 — 가장 강한 매수 신호.")),
        _grow("NEW ENTRY",   f"#{C['blue']}",  "+3 pts",
              ("Stock was absent from the prior quarter's filing — fresh position."
               if en else "이전 분기 공시에 없던 종목 — 신규 진입 포지션.")),
        _grow("HIGH CONC",   f"#{C['amber']}", "+2 pts",
              ("Position exceeds 5% of the Whale's total portfolio value."
               if en else "해당 종목이 Whale 포트폴리오의 5% 이상을 차지.")),
        _grow("HOLD",        "#4A5568",        "+0 pts",
              ("No significant change detected from the prior quarter."
               if en else "전 분기 대비 유의미한 변화 없음.")),
    )

    # ── Recommendation Levels ─────────────────────────────────────────────────
    recs = _gsec(
        "Recommendation Levels" if en else "추천 등급",
        _grow("🚀 STRONG BUY", f"#{C['green']}", "score ≥ 6  or  ≥ 4 with 2+ Whales",
              ("Highest institutional conviction — multiple Whales agree."
               if en else "최고 기관 확신도 — 복수 Whale이 동의한 종목.")),
        _grow("↑ BUY",         "#1DB954",        "score ≥ 3",
              ("Strong single-Whale signal worth watching."
               if en else "단일 Whale의 강한 신호 — 주목할 만한 종목.")),
        _grow("→ HOLD",        f"#{C['amber']}", "score ≥ 1",
              ("Mild institutional interest — monitor but don't rush."
               if en else "낮은 기관 관심도 — 모니터링 유지.")),
        _grow("↓ SELL",        f"#{C['red']}",   "score = 0",
              ("No institutional backing detected in this filing cycle."
               if en else "해당 공시 주기에 기관 지지 없음.")),
    )

    # ── How to use each tab ───────────────────────────────────────────────────
    tabs_guide = _gsec(
        "How to Use Each Tab" if en else "탭별 사용법",
        _gtab("🌊", "Whale Heatmap",
              ("View holdings for each institution sorted by signal strength. "
               "The Sector Rotation bar chart shows net institutional inflows by sector."
               if en else
               "각 기관의 보유 종목을 신호 강도별로 확인합니다. "
               "섹터 로테이션 차트는 섹터별 기관 순유입량을 보여줍니다.")),
        _gtab("💡", "Recommendations",
              ("Filter by recommendation type (ALL / STRONG BUY / BUY / HOLD / SELL). "
               "The conviction bar shows score out of a 12-point maximum."
               if en else
               "추천 유형(전체 / STRONG BUY / BUY / HOLD / SELL)으로 필터링합니다. "
               "컨빅션 바는 최대 12점 기준 점수를 나타냅니다.")),
        _gtab("📊", "My Portfolio",
              ("Compare your current sector weights against Whale-adjusted targets. "
               "Sectors drifting more than 5pp trigger a rebalancing suggestion."
               if en else
               "현재 섹터 비중을 Whale 신호가 반영된 목표 비중과 비교합니다. "
               "5%p 이상 이탈한 섹터는 리밸런싱 제안이 표시됩니다.")),
    )

    # ── Important Notes ───────────────────────────────────────────────────────
    notes = _gsec(
        "Important Notes" if en else "주요 참고사항",
        html.Ul([
            html.Li("13F filings have a ~45-day reporting lag after quarter end."
                    if en else "13F 공시는 분기 종료 후 약 45일의 보고 지연이 있습니다."),
            html.Li("MOCK MODE displays sample data — set DATA_MODE=live in .env for real data."
                    if en else "MOCK 모드는 샘플 데이터를 표시합니다. 실시간 데이터는 .env에서 DATA_MODE=live로 변경하세요."),
            html.Li("Edit my_portfolio.json to reflect your actual holdings."
                    if en else "my_portfolio.json을 편집해 실제 보유 종목을 입력하세요."),
            html.Li("Conviction score max = 12 (AGGRESSIVE_BUY × 3 whales)."
                    if en else "컨빅션 최대점수 = 12점 (AGGRESSIVE_BUY × 3 Whale)."),
        ], className="guide-notes"),
    )

    return html.Div([overview, whales, signals, recs, tabs_guide, notes],
                    className="guide-body")


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
)
server = app.server  # Gunicorn entry point

# ── LAYOUT ─────────────────────────────────────────────────────────────────────
app.layout = html.Div([

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
            html.Div([
                dcc.RadioItems(
                    id="rec-filter",
                    options=[{"label": v, "value": v}
                             for v in ["ALL", "STRONG BUY", "BUY", "HOLD", "SELL"]],
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
            html.Div(id="rec-cards", children=build_rec_cards("ALL")),
        ])
    if tab == "tab-port":
        return build_portfolio_tab()


@app.callback(Output("rec-cards", "children"), Input("rec-filter", "value"))
def update_rec_cards(filter_val: str):
    return build_rec_cards(filter_val)


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


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.getenv("PORT", 8050)))
