"""
app.py — WhaleTracker AI | Fintech Dashboard
---------------------------------------------
Run:  streamlit run app.py
"""

import os
import logging
from datetime import datetime

import streamlit as st
import plotly.graph_objects as go
from dotenv import load_dotenv

from src.data_collector import fetch_all_whale_filings
from src.analysis_engine import build_recommendations, get_sector_rotation_signals
from src.portfolio_manager import load_portfolio, suggest_rebalancing, get_current_sector_weights

load_dotenv()
logging.basicConfig(level=logging.INFO)
DATA_MODE = os.getenv("DATA_MODE", "mock")

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WhaleTracker AI",
    page_icon="🐋",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── DESIGN TOKENS ──────────────────────────────────────────────────────────────
C = {
    "bg":     "0D0F14",  "card":  "161922",  "card2": "1E2130",
    "text":   "E8ECF0",  "muted": "8892A4",  "border":"ffffff12",
    "green":  "00D09C",  "red":   "FF4757",  "blue":  "4B7BE5",
    "amber":  "FFB800",  "purple":"A78BFA",
}

SIG = {
    "NEW_ENTRY":          {"color": f"#{C['blue']}",   "label": "NEW ENTRY"},
    "AGGRESSIVE_BUY":     {"color": f"#{C['green']}",  "label": "AGG. BUY"},
    "HIGH_CONCENTRATION": {"color": f"#{C['amber']}",  "label": "HIGH CONC"},
    "HOLD":               {"color": "#4A5568",          "label": "HOLD"},
}

REC = {
    "STRONG BUY": {"color": f"#{C['green']}", "icon": "🚀"},
    "BUY":        {"color": "#1DB954",         "icon": "↑"},
    "HOLD":       {"color": f"#{C['amber']}", "icon": "→"},
    "SELL":       {"color": f"#{C['red']}",   "icon": "↓"},
}

PALETTE = [f"#{C['blue']}", f"#{C['green']}", f"#{C['amber']}",
           f"#{C['purple']}", f"#{C['red']}", "#20B2AA", "#FF8C00", "#9B59B6"]


def _plotly_base(**kwargs) -> dict:
    """Shared Plotly layout defaults — transparent dark theme."""
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, -apple-system, sans-serif", color=f"#{C['text']}"),
        margin=dict(l=0, r=0, t=36, b=0),
        **kwargs,
    )


# ── GLOBAL CSS ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Reset Streamlit chrome ── */
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding: 1.5rem 2rem 3rem !important; max-width: 100% !important; }}

/* ── Global ── */
html, body, [data-testid="stApp"] {{
    background: #{C["bg"]} !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
    color: #{C["text"]} !important;
}}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {{
    background: #{C["card"]} !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 2px !important;
    border: 1px solid #{C["border"]} !important;
    width: fit-content !important;
    margin-bottom: 0.5rem !important;
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent !important;
    border-radius: 8px !important;
    color: #{C["muted"]} !important;
    font-weight: 500 !important;
    font-size: 0.87rem !important;
    padding: 7px 20px !important;
    border: none !important;
    transition: all 0.15s ease !important;
}}
.stTabs [aria-selected="true"] {{
    background: #{C["card2"]} !important;
    color: #{C["text"]} !important;
    font-weight: 600 !important;
}}
.stTabs [data-baseweb="tab-border"] {{ display: none !important; }}
.stTabs [data-baseweb="tab-panel"] {{ padding-top: 1.2rem !important; }}

/* ── Radio buttons — pill filter style ── */
div[data-testid="stRadio"] > div {{
    flex-direction: row !important;
    flex-wrap: wrap !important;
    gap: 6px !important;
    align-items: center !important;
}}
div[data-testid="stRadio"] label {{
    background: #{C["card"]} !important;
    border: 1px solid #{C["border"]} !important;
    border-radius: 20px !important;
    padding: 5px 14px !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
    color: #{C["muted"]} !important;
    white-space: nowrap !important;
}}
div[data-testid="stRadio"] label:has(input:checked) {{
    background: #{C["card2"]} !important;
    border-color: #{C["blue"]} !important;
    color: #{C["text"]} !important;
}}
div[data-testid="stRadio"] input {{ display: none !important; }}

/* ── Expander ── */
details summary {{
    background: #{C["card"]} !important;
    border: 1px solid #{C["border"]} !important;
    border-radius: 10px !important;
    padding: 0.6rem 1rem !important;
    color: #{C["text"]} !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    cursor: pointer !important;
}}
details[open] summary {{ border-radius: 10px 10px 0 0 !important; }}

/* ── Spinner ── */
.stSpinner > div {{ border-top-color: #{C["green"]} !important; }}

/* ── Scrollbar ── */
::-webkit-scrollbar {{ width: 5px; height: 5px; }}
::-webkit-scrollbar-track {{ background: #{C["bg"]}; }}
::-webkit-scrollbar-thumb {{ background: #{C["border"]}; border-radius: 3px; }}
::-webkit-scrollbar-thumb:hover {{ background: #{C["muted"]}44; }}
</style>
""", unsafe_allow_html=True)


# ── DATA ───────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=21_600)
def load_data():
    filings        = fetch_all_whale_filings()
    recommendations = build_recommendations(filings)
    rotation       = get_sector_rotation_signals(filings)
    portfolio      = load_portfolio()
    rebalancing    = suggest_rebalancing(portfolio, rotation)
    return filings, recommendations, rotation, portfolio, rebalancing


with st.spinner("Fetching institutional intelligence..."):
    filings, recommendations, rotation, portfolio, rebalancing = load_data()

# Derived metrics
active_signals = sum(
    1 for holds in filings.values()
    for h in holds if h.get("signal", "HOLD") != "HOLD"
)
live_whales  = len([w for w, h in filings.items() if h])
port_value   = sum(
    h.get("quantity", 0) * h.get("avg_cost", 0.0)
    for h in portfolio.get("holdings", [])
)
top_rec      = recommendations[0] if recommendations else {}
current_weights = get_current_sector_weights(portfolio)


# ── HEADER ─────────────────────────────────────────────────────────────────────
mode_color = C["amber"] if DATA_MODE == "mock" else C["green"]
mode_label = "MOCK DATA" if DATA_MODE == "mock" else "● LIVE"
timestamp  = datetime.now().strftime("%b %d, %Y · %H:%M")

st.markdown(f"""
<div style="display:flex;align-items:center;justify-content:space-between;
     padding-bottom:1.2rem;border-bottom:1px solid #{C['border']};margin-bottom:1.4rem">
  <div style="display:flex;align-items:center;gap:14px">
    <div style="font-size:2rem;line-height:1;filter:drop-shadow(0 0 12px #{C['blue']}66)">🐋</div>
    <div>
      <div style="font-size:1.4rem;font-weight:800;color:#{C['text']};
                  letter-spacing:-0.5px;line-height:1.15">
        WhaleTracker <span style="color:#{C['blue']}">AI</span>
      </div>
      <div style="font-size:0.7rem;color:#{C['muted']};font-weight:400;
                  letter-spacing:0.4px;margin-top:2px">
        Institutional 13F Intelligence Platform
      </div>
    </div>
  </div>
  <div style="display:flex;align-items:center;gap:10px">
    <span style="background:#{mode_color}18;color:#{mode_color};
                 border:1px solid #{mode_color}44;border-radius:6px;
                 padding:3px 10px;font-size:0.67rem;font-weight:700;
                 letter-spacing:0.8px">
      {mode_label}
    </span>
    <span style="color:#{C['muted']};font-size:0.75rem">{timestamp}</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ── KPI STRIP ──────────────────────────────────────────────────────────────────
kpis = [
    ("WHALES TRACKED",   f"{live_whales}",             "active institutions",     C["blue"]),
    ("ACTIVE SIGNALS",   f"{active_signals}",           "NEW ENTRY · AGG. BUY",   C["green"]),
    ("PORTFOLIO VALUE",  f"${port_value:,.0f}",         "at avg cost basis",       C["purple"]),
    ("TOP CONVICTION",   top_rec.get("ticker", "—"),   top_rec.get("recommendation", "—"), C["amber"]),
]

for col, (label, value, sub, accent) in zip(st.columns(4), kpis):
    with col:
        st.markdown(f"""
<div style="background:#{C['card']};border:1px solid #{C['border']};
     border-radius:14px;padding:1.1rem 1.3rem;border-left:3px solid #{accent};
     position:relative;overflow:hidden">
  <div style="position:absolute;right:12px;top:50%;transform:translateY(-50%);
              font-size:2.8rem;opacity:0.04;color:#{accent};font-weight:900">◈</div>
  <div style="font-size:0.63rem;color:#{C['muted']};text-transform:uppercase;
              letter-spacing:1px;font-weight:600;margin-bottom:0.3rem">{label}</div>
  <div style="font-size:1.65rem;font-weight:800;color:#{C['text']};
              letter-spacing:-0.5px;line-height:1.05">{value}</div>
  <div style="font-size:0.68rem;color:#{C['muted']};margin-top:0.2rem">{sub}</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)


# ── TABS ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🌊  Whale Heatmap",
    "💡  Recommendations",
    "📊  My Portfolio",
])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — WHALE HEATMAP
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:

    # ── Sector Rotation Chart ──────────────────────────────────────────────────
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
        fig.update_layout(
            **_plotly_base(
                height=175,
                title=dict(
                    text="Sector Rotation — Net Whale Flow",
                    font=dict(size=12, color=f"#{C['muted']}"),
                    x=0, xanchor="left", y=0.98,
                ),
                xaxis=dict(showgrid=False, showticklabels=False,
                           zeroline=True, zerolinecolor=f"#{C['border']}", zerolinewidth=1),
                yaxis=dict(showgrid=False, tickfont=dict(size=11), autorange="reversed"),
                bargap=0.4,
            )
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown(f"<div style='height:0.3rem'></div>", unsafe_allow_html=True)

    # ── Per-Whale Card Sections ────────────────────────────────────────────────
    for whale, holdings in filings.items():
        if not holdings:
            continue

        non_hold  = sum(1 for h in holdings if h.get("signal", "HOLD") != "HOLD")
        sig_score = {"AGGRESSIVE_BUY": 4, "NEW_ENTRY": 3, "HIGH_CONCENTRATION": 2, "HOLD": 0}
        top_sig   = max((h.get("signal", "HOLD") for h in holdings), key=lambda s: sig_score.get(s, 0))
        si        = SIG.get(top_sig, SIG["HOLD"])

        st.markdown(f"""
<div style="display:flex;align-items:center;justify-content:space-between;
     margin:1.4rem 0 0.8rem;padding-bottom:0.6rem;border-bottom:1px solid #{C['border']}">
  <div style="display:flex;align-items:center;gap:10px">
    <span style="font-size:1rem;font-weight:700;color:#{C['text']}">{whale}</span>
    <span style="background:{si['color']}18;color:{si['color']};
                 border:1px solid {si['color']}44;border-radius:5px;
                 padding:2px 9px;font-size:0.62rem;font-weight:700;letter-spacing:0.5px">
      {si['label']}
    </span>
  </div>
  <span style="color:#{C['muted']};font-size:0.74rem">
    {len(holdings)} positions · <span style="color:#{C['green']}">{non_hold}</span> active signals
  </span>
</div>
""", unsafe_allow_html=True)

        # Holdings grid — 4 cards per row
        N = 4
        for i in range(0, len(holdings), N):
            row_cols = st.columns(N)
            for col, h in zip(row_cols, holdings[i:i + N]):
                sig  = h.get("signal", "HOLD")
                info = SIG.get(sig, SIG["HOLD"])
                val  = h.get("value_usd", 0)
                pct  = h.get("portfolio_pct", 0) * 100
                val_str = f"${val/1e9:.1f}B" if val >= 1e9 else f"${val/1e6:.0f}M"

                with col:
                    st.markdown(f"""
<div style="background:#{C['card']};border:1px solid #{C['border']};
     border-radius:12px;padding:0.9rem;margin-bottom:0.6rem;
     border-top:2px solid {info['color']}">
  <div style="display:flex;align-items:center;justify-content:space-between;
              margin-bottom:0.45rem">
    <span style="font-size:1.1rem;font-weight:800;color:#{C['text']};
                 letter-spacing:-0.3px">{h['ticker']}</span>
    <span style="background:{info['color']}18;color:{info['color']};
                 border-radius:4px;padding:1px 7px;
                 font-size:0.6rem;font-weight:700;letter-spacing:0.3px">
      {info['label']}
    </span>
  </div>
  <div style="font-size:0.71rem;color:#{C['muted']};margin-bottom:0.65rem;
              white-space:nowrap;overflow:hidden;text-overflow:ellipsis">
    {h.get('company', '')}
  </div>
  <div style="display:flex;justify-content:space-between;align-items:flex-end">
    <div>
      <div style="font-size:0.58rem;color:#{C['muted']};text-transform:uppercase;
                  letter-spacing:0.5px;margin-bottom:1px">Value</div>
      <div style="font-size:0.9rem;font-weight:600;color:#{C['text']}">{val_str}</div>
    </div>
    <div style="text-align:right">
      <div style="font-size:0.58rem;color:#{C['muted']};text-transform:uppercase;
                  letter-spacing:0.5px;margin-bottom:1px">Portfolio</div>
      <div style="font-size:0.9rem;font-weight:700;color:{info['color']}">{pct:.1f}%</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — RECOMMENDATIONS
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:

    # ── Filter bar ────────────────────────────────────────────────────────────
    f_col, _, count_col = st.columns([4, 2, 2])
    with f_col:
        rec_filter = st.radio(
            "", ["ALL", "STRONG BUY", "BUY", "HOLD", "SELL"],
            horizontal=True, label_visibility="collapsed",
        )
    with count_col:
        st.markdown(f"""
<div style="text-align:right;padding-top:5px">
  <span style="color:#{C['muted']};font-size:0.73rem">
    {len(recommendations)} tickers · {live_whales} whales
  </span>
</div>
""", unsafe_allow_html=True)

    # ── Filter logic ──────────────────────────────────────────────────────────
    filtered = (
        recommendations if rec_filter == "ALL"
        else [r for r in recommendations if r["recommendation"] == rec_filter]
    )

    if not filtered:
        st.markdown(f"""
<div style="background:#{C['card']};border:1px solid #{C['border']};
     border-radius:12px;padding:2.5rem;text-align:center;
     color:#{C['muted']};font-size:0.88rem;margin-top:1rem">
  No <strong style="color:#{C['text']}">{rec_filter}</strong> recommendations in the current dataset.
</div>
""", unsafe_allow_html=True)
    else:
        # ── Recommendation card grid — 3 per row ──────────────────────────────
        N = 3
        for i in range(0, len(filtered), N):
            row_cols = st.columns(N)
            for col, r in zip(row_cols, filtered[i:i + N]):
                rec_key   = r["recommendation"]
                ri        = REC.get(rec_key, {"color": "#4A5568", "icon": "?"})
                score     = r.get("conviction_score", 0)
                bar_pct   = min(100, int(score / 12 * 100))
                whales_str = " · ".join(r.get("supporting_whales", []))
                macro     = r.get("macro_note", "")

                # Build signal badge HTML
                sig_badges = "".join(
                    f'<span style="background:{SIG.get(s.strip(), SIG["HOLD"])["color"]}18;'
                    f'color:{SIG.get(s.strip(), SIG["HOLD"])["color"]};'
                    f'border-radius:4px;padding:2px 7px;'
                    f'font-size:0.6rem;font-weight:600;margin-right:4px;display:inline-block">'
                    f'{SIG.get(s.strip(), SIG["HOLD"])["label"]}</span>'
                    for s in (r.get("signal_summary") or "").split(",")
                    if s.strip()
                )
                macro_html = (
                    f'<div style="font-size:0.65rem;color:#{C["amber"]};'
                    f'margin-top:0.45rem">⚡ {macro}</div>'
                    if macro else ""
                )

                with col:
                    st.markdown(f"""
<div style="background:#{C['card']};border:1px solid #{C['border']};
     border-radius:14px;padding:1.1rem;margin-bottom:0.8rem">

  <!-- ticker + rec badge -->
  <div style="display:flex;align-items:flex-start;
              justify-content:space-between;margin-bottom:0.75rem">
    <div>
      <div style="font-size:1.5rem;font-weight:800;color:#{C['text']};
                  letter-spacing:-0.8px;line-height:1">{r['ticker']}</div>
      <div style="font-size:0.7rem;color:#{C['muted']};margin-top:3px;
                  max-width:120px;white-space:nowrap;overflow:hidden;
                  text-overflow:ellipsis">{r.get('company', '')}</div>
    </div>
    <div style="background:{ri['color']}1A;border:1px solid {ri['color']}55;
                border-radius:9px;padding:5px 11px;text-align:center;min-width:80px">
      <div style="font-size:1.1rem;line-height:1.2">{ri['icon']}</div>
      <div style="font-size:0.6rem;font-weight:700;color:{ri['color']};
                  letter-spacing:0.3px;margin-top:1px;white-space:nowrap">{rec_key}</div>
    </div>
  </div>

  <!-- conviction score bar -->
  <div style="margin-bottom:0.7rem">
    <div style="display:flex;justify-content:space-between;margin-bottom:4px">
      <span style="font-size:0.6rem;color:#{C['muted']};
                   text-transform:uppercase;letter-spacing:0.6px">Conviction</span>
      <span style="font-size:0.65rem;font-weight:700;color:{ri['color']}">{score} / 12</span>
    </div>
    <div style="background:#{C['card2']};border-radius:3px;height:4px;overflow:hidden">
      <div style="width:{bar_pct}%;height:100%;
                  background:linear-gradient(90deg,{ri['color']}88,{ri['color']});
                  border-radius:3px;transition:width 0.4s ease"></div>
    </div>
  </div>

  <!-- signal badges -->
  <div style="margin-bottom:0.6rem">{sig_badges}</div>

  <!-- whale footer -->
  <div style="font-size:0.67rem;color:#{C['muted']};
              border-top:1px solid #{C['border']};
              padding-top:0.5rem;line-height:1.4">
    🐋 {whales_str or "—"}
  </div>

  {macro_html}
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MY PORTFOLIO
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    holdings_list  = portfolio.get("holdings", [])
    target_weights = portfolio.get("target_sector_weights", {})
    top_sector     = max(current_weights, key=current_weights.get) if current_weights else "—"

    # ── Portfolio mini-KPIs ───────────────────────────────────────────────────
    p_kpis = [
        ("TOTAL VALUE",      f"${port_value:,.0f}", f"{len(holdings_list)} positions",          C["blue"]),
        ("SECTORS",          str(len(current_weights)), "GICS sectors covered",                 C["purple"]),
        ("DOMINANT SECTOR",  top_sector, f"{current_weights.get(top_sector, 0):.0%} of portfolio", C["amber"]),
    ]
    for col, (label, val, sub, accent) in zip(st.columns(3), p_kpis):
        with col:
            st.markdown(f"""
<div style="background:#{C['card']};border:1px solid #{C['border']};
     border-radius:14px;padding:0.9rem 1.1rem;border-left:3px solid #{accent}">
  <div style="font-size:0.62rem;color:#{C['muted']};text-transform:uppercase;
              letter-spacing:1px;font-weight:600;margin-bottom:0.25rem">{label}</div>
  <div style="font-size:1.4rem;font-weight:800;color:#{C['text']};
              letter-spacing:-0.3px">{val}</div>
  <div style="font-size:0.68rem;color:#{C['muted']};margin-top:0.15rem">{sub}</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

    # ── Charts row ────────────────────────────────────────────────────────────
    donut_col, bar_col = st.columns([1, 2])

    # Portfolio donut
    with donut_col:
        if current_weights:
            labels = list(current_weights.keys())
            values = [v * 100 for v in current_weights.values()]

            fig = go.Figure(go.Pie(
                labels=labels, values=values,
                hole=0.62,
                marker=dict(
                    colors=PALETTE[:len(labels)],
                    line=dict(color=f"#{C['bg']}", width=2),
                ),
                textinfo="label+percent",
                textfont=dict(size=11, color=f"#{C['text']}"),
                hovertemplate="<b>%{label}</b><br>%{value:.1f}%<extra></extra>",
                direction="clockwise",
            ))
            fig.update_layout(
                **_plotly_base(
                    height=250,
                    showlegend=False,
                    title=dict(
                        text="Current Allocation",
                        font=dict(size=12, color=f"#{C['muted']}"),
                        x=0.5, xanchor="center", y=0.97,
                    ),
                    annotations=[dict(
                        text=f"<b>${port_value / 1000:.1f}K</b>",
                        x=0.5, y=0.5, showarrow=False,
                        font=dict(size=15, color=f"#{C['text']}", family="Inter"),
                    )],
                ),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Current vs Target grouped bar
    with bar_col:
        all_sec   = sorted(set(list(current_weights) + list(target_weights)))
        curr_vals = [current_weights.get(s, 0) * 100 for s in all_sec]
        targ_vals = [target_weights.get(s, 0) * 100  for s in all_sec]

        fig = go.Figure([
            go.Bar(
                name="Current", x=all_sec, y=curr_vals,
                marker=dict(color=f"#{C['blue']}", line=dict(width=0)), opacity=0.85,
            ),
            go.Bar(
                name="Target", x=all_sec, y=targ_vals,
                marker=dict(color=f"#{C['green']}", line=dict(width=0)), opacity=0.85,
            ),
        ])
        fig.update_layout(
            **_plotly_base(
                height=250,
                barmode="group",
                title=dict(
                    text="Current vs Target Sector Weights (%)",
                    font=dict(size=12, color=f"#{C['muted']}"),
                    x=0, xanchor="left",
                ),
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.01,
                    xanchor="right", x=1,
                    font=dict(size=11), bgcolor="rgba(0,0,0,0)",
                ),
                xaxis=dict(showgrid=False, tickfont=dict(size=10)),
                yaxis=dict(
                    showgrid=True, gridcolor=f"#{C['border']}",
                    ticksuffix="%", tickfont=dict(size=10),
                ),
                bargap=0.25, bargroupgap=0.06,
            ),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ── Rebalancing Cards ─────────────────────────────────────────────────────
    st.markdown(f"""
<div style="display:flex;align-items:baseline;gap:8px;
     margin:0.6rem 0 0.8rem;padding-top:0.8rem;border-top:1px solid #{C['border']}">
  <span style="font-size:0.95rem;font-weight:700;color:#{C['text']}">Rebalancing Actions</span>
  <span style="font-size:0.7rem;color:#{C['muted']}">
    Whale-adjusted targets · ±5pp drift threshold
  </span>
</div>
""", unsafe_allow_html=True)

    if not rebalancing:
        st.markdown(f"""
<div style="background:#{C['card']};border:1px solid #{C['green']}44;
     border-radius:12px;padding:1rem 1.2rem;
     color:#{C['green']};font-size:0.85rem">
  ✓ &nbsp; Portfolio is within target weights — no rebalancing actions needed.
</div>
""", unsafe_allow_html=True)
    else:
        N = 3
        for i in range(0, len(rebalancing), N):
            row_cols = st.columns(N)
            for col, s in zip(row_cols, rebalancing[i:i + N]):
                is_up    = s["action"] == "INCREASE"
                ac_color = f"#{C['green']}" if is_up else f"#{C['red']}"
                arrow    = "↑" if is_up else "↓"
                drift_pp = abs(s["drift"] * 100)

                with col:
                    st.markdown(f"""
<div style="background:#{C['card']};border:1px solid #{C['border']};
     border-radius:12px;padding:0.9rem 1rem;margin-bottom:0.6rem;
     border-left:3px solid {ac_color}">
  <div style="display:flex;align-items:center;justify-content:space-between;
              margin-bottom:0.55rem">
    <span style="font-weight:700;font-size:0.9rem;color:#{C['text']}">
      {s['sector']}
    </span>
    <span style="background:{ac_color}18;color:{ac_color};
                 border-radius:5px;padding:2px 9px;
                 font-size:0.68rem;font-weight:700">
      {arrow} {s['action']}
    </span>
  </div>
  <div style="display:flex;gap:0.8rem;align-items:flex-end;margin-bottom:0.5rem">
    <div>
      <div style="font-size:0.58rem;color:#{C['muted']};text-transform:uppercase;
                  letter-spacing:0.5px;margin-bottom:1px">Current</div>
      <div style="font-size:1rem;font-weight:700;color:#{C['text']}">
        {s['current_weight'] * 100:.1f}%
      </div>
    </div>
    <div style="color:#{C['muted']};font-size:0.9rem;padding-bottom:2px">→</div>
    <div>
      <div style="font-size:0.58rem;color:#{C['muted']};text-transform:uppercase;
                  letter-spacing:0.5px;margin-bottom:1px">Target</div>
      <div style="font-size:1rem;font-weight:700;color:{ac_color}">
        {s['target_weight'] * 100:.1f}%
      </div>
    </div>
    <div style="margin-left:auto;text-align:right">
      <div style="font-size:0.58rem;color:#{C['muted']};text-transform:uppercase;
                  letter-spacing:0.5px;margin-bottom:1px">Drift</div>
      <div style="font-size:1rem;font-weight:700;color:{ac_color}">
        {drift_pp:.1f}pp
      </div>
    </div>
  </div>
  <div style="font-size:0.68rem;color:#{C['muted']};line-height:1.5">
    {s['rationale']}
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Raw Holdings expander ─────────────────────────────────────────────────
    with st.expander("📋  Raw Holdings"):
        th_style = (
            f"text-align:left;padding:7px 10px;color:#{C['muted']};"
            f"font-weight:600;font-size:0.63rem;text-transform:uppercase;"
            f"letter-spacing:0.6px;border-bottom:1px solid #{C['border']}"
        )
        th_right = th_style.replace("text-align:left", "text-align:right")

        rows_html = "".join(
            f"""
<tr style="border-bottom:1px solid #{C['border']}40">
  <td style="padding:7px 10px;font-weight:600;color:#{C['text']};font-size:0.82rem">{h['ticker']}</td>
  <td style="padding:7px 10px;color:#{C['muted']};font-size:0.82rem">{h.get('sector','—')}</td>
  <td style="padding:7px 10px;text-align:right;color:#{C['text']};font-size:0.82rem">{h['quantity']:,}</td>
  <td style="padding:7px 10px;text-align:right;color:#{C['muted']};font-size:0.82rem">${h['avg_cost']:,.2f}</td>
  <td style="padding:7px 10px;text-align:right;font-weight:600;
             color:#{C['green']};font-size:0.82rem">
    ${h['quantity'] * h['avg_cost']:,.0f}
  </td>
</tr>"""
            for h in holdings_list
        )

        st.markdown(f"""
<div style="background:#{C['card2']};border-radius:0 0 10px 10px;
     border:1px solid #{C['border']};border-top:none;overflow-x:auto">
  <table style="width:100%;border-collapse:collapse;font-family:Inter,sans-serif">
    <thead>
      <tr>
        <th style="{th_style}">Ticker</th>
        <th style="{th_style}">Sector</th>
        <th style="{th_right}">Qty</th>
        <th style="{th_right}">Avg Cost</th>
        <th style="{th_right}">Market Value</th>
      </tr>
    </thead>
    <tbody>{rows_html}</tbody>
  </table>
</div>
""", unsafe_allow_html=True)
