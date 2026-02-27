"""
app.py — WhaleTracker AI Dashboard
------------------------------------
Run with:  streamlit run app.py

Tabs:
  1. Whale Heatmap    — Signal overview across all tracked institutions.
  2. Recommendations  — Ranked Buy/Hold/Sell table.
  3. My Portfolio     — Current holdings vs Smart Money sector weights.
"""

import logging
import streamlit as st
import pandas as pd

from src.data_collector import fetch_all_whale_filings
from src.analysis_engine import build_recommendations, get_sector_rotation_signals
from src.portfolio_manager import load_portfolio, suggest_rebalancing, get_current_sector_weights

logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="WhaleTracker AI",
    page_icon=":whale:",
    layout="wide",
)

st.title(":whale: WhaleTracker AI — Institutional Investment Co-pilot")
st.caption("Tracking 13F filings | Signals: NEW ENTRY · AGGRESSIVE BUY · HIGH CONCENTRATION")

# ---------------------------------------------------------------------------
# Data loading (cached for 6 hours)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=21_600)
def load_data():
    filings = fetch_all_whale_filings()
    recommendations = build_recommendations(filings)
    rotation = get_sector_rotation_signals(filings)
    portfolio = load_portfolio()
    rebalancing = suggest_rebalancing(portfolio, rotation)
    return filings, recommendations, rotation, portfolio, rebalancing


with st.spinner("Loading Whale data..."):
    filings, recommendations, rotation, portfolio, rebalancing = load_data()

# ---------------------------------------------------------------------------
# Tab layout
# ---------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["Whale Heatmap", "Recommendations", "My Portfolio"])

# ── Tab 1: Whale Heatmap ────────────────────────────────────────────────────
with tab1:
    st.subheader("Institutional Holdings — Signal Heatmap")

    SIGNAL_COLORS = {
        "NEW_ENTRY":          "#1E88E5",
        "AGGRESSIVE_BUY":     "#43A047",
        "HIGH_CONCENTRATION": "#FB8C00",
        "HOLD":               "#9E9E9E",
    }

    for whale, holdings in filings.items():
        if not holdings:
            continue
        st.markdown(f"**{whale}**")
        df = pd.DataFrame(holdings)[["ticker", "company", "shares", "value_usd", "portfolio_pct", "signal"]]
        df.columns = ["Ticker", "Company", "Shares", "Value (USD)", "Portfolio %", "Signal"]
        df["Portfolio %"] = (df["Portfolio %"] * 100).round(2).astype(str) + "%"

        def highlight_signal(val):
            color = SIGNAL_COLORS.get(val, "#9E9E9E")
            return f"background-color: {color}; color: white; font-weight: bold"

        st.dataframe(
            df.style.applymap(highlight_signal, subset=["Signal"]),
            use_container_width=True,
            hide_index=True,
        )
        st.divider()

    # Sector rotation bar chart
    st.subheader("Sector Rotation — Net Whale Flow")
    if rotation:
        rot_df = pd.DataFrame(rotation.items(), columns=["Sector", "Net Score"])
        st.bar_chart(rot_df.set_index("Sector"))

# ── Tab 2: Recommendations ──────────────────────────────────────────────────
with tab2:
    st.subheader("Smart Money Recommendations")

    REC_ICONS = {
        "STRONG BUY": "🚀",
        "BUY":        "🟢",
        "HOLD":       "🟡",
        "SELL":       "🔴",
    }

    rec_df = pd.DataFrame(recommendations)
    if not rec_df.empty:
        rec_df["Rec"] = rec_df["recommendation"].map(lambda r: f"{REC_ICONS.get(r, '')} {r}")
        display_cols = ["ticker", "company", "Rec", "conviction_score", "whale_count", "signal_summary", "macro_note"]
        display_cols = [c for c in display_cols if c in rec_df.columns]
        st.dataframe(
            rec_df[display_cols].rename(columns={
                "ticker": "Ticker", "company": "Company", "Rec": "Recommendation",
                "conviction_score": "Score", "whale_count": "# Whales",
                "signal_summary": "Signals", "macro_note": "Macro Note",
            }),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No recommendations available. Check your data source.")

# ── Tab 3: My Portfolio ─────────────────────────────────────────────────────
with tab3:
    st.subheader("My Portfolio vs Smart Money")

    col1, col2 = st.columns(2)

    # Current sector weights
    current_weights = get_current_sector_weights(portfolio)
    target_weights = portfolio.get("target_sector_weights", {})

    with col1:
        st.markdown("**Current Sector Weights**")
        if current_weights:
            cw_df = pd.DataFrame(current_weights.items(), columns=["Sector", "Weight"])
            cw_df["Weight"] = (cw_df["Weight"] * 100).round(2)
            st.bar_chart(cw_df.set_index("Sector"))

    with col2:
        st.markdown("**Target Sector Weights**")
        if target_weights:
            tw_df = pd.DataFrame(target_weights.items(), columns=["Sector", "Weight"])
            tw_df["Weight"] = (tw_df["Weight"] * 100).round(2)
            st.bar_chart(tw_df.set_index("Sector"))

    st.subheader("Rebalancing Suggestions")
    if rebalancing:
        rb_df = pd.DataFrame(rebalancing)
        rb_df["current_weight"] = (rb_df["current_weight"] * 100).round(1).astype(str) + "%"
        rb_df["target_weight"]  = (rb_df["target_weight"]  * 100).round(1).astype(str) + "%"
        rb_df["drift"]          = (rb_df["drift"]          * 100).round(1).astype(str) + " pp"
        st.dataframe(
            rb_df.rename(columns={
                "sector": "Sector", "current_weight": "Current",
                "target_weight": "Target", "drift": "Drift",
                "action": "Action", "rationale": "Rationale",
            }),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.success("Portfolio is within target weights — no rebalancing needed.")

    # Raw holdings table
    with st.expander("View Raw Holdings"):
        holdings_df = pd.DataFrame(portfolio.get("holdings", []))
        if not holdings_df.empty:
            holdings_df["market_value"] = holdings_df["quantity"] * holdings_df["avg_cost"]
            st.dataframe(holdings_df, use_container_width=True, hide_index=True)
