from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st
import pandas as pd
import plotly.express as px

from risklens.data.demo_assets import load_demo_portfolio
from risklens.risk.stress import DEFAULT_SCENARIOS, run_scenario
from risklens.risk.crisis_replay import StressTestingEngine, CRISIS_TEMPLATES
from risklens.ui import header, status_banner, dark_plot, load_workspace_prices

header(
    st,
    "Stress Testing & Historical Crisis Replay",
    "Deterministic scenario shock vectors and historical crisis window / calibrated sector-beta proxy replays.",
)

prices, data_status, data_provider, data_ts = load_workspace_prices(st)
portfolio = st.session_state.get("shared_portfolio_holdings", load_demo_portfolio())
if prices.empty or portfolio.empty:
    st.error("Demo fixtures unavailable.")
    st.stop()

status_banner(st, data_status, data_provider, data_ts, "Scenario impacts are hypothetical and applied to the active portfolio.")

base = st.selectbox("Base currency", ["USD", "INR"])
fx = float(prices["USDINR=X"].iloc[-1])
rows = []
for _, row in portfolio.iterrows():
    ticker = row["ticker"]
    price = float(prices[ticker].iloc[-1]) if ticker in prices else None
    if price is None:
        continue
    value = float(row["units"]) * price
    if row["currency"] == "INR" and base == "USD":
        value /= fx
    elif row["currency"] == "USD" and base == "INR":
        value *= fx
    rows.append({"ticker": ticker, "units": row["units"], "currency": row["currency"], "sector": row["sector"], "market_value": value})

holdings = pd.DataFrame(rows)
if holdings.empty:
    st.error("No fixture assets found.")
    st.stop()

tab_det, tab_replay = st.tabs([
    "Scenario shocks",
    "Historical crisis replay",
])

with tab_det:
    c1, c2, c3 = st.columns(3)
    scenario = c1.selectbox("Scenario", list(DEFAULT_SCENARIOS))
    selected = c2.selectbox("Focus asset for single-stock event", holdings["ticker"].tolist())
    custom_shock = c3.slider("Custom broad shock (%)", -80.0, 20.0, -15.0, step=1.0)

    if scenario == "Single-stock crash":
        shocks = {ticker: (-0.35 if ticker == selected else -0.02) for ticker in holdings["ticker"]}
    elif scenario == "Broad-market decline":
        shocks = {ticker: custom_shock / 100.0 for ticker in holdings["ticker"]}
    else:
        shocks = None

    if st.button("Run stress scenario", type="primary"):
        try:
            result = run_scenario(holdings, scenario, shocks)
            st.session_state["stress_result"] = (result, base)
        except Exception as e:
            st.error(f"Stress run failed: {type(e).__name__}: {str(e)[:200]}")

    if "stress_result" in st.session_state:
        result, base_saved = st.session_state["stress_result"]
        a, b, c = st.columns(3)
        a.metric("Scenario", result["scenario"])
        b.metric("Portfolio P&L", f"{result['pnl']:,.2f} {base_saved}")
        c.metric("Loss as % of value", f"{result['loss_pct'] * 100:.2f}%")
        st.plotly_chart(dark_plot(px.bar(result["asset_results"], x="ticker", y="pnl", color="ticker", title="Hypothetical P&L Attribution")), use_container_width=True)
        st.dataframe(result["asset_results"], use_container_width=True, hide_index=True)

with tab_replay:
    r_c1, r_c2 = st.columns([2, 1])
    crisis = r_c1.selectbox("Crisis template", list(CRISIS_TEMPLATES), format_func=lambda x: CRISIS_TEMPLATES[x]["description"])
    severity = r_c2.slider("Replay severity multiplier", 0.0, 2.0, 1.0, step=0.1)
    if st.button("Run crisis replay", type="primary", key="run_crisis_replay"):
        replay_prices = prices[holdings["ticker"].tolist()] if all(t in prices.columns for t in holdings["ticker"].tolist()) else None
        try:
            replay = StressTestingEngine.run_historical_crisis_replay(holdings, replay_prices, crisis, severity)
            st.session_state["crisis_replay"] = replay
        except Exception as exc:
            st.error(f"Crisis replay failed: {type(exc).__name__}: {str(exc)[:180]}")

    if "crisis_replay" in st.session_state:
        replay = st.session_state["crisis_replay"]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Replay Mode", replay["replay_mode"].split(" ")[0])
        k2.metric("Portfolio P&L", f"{replay['total_pnl_currency']:,.2f} {base}")
        k3.metric("Portfolio Return", f"{replay['percentage_return'] * 100:.2f}%")
        k4.metric("Post-Shock HHI", f"{replay['post_shock_hhi']:.3f}")

        if "proxy" in replay["replay_mode"].lower():
            st.warning("No reliable exact crisis-window prices were available in the loaded timeframe; this is a hypothetical sector-beta proxy, not a historical replay.")

        df_ab = pd.DataFrame(replay["asset_breakdown"])
        st.plotly_chart(dark_plot(px.bar(df_ab, x="ticker", y=["initial_value", "post_shock_value"], barmode="group", title="Initial vs. Post-Shock Asset Values")), use_container_width=True)
        st.dataframe(df_ab, use_container_width=True, hide_index=True)
