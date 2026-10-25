from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from risklens.data.demo_assets import load_demo_portfolio
from risklens.simulation.engine import simulate_portfolio, terminal_pnl
from risklens.simulation.unified_engine import UnifiedMonteCarloEngine
from risklens.simulation.unified_event_generator import UnifiedEventGenerator
from risklens.risk.var import historical_var_es
from risklens.ui import header, status_banner, dark_plot, load_workspace_prices

header(
    st,
    "Monte Carlo & Forward Risk",
    "Seeded path simulation with standard baselines, the Unified Bates/Hawkes/Student-t Copula Engine, and quantile fan cones.",
)

prices, data_status, data_provider, data_ts = load_workspace_prices(st)
if prices.empty:
    st.error("Demo price fixtures are missing.")
    st.stop()

status_banner(st, data_status, data_provider, data_ts, "Simulations are hypothetical seeded scenarios, not predictions.")

avail = [c for c in prices.columns if c != "USDINR=X"]
assets = st.multiselect("Assets", avail, ["SPY", "QQQ", "AAPL"])
if not assets:
    st.error("Select at least one asset.")
    st.stop()

sample_pf = load_demo_portfolio().set_index("ticker")
raw_values = {
    a: float(sample_pf.loc[a, "units"]) * float(prices[a].iloc[-1])
    if a in sample_pf.index and a in prices.columns else 1.0
    for a in assets
}
raw_total = sum(raw_values.values())
if raw_total <= 0:
    st.error("No selected assets have sample portfolio units.")
    st.stop()

weights_in = {
    a: st.sidebar.number_input(
        f"Weight · {a}",
        min_value=0.0,
        max_value=1.0,
        value=float(raw_values[a] / raw_total),
        step=0.05,
        key=f"simweight_{a}",
    )
    for a in assets
}
weights = pd.Series(weights_in, dtype=float)
if weights.sum() <= 0:
    st.error("At least one weight must be positive.")
    st.stop()
weights = weights / weights.sum()

m_col1, m_col2, m_col3 = st.columns([2, 1, 1])
mode = m_col1.selectbox(
    "Simulation Engine Mode",
    [
        "Unified Engine: full_bates_hawkes_copula",
        "Unified Engine: bates_static_jumps",
        "Gaussian baseline",
        "Historical bootstrap",
        "Merton-only",
        "Heston-only",
        "Integrated (available components)",
    ],
)
copula_df = m_col2.slider("Student-t Copula df (ν)", 2.5, 25.0, 5.0, step=0.5, disabled=not mode.startswith("Unified"))
regime_mult = m_col3.slider("Regime Vol Multiplier", 0.5, 3.0, 1.0, step=0.1, disabled=not mode.startswith("Unified"))

c1, c2, c3, c4 = st.columns(4)
paths = c1.select_slider("Paths", options=[1000, 2500, 5000, 10000, 20000, 50000], value=10000)
horizon = c2.slider("Trading-day horizon", 1, 30, 5)
seed = c3.number_input("Random seed", min_value=0, max_value=2**31 - 1, value=42, step=1)
value = c4.number_input("Current portfolio value", min_value=100.0, value=100000.0, step=1000.0)

returns = prices[assets].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).dropna(how="any")

if st.button("Run Monte Carlo", type="primary"):
    try:
        with st.spinner("Simulating deterministic-seed paths..."):
            if mode.startswith("Unified Engine:"):
                u_mode = mode.split(": ")[1].strip()
                ev_gen = UnifiedEventGenerator(seed=int(seed))
                u_mc = UnifiedMonteCarloEngine(n_paths=int(paths), horizon_days=int(horizon), seed=int(seed))
                u_res = u_mc.run_portfolio_simulation(
                    returns_df=returns,
                    weights=weights.to_numpy(),
                    portfolio_value=float(value),
                    model_mode=u_mode,
                    copula_df=float(copula_df),
                    event_gen=ev_gen,
                    regime_vol_multiplier=float(regime_mult),
                )
                pnl = u_res["portfolio_pnl"]
                m95 = historical_var_es(pnl, 0.95)
                m99 = historical_var_es(pnl, 0.99)
                st.session_state["forward_result"] = {
                    "is_unified": True,
                    "u_res": u_res,
                    "pnl": pnl,
                    "m95": m95,
                    "m99": m99,
                    "value": value,
                    "seed": int(seed),
                    "path_count": int(paths),
                    "horizon_days": int(horizon),
                    "mode_used": u_mode,
                    "warnings": u_res.get("warnings", []),
                }
            else:
                result = simulate_portfolio(returns, weights, int(paths), int(horizon), mode, int(seed))
                pnl = terminal_pnl(result, value)
                m95 = historical_var_es(pnl, 0.95)
                m99 = historical_var_es(pnl, 0.99)
                st.session_state["forward_result"] = {
                    "is_unified": False,
                    "result": result,
                    "pnl": pnl,
                    "m95": m95,
                    "m99": m99,
                    "value": value,
                    "seed": result.seed,
                    "path_count": result.path_count,
                    "horizon_days": result.horizon_days,
                    "mode_used": result.mode_used,
                    "warnings": result.warnings,
                }
    except Exception as e:
        st.error(f"Simulation failed: {type(e).__name__}: {str(e)[:250]}")

stored = st.session_state.get("forward_result")
if stored:
    pnl = stored["pnl"]
    a, b, c, d = st.columns(4)
    a.metric("Mode used", stored["mode_used"])
    b.metric("95% VaR", f"{stored['m95'].var:,.2f}")
    c.metric("95% ES", f"{stored['m95'].es:,.2f}")
    d.metric("99% VaR", f"{stored['m99'].var:,.2f}")
    st.caption(f"Seed {stored['seed']} · {stored['path_count']:,} paths · {stored['horizon_days']} trading days")

    for w in stored.get("warnings", []):
        st.warning(w)

    if not stored["is_unified"]:
        result = stored["result"]
        for a_ in result.assumptions:
            st.caption("Assumption: " + a_)
        cum_wealth = stored["value"] * np.column_stack([np.ones(result.path_count), np.cumprod(1.0 + result.paths, axis=1)])
        days_axis = np.arange(cum_wealth.shape[1])
        q05, q25, q50, q75, q95 = np.percentile(cum_wealth, [5, 25, 50, 75, 95], axis=0)

        fig_fan = go.Figure()
        fig_fan.add_trace(go.Scatter(x=days_axis, y=q95, mode="lines", line=dict(width=0), showlegend=False))
        fig_fan.add_trace(go.Scatter(x=days_axis, y=q05, mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(181, 194, 142, 0.28)", name="5th–95th Band"))
        fig_fan.add_trace(go.Scatter(x=days_axis, y=q75, mode="lines", line=dict(width=0), showlegend=False))
        fig_fan.add_trace(go.Scatter(x=days_axis, y=q25, mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(95, 135, 119, 0.26)", name="25th–75th Band"))
        fig_fan.add_trace(go.Scatter(x=days_axis, y=q50, mode="lines", line=dict(color="#315F49", width=2.4), name="Median Wealth"))
        fig_fan.update_layout(title="Simulated Portfolio Wealth Quantile Fan Cone", xaxis_title="Trading Day", yaxis_title="Portfolio Value")
        st.plotly_chart(dark_plot(fig_fan), use_container_width=True)

        pathsdf = pd.DataFrame(result.paths[:80].T, columns=[f"Path {i+1}" for i in range(min(80, result.path_count))])
        pathsdf.index.name = "Trading day"
        st.plotly_chart(dark_plot(px.line(pathsdf, title="Sample simulated daily portfolio returns (first 80 paths)")), use_container_width=True)
    else:
        st.json(stored["u_res"]["jump_diagnostics"])

    fig_hist = px.histogram(x=pnl, nbins=70, title="Terminal portfolio P&L distribution", labels={"x": "Terminal P&L"})
    fig_hist.add_vline(x=-stored["m95"].var, line_dash="dash", line_color="#C58B4E", annotation_text="95% VaR")
    fig_hist.add_vline(x=-stored["m99"].var, line_dash="dot", line_color="#EF4444", annotation_text="99% VaR")
    st.plotly_chart(dark_plot(fig_hist), use_container_width=True)

    st.dataframe(
        {
            "Metric": ["95% VaR", "95% Expected Shortfall", "99% VaR", "99% Expected Shortfall"],
            "Value": [stored["m95"].var, stored["m95"].es, stored["m99"].var, stored["m99"].es],
            "Unit": ["selected base-currency units"] * 4,
        },
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        "Download terminal P&L sample CSV",
        pd.DataFrame({"terminal_pnl": pnl}).to_csv(index=False).encode(),
        file_name="risklens_terminal_pnl.csv",
        mime="text/csv",
    )
else:
    st.info("Choose settings and click Run Monte Carlo. No simulated values are shown until the engine runs.")
