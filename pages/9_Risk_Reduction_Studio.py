from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from risklens.data.demo_assets import load_demo_portfolio
from risklens.portfolio.institutional_optimizer import InstitutionalPortfolioOptimizer
from risklens.ui import header, status_banner, dark_plot, pct, load_workspace_prices

header(
    st,
    "Risk Reduction Studio",
    "Hypothetical portfolio optimization with Rockafellar–Uryasev CVaR, HRP, turnover limits, transaction-cost drag, and frontier cloud.",
)

prices, data_status, data_provider, data_ts = load_workspace_prices(st)
pf = load_demo_portfolio()
if prices.empty:
    st.error("Demo fixtures unavailable.")
    st.stop()

status_banner(st, data_status, data_provider, data_ts, "Optimized weights are hypothetical analytical outputs, not investment advice.")

assets = st.multiselect("Assets", list(prices.columns.drop("USDINR=X")), ["SPY", "QQQ", "AAPL", "TCS.NS"])
if len(assets) < 2:
    st.error("Choose at least two assets.")
    st.stop()

rets = prices[assets].pct_change(fill_method=None).dropna(how="any")
unit_map = pf.set_index("ticker")["units"].to_dict()
units = np.array([float(unit_map.get(a, 0.0)) for a in assets])
vals = units * np.array([prices[a].iloc[-1] for a in assets])
if vals.sum() <= 0:
    st.error("Selected assets need nonzero holdings in the sample portfolio.")
    st.stop()

current = pd.Series(vals / vals.sum(), index=assets)

c1, c2, c3, c4 = st.columns(4)
method = c1.selectbox(
    "Objective",
    ["cvar_es", "min_variance", "max_sharpe", "hrp", "regime_kelly"],
    format_func=lambda x: {
        "cvar_es": "CVaR / Expected Shortfall",
        "min_variance": "Minimum variance",
        "max_sharpe": "Maximum Sharpe",
        "hrp": "Hierarchical Risk Parity",
        "regime_kelly": "Conservative regime-aware Kelly proxy",
    }[x],
)
low, high = c2.slider("Position bounds", 0.0, 1.0, (0.0, 0.70), step=0.05)
turn = c3.slider("Maximum one-way turnover", 0.0, 1.0, 0.5, step=0.05)
cost = c4.number_input("Transaction cost (basis points)", min_value=0.0, max_value=500.0, value=15.0)
conf = st.selectbox("CVaR confidence", [0.95, 0.99], index=0)

if st.button("Optimize hypothetical weights", type="primary"):
    try:
        engine = InstitutionalPortfolioOptimizer(rets)
        result = engine.optimize(
            current.to_numpy(),
            method=method,
            min_weight=low,
            max_weight=high,
            max_turnover=turn,
            tc_bps=cost,
            confidence_level=conf,
        )
        st.session_state["advanced_opt"] = (result, assets, current.copy(), engine)
    except Exception as exc:
        st.error(f"Optimization failed safely: {type(exc).__name__}: {str(exc)[:220]}")

if "advanced_opt" in st.session_state:
    result, assets_saved, current_saved, engine = st.session_state["advanced_opt"]
    ow = np.asarray(result["optimized_weights"])
    cw = np.asarray(result["current_weights"])
    comp = pd.DataFrame({"ticker": result["tickers"], "Current weight": cw, "Suggested weight": ow, "Weight Delta": ow - cw})

    a, b, c, d = st.columns(4)
    a.metric("Current annualized volatility", pct(result["current_metrics"]["annual_volatility"]))
    b.metric("Suggested annualized volatility", pct(result["optimized_metrics"]["annual_volatility"]))
    c.metric("One-way turnover", pct(result["turnover"]))
    d.metric("Transaction Cost Drag", f"{result['transaction_cost_pct'] * 10000:.2f} bps")

    col_bar, col_front = st.columns(2)
    with col_bar:
        st.plotly_chart(
            dark_plot(px.bar(comp, x="ticker", y=["Current weight", "Suggested weight"], barmode="group", title="Current vs suggested weights")),
            use_container_width=True,
        )
    with col_front:
        rng = np.random.default_rng(42)
        rand_w = rng.dirichlet(np.ones(len(assets_saved)), size=500)
        c_vols = np.sqrt(np.einsum("ij,jk,ik->i", rand_w, engine.cov_annual, rand_w))
        c_rets = rand_w @ engine.mean_annual
        fig_ef = go.Figure()
        fig_ef.add_trace(go.Scatter(x=c_vols * 100, y=c_rets * 100, mode="markers", marker=dict(size=5, color=c_rets / np.maximum(c_vols, 1e-8), colorscale="Viridis", opacity=0.45), name="Feasible Portfolios"))
        fig_ef.add_trace(go.Scatter(x=[result["current_metrics"]["annual_volatility"] * 100], y=[result["current_metrics"]["annual_return"] * 100], mode="markers+text", text=["Current"], textposition="top center", marker=dict(size=13, color="#C58B4E", symbol="diamond"), name="Current"))
        fig_ef.add_trace(go.Scatter(x=[result["optimized_metrics"]["annual_volatility"] * 100], y=[result["optimized_metrics"]["annual_return"] * 100], mode="markers+text", text=["Optimized"], textposition="top center", marker=dict(size=15, color="#315F49", symbol="star"), name="Optimized"))
        fig_ef.update_layout(title="Risk-Return Frontier Cloud", xaxis_title="Annualized Volatility (%)", yaxis_title="Annualized Return (%)")
        st.plotly_chart(dark_plot(fig_ef), use_container_width=True)

    st.dataframe(comp, use_container_width=True, hide_index=True)
    with st.expander("Full Optimizer Metrics & Diagnostics"):
        st.write({
            "Method": result["method"],
            "Estimated transaction cost (% of portfolio)": result["transaction_cost_pct"] * 100,
            "Status": result["status"],
            "Current metrics": result["current_metrics"],
            "Suggested metrics": result["optimized_metrics"],
        })
    st.caption("CVaR uses a Rockafellar–Uryasev epigraph objective. HRP is hierarchical clustering with recursive bisection. Constraints that cannot be satisfied result in retaining current weights.")
