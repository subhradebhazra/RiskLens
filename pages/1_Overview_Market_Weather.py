from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from risklens.features.returns import calculate_returns, annualized_volatility, drawdown_series
from risklens.models.regime_switching import fit_regimes
from risklens.models.hamilton_regime import HamiltonRegimeSwitchingModel
from risklens.models.pca_absorption import absorption_ratio
from risklens.risk.var import historical_var_es
from risklens.risk.contagion import market_weather
from risklens.news.providers import fetch_market_news
from risklens.reporting.reports import build_report, report_json
from risklens.ui import header, status_banner, dark_plot, pct, load_workspace_prices, evaluate_and_render_alerts

header(
    st,
    "Overview & Market Weather",
    "Reference market condition, regime-switching telemetry, news context, and data provenance.",
)

prices, data_status, data_provider, data_ts = load_workspace_prices(st)
if prices.empty:
    st.error("Demo price fixtures missing.")
    st.stop()

status_banner(st, data_status, data_provider, data_ts, "All market weather classifications are observation heuristics, not deterministic forecasts.")

c1, c2 = st.columns([1, 2])
ref_symbol = c1.selectbox("Reference Market Benchmark", [c for c in prices.columns if c != "USDINR=X"], index=0)
regime_mode = c2.radio(
    "Regime Classification Engine",
    ["Gaussian-Mixture Volatility Proxy", "Hamilton Markov-Switching MLE (with GMM Fallback)"],
    horizontal=True,
)

r = calculate_returns(prices[ref_symbol]).dropna()
dd = drawdown_series(r, input_is_returns=True)
var = historical_var_es((r * 100_000).to_numpy(), 0.95)
vol = annualized_volatility(r)

multi_ret = prices[[c for c in prices.columns if c != "USDINR=X"]].pct_change(fill_method=None).dropna()
pca = absorption_ratio(multi_ret) if len(multi_ret) >= 20 else None

if regime_mode.startswith("Hamilton") and len(r) >= 120:
    hm = HamiltonRegimeSwitchingModel(n_regimes=4)
    h_diag = hm.fit(r)
    probs_df = hm.smoothed_probs
    method_used = h_diag["method_used"]
    reg_warnings = h_diag.get("warnings", [])
else:
    regime = fit_regimes(r)
    probs_df = regime.probabilities
    method_used = regime.method
    reg_warnings = regime.warnings

crisis = float(probs_df.iloc[-1]["Crisis / Stress"]) if not probs_df.empty else None
weather, weather_note = market_weather(crisis, vol, float(dd.min()), pca.ratio if pca else None)

a, b, c, d, e = st.columns(5)
a.metric("Market Weather", weather)
b.metric("Annualized Volatility", pct(vol))
c.metric("Max Drawdown", pct(dd.min()))
d.metric("95% Historical VaR ($100k)", f"${var.var:,.0f}")
e.metric("Crisis State Prob.", pct(crisis) if crisis is not None else "N/A")
st.caption(weather_note)

evaluate_and_render_alerts(
    st,
    {
        "crisis_regime_prob": crisis or 0.0,
        "current_drawdown_pct": abs(float(dd.iloc[-1])) if len(dd) else 0.0,
        "portfolio_var_99_pct": historical_var_es((r * 100_000).to_numpy(), 0.99).var / 100_000.0,
    },
)

rebased = prices[["SPY", "QQQ", "AAPL"]].div(prices[["SPY", "QQQ", "AAPL"]].iloc[0]).mul(100)
fig = px.line(rebased, title="Reference Equity Paths · Rebased to 100")
st.plotly_chart(dark_plot(fig, show_rangeslider=True), use_container_width=True)

left, right = st.columns(2)
with left:
    st.subheader("Regime-Switching Probabilities")
    st.caption(f"Engine: `{method_used}`")
    if not probs_df.empty:
        st.plotly_chart(dark_plot(px.area(probs_df, title="Historical Classified State Probabilities")), use_container_width=True)
    for warning in reg_warnings:
        st.warning(warning)

with right:
    st.subheader("Latest Configured News Context")
    news, warns = fetch_market_news(limit=8)
    if news.empty:
        st.info("No news available.")
    else:
        st.dataframe(
            news[[c for c in ["published_at", "title", "source", "provider"] if c in news]].head(8),
            use_container_width=True,
            hide_index=True,
        )
    for warn in warns:
        st.caption(warn)

rep = build_report(
    "Market Weather Snapshot",
    {"benchmark": ref_symbol, "weather": weather, "volatility": vol, "max_drawdown": float(dd.min()), "var_95": var.var, "regime_engine": method_used},
)
st.download_button("Export weather telemetry (JSON)", report_json(rep).encode(), file_name="risklens_market_weather.json", mime="application/json")
