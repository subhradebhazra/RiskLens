from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from risklens.features.returns import calculate_returns, annualized_volatility, drawdown_series, sharpe_ratio, sortino_ratio
from risklens.models.pca_absorption import absorption_ratio, standardized_delta_ar
from risklens.models.regime_switching import fit_regimes
from risklens.risk.var import historical_var_es
from risklens.risk.contagion import calculate_contagion_score
from risklens.reporting.reports import build_report, report_json
from risklens.ui import header, status_banner, dark_plot, pct, load_workspace_prices, evaluate_and_render_alerts, NAV_PAGES

header(
    st,
    "Executive risk overview",
    "A working view of market conditions, portfolio exposure and modeled downside risk.",
)

st.markdown("""
<section class="rl-photo-hero">
  <div class="rl-photo-hero-copy">
    <div class="rl-hero-eyebrow">MARKET INTELLIGENCE / RISK RESEARCH</div>
    <h2>See the risk behind<br>the numbers.</h2>
    <p>Explore portfolio exposures, market regimes and potential downside with one connected research workspace.</p>
    <div class="rl-hero-foot"><span>QUANTITATIVE ANALYSIS</span><span>·</span><span>PORTFOLIO CONTEXT</span><span>·</span><span>SCENARIO TESTING</span></div>
  </div>
  <div class="rl-hero-index"><span>RISK / 02</span><strong>Research<br>before reaction.</strong></div>
</section>
<style>
.rl-photo-hero { position:relative; overflow:hidden; min-height:330px; border-radius:13px; margin:0 0 22px; padding:36px 40px; display:flex; align-items:center; justify-content:space-between; gap:25px; color:#fff; background-image:linear-gradient(90deg,rgba(5,16,23,.91) 0%,rgba(5,16,23,.75) 46%,rgba(5,16,23,.20) 100%),url("https://images.pexels.com/photos/36598869/pexels-photo-36598869.jpeg?auto=compress&cs=tinysrgb&w=1800"); background-size:cover; background-position:center 48%; box-shadow:0 12px 32px rgba(18,34,30,.12); }
.rl-photo-hero-copy { max-width:690px; position:relative; z-index:1; }
.rl-hero-eyebrow { font-size:.68rem; letter-spacing:.18em; font-weight:800; color:#C8E3D1; margin-bottom:18px; }
.rl-photo-hero h2 { color:#fff !important; font-family:Georgia,serif !important; font-weight:500 !important; font-size:clamp(2.25rem,4.2vw,4rem) !important; letter-spacing:-.055em !important; line-height:1.02 !important; margin:0 0 15px; }
.rl-photo-hero p { color:rgba(255,255,255,.86) !important; font-size:1rem; line-height:1.6; max-width:570px; margin:0 0 22px; }
.rl-hero-foot { display:flex; flex-wrap:wrap; gap:10px; align-items:center; color:#D3E1D9; font-size:.62rem; font-weight:750; letter-spacing:.11em; }
.rl-hero-index { position:relative; z-index:1; align-self:flex-end; min-width:145px; border-left:1px solid rgba(255,255,255,.45); padding:10px 0 10px 17px; color:white; }
.rl-hero-index span { display:block; font-size:.65rem; letter-spacing:.16em; color:#C8E3D1; margin-bottom:9px; }
.rl-hero-index strong { font-family:Georgia,serif; font-size:1.25rem; line-height:1.25; font-weight:500; }
@media(max-width:700px){.rl-photo-hero{padding:26px 22px;min-height:360px}.rl-hero-index{display:none}.rl-photo-hero h2{font-size:2.35rem !important}}
</style>
""", unsafe_allow_html=True)

prices, data_status, data_provider, data_ts = load_workspace_prices(st)
if prices.empty:
    st.error("Bundled demo fixtures are missing. Restore data/demo/demo_asset_prices.csv.")
    st.stop()

status_banner(
    st,
    status=data_status,
    provider=data_provider,
    timestamp=f"{prices.index.min().date()} to {prices.index.max().date()}",
    message="Check the source, observation window and model assumptions before interpreting these estimates.",
)

available_assets = [c for c in prices.columns if c != "USDINR=X"]
c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
selected_assets = c1.multiselect(
    "Assets to compare",
    options=available_assets,
    default=["SPY", "QQQ", "AAPL", "RELIANCE.NS", "TCS.NS"],
)
ref_asset = c2.selectbox("Reference asset", options=selected_assets or ["SPY"], index=0)
notional = c3.number_input("Position value (USD equivalent)", min_value=10_000.0, max_value=50_000_000.0, value=100_000.0, step=25_000.0)
window_days = c4.select_slider("Lookback (trading days)", options=[60, 126, 252, 504, len(prices)], value=len(prices))

window_prices = prices.tail(int(window_days))
r = calculate_returns(window_prices[ref_asset]).dropna()
dd = drawdown_series(r, input_is_returns=True)
port = r * notional
rm95 = historical_var_es(port.to_numpy(), 0.95)
rm99 = historical_var_es(port.to_numpy(), 0.99)
vol = annualized_volatility(r)
sr = sharpe_ratio(r)
sor = sortino_ratio(r)

multi_ret = window_prices[available_assets].pct_change(fill_method=None).dropna()
pca_fit = absorption_ratio(multi_ret) if len(multi_ret) >= 20 else None
delta_ar = standardized_delta_ar(pca_fit.rolling) if pca_fit and not pca_fit.rolling.empty else 0.0
reg_fit = fit_regimes(r)
crisis_prob = float(reg_fit.probabilities.iloc[-1]["Crisis / Stress"]) if not reg_fit.probabilities.empty else 0.0
c_score = calculate_contagion_score(
    absorption_ratio=pca_fit.ratio if pca_fit else None,
    crisis_probability=crisis_prob,
)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Reference asset", f"{ref_asset} · {data_status}", f"{len(r):,} obs")
k2.metric("Annualized Volatility", pct(vol), f"Sharpe: {sr:.2f} | Sortino: {sor:.2f}")
k3.metric("Historical Max Drawdown", pct(dd.min()), f"Current: {pct(dd.iloc[-1])}")
k4.metric("95% historical VaR", f"${rm95.var:,.0f}", f"99% VaR: ${rm99.var:,.0f}", delta_color="inverse")
k5.metric("Systemic risk score", f"{c_score.score:.1f}/100" if np.isfinite(c_score.score) else "N/A", f"Absorption: {pca_fit.ratio:.2f}" if pca_fit else "")

alert_telemetry = {
    "pca_delta_ar": delta_ar if np.isfinite(delta_ar) else 0.0,
    "contagion_score": c_score.score if np.isfinite(c_score.score) else 0.0,
    "crisis_regime_prob": crisis_prob,
    "portfolio_var_99_pct": rm99.var / notional if notional > 0 else 0.0,
    "current_drawdown_pct": abs(float(dd.iloc[-1])) if len(dd) else 0.0,
}
triggered_alerts = evaluate_and_render_alerts(st, alert_telemetry)

tab_paths, tab_dd, tab_modules = st.tabs([
    "Relative performance",
    "Drawdown history",
    "Explore the workspace",
])

with tab_paths:
    if selected_assets:
        rebased = window_prices[selected_assets].div(window_prices[selected_assets].iloc[0]).mul(100)
        fig = px.line(rebased, title="Normalized Equity Paths (Rebased to 100 at Window Start)")
        st.plotly_chart(dark_plot(fig, show_rangeslider=True), use_container_width=True)
    st.caption(f"Observation range: {window_prices.index.min().date()} to {window_prices.index.max().date()} · Source: {data_provider}")

with tab_dd:
    if selected_assets:
        dd_df = pd.DataFrame({
            sym: drawdown_series(calculate_returns(window_prices[sym]), input_is_returns=True) * 100.0
            for sym in selected_assets
        })
        st.plotly_chart(dark_plot(px.area(dd_df, title="Multi-Asset Drawdown Comparison (%)")), use_container_width=True)

with tab_modules:
    st.markdown('<div class="rl-section-label">Explore RiskLens</div><h3>Analysis modules</h3>', unsafe_allow_html=True)
    cols = st.columns(3)
    for idx, (path, short_label, full_label) in enumerate(NAV_PAGES[1:]):
        with cols[idx % 3]:
            with st.container(border=True):
                st.markdown(f"**{short_label}** — {full_label}")
                st.page_link(path, label=f"Open {full_label} →", use_container_width=True)

exec_report = build_report(
    "RiskLens Executive Snapshot",
    {
        "reference_asset": ref_asset,
        "data_status": data_status,
        "annualized_volatility": vol,
        "max_drawdown": float(dd.min()),
        "var_95": rm95.var,
        "es_95": rm95.es,
        "contagion_score": c_score.score,
        "triggered_alerts": triggered_alerts,
    },
)
st.download_button(
    "Download executive snapshot (JSON)",
    data=report_json(exec_report).encode("utf-8"),
    file_name="risklens_executive_snapshot.json",
    mime="application/json",
)
