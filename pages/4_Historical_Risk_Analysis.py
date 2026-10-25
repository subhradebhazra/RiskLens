from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

from risklens.features.returns import calculate_returns, drawdown_series, annualized_volatility, sharpe_ratio, sortino_ratio
from risklens.features.volatility import ewma_volatility
from risklens.models.regime_switching import fit_regimes
from risklens.models.hamilton_regime import HamiltonRegimeSwitchingModel
from risklens.models.bates import fit_bates
from risklens.risk.var import historical_var_es, gaussian_var_es, cornish_fisher_var_es
from risklens.ui import header, status_banner, dark_plot, pct, load_workspace_prices

header(
    st,
    "Historical Risk, Drawdowns & Regimes",
    "Tail risk comparison, EWMA volatility, Markov-switching vs. GMM regime classification, and Merton/Heston/Bates structural diagnostics.",
)

prices, data_status, data_provider, data_ts = load_workspace_prices(st)
if prices.empty:
    st.error("Demo fixture unavailable.")
    st.stop()

status_banner(st, data_status, data_provider, data_ts, "Historical tail metrics and physical-measure structural fits.")

c1, c2, c3, c4 = st.columns(4)
symbol = c1.selectbox("Reference Asset", [c for c in prices.columns if c != "USDINR=X"])
portfolio_size = c2.number_input("Illustrative Position Value ($)", min_value=1_000.0, max_value=100_000_000.0, value=100_000.0, step=10_000.0)
ewma_lam = c3.slider("EWMA Decay (λ)", 0.85, 0.99, 0.94, step=0.01)
reg_choice = c4.selectbox("Regime Engine", ["Hamilton Markov-Switching (Statsmodels + Fallback)", "Gaussian-Mixture Proxy"])

r = calculate_returns(prices[symbol]).dropna()
dd = drawdown_series(r, input_is_returns=True)
ew = ewma_volatility(r, lam=ewma_lam)
pnl = r * portfolio_size

metrics = []
for conf in [0.95, 0.99]:
    metrics += [
        historical_var_es(pnl.to_numpy(), conf),
        gaussian_var_es(pnl.to_numpy(), conf),
        cornish_fisher_var_es(pnl.to_numpy(), conf),
    ]

a, b, c, d = st.columns(4)
a.metric("Annualized Volatility", pct(annualized_volatility(r)))
b.metric("Max Drawdown", pct(dd.min()))
c.metric("Sharpe / Sortino", f"{sharpe_ratio(r):.2f} / {sortino_ratio(r):.2f}")
d.metric("Total Return", pct((1 + r).prod() - 1))

tab_var, tab_dd, tab_reg, tab_bates = st.tabs([
    "VaR & expected shortfall",
    "Drawdown & volatility",
    "Regime classification",
    "Structural model diagnostics",
])

with tab_var:
    rows = [
        {
            "confidence": m.confidence,
            "method": m.method,
            "VaR": m.var,
            "Expected shortfall": m.es,
            "observations": m.sample_size,
            "notes": "; ".join(m.warnings or []),
        }
        for m in metrics
    ]
    df_var = pd.DataFrame(rows)
    st.plotly_chart(
        dark_plot(px.bar(df_var, x="method", y=["VaR", "Expected shortfall"], facet_col="confidence", barmode="group", title="Tail Risk Estimator Comparison")),
        use_container_width=True,
    )
    st.dataframe(df_var, use_container_width=True, hide_index=True)

with tab_dd:
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(dark_plot(px.area(dd.rename("Drawdown"), title="Historical Drawdown")), use_container_width=True)
    with col2:
        st.plotly_chart(dark_plot(px.line((ew * np.sqrt(252)).rename("Annualized EWMA volatility"), title="EWMA Volatility Estimate")), use_container_width=True)

with tab_reg:
    if reg_choice.startswith("Hamilton") and len(r) >= 120:
        hm = HamiltonRegimeSwitchingModel(n_regimes=4)
        diag = hm.fit(r)
        probs = hm.smoothed_probs
        tmat = np.asarray(diag["transition_matrix"], dtype=float)
        m_name = diag["method_used"]
        warns = diag.get("warnings", [])
        dur_df = pd.DataFrame(diag.get("regime_statistics", []))
    else:
        reg = fit_regimes(r)
        probs = reg.probabilities
        tmat = reg.transition_matrix
        m_name = reg.method
        warns = reg.warnings
        dur_df = pd.DataFrame({"regime": [*probs.columns], "expected_duration_observations": reg.expected_durations})

    st.caption(f"Method: `{m_name}`. Probabilistic classifications, not deterministic forecasts.")
    for w in warns:
        st.warning(w)
    col_p, col_t = st.columns([3, 2])
    with col_p:
        if not probs.empty:
            st.plotly_chart(dark_plot(px.area(probs, title="Regime Posterior Probabilities")), use_container_width=True)
    with col_t:
        st.plotly_chart(
            dark_plot(px.imshow(tmat, x=list(probs.columns), y=list(probs.columns), text_auto=".2f", aspect="auto", title="Transition Matrix")),
            use_container_width=True,
        )
    st.dataframe(dur_df, use_container_width=True, hide_index=True)

with tab_bates:
    st.markdown("#### Independent Merton Jump-Diffusion & Heston Volatility Proxy (`bates.py`)")
    bates = fit_bates(np.log1p(r).dropna())
    st.info(f"Selected Structural Mode: **{bates.selected_model}**")
    for w in bates.warnings:
        st.caption(f"• {w}")
    b1, b2 = st.columns(2)
    with b1:
        st.markdown("**Merton Jump-Diffusion Fit (`merton.py`)**")
        if bates.merton:
            st.json(bates.merton.to_dict())
    with b2:
        st.markdown("**Heston Volatility Proxy Fit (`heston.py`)**")
        if bates.heston:
            st.json(bates.heston.to_dict())
