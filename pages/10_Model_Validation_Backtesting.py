from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from risklens.backtesting.walk_forward import walk_forward_historical_var
from risklens.backtesting.validation import kupiec_pof, christoffersen_independence, acerbi_szekely_es_test
from risklens.backtesting.walk_forward_suite import WalkForwardBacktester
from risklens.ui import header, status_banner, dark_plot, load_workspace_prices

header(
    st,
    "Walk-Forward Validation & Ablation",
    "Strictly chronological VaR validation, breach clustering diagnostics, and multi-model ablation suite.",
)

prices, data_status, data_provider, data_ts = load_workspace_prices(st)
if prices.empty:
    st.error("Demo fixture unavailable.")
    st.stop()

status_banner(st, data_status, data_provider, data_ts, "Forecasts use training windows ending strictly before each forecast observation.")

c1, c2, c3 = st.columns(3)
asset = c1.selectbox("Reference series", [c for c in prices.columns if c != "USDINR=X"])
lookback = c2.slider("Training lookback", 60, 500, 252, step=12)
conf = c3.selectbox("VaR confidence", [0.95, 0.99], index=0)

r = prices[asset].pct_change(fill_method=None).dropna()
if st.button("Run walk-forward backtest", type="primary"):
    try:
        st.session_state["backtest_result"] = (walk_forward_historical_var(r, lookback, conf), conf, asset, lookback)
    except Exception as e:
        st.error(f"Backtest failed: {type(e).__name__}: {str(e)[:200]}")

if "backtest_result" in st.session_state:
    result, conf, asset, lookback = st.session_state["backtest_result"]
    if result.empty:
        st.info("Not enough data for selected lookback.")
    else:
        pof = kupiec_pof(result["breach"], conf)
        ind = christoffersen_independence(result["breach"])
        es_test = acerbi_szekely_es_test(result["actual_loss"], result["var"], result["es"], conf)

        a, b, c, d = st.columns(4)
        a.metric("Forecast observations", f"{len(result):,}")
        b.metric("VaR exceedances", str(int(result["breach"].sum())))
        c.metric("Kupiec POF p-value", f"{pof['p_value']:.4f}" if pd.notna(pof["p_value"]) else "N/A")
        d.metric("Independence p-value", f"{ind['p_value']:.4f}" if pd.notna(ind["p_value"]) else "N/A")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=result.index, y=result["actual_loss"], mode="lines", name="Realized Loss", line=dict(color="#6F9B8A", width=1.3)))
        fig.add_trace(go.Scatter(x=result.index, y=result["var"], mode="lines", name="Forecast VaR", line=dict(color="#315F49", width=2)))
        breaches = result[result["breach"]]
        if not breaches.empty:
            fig.add_trace(go.Scatter(
                x=breaches.index, y=breaches["actual_loss"], mode="markers", name="VaR Exceedance",
                marker=dict(color="#EF4444", size=8, symbol="x"),
            ))
        fig.update_layout(title="Walk-Forward Forecast VaR vs. Realized Loss (Red X = Breach)")
        st.plotly_chart(dark_plot(fig, show_rangeslider=True), use_container_width=True)

        st.subheader("Validation diagnostics")
        st.write({
            "Kupiec Proportion of Failures": pof,
            "Christoffersen independence": ind,
            "Acerbi–Szekely-style ES calibration diagnostic": es_test,
        })

        st.subheader("Multi-model walk-forward ablation")
        try:
            suite = WalkForwardBacktester(conf, lookback).run_ablation_suite(r)
            st.dataframe(suite, use_container_width=True, hide_index=True)
            st.download_button("Download model ablation", suite.to_csv(index=False).encode(), file_name="risklens_ablation.csv", mime="text/csv")
        except Exception as exc:
            st.info(f"Ablation unavailable for this sample: {type(exc).__name__}: {str(exc)[:180]}")

        st.warning("Structural models are represented by explicitly labeled proxies in this ablation suite; these results are not empirical evidence that an uncalibrated model outperforms another.")
        st.download_button("Download walk-forward outputs", result.to_csv().encode(), file_name="risklens_walk_forward.csv", mime="text/csv")
else:
    st.info("Run the walk-forward backtest to produce strictly chronological one-step forecasts.")
