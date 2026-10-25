from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from risklens.data.demo_assets import load_demo_portfolio
from risklens.models.student_t_copula import fit_student_t_copula
from risklens.models.hawkes import fit_hawkes, event_times_from_returns
from risklens.models.pca_absorption import absorption_ratio, standardized_delta_ar
from risklens.models.regime_switching import fit_regimes
from risklens.risk.contagion import calculate_contagion_score
from risklens.risk.hackathon_differentiators import (
    ContagionNetwork3DEngine,
    ReverseStressTestSolver,
    IndianRegulatoryFrictionEngine,
    JudgeChaosHeadlineSimulator,
)
from risklens.ui import header, status_banner, dark_plot, load_workspace_prices

header(
    st,
    "Tail Dependence, Systemic Risk & Institutional Command Suite",
    "Student-t copula tail dependence, PCA absorption, Hawkes self-exciting processes, 3D cascade network, reverse stress, and India friction.",
)

prices, data_status, data_provider, data_ts = load_workspace_prices(st)
if prices.empty:
    st.error("Demo fixture unavailable.")
    st.stop()

status_banner(st, data_status, data_provider, data_ts, "Systemic risk diagnostics and scenario simulators.")

ret = prices[["SPY", "QQQ", "AAPL", "RELIANCE.NS", "TCS.NS"]].pct_change(fill_method=None).dropna()
with st.spinner("Estimating dependence and factor concentration diagnostics..."):
    cop = fit_student_t_copula(ret)
    pca = absorption_ratio(ret)
    reg = fit_regimes(ret["SPY"])
    events = event_times_from_returns(ret["SPY"], threshold=1.5)
    hawkes = fit_hawkes(events, horizon=max(len(ret) - 1, 1))

score = calculate_contagion_score(
    float(np.nanmean(cop.lower_tail_dependence[np.triu_indices(len(cop.symbols), 1)])) if cop.converged else None,
    pca.ratio if np.isfinite(pca.ratio) else None,
    hawkes.branching_ratio if hawkes.converged else None,
    float((ret < 0).all(axis=1).mean()) if len(ret) else None,
    float(reg.probabilities.iloc[-1]["Crisis / Stress"]) if reg.converged and len(reg.probabilities) else None,
)

a, b, c, d = st.columns(4)
a.metric("Contagion score", f"{score.score:.1f}/100" if np.isfinite(score.score) else "Unavailable")
b.metric("Tail copula df", f"{cop.degrees_freedom:.1f}" if cop.converged else "Not fitted")
c.metric("Absorption ratio", f"{pca.ratio:.3f}" if np.isfinite(pca.ratio) else "N/A")
d.metric("Hawkes branching ratio", f"{hawkes.branching_ratio:.3f}" if hawkes.converged else "Unavailable")

for warn in score.warnings + cop.warnings + pca.warnings + hawkes.warnings + reg.warnings:
    st.warning(warn)

left, right = st.columns(2)
with left:
    st.subheader("Student-t copula lower-tail dependence")
    if cop.converged:
        st.plotly_chart(
            dark_plot(px.imshow(cop.lower_tail_dependence, x=cop.symbols, y=cop.symbols, text_auto=".2f", aspect="auto", title="Asymptotic lower-tail dependence")),
            use_container_width=True,
        )
    st.caption("Estimate assumes the fitted Student-t copula and empirical rank-based dependence.")

with right:
    st.subheader("PCA absorption diagnostics")
    if not pca.rolling.empty:
        st.plotly_chart(dark_plot(px.line(pca.rolling, title="Rolling absorption ratio")), use_container_width=True)
    delta = standardized_delta_ar(pca.rolling)
    m_k, m_d = st.columns(2)
    m_k.metric("Effective top-k factors", str(pca.top_k))
    m_d.metric("Standardized ΔAR", f"{delta:.2f}" if np.isfinite(delta) else "Unavailable")
    st.caption(f"Covariance method: {pca.covariance_method}; sample={pca.sample_size}; assets={pca.asset_count}.")

with st.expander("Contagion components and event-process details", expanded=False):
    rows = [{"component": k, "percentile-standardized score": v, "effective weight": score.effective_weights.get(k)} for k, v in score.components.items()]
    st.dataframe(rows, use_container_width=True, hide_index=True)
    st.write({
        "event_definition": "SPY daily returns below 1.5 × historical standard deviation threshold",
        "events": len(events),
        "converged": hawkes.converged,
        "mu": hawkes.mu,
        "alpha": hawkes.alpha,
        "beta": hawkes.beta,
        "branching_ratio": hawkes.branching_ratio,
        "log_likelihood": hawkes.log_likelihood,
    })

st.divider()
st.subheader("Institutional Command Suite")
st.caption(
    "Dependence cascades, reverse-stress solutions, liquidity add-ons, and headline reactions are analytical scenarios, "
    "not live forecasts, regulatory certification, or individualized tax advice."
)

try:
    suite_prices = prices.copy()
    suite_tickers = [ticker for ticker in ["SPY", "QQQ", "AAPL", "RELIANCE.NS", "TCS.NS"] if ticker in suite_prices.columns]
    suite_returns = suite_prices[suite_tickers].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).dropna()
    if len(suite_returns) < 30 or not suite_tickers:
        st.warning("Not enough valid demonstration returns to render the command suite.")
    else:
        demo_holdings = load_demo_portfolio()
        fx_rate = float(suite_prices["USDINR=X"].dropna().iloc[-1]) if "USDINR=X" in suite_prices else 83.0
        position_values = {}
        for ticker in suite_tickers:
            matching = demo_holdings.loc[demo_holdings["ticker"] == ticker] if not demo_holdings.empty else pd.DataFrame()
            if not matching.empty:
                holding = matching.iloc[0]
                last_price = float(suite_prices[ticker].dropna().iloc[-1])
                value = float(holding["units"]) * last_price
                if str(holding["currency"]).upper() == "USD":
                    value *= fx_rate
                position_values[ticker] = max(value, 0.0)
            else:
                position_values[ticker] = 1.0
        value_vector = np.array([position_values[t] for t in suite_tickers], dtype=float)
        suite_weights = value_vector / value_vector.sum() if value_vector.sum() > 0 else np.full(len(suite_tickers), 1 / len(suite_tickers))

        portfolio_value_inr = st.number_input(
            "Illustrative portfolio value (INR)",
            min_value=100_000.0,
            max_value=1_000_000_000.0,
            value=10_000_000.0,
            step=500_000.0,
            key="diff_suite_portfolio_value",
        )
        tab_3d, tab_chaos, tab_rev, tab_india = st.tabs([
            "3D dependence network",
            "Scenario stress controls",
            "Reverse stress solver",
            "India: liquidity and tax friction",
        ])

        with tab_3d:
            epi_col, shock_col, nu_col, threshold_col = st.columns(4)
            epicenter = epi_col.selectbox("Epicenter stock", suite_tickers, index=min(2, len(suite_tickers) - 1), key="diff_epi")
            epicenter_shock = shock_col.slider("Epicenter shock (%)", -35.0, -5.0, -15.0, step=1.0, key="diff_epi_shock")
            copula_nu = nu_col.slider("Student-t df (ν)", 2.5, 15.0, 4.5, step=0.5, key="diff_nu")
            edge_threshold = threshold_col.slider("Show edge if λL ≥", 0.0, 0.8, 0.15, step=0.025, key="diff_edge_threshold")
            try:
                cascade = ContagionNetwork3DEngine.simulate_epicenter_cascade(
                    returns_df=suite_returns,
                    weights=suite_weights,
                    epicenter_ticker=epicenter,
                    initial_shock_pct=epicenter_shock / 100.0,
                    nu=copula_nu,
                    edge_threshold=edge_threshold,
                )
                impact = cascade["portfolio_cascade_loss_pct"]
                loss_amount = -impact * portfolio_value_inr
                impact_col, edge_col, source_col = st.columns(3)
                impact_col.metric("Scenario cascade P&L", f"-₹{loss_amount:,.0f}", f"{impact * 100:.2f}% impact", delta_color="inverse")
                edge_count = int(np.sum(np.triu(cascade["lambda_matrix"] >= edge_threshold, k=1)))
                edge_col.metric("Displayed dependence links", str(edge_count))
                source_col.metric("Epicenter", cascade["epicenter_ticker"])
                st.plotly_chart(ContagionNetwork3DEngine.build_3d_plotly_figure(cascade), use_container_width=True)
                st.caption(cascade["interpretation"] + " Hop 0 through hop 3 are deterministic propagation steps; edge weights are not causal estimates.")
                with st.expander("Cascade shock-vector detail"):
                    detail = pd.DataFrame({
                        "ticker": suite_tickers,
                        "portfolio_weight_pct": suite_weights * 100,
                        "hop_0_epicenter_pct": cascade["hop0_shock"] * 100,
                        "hop_1_pct": cascade["hop1_shock"] * 100,
                        "hop_2_pct": cascade["hop2_shock"] * 100,
                        "hop_3_final_pct": cascade["hop3_final_shock"] * 100,
                    })
                    st.dataframe(detail, use_container_width=True, hide_index=True)
            except Exception as exc:
                st.error(f"Contagion cascade could not be calculated: {type(exc).__name__}: {str(exc)[:180]}")

        with tab_chaos:
            st.markdown("Enter a **hypothetical** headline to demonstrate a deterministic keyword-severity shock.")
            st.warning("This does not query a live news feed, run FinBERT, or rerun the portfolio simulation. Tail-risk changes are scenario multipliers on the base values below.")
            custom_headline = st.text_input(
                "Hypothetical breaking headline",
                value="Emergency RBI liquidity action after a banking cyberattack and global semiconductor supply halt",
                key="diff_chaos_headline",
            )
            c_mu, c_alpha, c_beta = st.columns(3)
            base_mu = c_mu.number_input("Baseline event intensity μ₀", min_value=0.001, max_value=5.0, value=0.05, step=0.01, key="diff_mu")
            hawkes_alpha = c_alpha.number_input("Hawkes α", min_value=0.0, max_value=0.95, value=0.14, step=0.01, key="diff_alpha")
            hawkes_beta = c_beta.number_input("Hawkes β", min_value=0.01, max_value=5.0, value=0.32, step=0.01, key="diff_beta")
            if hawkes_alpha >= hawkes_beta:
                st.warning("For a stationary exponential Hawkes process, select α < β before running the scenario.")
            base_var = st.number_input("Baseline 99% VaR (INR)", min_value=0.0, max_value=float(portfolio_value_inr), value=min(342_000.0, float(portfolio_value_inr)), step=10_000.0, key="diff_base_var")
            base_es = st.number_input("Baseline 99% ES (INR)", min_value=0.0, max_value=float(portfolio_value_inr) * 2, value=min(468_900.0, float(portfolio_value_inr) * 2), step=10_000.0, key="diff_base_es")
            if st.button("Inject hypothetical headline shock", type="primary", key="diff_run_chaos"):
                try:
                    st.session_state["diff_chaos_result"] = JudgeChaosHeadlineSimulator.execute_flash_crash_trace(
                        custom_headline, base_mu, hawkes_alpha, hawkes_beta, base_var, base_es
                    )
                except Exception as exc:
                    st.error(f"Headline scenario failed: {type(exc).__name__}: {str(exc)[:180]}")
            if "diff_chaos_result" in st.session_state:
                trace = st.session_state["diff_chaos_result"]
                scores = trace["heuristic_sentiment_scores"]
                telemetry = trace["stage3_hawkes_telemetry"]
                pnl_impact = trace["stage4_portfolio_impact"]
                hc1, hc2, hc3, hc4 = st.columns(4)
                hc1.metric("Keyword P(negative)*", f"{scores['P_negative_heuristic'] * 100:.1f}%", f"Z-shock {trace['stage2_sentiment_z_shock']}σ")
                hc2.metric("Scenario intensity μ", f"{telemetry['baseline_mu_after_scenario']:.3f}", f"+{telemetry['intensity_surge_pct']:.1f}%")
                hc3.metric("Scenario 99% VaR", f"₹{pnl_impact['var_99_after_scenario']:,.0f}", f"+₹{pnl_impact['var_99_after_scenario'] - pnl_impact['var_99_before']:,.0f}")
                hc4.metric("Scenario 99% ES", f"₹{pnl_impact['es_99_after_scenario']:,.0f}", f"+₹{pnl_impact['additional_tail_capital_at_risk']:,.0f}")
                st.json(trace)
                st.caption("*Uncalibrated heuristic score—not a real probability forecast or FinBERT output.")

        with tab_rev:
            max_target = max(200_000.0, min(portfolio_value_inr * 0.50, 2_500_000.0))
            default_target = min(750_000.0, max_target)
            target_loss = st.number_input(
                "Target portfolio loss (INR)",
                min_value=1_000.0,
                max_value=float(max_target),
                value=float(default_target),
                step=50_000.0,
                key="diff_reverse_target",
            )
            if st.button("Solve minimum-distance joint shock", type="primary", key="diff_run_reverse"):
                try:
                    st.session_state["diff_reverse_result"] = ReverseStressTestSolver.solve_most_probable_breaking_shock(
                        suite_returns, suite_weights, float(portfolio_value_inr), float(target_loss)
                    )
                except Exception as exc:
                    st.error(f"Reverse stress solver failed: {type(exc).__name__}: {str(exc)[:180]}")
            if "diff_reverse_result" in st.session_state:
                reverse = st.session_state["diff_reverse_result"]
                if not reverse.get("feasible", False):
                    st.error(f"No feasible solution: {reverse.get('message', reverse.get('solver_status'))}")
                else:
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Mahalanobis distance", f"{reverse['mahalanobis_distance_sigma']:.3f}")
                    m2.metric("Heuristic plausibility index", f"{reverse['statistical_plausibility_index']:.1f}/100")
                    m3.metric("Achieved loss", f"{reverse['actual_loss_pct'] * 100:.2f}%")
                    st.caption(f"Solver status: {reverse['solver_status']}. {reverse['plausibility_index_note']}")
                    rev_df = pd.DataFrame(reverse["asset_shock_vector"])
                    st.plotly_chart(dark_plot(px.bar(rev_df, x="ticker", y="required_return_shock_pct", title="Minimum-Distance Asset Shock Vector (%)")), use_container_width=True)
                    st.dataframe(rev_df, use_container_width=True, hide_index=True)

        with tab_india:
            st.markdown("**Illustrative cash-equity delivery costs, aggregate capital-gains assumptions, and market-impact add-on.**")
            st.caption("Rates should be rechecked for the actual transaction date, exchange, segment, broker and investor circumstances. The liquidity add-on is not a universal SEBI-prescribed formula.")
            initial_target = np.full(len(suite_tickers), 1.0 / len(suite_tickers))
            editor_df = pd.DataFrame({
                "ticker": suite_tickers,
                "current_weight": suite_weights,
                "target_weight": initial_target,
                "estimated_gain_on_cost_pct": np.full(len(suite_tickers), 14.0),
                "holding_days": np.full(len(suite_tickers), 210, dtype=int),
                "market_value": suite_weights * portfolio_value_inr,
                "adtv_currency": np.full(len(suite_tickers), 85_000_000.0),
            })
            edited = st.data_editor(
                editor_df,
                key="diff_india_editor",
                use_container_width=True,
                hide_index=True,
                disabled=["ticker", "current_weight", "market_value"],
                column_config={
                    "current_weight": st.column_config.NumberColumn("Current weight", format="%.4f"),
                    "target_weight": st.column_config.NumberColumn("Target weight", min_value=0.0, max_value=1.0, format="%.4f"),
                    "estimated_gain_on_cost_pct": st.column_config.NumberColumn("Estimated gain on cost (%)", min_value=-99.0, max_value=10000.0, format="%.2f"),
                    "holding_days": st.column_config.NumberColumn("Holding days", min_value=0, max_value=20000, step=1),
                    "market_value": st.column_config.NumberColumn("Position market value (INR)", format="%.2f"),
                    "adtv_currency": st.column_config.NumberColumn("Assumed ADTV (INR)", min_value=1.0, format="%.2f"),
                },
            )
            target_sum = float(pd.to_numeric(edited["target_weight"], errors="coerce").sum())
            target_w = pd.to_numeric(edited["target_weight"], errors="coerce").to_numpy(dtype=float)
            current_w = suite_weights
            if not np.isfinite(target_w).all() or not np.isclose(target_sum, 1.0, atol=1e-4):
                st.error(f"Target weights must be finite, non-negative and sum to 1. Current total: {target_sum:.4f}.")
            else:
                gain_pct = pd.to_numeric(edited["estimated_gain_on_cost_pct"], errors="coerce").to_numpy(dtype=float) / 100.0
                holding_days = pd.to_numeric(edited["holding_days"], errors="coerce").to_numpy(dtype=int)
                adtv_df = edited[["ticker", "market_value", "adtv_currency"]].copy()
                d1, d2, d3 = st.columns(3)
                base_lvar = d1.number_input("Base 99% VaR (INR)", min_value=0.0, max_value=float(portfolio_value_inr) * 2, value=min(342_000.0, float(portfolio_value_inr) * 2), step=10_000.0, key="diff_lvar_base")
                participation = d2.slider("Max liquidation participation (%)", 1.0, 30.0, 10.0, step=1.0, key="diff_lvar_participation")
                ltcg_used = d3.number_input("LTCG exemption already used (INR)", min_value=0.0, max_value=125_000.0, value=0.0, step=5_000.0, key="diff_ltcg_used")
                if st.button("Calculate liquidity and rebalance friction", type="primary", key="diff_run_india"):
                    try:
                        tax_out = IndianRegulatoryFrictionEngine.compute_rebalancing_tax_and_stt(
                            float(portfolio_value_inr), current_w, target_w, gain_pct, holding_days, ltcg_gain_already_used_currency=float(ltcg_used)
                        )
                        lvar_out = IndianRegulatoryFrictionEngine.compute_sebi_liquidity_adjusted_var(
                            float(base_lvar), adtv_df, max_participation_rate=float(participation / 100.0)
                        )
                        st.session_state["diff_india_result"] = {"tax": tax_out, "lvar": lvar_out}
                    except Exception as exc:
                        st.error(f"Friction calculation failed: {type(exc).__name__}: {str(exc)[:180]}")
                if "diff_india_result" in st.session_state:
                    result = st.session_state["diff_india_result"]
                    tax_out, lvar_out = result["tax"], result["lvar"]
                    k1, k2, k3 = st.columns(3)
                    k1.metric("Liquidity-adjusted VaR estimate", f"₹{lvar_out['liquidity_adjusted_lvar_99_currency']:,.0f}", f"+₹{lvar_out['liquidity_slippage_cost_currency']:,.0f} impact")
                    k2.metric("Estimated STT + stamp + fee + GST", f"₹{tax_out['stt_currency'] + tax_out['stamp_and_sebi_currency']:,.0f}")
                    k3.metric("Estimated capital-gains tax", f"₹{tax_out['stcg_tax_20pct_currency'] + tax_out['ltcg_tax_12_5pct_currency']:,.0f}", f"{tax_out['total_drag_bps']:.2f} bps total modeled drag")
                    st.write({
                        "Maximum days to liquidate full position at selected participation": lvar_out["max_days_to_liquidate_100pct"],
                        "Turnover (buy + sell value)": tax_out["total_turnover_currency"],
                        "Estimated total modeled friction (INR)": tax_out["total_statutory_and_tax_drag_currency"],
                    })
                    with st.expander("Assumptions / exclusions"):
                        st.write(tax_out["assumptions"])
                        st.write(lvar_out["method"])
except Exception as exc:
    st.error(f"Institutional Command Suite could not initialize: {type(exc).__name__}: {str(exc)[:180]}")
