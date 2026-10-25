from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from risklens.data.demo_assets import load_demo_portfolio
from risklens.portfolio.analytics import normalize_holdings, portfolio_weights, concentration_hhi, portfolio_return_series, portfolio_summary
from risklens.portfolio.contributions import volatility_contributions
from risklens.risk.var import historical_var_es
from risklens.reporting.reports import build_report, report_json
from risklens.ui import header, status_banner, dark_plot, pct, load_workspace_prices

header(
    st,
    "Multi-Currency Portfolio X-Ray",
    "Editable holdings, base-currency valuation, hierarchical sunburst concentration, dependence, and downside risk contributors.",
)

prices, data_status, data_provider, data_ts = load_workspace_prices(st)
if prices.empty:
    st.error("Demo fixtures unavailable.")
    st.stop()

status_banner(st, data_status, data_provider, data_ts, "Edit holdings inline or upload a CSV. Modifications sync across your active session.")

c_base, c_up = st.columns([1, 3])
base = c_base.selectbox("Base currency", ["USD", "INR"], index=0)
upload = c_up.file_uploader("Upload holdings CSV (ticker, units, currency; optional price, sector, benchmark)", type=["csv"])

if upload:
    try:
        raw = pd.read_csv(upload)
    except Exception as e:
        st.error(f"Could not parse CSV: {type(e).__name__}")
        st.stop()
else:
    raw = load_demo_portfolio()

if "current_price" not in raw.columns:
    if "price" in raw.columns:
        raw = raw.rename(columns={"price": "current_price"})
    elif "latest_price" in raw.columns:
        raw = raw.rename(columns={"latest_price": "current_price"})
    else:
        raw["current_price"] = raw["ticker"].map({a: float(prices[a].iloc[-1]) for a in prices.columns if a != "USDINR=X"})

if "sector" not in raw.columns:
    raw["sector"] = "Unclassified"
if "benchmark" not in raw.columns:
    raw["benchmark"] = "SPY"
if "currency" not in raw.columns:
    raw["currency"] = "USD"
if "units" not in raw.columns:
    raw["units"] = 0.0

edited = st.data_editor(
    raw[["ticker", "units", "current_price", "currency", "sector", "benchmark"]],
    num_rows="dynamic",
    use_container_width=True,
    key="xray_holdings",
)

try:
    holdings = normalize_holdings(edited.rename(columns={"current_price": "price"}).rename(columns={"price": "current_price"}))
    holdings["current_price"] = pd.to_numeric(holdings["current_price"], errors="coerce")
    if holdings["current_price"].isna().any() or (holdings["current_price"] <= 0).any():
        st.error("Enter a positive numeric current_price for each holding.")
        st.stop()

    fx = float(prices["USDINR=X"].iloc[-1])
    vals, missing = [], []
    for _, row in holdings.iterrows():
        ccy = str(row["currency"]).upper()
        v = float(row["units"]) * float(row["current_price"])
        if ccy == base:
            vals.append(v)
        elif ccy == "USD" and base == "INR":
            vals.append(v * fx)
        elif ccy == "INR" and base == "USD":
            vals.append(v / fx)
        else:
            vals.append(np.nan)
            missing.append(str(row["ticker"]))

    holdings["market_value"] = vals
    if missing:
        st.error("Missing or unsupported FX conversion for: " + ", ".join(missing) + ". Combined portfolio totals are blocked.")
        st.stop()

    holdings["weight"] = portfolio_weights(holdings["market_value"])
    st.session_state["shared_portfolio_holdings"] = holdings.copy()

    total = float(holdings["market_value"].sum())
    hhi = concentration_hhi(holdings["weight"])
    a, b, c, d = st.columns(4)
    a.metric("Portfolio value", f"{total:,.2f} {base}")
    b.metric("Holdings", f"{len(holdings)} (Eff. N: {1/hhi:.1f})" if hhi > 0 else str(len(holdings)))
    c.metric("HHI concentration", f"{hhi:.3f}")
    d.metric("Top holding", str(holdings.loc[holdings["weight"].idxmax(), "ticker"]) if len(holdings) else "N/A")

    st.subheader("Asset and sector allocation")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.plotly_chart(dark_plot(px.sunburst(holdings, path=["sector", "ticker"], values="market_value", title="Hierarchical Sector → Asset Sunburst")), use_container_width=True)
    with col_s2:
        st.plotly_chart(dark_plot(px.pie(holdings, names="ticker", values="market_value", hole=0.45, title="Allocation by Asset")), use_container_width=True)

    st.dataframe(holdings, use_container_width=True, hide_index=True)

    known = [s for s in holdings["ticker"] if s in prices.columns and s != "USDINR=X"]
    if len(known) >= 1:
        asset_prices = prices[known].copy()
        for _, row in holdings.iterrows():
            if row["ticker"] in known and str(row["currency"]).upper() != base:
                if str(row["currency"]).upper() == "INR" and base == "USD":
                    asset_prices[row["ticker"]] = asset_prices[row["ticker"]] / prices["USDINR=X"]
                elif str(row["currency"]).upper() == "USD" and base == "INR":
                    asset_prices[row["ticker"]] = asset_prices[row["ticker"]] * prices["USDINR=X"]

        rets = asset_prices.pct_change(fill_method=None).dropna(how="any")
        weight_map = holdings.set_index("ticker").loc[known, "weight"]
        pr = portfolio_return_series(rets, weight_map)
        summary = portfolio_summary(pr)
        rp = pr * total
        v95 = historical_var_es(rp.to_numpy(), 0.95)
        v99 = historical_var_es(rp.to_numpy(), 0.99)

        x, y, z, q = st.columns(4)
        x.metric("Annualized volatility", pct(summary["annualized_volatility"]))
        y.metric("Sharpe ratio", f"{summary['sharpe']:.2f}" if np.isfinite(summary["sharpe"]) else "N/A")
        z.metric("Max drawdown", pct(summary["max_drawdown"]))
        q.metric("99% historical VaR", f"{v99.var:,.2f} {base}")

        c_corr, c_rc = st.columns(2)
        with c_corr:
            st.plotly_chart(dark_plot(px.imshow(rets.corr(), text_auto=".2f", aspect="auto", title="Return correlation matrix")), use_container_width=True)
        contrib = volatility_contributions(rets, weight_map)
        with c_rc:
            if not contrib.empty:
                c_df = contrib.reset_index().rename(columns={"index": "ticker"})
                st.plotly_chart(dark_plot(px.bar(c_df, x="ticker", y=["weight", "component_risk", "marginal_risk"], barmode="group", title="Marginal & Component Risk Attribution")), use_container_width=True)
        if not contrib.empty:
            st.dataframe(contrib, use_container_width=True)
        st.caption(f"Historical calculations use {len(rets)} aligned observations. FX conversion uses bundled USDINR demo series aligned by date; base: {base}.")

    unknown = set(holdings["ticker"]) - set(known)
    if unknown:
        st.warning("Historical risk metrics are unavailable for non-fixture tickers: " + ", ".join(sorted(unknown)) + ". Add a provider or matching historical data; no returns were invented.")

    d1, d2 = st.columns(2)
    d1.download_button("Download X-Ray CSV", holdings.to_csv(index=False).encode(), file_name="risklens_portfolio_xray.csv", mime="text/csv")
    xray_rep = build_report("Portfolio X-Ray Report", {"base_currency": base, "total_value": total, "hhi": hhi, "holdings": holdings.to_dict("records")})
    d2.download_button("Download Audit JSON Report", report_json(xray_rep).encode(), file_name="risklens_portfolio_xray.json", mime="application/json")
except Exception as e:
    if not st.session_state.get("_xray_stopped", False):
        st.error(f"Portfolio analysis failed validation: {type(e).__name__}: {str(e)[:250]}")
