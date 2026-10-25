from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from risklens.data.market_data import fetch_historical_prices
from risklens.data.validation import normalize_ticker, validate_price_data
from risklens.features.returns import calculate_returns, annualized_volatility, drawdown_series, sharpe_ratio, sortino_ratio
from risklens.features.volatility import parkinson_variance, garman_klass_variance, ewma_volatility
from risklens.ui import header, status_banner, dark_plot, pct

header(
    st,
    "Stock & Index Deep Dive",
    "Select a market symbol and inspect interactive candlesticks, drawdowns, OHLC range volatility estimators, and return distributions.",
)

c_sym, c_mkt, c_start, c_end = st.columns([2, 1, 1, 1])
symbol = c_sym.text_input("Ticker (examples: SPY, AAPL, RELIANCE.NS, TCS.BO, ^VIX)", "SPY").strip().upper()
market = c_mkt.selectbox("Market Suffix Helper", ["Auto", "NSE", "BSE", "US"])
start = c_start.date_input("Start Date", value=pd.to_datetime("2023-01-02").date())
end = c_end.date_input("End Date", value=pd.Timestamp.now().date())

opt_1, opt_2 = st.columns(2)
chart_mode = opt_1.radio("Price Chart Type", ["OHLC Candlestick + Moving Averages", "Adjusted Close Line"], horizontal=True)
force_refresh = opt_2.checkbox("Force refresh from provider (bypass cache)", value=False)

if start >= end:
    st.error("Start date must precede end date.")
    st.stop()

try:
    norm = normalize_ticker(symbol, None if market in ("Auto", "US") else market)
    result = fetch_historical_prices(norm, str(start), str(end + pd.Timedelta(days=1)), refresh=force_refresh)
    frame, warnings = validate_price_data(result.data)
    status_banner(st, result.status, result.provider, result.timestamp_utc, result.message)

    if result.status == "DEMO":
        st.warning(
            "Demo fixture data is not symbol-specific unless the requested ticker is present in the bundled multi-asset set. "
            "The OHLC demo series is a broad reference fixture, not a price claim for your symbol."
        )
    if frame.empty:
        st.error("No usable observations. Check the ticker, date range, provider setup, or demo fixture range.")
        st.stop()

    close = frame["Adj Close"] if "Adj Close" in frame.columns else frame.get("Close")
    if close is None:
        st.error("Provider data has no Close/Adj Close column.")
        st.stop()

    r = calculate_returns(close).dropna()
    dd = drawdown_series(r, input_is_returns=True)
    vol = annualized_volatility(r)
    sr = sharpe_ratio(r)
    sor = sortino_ratio(r)

    a, b, c, d, e = st.columns(5)
    a.metric("Observations", f"{len(close):,}")
    b.metric("Latest Close", f"{close.iloc[-1]:,.2f}")
    c.metric("Annualized Volatility", pct(vol))
    d.metric("Max Drawdown", pct(dd.min()))
    e.metric("Sharpe / Sortino", f"{sr:.2f} / {sor:.2f}")

    tab_price, tab_vol, tab_dist, tab_raw = st.tabs([
        "Price & drawdown",
        "Volatility measures",
        "Return distribution",
        "Validated OHLCV data",
    ])

    with tab_price:
        if chart_mode.startswith("OHLC") and all(k in frame.columns for k in ["Open", "High", "Low", "Close"]):
            fig_candle = go.Figure()
            fig_candle.add_trace(go.Candlestick(
                x=frame.index, open=frame["Open"], high=frame["High"], low=frame["Low"], close=frame["Close"], name=norm
            ))
            fig_candle.add_trace(go.Scatter(x=close.index, y=close.rolling(20).mean(), mode="lines", name="20D SMA", line=dict(width=1.5, color="#6F9B8A")))
            fig_candle.add_trace(go.Scatter(x=close.index, y=close.rolling(50).mean(), mode="lines", name="50D SMA", line=dict(width=1.5, color="#B88A55")))
            fig_candle.update_layout(title=f"{norm} · OHLC Candlestick & Moving Averages")
            st.plotly_chart(dark_plot(fig_candle, show_rangeslider=True), use_container_width=True)
        else:
            st.plotly_chart(dark_plot(px.line(close.rename("Price"), title=f"{norm} · Selected Close Series"), show_rangeslider=True), use_container_width=True)

        st.plotly_chart(dark_plot(px.area(dd.rename("Drawdown"), title="Historical Drawdown Profile")), use_container_width=True)

    with tab_vol:
        if all(k in frame for k in ["Open", "High", "Low", "Close"]):
            vol_df = pd.DataFrame({
                "Parkinson Ann. Vol": np.sqrt(parkinson_variance(frame) * 252),
                "Garman–Klass Ann. Vol": np.sqrt(garman_klass_variance(frame) * 252),
                "EWMA Ann. Vol (λ=0.94)": ewma_volatility(r) * np.sqrt(252),
            }).dropna()
            st.plotly_chart(dark_plot(px.line(vol_df, title="Annualized Range & Return Volatility Estimators")), use_container_width=True)
        else:
            st.plotly_chart(dark_plot(px.line((ewma_volatility(r) * np.sqrt(252)).rename("EWMA Vol"), title="EWMA Annualized Volatility")), use_container_width=True)

    with tab_dist:
        st.plotly_chart(dark_plot(px.histogram(r * 100, nbins=65, title=f"{norm} Daily Return Distribution (%)")), use_container_width=True)

    with tab_raw:
        st.dataframe(frame.tail(50), use_container_width=True)
        st.download_button("Download Validated Series CSV", frame.to_csv().encode(), file_name=f"{norm}_validated_ohlc.csv", mime="text/csv")

    for w in warnings:
        st.warning(w)
except Exception as e:
    st.error(f"Analysis could not be completed: {type(e).__name__}: {str(e)[:250]}")
