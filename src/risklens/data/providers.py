"""Provider adapters returning normalized frames and provenance metadata."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import pandas as pd
import requests
from risklens.data.http import request_with_retry
from risklens.config import get_setting, SETTINGS
from risklens.data.caching import cache_get, cache_set
from risklens.data.validation import normalize_ticker, validate_price_data

@dataclass
class DataResult:
    data: Any
    provider: str
    status: str
    timestamp_utc: str
    message: str = ""

    @property
    def is_demo(self) -> bool:
        return self.status == "DEMO"


class MarketDataProvider:
    name = "base"
    def historical(self, symbol: str, start: str, end: str) -> DataResult:
        raise NotImplementedError

class YFinanceProvider(MarketDataProvider):
    name = "yfinance"
    def historical(self, symbol: str, start: str, end: str) -> DataResult:
        try:
            import yfinance as yf  # optional dependency
        except Exception:
            raise RuntimeError("yfinance is not installed; use demo mode or install requirements.txt.")
        df = yf.download(symbol, start=start, end=end, auto_adjust=False, progress=False, threads=False, timeout=float(get_setting("REQUEST_TIMEOUT_SECONDS", SETTINGS.request_timeout_seconds)))
        if df is None or df.empty:
            raise RuntimeError(f"Provider returned no historical data for {symbol}.")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        df, warnings = validate_price_data(df)
        if df.empty:
            raise RuntimeError("Provider returned data that failed validation.")
        return DataResult(df, self.name, "FETCHED", str(df.index.max()) if not df.empty else "", "; ".join(warnings))

class FinnhubProvider:
    """Finnhub REST client for historical candles; only used when a key is configured."""
    def historical(self, symbol: str, start: str, end: str) -> DataResult:
        key = get_setting("FINNHUB_API_KEY")
        if not key: raise RuntimeError("FINNHUB_API_KEY is not configured.")
        import calendar, datetime as dt
        start_ts = int(pd.Timestamp(start, tz="UTC").timestamp())
        end_ts = int(pd.Timestamp(end, tz="UTC").timestamp())
        response = request_with_retry("GET", "https://finnhub.io/api/v1/stock/candle", params={"symbol":symbol,"resolution":"D","from":start_ts,"to":end_ts,"token":key}, timeout=float(get_setting("REQUEST_TIMEOUT_SECONDS", SETTINGS.request_timeout_seconds)))
        if response.status_code != 200: raise RuntimeError(f"Finnhub returned HTTP {response.status_code}.")
        payload = response.json()
        if payload.get("s") != "ok": raise RuntimeError("Finnhub returned no usable candles for this symbol.")
        df = pd.DataFrame({"Open":payload["o"],"High":payload["h"],"Low":payload["l"],"Close":payload["c"],"Volume":payload["v"]}, index=pd.to_datetime(payload["t"], unit="s", utc=True))
        df, warnings = validate_price_data(df)
        return DataResult(df, "Finnhub", "FETCHED", str(df.index.max()) if not df.empty else "", "; ".join(warnings))


class AlphaVantageProvider:
    """Alpha Vantage daily adjusted-price adapter; quota/API errors fail closed."""
    name = "Alpha Vantage"
    def historical(self, symbol: str, start: str, end: str) -> DataResult:
        key = get_setting("ALPHA_VANTAGE_API_KEY")
        if not key: raise RuntimeError("ALPHA_VANTAGE_API_KEY is not configured.")
        response = request_with_retry("GET", "https://www.alphavantage.co/query", params={"function":"TIME_SERIES_DAILY_ADJUSTED","symbol":symbol,"outputsize":"full","apikey":key}, timeout=float(get_setting("REQUEST_TIMEOUT_SECONDS", SETTINGS.request_timeout_seconds)))
        if response.status_code != 200: raise RuntimeError(f"Alpha Vantage returned HTTP {response.status_code}.")
        payload = response.json()
        if "Note" in payload or "Information" in payload: raise RuntimeError("Alpha Vantage rate limit or entitlement message returned.")
        if "Error Message" in payload: raise RuntimeError("Alpha Vantage rejected the symbol or request.")
        series = payload.get("Time Series (Daily)")
        if not series: raise RuntimeError("Alpha Vantage response did not contain a daily time series.")
        frame = pd.DataFrame.from_dict(series, orient="index")
        frame.index = pd.to_datetime(frame.index, utc=True, errors="coerce")
        frame = frame.rename(columns={"1. open":"Open","2. high":"High","3. low":"Low","4. close":"Close","5. adjusted close":"Adj Close","6. volume":"Volume"})
        frame = frame.loc[(frame.index >= pd.Timestamp(start, tz="UTC")) & (frame.index <= pd.Timestamp(end, tz="UTC"))]
        frame, warnings = validate_price_data(frame)
        if frame.empty: raise RuntimeError("Alpha Vantage returned no observations in the requested range.")
        return DataResult(frame, self.name, "FETCHED", str(frame.index.max()) if not frame.empty else "", "; ".join(warnings))
