"""Market data orchestration with explicit provider provenance and demo fallback."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from risklens.data.providers import DataResult, YFinanceProvider, FinnhubProvider, AlphaVantageProvider
from risklens.data.caching import cache_get, cache_set
from risklens.data.validation import normalize_ticker, validate_price_data
from risklens.security import sanitize_text
from risklens.config import ROOT, SETTINGS, get_setting


def fetch_historical_prices(symbol: str = "SPY", start: str = "2023-01-01", end: str | None = None, market: str | None = None, allow_demo: bool = True, refresh: bool = False) -> DataResult:
    normalized = normalize_ticker(symbol, market)
    end = end or (pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    priority = (get_setting("MARKET_DATA_PRIORITY", "yfinance -> Finnhub -> Alpha Vantage -> Demo") or "").lower()
    ttl = max(0, int(float(get_setting("CACHE_TTL_SECONDS", SETTINGS.cache_ttl_seconds))))
    key = f"hist|{priority}|{normalized}|{start}|{end}"
    if not refresh and ttl > 0:
        cached = cache_get(key, ttl)
        if cached is not None:
            return DataResult(cached.copy(), "local cache", "CACHED", str(cached.index.max()), "Cached result; not a live quote.")
    errors = []
    if priority.strip() == "demo only":
        providers = []
    else:
        candidates = {"yfinance": YFinanceProvider(), "finnhub": FinnhubProvider(), "alpha vantage": AlphaVantageProvider()}
        # Parse the visible priority in order; configured providers only are attempted.
        keys = []
        for token in [part.strip() for part in priority.split("->")]:
            match = next((key for key in candidates if key in token), None)
            if match and match not in keys: keys.append(match)
        if not keys: keys = ["yfinance", "finnhub", "alpha vantage"]
        providers = [candidates[k] for k in keys if k == "yfinance" or (k == "finnhub" and get_setting("FINNHUB_API_KEY")) or (k == "alpha vantage" and get_setting("ALPHA_VANTAGE_API_KEY"))]
    for provider in providers:
        try:
            result = provider.historical(normalized, start, end)
            # Only label a request stale when the requested window includes the recent period.
            try:
                requested_end = pd.Timestamp(end, tz="UTC")
                now = pd.Timestamp.now(tz="UTC")
                latest = pd.Timestamp(result.data.index.max()) if not result.data.empty else None
                if latest is not None and requested_end >= now - pd.Timedelta(days=10) and (now-latest).total_seconds() > 5*86400:
                    result.status = "STALE"
                    result.message = (result.message + " " if result.message else "") + "Latest returned observation is over five days old although the requested window includes the recent period."
            except Exception:
                result.message = (result.message + " " if result.message else "") + "Freshness could not be assessed from the returned timestamps."
            cache_set(key, result.data.copy())
            return result
        except Exception as exc:
            errors.append(sanitize_text(f"{provider.name}: {str(exc)[:220]}", [get_setting(n, "") or "" for n in ("GEMINI_API_KEY","NEWSDATA_API_KEY","CURRENTS_API_KEY","FINNHUB_API_KEY","ALPHA_VANTAGE_API_KEY")]))
    if allow_demo:
        demo = load_demo_prices()
        # Map the demo timeline onto requested symbol; this remains explicitly demo data.
        if demo.empty:
            return DataResult(pd.DataFrame(), "none", "UNAVAILABLE", "", "No provider succeeded and no demo fixture is available.")
        start_ts = pd.Timestamp(start, tz="UTC")
        end_ts = pd.Timestamp(end, tz="UTC")
        demo = demo.loc[(demo.index >= start_ts) & (demo.index <= end_ts)]
        return DataResult(demo.copy(), "bundled broad-market OHLC fixture (not symbol-specific)", "DEMO", str(demo.index.max()) if not demo.empty else "", "Requested symbol-specific market history is unavailable. Demo fixture must not be interpreted as pricing for the requested symbol. " + " | ".join(errors))
    return DataResult(pd.DataFrame(), "none", "UNAVAILABLE", "", " | ".join(errors))


def load_demo_prices() -> pd.DataFrame:
    path = ROOT / "data" / "demo" / "demo_prices.csv"
    if not path.exists(): return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    if df.index.tz is None: df.index = df.index.tz_localize("UTC")
    return df


def fetch_latest_quotes(symbols: list[str]) -> dict[str, DataResult]:
    """Return each provider's most recent daily historical row, not an intraday quote.

    This method intentionally labels the returned observation by provider freshness;
    a daily close must not be represented as a real-time quote.
    """
    results = {}
    for symbol in symbols:
        result = fetch_historical_prices(symbol, start=(pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=45)).strftime("%Y-%m-%d"))
        if not result.data.empty:
            result.data = result.data.tail(1).copy()
            result.message = (result.message + " " if result.message else "") + "Latest daily historical observation; not a guaranteed real-time quote."
        results[symbol] = result
    return results


def fetch_company_metadata(symbol: str) -> dict[str, str]:
    return {"symbol": normalize_ticker(symbol), "sector": "Unavailable unless supported by provider", "source": "No metadata adapter configured"}

def fetch_macro_data(series: str = "^VIX") -> DataResult:
    return fetch_historical_prices(series, start=(pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=365)).strftime("%Y-%m-%d"))

def cache_market_data(key: str, data: pd.DataFrame) -> None:
    cache_set(key, data.copy())


def align_trading_calendars(frames: dict[str, pd.DataFrame], join: str = "inner") -> pd.DataFrame:
    if not frames: return pd.DataFrame()
    series = []
    for symbol, frame in frames.items():
        col = "Adj Close" if "Adj Close" in frame else "Close"
        if col not in frame: continue
        s = frame[col].rename(symbol)
        series.append(s)
    return pd.concat(series, axis=1, join=join).sort_index() if series else pd.DataFrame()
