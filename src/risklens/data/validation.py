"""Validation and normalization for market time series."""
from __future__ import annotations
import numpy as np
import pandas as pd

REQUIRED_OHLC = ("Open", "High", "Low", "Close")

def normalize_ticker(symbol: str, market: str | None = None) -> str:
    """Normalize common NSE/BSE suffixes without asserting that a symbol exists."""
    s = symbol.strip().upper().replace(" ", "")
    if not s:
        raise ValueError("Enter a ticker symbol.")
    if s.endswith((".NS", ".BO", "=X", "=F")) or s.startswith("^"):
        return s
    if market:
        m = market.strip().lower()
        if m in {"nse", "india nse"}: return f"{s}.NS"
        if m in {"bse", "india bse"}: return f"{s}.BO"
    return s

def validate_price_data(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return a normalized copy and warnings; questionable rows are not silently repaired."""
    warnings: list[str] = []
    if frame is None or frame.empty:
        return pd.DataFrame(), ["No price observations were returned."]
    df = frame.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        if "Date" in df.columns or "Datetime" in df.columns:
            col = "Date" if "Date" in df.columns else "Datetime"
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
            df = df.set_index(col)
        else:
            return pd.DataFrame(), ["Price data must have a DatetimeIndex or Date/Datetime column."]
    idx = pd.to_datetime(df.index, utc=True, errors="coerce")
    df.index = idx
    invalid_index = df.index.isna().sum()
    if invalid_index:
        warnings.append(f"Dropped {invalid_index} rows with invalid timestamps.")
        df = df.loc[~df.index.isna()]
    if df.index.has_duplicates:
        warnings.append(f"Found {int(df.index.duplicated(keep='last').sum())} duplicate timestamps; retained last row per timestamp.")
        df = df[~df.index.duplicated(keep="last")]
    if not df.index.is_monotonic_increasing:
        warnings.append("Input timestamps were out of order; sorted chronologically.")
        df = df.sort_index()
    # Standardize case-insensitive OHLC names.
    rename = {c: c.title() for c in df.columns if c.lower() in {"open", "high", "low", "close", "adj close", "volume"}}
    df = df.rename(columns=rename)
    for col in [c for c in (*REQUIRED_OHLC, "Adj Close", "Volume") if c in df.columns]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    px_cols = [c for c in (*REQUIRED_OHLC, "Adj Close") if c in df.columns]
    if px_cols:
        bad = (df[px_cols] <= 0).any(axis=1) | ~np.isfinite(df[px_cols]).all(axis=1)
        if bad.any(): warnings.append(f"{int(bad.sum())} rows contain non-positive/non-finite prices and remain flagged for downstream exclusion.")
    if all(c in df.columns for c in REQUIRED_OHLC):
        bad_ohlc = (df["High"] < df[["Open", "Close", "Low"]].max(axis=1)) | (df["Low"] > df[["Open", "Close", "High"]].min(axis=1))
        if bad_ohlc.any(): warnings.append(f"{int(bad_ohlc.sum())} rows violate OHLC relationships.")
    if df.isna().any().any(): warnings.append("Missing values are present; no interpolation was applied.")
    return df, warnings

def detect_stale_data(frame: pd.DataFrame, max_age_days: int = 5, now: pd.Timestamp | None = None) -> bool:
    if frame is None or frame.empty:
        return True
    now = now or pd.Timestamp.now(tz="UTC")
    last = pd.Timestamp(frame.index.max())
    if last.tzinfo is None: last = last.tz_localize("UTC")
    return (now - last).total_seconds() > max_age_days * 86400
