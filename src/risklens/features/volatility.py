"""Range-based and return-based volatility estimators."""
from __future__ import annotations
import numpy as np
import pandas as pd

def parkinson_variance(ohlc: pd.DataFrame) -> pd.Series:
    if not {"High", "Low"}.issubset(ohlc.columns): raise ValueError("Parkinson estimator requires High and Low.")
    ratio = np.log(ohlc["High"].astype(float) / ohlc["Low"].astype(float))
    return ratio.pow(2) / (4 * np.log(2))

def garman_klass_variance(ohlc: pd.DataFrame) -> pd.Series:
    required = {"Open", "High", "Low", "Close"}
    if not required.issubset(ohlc.columns): raise ValueError("Garman-Klass estimator requires OHLC columns.")
    o, h, l, c = (np.log(ohlc[k].astype(float)) for k in ("Open", "High", "Low", "Close"))
    return (0.5 * (h-l)**2 - (2*np.log(2)-1)*(c-o)**2).clip(lower=0)

def ewma_volatility(returns: pd.Series, lam: float = 0.94) -> pd.Series:
    r = pd.Series(returns).dropna().astype(float)
    if not 0 < lam < 1: raise ValueError("EWMA lambda must be between 0 and 1.")
    return r.pow(2).ewm(alpha=1-lam, adjust=False).mean().pow(0.5)
