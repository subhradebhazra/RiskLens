"""Return and drawdown calculations, implemented as pure functions."""
from __future__ import annotations
import numpy as np
import pandas as pd

def calculate_returns(prices: pd.Series | pd.DataFrame, log: bool = False) -> pd.Series | pd.DataFrame:
    px = prices.astype(float)
    if log:
        out = np.log(px / px.shift(1))
    else:
        out = px.pct_change(fill_method=None)
    return out.replace([np.inf, -np.inf], np.nan).dropna(how="all")

def annualized_volatility(returns: pd.Series, periods: int = 252) -> float:
    r = pd.Series(returns).dropna().astype(float)
    return float(r.std(ddof=1) * np.sqrt(periods)) if len(r) > 1 else float("nan")

def cumulative_returns(returns: pd.Series) -> pd.Series:
    r = pd.Series(returns).fillna(0.0).astype(float)
    return (1 + r).cumprod() - 1

def drawdown_series(prices_or_returns: pd.Series, input_is_returns: bool = False) -> pd.Series:
    s = pd.Series(prices_or_returns).dropna().astype(float)
    wealth = (1 + s).cumprod() if input_is_returns else s
    if wealth.empty: return wealth
    return wealth / wealth.cummax() - 1

def sharpe_ratio(returns: pd.Series, risk_free_annual: float = 0.0, periods: int = 252) -> float:
    r = pd.Series(returns).dropna().astype(float)
    if len(r) < 2 or r.std(ddof=1) <= 1e-15: return float("nan")
    rf_period = (1 + risk_free_annual) ** (1 / periods) - 1
    return float((r.mean() - rf_period) / r.std(ddof=1) * np.sqrt(periods))

def sortino_ratio(returns: pd.Series, risk_free_annual: float = 0.0, periods: int = 252) -> float:
    r = pd.Series(returns).dropna().astype(float)
    if len(r) < 2: return float("nan")
    rf_period = (1 + risk_free_annual) ** (1 / periods) - 1
    downside = np.minimum(r.to_numpy() - rf_period, 0)
    dd = float(np.sqrt(np.mean(downside**2)))
    return float((r.mean() - rf_period) / dd * np.sqrt(periods)) if dd > 1e-15 else float("nan")
