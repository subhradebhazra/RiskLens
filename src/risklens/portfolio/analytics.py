"""Portfolio weights and historical analytics."""
from __future__ import annotations
import numpy as np
import pandas as pd
from risklens.features.returns import annualized_volatility, sharpe_ratio, sortino_ratio, drawdown_series

def normalize_holdings(holdings: pd.DataFrame) -> pd.DataFrame:
    df=holdings.copy()
    if "units" not in df.columns and "shares" in df.columns:
        df=df.rename(columns={"shares":"units"})
    if "ticker" not in df.columns and "symbol" in df.columns:
        df=df.rename(columns={"symbol":"ticker"})
    required={"ticker","units","currency"}
    missing=required-set(df.columns)
    if missing: raise ValueError(f"Missing holdings columns: {', '.join(sorted(missing))}")
    df["ticker"]=df["ticker"].astype(str).str.strip().str.upper()
    df["units"]=pd.to_numeric(df["units"],errors="coerce")
    if df["units"].isna().any() or (~np.isfinite(df["units"])).any(): raise ValueError("Units must be finite numbers.")
    if (df["units"]<0).any(): raise ValueError("Short/negative units are not supported by this long-only portfolio workflow.")
    return df

def portfolio_weights(values: pd.Series) -> pd.Series:
    v=pd.Series(values,dtype=float)
    total=float(v.sum())
    return v/total if abs(total)>1e-12 else pd.Series(np.nan,index=v.index)

def concentration_hhi(weights: pd.Series) -> float:
    w=pd.Series(weights).dropna().astype(float)
    return float(np.sum(w*w)) if len(w) else float("nan")

def portfolio_return_series(asset_returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    cols=[c for c in asset_returns.columns if c in weights.index]
    if not cols: return pd.Series(dtype=float)
    w=weights.loc[cols].astype(float); aligned=asset_returns[cols].dropna(how="any")
    # Re-normalize only among selected columns, not across missing-data gaps.
    if abs(w.sum())>1e-12: w=w/w.sum()
    return aligned.mul(w,axis=1).sum(axis=1)

def portfolio_summary(returns: pd.Series, risk_free_annual: float=0.0) -> dict:
    r=pd.Series(returns).dropna()
    dd=drawdown_series(r,input_is_returns=True)
    return {"observations":len(r),"annualized_volatility":annualized_volatility(r),"sharpe":sharpe_ratio(r,risk_free_annual),"sortino":sortino_ratio(r,risk_free_annual),"max_drawdown":float(dd.min()) if len(dd) else float("nan"),"cumulative_return":float((1+r).prod()-1) if len(r) else float("nan")}
