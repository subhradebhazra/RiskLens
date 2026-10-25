"""Deterministic scenario stress testing for supplied portfolio weights."""
from __future__ import annotations
import pandas as pd

DEFAULT_SCENARIOS={
 "Broad-market decline":-0.15,
 "Volatility shock":-0.08,
 "Single-stock crash":-0.35,
 "Crisis correlation increase":-0.12,
 "Jump-intensity surge":-0.18,
 "Crisis-regime transition":-0.22,
 "Clustered downside events":-0.25,
 "Multi-factor combined shock":-0.30,
}

def run_scenario(holdings: pd.DataFrame, scenario: str, shocks: dict[str,float]|None=None) -> dict:
    """Return mark-to-market P&L from user-defined holdings and deterministic shock vector."""
    if holdings.empty: raise ValueError("Portfolio holdings are empty.")
    df=holdings.copy()
    if "market_value" not in df.columns: raise ValueError("Holdings require a market_value column in base currency.")
    default=DEFAULT_SCENARIOS.get(scenario,-0.10)
    if shocks is None: shocks={str(t):default for t in df.get("ticker",df.index)}
    returns=[]
    for _,row in df.iterrows(): returns.append(float(shocks.get(str(row.get("ticker","")),default)))
    values=df["market_value"].astype(float).to_numpy(); p_and_l=float((values*returns).sum()); total=float(values.sum())
    return {"scenario":scenario,"pnl":p_and_l,"loss_pct":(-p_and_l/total if total else float("nan")),"asset_results":pd.DataFrame({"ticker":df.get("ticker",df.index),"market_value":values,"shock_return":returns,"pnl":values*np.asarray(returns)}),"type":"Hypothetical deterministic scenario"}
