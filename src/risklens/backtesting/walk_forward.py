"""Leakage-resistant simple walk-forward VaR backtest."""
from __future__ import annotations
import numpy as np
import pandas as pd

def walk_forward_historical_var(returns: pd.Series, lookback: int=252, confidence: float=0.95) -> pd.DataFrame:
    r=pd.Series(returns).replace([np.inf,-np.inf],np.nan).dropna().astype(float)
    if lookback<20: raise ValueError("lookback must be >=20")
    rows=[]
    for i in range(lookback,len(r)):
        training=r.iloc[i-lookback:i]  # strictly before the forecast observation
        losses=-training.to_numpy(); var=float(np.quantile(losses,confidence,method="higher"))
        tail=losses[losses>=var]
        es=float(tail.mean()) if len(tail) else var
        actual_loss=float(-r.iloc[i]); rows.append({"date":r.index[i],"var":var,"es":es,"actual_loss":actual_loss,"breach":actual_loss>var,"training_end":r.index[i-1]})
    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame(columns=["var","es","actual_loss","breach","training_end"])
