"""Marginal and component volatility contributions."""
from __future__ import annotations
import numpy as np
import pandas as pd

def volatility_contributions(returns: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    x=returns.dropna(how="any"); cols=[c for c in x if c in weights.index]
    if len(cols)<2: return pd.DataFrame(columns=["weight","marginal_risk","component_risk"])
    w=weights[cols].to_numpy(dtype=float); w=w/w.sum(); cov=x[cols].cov().to_numpy()*252
    port_var=float(w@cov@w); port_vol=float(np.sqrt(max(port_var,0)))
    marginal=(cov@w)/port_vol if port_vol>1e-12 else np.zeros(len(w))
    component=w*marginal
    return pd.DataFrame({"weight":w,"marginal_risk":marginal,"component_risk":component},index=cols)
