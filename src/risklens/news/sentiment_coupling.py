"""Timestamp-safe, lagged sentiment/volatility coupling diagnostics."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

def evaluate_sentiment_coupling(daily_sentiment: pd.Series, returns: pd.Series, min_observations: int=60) -> dict:
    """Test lagged negative sentiment against subsequent absolute returns; no same-day forward leakage."""
    sentiment=pd.Series(daily_sentiment).copy(); sentiment.index=pd.to_datetime(sentiment.index,utc=True)
    r=pd.Series(returns).copy(); r.index=pd.to_datetime(r.index,utc=True)
    # Shift sentiment one trading observation before associating with realized absolute return.
    frame=pd.concat([sentiment.rename("sentiment"),r.abs().rename("future_abs_return")],axis=1).sort_index()
    frame["sentiment_lagged"]=frame["sentiment"].shift(1)
    frame=frame[["sentiment_lagged","future_abs_return"]].dropna()
    if len(frame)<min_observations:
        return {"validated":False,"sample_size":len(frame),"correlation":None,"p_value":None,"reason":f"Need at least {min_observations} lagged aligned observations; sentiment remains contextual only."}
    corr,p=spearmanr(frame["sentiment_lagged"],frame["future_abs_return"])
    # Conservative predeclared gate: negative sentiment score should positively associate with future volatility score only if evidence is stable.
    stable=bool(np.isfinite(corr) and np.isfinite(p) and p<0.05 and corr>0)
    return {"validated":stable,"sample_size":len(frame),"correlation":float(corr),"p_value":float(p),"reason":"Nominal Spearman gate; inspect multiple-testing, regime stability and economic magnitude before changing Hawkes intensity." if stable else "Validation gate failed; sentiment remains contextual only."}
