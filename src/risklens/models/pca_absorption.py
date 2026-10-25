"""PCA absorption ratio with Ledoit-Wolf covariance and rolling diagnostics."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

@dataclass
class AbsorptionFit:
    ratio: float
    eigenvalues: np.ndarray
    top_k: int
    sample_size: int
    asset_count: int
    covariance_method: str
    rolling: pd.Series
    warnings: list[str]

def absorption_ratio(returns: pd.DataFrame, top_k: int | None=None, window: int=60) -> AbsorptionFit:
    x=returns.replace([np.inf,-np.inf],np.nan).dropna(how="any")
    n,d=x.shape; warns=[]
    if d<2 or n<10:
        return AbsorptionFit(float("nan"),np.array([]),top_k or 1,n,d,"unavailable",pd.Series(dtype=float),["At least two assets and ten aligned observations are required."])
    k=top_k or max(1,int(np.ceil(np.sqrt(d))))
    if not 1<=k<=d: raise ValueError("top_k must be between 1 and number of assets.")
    try:
        cov=LedoitWolf().fit(x.to_numpy()).covariance_
        eig=np.maximum(np.linalg.eigvalsh(cov),0)[::-1]
        ratio=float(eig[:k].sum()/eig.sum()) if eig.sum()>0 else float("nan")
        method="Ledoit-Wolf covariance"
    except Exception as exc:
        cov=x.cov().to_numpy(); eig=np.maximum(np.linalg.eigvalsh(cov),0)[::-1]
        ratio=float(eig[:k].sum()/eig.sum()) if eig.sum()>0 else float("nan")
        method="sample covariance fallback"; warns.append(f"Ledoit-Wolf failed ({type(exc).__name__}); sample covariance used.")
    vals=[]
    if n>=window:
        for end in range(window,n+1):
            block=x.iloc[end-window:end]
            c=LedoitWolf().fit(block.to_numpy()).covariance_
            e=np.maximum(np.linalg.eigvalsh(c),0)[::-1]
            vals.append(float(e[:k].sum()/e.sum()) if e.sum()>0 else np.nan)
        rolling=pd.Series(vals,index=x.index[window-1:],name="absorption_ratio")
    else:
        rolling=pd.Series(dtype=float); warns.append(f"Not enough observations for rolling {window}-row absorption ratio.")
    if n<max(60,5*d): warns.append("Sample is short relative to asset count; eigenvalue/noise-bound inference is limited.")
    return AbsorptionFit(ratio,eig,k,n,d,method,rolling,warns)

def standardized_delta_ar(rolling: pd.Series, short_window: int=5, long_window: int=60) -> float:
    s=pd.Series(rolling).dropna()
    if len(s)<long_window+short_window: return float("nan")
    short=float(s.iloc[-short_window:].mean()); long=float(s.iloc[-long_window:-short_window].mean())
    hist=s.iloc[:-short_window]
    sd=float(hist.std(ddof=1)) if len(hist)>1 else 0.0
    return (short-long)/sd if sd>1e-12 else float("nan")
