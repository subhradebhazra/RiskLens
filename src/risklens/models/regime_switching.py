"""Four-state volatility regime classifier with an explicitly labeled fallback."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

REGIME_LABELS=["Calm / Low Volatility","Normal / Transitioning","Elevated Volatility","Crisis / Stress"]
@dataclass
class RegimeFit:
    probabilities: pd.DataFrame
    transition_matrix: np.ndarray
    expected_durations: np.ndarray
    method: str
    converged: bool
    warnings: list[str]

def fit_regimes(returns: pd.Series) -> RegimeFit:
    r=pd.Series(returns).replace([np.inf,-np.inf],np.nan).dropna().astype(float)
    if len(r)<40:
        probs=pd.DataFrame(np.tile([1.,0,0,0],(len(r),1)),index=r.index,columns=REGIME_LABELS)
        return RegimeFit(probs,np.eye(4),np.full(4,np.inf),"insufficient-data placeholder",False,["At least 40 observations are required; probabilities are not a fitted model."])
    feat=np.column_stack([r.to_numpy(),np.abs(r.to_numpy())])
    warnings=[]; method="Gaussian-mixture proxy"; converged=True
    try:
        g=GaussianMixture(n_components=4,random_state=42,n_init=3,reg_covar=1e-6,max_iter=500).fit(feat)
        raw=g.predict_proba(feat)
        vol=np.array([np.average(feat[:,1],weights=raw[:,i]) for i in range(4)])
        draw=np.array([np.average(np.minimum(feat[:,0],0)**2,weights=raw[:,i]) for i in range(4)])
        order=np.lexsort((draw,vol)); probs=raw[:,order]
        warnings.append("Fallback Gaussian-mixture classifier; not a Hamilton Markov-switching fit.")
        method="Gaussian-mixture volatility proxy"
    except Exception as exc:
        # deterministic volatility-quartile fallback
        rank=pd.Series(np.abs(r)).rank(pct=True).to_numpy()
        bins=np.minimum((rank*4).astype(int),3)
        probs=np.eye(4)[bins]; converged=False; warnings.append(f"Mixture optimization failed; quartile fallback used ({type(exc).__name__}).")
    labels=REGIME_LABELS
    pframe=pd.DataFrame(probs,index=r.index,columns=labels)
    states=np.argmax(probs,axis=1); trans=np.ones((4,4))*0.5
    for a,b in zip(states[:-1],states[1:]): trans[a,b]+=1
    trans=trans/trans.sum(axis=1,keepdims=True)
    dur=np.where(1-np.diag(trans)>1e-12,1/(1-np.diag(trans)),np.inf)
    if len(r)<252: warnings.append("Fewer than 252 observations: state estimates may be unstable.")
    return RegimeFit(pframe,trans,dur,method,converged,warnings)
