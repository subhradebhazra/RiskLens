"""Heston stochastic-volatility proxy estimation and full-truncation simulation.

Calibration is explicitly a method-of-moments proxy, not full likelihood estimation.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

@dataclass
class HestonFit:
    kappa: float
    theta: float
    xi: float
    rho: float
    v0: float
    sample_size: int
    converged: bool
    method: str
    feller_satisfied: bool
    warnings: list[str]
    def to_dict(self): return asdict(self)

def fit_heston(returns: pd.Series) -> HestonFit:
    r=pd.Series(returns).replace([np.inf,-np.inf],np.nan).dropna().astype(float)
    warns=[]
    if len(r)<60: warns.append("Short sample: volatility dynamics are weakly identified.")
    if len(r)<2:
        return HestonFit(1.0,0.0001,0.2,0.0,0.0001,len(r),False,"unavailable",False,warns+["At least two returns are required."])
    sq=r.pow(2); theta=float(np.clip(sq.mean()*252,1e-6,4.0)); v0=float(np.clip(sq.iloc[-1]*252,1e-6,4.0))
    lag=sq.shift(1).dropna(); next_=sq.loc[lag.index]
    corr=float(spearmanr(lag,next_).statistic) if len(lag)>4 else 0.0
    if not np.isfinite(corr): corr=0.0
    # Autocorrelation proxy for mean-reversion speed, capped rather than pretending to be QMLE.
    kappa=float(np.clip(-np.log(np.clip(corr,0.02,0.98))*252,0.1,10.0))
    xi=float(np.clip(np.std(sq.diff().dropna())*252/np.sqrt(max(theta,1e-12)),0.05,3.0))
    lag_r=r.shift(1).dropna(); r_aligned=r.loc[lag_r.index]
    rho=float(np.clip(np.corrcoef(lag_r, r_aligned)[0,1] if len(r_aligned)>4 else 0.0,-0.95,0.95))
    if len(r)<252: warns.append("Fewer than 252 observations: calibration stability is limited.")
    warns.append("Method-of-moments proxy; not full maximum likelihood. Parameter uncertainty is not estimated.")
    return HestonFit(kappa,theta,xi,rho,v0,len(r),True,"method-of-moments volatility proxy",2*kappa*theta > xi*xi,warns)

def simulate_heston(fit: HestonFit, paths: int, horizon: int, seed: int=42, initial_price: float=1.0) -> np.ndarray:
    if not fit.converged: raise ValueError("Heston parameters unavailable.")
    rng=np.random.default_rng(seed); dt=1/252
    prices=np.full(paths,initial_price); var=np.full(paths,max(fit.v0,1e-10)); out=np.empty((paths,horizon))
    rho=np.clip(fit.rho,-0.999,0.999)
    for t in range(horizon):
        z1=rng.normal(size=paths); z2=rho*z1+np.sqrt(1-rho*rho)*rng.normal(size=paths)
        vp=np.maximum(var,0)
        var_next=np.maximum(var+fit.kappa*(fit.theta-vp)*dt+fit.xi*np.sqrt(vp*dt)*z2,0)
        lr=-0.5*vp*dt+np.sqrt(vp*dt)*z1
        prices=prices*np.exp(lr); out[:,t]=lr; var=var_next
    return out
