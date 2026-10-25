"""Exponential Hawkes process likelihood and Ogata thinning simulation."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize

@dataclass
class HawkesFit:
    mu: float
    alpha: float
    beta: float
    branching_ratio: float
    log_likelihood: float
    sample_size: int
    horizon: float
    converged: bool
    warnings: list[str]

def fit_hawkes(event_times: np.ndarray, horizon: float | None = None) -> HawkesFit:
    t=np.sort(np.asarray(event_times,dtype=float)); t=t[np.isfinite(t)]
    if horizon is None: horizon=float(t[-1]) if len(t) else 0.0
    horizon=max(float(horizon),1e-6); t=t[(t>=0)&(t<=horizon)]
    if len(t)<5:
        return HawkesFit(float("nan"),float("nan"),float("nan"),float("nan"),float("nan"),len(t),horizon,False,["At least five events are needed for Hawkes calibration."])
    def negll(x):
        mu,alpha,beta=x
        if mu<=0 or beta<=0 or alpha<0 or alpha>=0.98*beta: return 1e100
        intens=[]
        for i,ti in enumerate(t):
            exc=np.sum(alpha*np.exp(-beta*(ti-t[:i]))) if i else 0.0
            intens.append(mu+exc)
        integral=mu*horizon + (alpha/beta)*np.sum(1-np.exp(-beta*(horizon-t)))
        return float(integral-np.sum(np.log(np.maximum(intens,1e-12))))
    rate=len(t)/horizon
    opt=minimize(negll,[max(rate*0.7,1e-5),max(rate*0.1,1e-5),max(1/horizon,0.1)],method="Nelder-Mead",options={"maxiter":2000,"xatol":1e-8})
    p=opt.x
    if p[0]<=0 or p[1]<0 or p[2]<=0 or p[1]>=p[2]:
        return HawkesFit(float(p[0]),float(p[1]),float(p[2]),float(p[1]/p[2] if p[2] else np.inf),-float(opt.fun),len(t),horizon,False,["Optimizer returned non-stationary parameters."])
    return HawkesFit(float(p[0]),float(p[1]),float(p[2]),float(p[1]/p[2]),-float(opt.fun),len(t),horizon,bool(opt.success),([] if opt.success else ["Likelihood optimizer did not report convergence."]))

def simulate_ogata(fit: HawkesFit, horizon: float, seed: int=42, max_events: int=10000) -> np.ndarray:
    if not fit.converged or fit.branching_ratio>=1: raise ValueError("Requires a converged stationary Hawkes fit.")
    rng=np.random.default_rng(seed); t=0.; events=[]; intensity=fit.mu
    while len(events)<max_events:
        bound=fit.mu + (fit.alpha*np.sum(np.exp(-fit.beta*(t-np.asarray(events)))) if events else 0.0)
        if bound<=0: break
        t += rng.exponential(1/bound)
        if t>=horizon: break
        intensity=fit.mu + (fit.alpha*np.sum(np.exp(-fit.beta*(t-np.asarray(events)))) if events else 0.0)
        if rng.random() <= min(intensity/bound,1): events.append(t)
    return np.asarray(events)

def event_times_from_returns(returns, threshold: float=2.5):
    import pandas as pd
    r=pd.Series(returns).dropna(); scale=float(r.std(ddof=1)) if len(r)>1 else 0.0
    flags=(r < -threshold*scale) if scale>0 else pd.Series(False,index=r.index)
    return np.flatnonzero(flags.to_numpy()).astype(float)
