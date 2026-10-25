"""Merton jump-diffusion estimation on daily log returns.

The likelihood is a truncated Poisson mixture of normals. Estimates are physical-
measure historical estimates, not risk-neutral pricing parameters.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln, logsumexp

@dataclass
class MertonFit:
    mu: float
    sigma: float
    jump_intensity: float
    jump_mu: float
    jump_sigma: float
    log_likelihood: float
    aic: float
    sample_size: int
    converged: bool
    message: str
    jump_dates: list[str]
    threshold: float
    convention: str = "Daily log returns; compensator applied to arithmetic drift in likelihood; parameters are historical physical-measure estimates."
    def to_dict(self): return asdict(self)

def flag_jump_candidates(returns: pd.Series, threshold: float = 3.5) -> pd.Series:
    r = pd.Series(returns).dropna().astype(float)
    med = r.median(); mad = float((r-med).abs().median())
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < 1e-12: scale = float(r.std(ddof=1)) if len(r)>1 else 0.0
    if scale < 1e-12: return pd.Series(False, index=r.index)
    return ((r-med).abs()/scale >= threshold).rename("candidate_jump")

def _mixture_loglik(p: np.ndarray, r: np.ndarray, max_jumps: int) -> float:
    mu, sigma, lam, muj, sigmaj = p
    if sigma <= 0 or sigmaj <= 0 or lam < 0: return -1e100
    nobs=len(r); ks=np.arange(max_jumps+1)
    logp = -lam + ks*np.log(max(lam, 1e-12)) - gammaln(ks+1)
    # Compensated arithmetic drift: dS/S = (mu - lambda*kappa)dt + sigma dW + jumps.
    kappa=np.exp(muj + 0.5*sigmaj*sigmaj)-1
    means=(mu-lam*kappa-0.5*sigma*sigma)+ks*muj
    vars_=sigma*sigma+ks*sigmaj*sigmaj
    comps = logp[None,:] - 0.5*np.log(2*np.pi*vars_[None,:]) - (r[:,None]-means[None,:])**2/(2*vars_[None,:])
    return float(np.sum(logsumexp(comps,axis=1)))

def fit_merton(returns: pd.Series, threshold: float = 3.5, max_jumps: int = 8) -> MertonFit:
    s=pd.Series(returns).replace([np.inf,-np.inf],np.nan).dropna().astype(float)
    if len(s)<30: return MertonFit(float("nan"),float("nan"),float("nan"),float("nan"),float("nan"),float("nan"),float("nan"),len(s),False,"At least 30 return observations are required; calibration not run.",[],threshold)
    r=s.to_numpy(); sigma0=max(float(np.std(r,ddof=1)),1e-4)
    flags=flag_jump_candidates(s,threshold)
    tails=r[np.asarray(flags)]
    params0=np.array([float(np.mean(r))*252, sigma0*np.sqrt(252), max(len(tails)/len(r)*252,0.05), float(np.mean(tails) if len(tails)>1 else np.mean(r)), max(float(np.std(tails,ddof=1)) if len(tails)>2 else sigma0, 0.005)])
    bounds=[(-2,2),(0.005,5),(0.001,100),(-1,1),(0.001,1.5)]
    def objective(p):
        daily=p.copy(); daily[0] /=252; daily[1]/=np.sqrt(252); daily[2]/=252
        return -_mixture_loglik(daily,r,max_jumps)
    opt=minimize(objective,params0,bounds=bounds,method="L-BFGS-B",options={"maxiter":1200,"ftol":1e-10})
    ll=-float(opt.fun); p=opt.x
    return MertonFit(*map(float,p),ll,float(2*5-2*ll),len(s),bool(opt.success),str(opt.message),[str(x) for x in s.index[flags]],threshold)

def simulate_merton(fit: MertonFit, paths: int, horizon: int, seed: int=42) -> np.ndarray:
    if not fit.converged: raise ValueError("Merton calibration did not converge; choose a transparent alternative.")
    rng=np.random.default_rng(seed); dt=1/252
    mu=fit.mu; sigma=fit.sigma; lam=fit.jump_intensity; muj=fit.jump_mu; sj=fit.jump_sigma
    k=np.exp(muj+0.5*sj*sj)-1
    z=rng.normal(size=(paths,horizon)); counts=rng.poisson(lam*dt,size=(paths,horizon))
    jump_sum=rng.normal(counts*muj,np.sqrt(np.maximum(counts,0))*sj)
    return (mu-lam*k-0.5*sigma*sigma)*dt + sigma*np.sqrt(dt)*z + jump_sum
