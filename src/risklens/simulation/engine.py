"""Reproducible Monte Carlo engine; unsupported integrations fall back visibly."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from risklens.models.merton import fit_merton, simulate_merton
from risklens.models.heston import fit_heston, simulate_heston

@dataclass
class SimulationResult:
    paths: np.ndarray
    mode_requested: str
    mode_used: str
    seed: int
    horizon_days: int
    path_count: int
    converged: bool
    warnings: list[str]
    assumptions: list[str]

def simulate_portfolio(returns: pd.DataFrame, weights: pd.Series, path_count: int=10_000, horizon_days: int=5, mode: str="Gaussian baseline", seed: int=42, batch_size: int=2_000) -> SimulationResult:
    if not 1000<=path_count<=50000: raise ValueError("path_count must be between 1,000 and 50,000.")
    if not 1<=horizon_days<=30: raise ValueError("horizon_days must be between 1 and 30.")
    x=returns.dropna(how="any"); cols=[c for c in weights.index if c in x.columns]
    if len(cols)<1 or len(x)<20: raise ValueError("Need at least one asset and 20 aligned returns.")
    w=weights[cols].astype(float).to_numpy(); w=w/w.sum()
    r=x[cols].to_numpy(dtype=float); warnings=[]; assumptions=[]
    rng=np.random.default_rng(seed); out=np.empty((path_count,horizon_days),dtype=float)
    mu=np.mean(r,axis=0); cov=np.cov(r,rowvar=False) if len(cols)>1 else np.array([[np.var(r[:,0],ddof=1)]])
    cov=np.atleast_2d(cov); cov=(cov+cov.T)/2
    try:
        ev,vec=np.linalg.eigh(cov); cov=(vec@np.diag(np.maximum(ev,1e-12))@vec.T)
    except Exception: cov=np.diag(np.maximum(np.var(r,axis=0,ddof=1),1e-12))
    used=mode
    if mode in {"Merton-only","Heston-only"} and len(cols)!=1:
        warnings.append(f"{mode} is currently single-asset only; switched to Gaussian baseline for {len(cols)} assets.")
        used="Gaussian baseline"
    if mode.startswith("Integrated"):
        warnings.append("Integrated Bates/Hawkes/copula calibration is not represented as complete here; Gaussian baseline selected explicitly.")
        used="Gaussian baseline"
    for start in range(0,path_count,batch_size):
        stop=min(start+batch_size,path_count); n=stop-start
        if used=="Historical bootstrap":
            idx=rng.integers(0,len(r),size=(n,horizon_days)); sim=r[idx,:]
            port=sim@w
        elif used=="Merton-only":
            fit=fit_merton(pd.Series(r[:,0]))
            if fit.converged:
                sim=np.expm1(simulate_merton(fit,n,horizon_days,seed+start))
            else:
                warnings.append("Merton fit failed; Gaussian baseline used."); used="Gaussian baseline"
                sim=rng.multivariate_normal(mu,cov,size=(n,horizon_days))
            port=np.asarray(sim).reshape(n,horizon_days)
        elif used=="Heston-only":
            fit=fit_heston(pd.Series(r[:,0]))
            sim=np.expm1(simulate_heston(fit,n,horizon_days,seed+start)) if fit.converged else rng.normal(mu[0],np.sqrt(cov[0,0]),size=(n,horizon_days))
            if not fit.converged: warnings.append("Heston proxy unavailable; Gaussian baseline used.")
            port=np.asarray(sim).reshape(n,horizon_days)
        else:
            sim=rng.multivariate_normal(mu,cov,size=(n,horizon_days))
            # Simulated simple asset returns are converted to portfolio one-period returns.
            port=np.einsum("pth,t->ph",sim,np.zeros(len(cols))) if False else np.tensordot(sim,w,axes=([2],[0]))
        out[start:stop,:]=port
    assumptions += [f"{len(cols)} aligned assets; weights applied as supplied then normalized to sum to one.","Returns modeled per trading day; terminal wealth compounds simple portfolio returns.","Covariance is historical sample covariance; this engine does not jointly calibrate all advanced components."]
    if used=="Gaussian baseline": assumptions.append("Multivariate Gaussian daily returns; no explicit jumps or stochastic volatility.")
    terminal=(1+out).prod(axis=1)-1
    if np.any(terminal < -1): warnings.append("Some simulated compound paths crossed the -100% simple-return floor; inspect model assumptions.")
    return SimulationResult(out,mode,used,seed,horizon_days,path_count,not warnings, warnings,assumptions)

def terminal_pnl(result: SimulationResult, portfolio_value: float) -> np.ndarray:
    return portfolio_value*((1+result.paths).prod(axis=1)-1)
