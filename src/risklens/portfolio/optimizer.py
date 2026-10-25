"""Long-only portfolio optimization using documented covariance scenarios."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import minimize

@dataclass
class OptimizationResult:
    weights: pd.Series
    method: str
    success: bool
    message: str
    annualized_volatility: float
    expected_return: float
    turnover: float
    warnings: list[str]

def optimize_portfolio(returns: pd.DataFrame, current_weights: pd.Series, method: str="minimum_variance", bounds: tuple[float,float]=(0.0,1.0), max_turnover: float|None=None, risk_free_annual: float=0.0) -> OptimizationResult:
    x=returns.dropna(how="any"); cols=[c for c in current_weights.index if c in x.columns]
    if len(cols)<2 or len(x)<20: return OptimizationResult(current_weights.copy(),method,False,"Need two or more assets and at least 20 aligned returns; current portfolio retained.",float("nan"),float("nan"),0.0,["Insufficient data."])
    w0=current_weights[cols].to_numpy(dtype=float); w0=w0/w0.sum()
    mu=x[cols].mean().to_numpy()*252; cov=x[cols].cov().to_numpy()*252
    lo,hi=bounds
    if lo<0 or hi>1 or lo>hi: raise ValueError("Bounds must satisfy 0 <= lower <= upper <= 1.")
    def vol(w): return float(np.sqrt(max(w@cov@w,0)))
    def objective(w):
        if method=="max_sharpe":
            v=vol(w); return -((w@mu-risk_free_annual)/v) if v>1e-12 else 1e6
        if method=="target_return": return float(w@cov@w)
        return float(w@cov@w)
    cons=[{"type":"eq","fun":lambda w: float(w.sum()-1)}]
    if method=="target_return": cons.append({"type":"ineq","fun":lambda w:float(w@mu-mu.mean())})
    if max_turnover is not None: cons.append({"type":"ineq","fun":lambda w:float(max_turnover-np.sum(np.abs(w-w0))/2)})
    opt=minimize(objective,w0,method="SLSQP",bounds=[(lo,hi)]*len(cols),constraints=cons,options={"maxiter":1000,"ftol":1e-10})
    if not opt.success or not np.isfinite(opt.x).all() or abs(opt.x.sum()-1)>1e-4:
        return OptimizationResult(pd.Series(w0,index=cols),method,False,str(opt.message)+"; current weights retained.",vol(w0),float(w0@mu),0.0,["Optimization failed or constraints infeasible; no suggested weights emitted."])
    w=np.clip(opt.x,lo,hi); w=w/w.sum(); turnover=float(np.sum(np.abs(w-w0))/2)
    return OptimizationResult(pd.Series(w,index=cols),method,True,str(opt.message),vol(w),float(w@mu),turnover,[])
