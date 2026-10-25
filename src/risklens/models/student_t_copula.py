"""Empirical-marginal Student-t copula dependence model."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import kendalltau, rankdata, multivariate_t, t as student_t
from risklens.features.dependence import nearest_correlation, align_returns, kendall_to_pearson

@dataclass
class CopulaFit:
    symbols: list[str]
    correlation: np.ndarray
    degrees_freedom: float
    sample_size: int
    aligned_dropped: int
    converged: bool
    lower_tail_dependence: np.ndarray
    warnings: list[str]

def fit_student_t_copula(returns: pd.DataFrame) -> CopulaFit:
    aligned,dropped=align_returns(returns)
    n,d=aligned.shape
    warnings=[]
    if d<2 or n<30:
        warnings.append("Copula needs at least two assets and 30 aligned observations.")
        corr=np.eye(max(d,1)); return CopulaFit(list(aligned.columns),corr,8.0,n,dropped,False,np.zeros_like(corr),warnings)
    corr=nearest_correlation(kendall_to_pearson(aligned).to_numpy())
    u=np.column_stack([(rankdata(aligned.iloc[:,j].to_numpy())-0.5)/n for j in range(d)])
    z=student_t.ppf(np.clip(u,1e-6,1-1e-6),df=8)
    # Pseudo-likelihood grid/profile over nu; determinant guard from PD projection.
    def nll(nu):
        try:
            ll=multivariate_t.logpdf(z,loc=np.zeros(d),shape=corr,df=nu) - np.sum(student_t.logpdf(z,df=nu),axis=1)
            return -float(np.sum(ll[np.isfinite(ll)]))
        except Exception: return 1e100
    opt=minimize_scalar(nll,bounds=(2.1,50),method="bounded",options={"xatol":1e-2})
    nu=float(opt.x) if opt.success else 8.0
    # Asymptotic lower-tail dependence coefficient for a bivariate t-copula.
    ltd=np.zeros((d,d))
    from scipy.stats import t as tdist
    for i in range(d):
        for j in range(d):
            rho=float(np.clip(corr[i,j],-0.999,0.999))
            if i==j: ltd[i,j]=1.0
            else:
                arg=-np.sqrt((nu+1)*(1-rho)/(1+rho))
                ltd[i,j]=2*tdist.cdf(arg,df=nu+1)
    if dropped: warnings.append(f"Aligned on common timestamps; dropped {dropped} rows containing any missing asset return.")
    if n<252: warnings.append("Fewer than 252 aligned observations; tail estimates may be noisy.")
    if not opt.success: warnings.append("Degrees-of-freedom profile optimization did not converge; default nu=8 used.")
    return CopulaFit(list(aligned.columns),corr,nu,n,dropped,bool(opt.success),ltd,warnings)

def simulate_copula(fit: CopulaFit, returns: pd.DataFrame, n_samples: int, seed: int=42) -> pd.DataFrame:
    if not fit.converged: raise ValueError("Copula fit did not converge; result not simulated.")
    rng=np.random.default_rng(seed); d=len(fit.symbols)
    z=rng.multivariate_normal(np.zeros(d),fit.correlation,size=n_samples)
    chi=rng.chisquare(fit.degrees_freedom,size=n_samples)
    tdraw=z/np.sqrt(chi[:,None]/fit.degrees_freedom)
    u=student_t.cdf(tdraw,df=fit.degrees_freedom)
    aligned=returns[fit.symbols].dropna()
    out={sym:np.quantile(aligned[sym].to_numpy(),np.clip(u[:,j],1e-6,1-1e-6)) for j,sym in enumerate(fit.symbols)}
    return pd.DataFrame(out)
