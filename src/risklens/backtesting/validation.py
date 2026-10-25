"""Coverage and expected-shortfall calibration diagnostics.

The Acerbi–Szekely-style moment diagnostic below is a finite-sample calibration
check: for correctly specified tail forecasts, E[L * 1(L >= VaR) / ES] is
approximately 1-confidence. It is not evidence of model correctness and can have
low power with few exceedances.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import chi2, norm

def kupiec_pof(breaches, confidence: float=0.95) -> dict:
    b=np.asarray(breaches,dtype=bool); n=len(b); x=int(b.sum()); p0=1-confidence
    if n==0: return {"lr_stat":float("nan"),"p_value":float("nan"),"breaches":0,"sample_size":0,"note":"No forecast observations."}
    phat=np.clip(x/n,1e-12,1-1e-12); p0=np.clip(p0,1e-12,1-1e-12)
    ll0=x*np.log(p0)+(n-x)*np.log(1-p0); ll1=x*np.log(phat)+(n-x)*np.log(1-phat)
    lr=float(-2*(ll0-ll1)); p=float(chi2.sf(lr,df=1))
    return {"lr_stat":lr,"p_value":p,"breaches":x,"sample_size":n,"expected_breaches":n*p0,"note":"Non-rejection is not proof the model is correct; p-values have limited power in short samples."}

def christoffersen_independence(breaches) -> dict:
    b=np.asarray(breaches,dtype=bool)
    if len(b)<2: return {"lr_stat":float("nan"),"p_value":float("nan"),"note":"At least two observations required."}
    n00=n01=n10=n11=0
    for a,c in zip(b[:-1],b[1:]):
        if not a and not c:n00+=1
        elif not a and c:n01+=1
        elif a and not c:n10+=1
        else:n11+=1
    p01=n01/max(n00+n01,1); p11=n11/max(n10+n11,1); p=(n01+n11)/max(n00+n01+n10+n11,1)
    def ll(a,b,p): return a*np.log(max(p,1e-12))+b*np.log(max(1-p,1e-12))
    ll_uncond=ll(n01+n11,n00+n10,p); ll_cond=ll(n01,n00,p01)+ll(n11,n10,p11)
    lr=float(max(0,-2*(ll_uncond-ll_cond)))
    return {"lr_stat":lr,"p_value":float(chi2.sf(lr,1)),"transitions":{"n00":n00,"n01":n01,"n10":n10,"n11":n11},"note":"Independence test; small breach counts weaken inference."}

def acerbi_szekely_es_test(actual_losses, var_forecasts, es_forecasts, confidence: float=0.95) -> dict:
    losses=np.asarray(actual_losses,dtype=float); var=np.asarray(var_forecasts,dtype=float); es=np.asarray(es_forecasts,dtype=float)
    n=min(len(losses),len(var),len(es)); losses,var,es=losses[:n],var[:n],es[:n]
    valid=np.isfinite(losses)&np.isfinite(var)&np.isfinite(es)&(es>0)
    losses,var,es=losses[valid],var[valid],es[valid]; n=len(losses)
    if n<20: return {"statistic":float("nan"),"z_score":float("nan"),"p_value":float("nan"),"sample_size":n,"note":"At least 20 valid ES forecasts are required; insufficient sample."}
    tail=(losses>=var).astype(float)
    terms=losses*tail/es-(1-confidence)
    mean=float(np.mean(terms)); sd=float(np.std(terms,ddof=1))
    z=mean/(sd/np.sqrt(n)) if sd>1e-12 else float("nan")
    p=float(2*norm.sf(abs(z))) if np.isfinite(z) else float("nan")
    return {"statistic":mean,"z_score":float(z),"p_value":p,"sample_size":n,"exceedances":int(tail.sum()),"note":"Acerbi–Szekely-style ES calibration moment diagnostic; finite-sample approximation, not a guarantee. Interpret cautiously with few exceedances, overlapping horizons or model-selection effects."}
