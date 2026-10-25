"""Value-at-Risk methods with explicit loss sign convention."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.stats import norm, skew, kurtosis

@dataclass
class RiskMetric:
    var: float
    es: float
    confidence: float
    method: str
    sample_size: int
    unit: str = "portfolio currency units"
    warnings: list[str] | None = None

def historical_var_es(pnl: np.ndarray, confidence: float=0.95) -> RiskMetric:
    x=np.asarray(pnl,dtype=float); x=x[np.isfinite(x)]
    if not 0.5 < confidence < 1: raise ValueError("confidence must be between 0.5 and 1.")
    loss=-x
    if not len(loss): return RiskMetric(float("nan"),float("nan"),confidence,"historical",0,warnings=["No valid P&L observations."])
    var=float(np.quantile(loss,confidence,method="higher")); tail=loss[loss>=var]
    es=float(tail.mean()) if len(tail) else var
    warns=["Finite-sample ES averages observed losses at or beyond the empirical VaR threshold."]
    if len(loss)<100: warns.append("Small sample; tail estimates may be unstable.")
    return RiskMetric(var,es,confidence,"historical",len(loss),warnings=warns)

def gaussian_var_es(pnl: np.ndarray, confidence: float=0.95) -> RiskMetric:
    x=np.asarray(pnl,dtype=float); x=x[np.isfinite(x)]
    if len(x)<2: return RiskMetric(float("nan"),float("nan"),confidence,"Gaussian",len(x),warnings=["At least two P&L observations are required."])
    loss=-x; mu=float(loss.mean()); sd=float(loss.std(ddof=1)); z=float(norm.ppf(confidence))
    v=mu+sd*z; es=mu+sd*norm.pdf(z)/(1-confidence)
    return RiskMetric(v,es,confidence,"Gaussian parametric",len(x),warnings=["Assumes normally distributed portfolio P&L."])

def cornish_fisher_var_es(pnl: np.ndarray, confidence: float=0.95) -> RiskMetric:
    x=np.asarray(pnl,dtype=float); x=x[np.isfinite(x)]
    if len(x)<30: return RiskMetric(float("nan"),float("nan"),confidence,"Cornish-Fisher",len(x),warnings=["At least 30 observations are required for this approximation."])
    loss=-x; mu=float(loss.mean()); sd=float(loss.std(ddof=1)); z=float(norm.ppf(confidence)); s=float(skew(loss)); k=float(kurtosis(loss,fisher=True))
    zcf=z+(z*z-1)*s/6+(z**3-3*z)*k/24-(2*z**3-5*z)*s*s/36
    v=mu+sd*zcf
    if not np.isfinite(zcf) or abs(zcf)>8: return RiskMetric(float("nan"),float("nan"),confidence,"Cornish-Fisher",len(x),warnings=["Approximation is numerically unstable for this distribution."])
    # ES is not reliably recovered by a single CF quantile; use empirical conditional tail and label it.
    tail=loss[loss>=v]; es=float(tail.mean()) if len(tail) else v
    return RiskMetric(float(v),es,confidence,"Cornish-Fisher adjusted VaR / empirical tail ES",len(x),warnings=["Skew/kurtosis approximation can be unstable; ES uses empirical losses beyond adjusted VaR."])
