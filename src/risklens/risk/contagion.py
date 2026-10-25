"""Transparent 0-100 contagion composite with explicit missing-feature weights."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

DEFAULT_WEIGHTS={"tail_dependence":0.25,"absorption_ratio":0.25,"hawkes_branching":0.20,"downside_comovement":0.15,"crisis_probability":0.15}
@dataclass
class ContagionScore:
    score: float
    components: dict[str,float|None]
    effective_weights: dict[str,float]
    confidence: str
    warnings: list[str]

def _pct(x, lo, hi):
    return float(np.clip((x-lo)/(hi-lo)*100,0,100))

def calculate_contagion_score(tail_dependence=None, absorption_ratio=None, hawkes_branching=None, downside_comovement=None, crisis_probability=None) -> ContagionScore:
    raw={"tail_dependence":tail_dependence,"absorption_ratio":absorption_ratio,"hawkes_branching":hawkes_branching,"downside_comovement":downside_comovement,"crisis_probability":crisis_probability}
    transform={"tail_dependence":lambda x:_pct(x,0,1),"absorption_ratio":lambda x:_pct(x,0.2,1),"hawkes_branching":lambda x:_pct(x,0,1),"downside_comovement":lambda x:_pct(x,-1,1),"crisis_probability":lambda x:_pct(x,0,1)}
    components={k:(transform[k](float(v)) if v is not None and np.isfinite(v) else None) for k,v in raw.items()}
    w={k:DEFAULT_WEIGHTS[k] for k,v in components.items() if v is not None}
    if not w: return ContagionScore(float("nan"),components,{},"UNAVAILABLE",["All score components are unavailable; no reassuring zero was imputed."])
    total=sum(w.values()); w={k:v/total for k,v in w.items()}
    score=float(sum(components[k]*w[k] for k in w))
    warns=[]
    if len(w)<5: warns.append(f"{5-len(w)} components unavailable; remaining weights renormalized.")
    return ContagionScore(score,components,w,"LOW" if len(w)<4 else "MODERATE" if len(w)<5 else "FULL",warns)

def market_weather(regime_prob: float|None, realized_vol: float|None, drawdown: float|None, absorption_ratio: float|None=None) -> tuple[str,str]:
    """Rule-based observation classifier; not a deterministic forecast."""
    vals=[v for v in [regime_prob,realized_vol,drawdown,absorption_ratio] if v is not None and np.isfinite(v)]
    if not vals: return "Unavailable","Insufficient current telemetry to classify market conditions."
    score=0
    if regime_prob is not None: score+=2*float(regime_prob)
    if realized_vol is not None: score+=min(float(realized_vol)/0.5,1.5)
    if drawdown is not None: score+=min(abs(float(drawdown))/0.25,1.5)
    if absorption_ratio is not None: score+=min(max(float(absorption_ratio)-0.3,0)/0.5,1)
    if score>=3.3: label="Stress"
    elif score>=2.0: label="Elevated Risk"
    elif score>=1.0: label="Normal"
    else: label="Calm"
    return label,"Rule-based classification from available observations; thresholds are configurable heuristics, not forecasts."
