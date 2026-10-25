"""Configurable in-app telemetry alerts. Webhooks are optional and opt-in."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
import math
from typing import Any
@dataclass
class AlertRule:
 rule_id:str; metric_key:str; label:str; threshold:float; comparison:str='>'; severity:str='WARNING'; enabled:bool=True
 def __post_init__(self):
  if self.comparison not in ('>','<','>=','<='): raise ValueError('comparison must be >, <, >=, or <=')
  if not math.isfinite(float(self.threshold)): raise ValueError('threshold must be finite')
class ConfigurableAlertEngine:
 DEFAULT_RULES=[AlertRule('vix_us','cboe_vix','CBOE VIX',25,'>','WARNING'),AlertRule('vix_in','india_vix','India VIX',22,'>','WARNING'),AlertRule('pca_ar','pca_delta_ar','PCA absorption ratio shift',1.5,'>','CRITICAL'),AlertRule('contagion','contagion_score','Contagion score',70,'>','CRITICAL'),AlertRule('regime','crisis_regime_prob','Crisis-regime probability',.5,'>','CRITICAL'),AlertRule('var_limit','portfolio_var_99_pct','Portfolio 99% VaR',.045,'>','WARNING'),AlertRule('drawdown','current_drawdown_pct','Drawdown',.12,'>','CRITICAL')]
 def __init__(self,rules=None): self.rules=list(rules) if rules is not None else [AlertRule(**vars(x)) for x in self.DEFAULT_RULES]
 def evaluate_telemetry(self,telemetry:dict[str,Any])->list[dict[str,Any]]:
  out=[]; now=datetime.now(timezone.utc).isoformat()
  for r in self.rules:
   if not r.enabled or r.metric_key not in telemetry or telemetry[r.metric_key] is None: continue
   try: v=float(telemetry[r.metric_key])
   except (TypeError,ValueError): continue
   if not math.isfinite(v): continue
   hit={'>':v>r.threshold,'<':v<r.threshold,'>=':v>=r.threshold,'<=':v<=r.threshold}[r.comparison]
   if hit: out.append({'timestamp_utc':now,'rule_id':r.rule_id,'label':r.label,'observed_value':v,'threshold':r.threshold,'severity':r.severity,'source':telemetry.get(r.metric_key+'_source','RiskLens_Engine')})
  return out
