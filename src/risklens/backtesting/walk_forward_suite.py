"""Walk-forward VaR diagnostics and transparent model ablation suite."""
from __future__ import annotations
import numpy as np, pandas as pd
from scipy.stats import chi2, norm, t as student_t
class WalkForwardBacktester:
 def __init__(self,confidence_level=.95,window_size=252):
  if not 0<confidence_level<1 or window_size<30: raise ValueError('Invalid confidence_level/window_size')
  self.q=float(confidence_level); self.alpha=1-self.q; self.window_size=int(window_size)
 @staticmethod
 def kupiec_pof_test(exceedances,alpha):
  x=np.asarray(exceedances,dtype=bool); N=len(x)
  if N==0: raise ValueError('At least one observation is required.')
  n=int(x.sum()); ph=np.clip(n/N,1e-12,1-1e-12); a=np.clip(alpha,1e-12,1-1e-12)
  ll0=(N-n)*np.log(1-a)+n*np.log(a); ll1=(N-n)*np.log(1-ph)+n*np.log(ph); lr=max(0.,-2*(ll0-ll1))
  return {'exceptions':n,'total_obs':N,'exception_rate':n/N,'lr_pof':float(lr),'p_value':float(chi2.sf(lr,1))}
 @staticmethod
 def christoffersen_test(exceedances,alpha):
  x=np.asarray(exceedances,dtype=int)
  if len(x)<3: raise ValueError('At least three exceedance indicators required.')
  n00=int(((x[:-1]==0)&(x[1:]==0)).sum()); n01=int(((x[:-1]==0)&(x[1:]==1)).sum()); n10=int(((x[:-1]==1)&(x[1:]==0)).sum()); n11=int(((x[:-1]==1)&(x[1:]==1)).sum())
  def ll(a,b): return (a*np.log(max(b,1e-12)) if a else 0)
  p0=n01/max(n00+n01,1); p1=n11/max(n10+n11,1); p=(n01+n11)/max(n00+n01+n10+n11,1)
  ll_null=ll(n00+n10,1-p)+ll(n01+n11,p); ll_alt=ll(n00,1-p0)+ll(n01,p0)+ll(n10,1-p1)+ll(n11,p1); lrind=max(0.,-2*(ll_null-ll_alt)); pof=WalkForwardBacktester.kupiec_pof_test(x,alpha); lrcc=pof['lr_pof']+lrind
  return {'lr_ind':lrind,'p_value_ind':float(chi2.sf(lrind,1)),'lr_cc':lrcc,'p_value_cc':float(chi2.sf(lrcc,2))}
 @staticmethod
 def acerbi_szekely_es_score(losses,var_forecasts,es_forecasts,alpha):
  l=np.asarray(losses,float); v=np.asarray(var_forecasts,float); e=np.asarray(es_forecasts,float)
  if not (len(l)==len(v)==len(e)) or len(l)==0 or not 0<alpha<1: raise ValueError('Inputs must have equal nonzero length and valid alpha.')
  if np.any(e<=0): raise ValueError('ES forecasts must be positive losses for this score convention.')
  # Diagnostic score only; formal inference requires an appropriate bootstrap/reference distribution.
  return float(1-np.mean((l*(l>v))/e)/alpha)
 def run_ablation_suite(self,portfolio_returns:pd.Series)->pd.DataFrame:
  r=pd.Series(portfolio_returns).replace([np.inf,-np.inf],np.nan).dropna().to_numpy(); W=self.window_size
  if len(r)<=W+30: raise ValueError('Series too short for rolling walk-forward backtest.')
  names=['Historical','Gaussian','EWMA volatility proxy','Jump-adjusted volatility proxy','RiskLens heavy-tail proxy']; vp={k:[] for k in names}; ep={k:[] for k in names}; realized=-r[W:]; varz=norm.ppf(self.q); esz=norm.pdf(varz)/self.alpha; ewvar=np.var(r[:W]);
  for t in range(W,len(r)):
   train=r[t-W:t]; mu=float(train.mean()); sig=max(float(train.std(ddof=1)),1e-8); losses=-train; vh=float(np.quantile(losses,self.q)); tail=losses[losses>=vh]; eh=float(tail.mean()) if len(tail) else vh
   ewvar=.94*ewvar+.06*r[t-1]**2; ew=np.sqrt(max(ewvar,1e-12)); jump=max(sig,ew)*1.15; tq=float(student_t.ppf(self.q,df=5)*np.sqrt(3/5))
   preds=[(vh,eh),(-mu+varz*sig,-mu+esz*sig),(-mu+varz*ew,-mu+esz*ew),(-mu+varz*jump,-mu+1.3*varz*jump),(-mu+tq*jump,-mu+1.4*tq*jump)]
   for name,(v,e) in zip(names,preds): vp[name].append(max(float(v),1e-10)); ep[name].append(max(float(e),1e-10))
  rows=[]
  for name in names:
   v=np.array(vp[name]); e=np.array(ep[name]); ex=realized>v; p=self.kupiec_pof_test(ex,self.alpha); cc=self.christoffersen_test(ex,self.alpha); rows.append({'Model':name,'Target_Exceed_Rate':self.alpha,'Actual_Exceed_Rate':p['exception_rate'],'Exceptions':p['exceptions'],'Kupiec_LR_POF':p['lr_pof'],'Kupiec_p_value':p['p_value'],'Christoffersen_LR_CC':cc['lr_cc'],'Christoffersen_p_value':cc['p_value_cc'],'Acerbi_Szekely_ES_Z2_diagnostic':self.acerbi_szekely_es_score(realized,v,e,self.alpha),'Status':'NOT_REJECTED' if p['p_value']>=.05 and cc['p_value_cc']>=.05 else 'REJECTED','Model_scope':'Proxy, not calibrated structural model' if 'proxy' in name.lower() or 'jump' in name.lower() else 'Historical/Gaussian baseline'})
  return pd.DataFrame(rows)
