"""Historical-window replay when observed prices exist; otherwise labeled scenario proxy."""
from __future__ import annotations
import numpy as np, pandas as pd

CRISIS_TEMPLATES={
 '2008_GFC_Lehman_Collapse':{'window':('2008-09-01','2008-11-30'),'description':'Global Financial Crisis, Sep-Nov 2008','shock':-.38,'vol_multiplier':3.4,'sector_tilts':{'Financials':1.35,'Technology':1.05,'Consumer':.8,'Energy':1.15}},
 '2020_COVID_Liquidity_Crash':{'window':('2020-02-19','2020-03-23'),'description':'COVID-19 sell-off, Feb-Mar 2020','shock':-.34,'vol_multiplier':4.1,'sector_tilts':{'Financials':1.2,'Energy':1.45,'Technology':.78,'Healthcare':.65}},
 '2022_Global_Rate_Tightening':{'window':('2022-01-03','2022-06-17'),'description':'Global rate-tightening period, H1 2022','shock':-.23,'vol_multiplier':1.85,'sector_tilts':{'Technology':1.4,'Financials':.95,'Energy':.4}}}
class StressTestingEngine:
 @staticmethod
 def run_historical_crisis_replay(portfolio_df:pd.DataFrame,historical_prices:pd.DataFrame|None,crisis_key:str,severity_multiplier:float=1.):
  if crisis_key not in CRISIS_TEMPLATES: raise ValueError(f'Unknown crisis_key. Choose from {list(CRISIS_TEMPLATES)}')
  if not np.isfinite(severity_multiplier) or severity_multiplier<0: raise ValueError('severity_multiplier must be finite and nonnegative.')
  t=CRISIS_TEMPLATES[crisis_key]; start,end=t['window']; rows=[]; exact=False
  sub=None
  if historical_prices is not None and not historical_prices.empty:
   hp=historical_prices.copy(); hp.index=pd.to_datetime(hp.index); sub=hp.loc[start:end]; exact=len(sub)>=2
  for _,r in portfolio_df.iterrows():
   ticker=str(r['ticker']); mv=float(r['market_value']); sector=str(r.get('sector','General'))
   if not np.isfinite(mv) or mv<0: raise ValueError('Market values must be finite and nonnegative.')
   if exact and ticker in sub.columns and sub[ticker].dropna().shape[0]>=2:
    s=sub[ticker].dropna(); shock=float(s.iloc[-1]/s.iloc[0]-1)
   else: shock=float(t['shock']*t['sector_tilts'].get(sector,1.))
   shock=float(np.clip(shock*severity_multiplier,-.95,2.)); pnl=mv*shock
   rows.append({'ticker':ticker,'sector':sector,'initial_value':mv,'shock_return':shock,'pnl_currency':pnl,'post_shock_value':mv+pnl})
  out=pd.DataFrame(rows); total=float(out.initial_value.sum()); pnl=float(out.pnl_currency.sum()); post=float(out.post_shock_value.sum())
  out['loss_contribution_pct']=out.pnl_currency/pnl if abs(pnl)>1e-12 else 0.; out['post_shock_weight']=out.post_shock_value/post if post>0 else 0.
  return {'crisis_key':crisis_key,'description':t['description'],'replay_mode':'Exact_Historical_Window' if exact else 'Calibrated_Sector_Beta_Replay (hypothetical proxy)','total_initial_value':total,'total_pnl_currency':pnl,'percentage_return':pnl/total if total else 0.,'post_shock_hhi':float(np.square(out.post_shock_weight).sum()),'asset_breakdown':out.to_dict('records'),'sector_breakdown':out.groupby('sector')[['initial_value','pnl_currency','post_shock_value']].sum().reset_index().to_dict('records'),'vol_multiplier_context':t['vol_multiplier']}
