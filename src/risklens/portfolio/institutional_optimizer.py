"""Long-only optimizer with CVaR linear programming, HRP, turnover and costs."""
from __future__ import annotations
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf

class InstitutionalPortfolioOptimizer:
    def __init__(self,returns_df:pd.DataFrame,risk_free_rate=.05):
        self.returns_df=returns_df.replace([np.inf,-np.inf],np.nan).dropna(how='any'); self.tickers=list(self.returns_df.columns); self.n_assets=len(self.tickers); self.rf=float(risk_free_rate)
        if self.n_assets<2 or len(self.returns_df)<20: raise ValueError('At least two assets and 20 complete return observations are required.')
        self.cov_annual=LedoitWolf().fit(self.returns_df.to_numpy()).covariance_*252; self.mean_annual=self.returns_df.mean().to_numpy()*252
    def optimize(self,current_weights,method='cvar_es',min_weight=0.,max_weight=.35,sector_map=None,sector_caps=None,max_turnover=.5,tc_bps=15.,confidence_level=.95,crisis_regime_prob=.1):
        w0=np.asarray(current_weights,dtype=float)
        if len(w0)!=self.n_assets or not np.isfinite(w0).all() or w0.sum()<=0: raise ValueError('Current weights must be finite, positive-sum, and match assets.')
        w0=w0/w0.sum(); lo,hi=float(min_weight),float(max_weight)
        if lo<0 or hi>1 or lo>hi or self.n_assets*lo>1 or self.n_assets*hi<1: raise ValueError('Infeasible position bounds.')
        if not 0<confidence_level<1 or max_turnover<0 or tc_bps<0: raise ValueError('Invalid confidence, turnover, or transaction cost.')
        if method=='hrp': target=self._hrp()
        elif method=='regime_kelly':
            k=np.maximum(np.linalg.pinv(self.cov_annual)@(self.mean_annual-self.rf),0); k=k/k.sum() if k.sum()>0 else np.repeat(1/self.n_assets,self.n_assets); target=(1-crisis_regime_prob)*k+crisis_regime_prob*self._hrp()
        else: target=self._solve(w0,method,lo,hi,sector_map,sector_caps,max_turnover,tc_bps/10000,confidence_level)
        target=np.asarray(target); target=np.clip(target,lo,hi)
        # Only accept feasible normalized results; otherwise retain original portfolio.
        if not np.isfinite(target).all() or target.sum()<=0: target=w0.copy()
        else: target=target/target.sum()
        turn=float(np.abs(target-w0).sum()/2)
        if turn>max_turnover+1e-6 or np.any(target<lo-1e-6) or np.any(target>hi+1e-6): target=w0.copy(); turn=0.
        return {'method':method,'tickers':self.tickers,'current_weights':w0.tolist(),'optimized_weights':target.tolist(),'turnover':turn,'transaction_cost_pct':turn*tc_bps/10000,'current_metrics':self._metrics(w0),'optimized_metrics':self._metrics(target),'status':'suggested' if not np.allclose(target,w0) else 'retained_current_weights'}
    def _hrp(self):
        corr=self.returns_df.corr().fillna(0).to_numpy(); dist=np.sqrt(np.maximum(.5*(1-corr),0)); np.fill_diagonal(dist,0); order=leaves_list(linkage(squareform(dist,checks=False),method='average')); cov=self.cov_annual; weights=np.ones(len(order)); clusters=[list(order)]
        while clusters:
            next_clusters=[]
            for c in clusters:
                if len(c)<=1: continue
                mid=len(c)//2; left,right=c[:mid],c[mid:]
                def cv(ix):
                    sub=cov[np.ix_(ix,ix)]; iv=1/np.maximum(np.diag(sub),1e-12); ww=iv/iv.sum(); return float(ww@sub@ww)
                vl,vr=cv(left),cv(right); a=vr/max(vl+vr,1e-12); weights[left]*=a; weights[right]*=(1-a); next_clusters.extend([left,right])
            clusters=next_clusters
        out=np.zeros(len(order)); out[order]=weights; return out/out.sum()
    def _solve(self,w0,method,lo,hi,sector_map,sector_caps,max_turnover,tc,alpha):
        R=self.returns_df.to_numpy(); losses=-R; n=len(w0); bounds=[(lo,hi)]*n
        def constraints_for(w):
            c=[{'type':'eq','fun':lambda x:x.sum()-1},{'type':'ineq','fun':lambda x:max_turnover-np.abs(x-w0).sum()/2}]
            for sec,cap in (sector_caps or {}).items():
                ids=[i for i,t in enumerate(self.tickers) if (sector_map or {}).get(t)==sec]
                if ids: c.append({'type':'ineq','fun':lambda x,ids=ids,cap=cap:cap-x[ids].sum()})
            return c
        if method=='cvar_es':
            # Rockafellar-Uryasev objective: minimize eta + mean(max(loss - eta, 0))/(1-alpha).
            eta0=float(np.quantile(losses@w0,alpha)); x0=np.r_[w0,eta0,np.zeros(len(R))]
            def obj(x): return x[n]+np.mean(x[n+1:])/(1-alpha)+tc*np.abs(x[:n]-w0).sum()/2
            cons=[{'type':'eq','fun':lambda x:x[:n].sum()-1},{'type':'ineq','fun':lambda x:max_turnover-np.abs(x[:n]-w0).sum()/2},{'type':'ineq','fun':lambda x:x[n+1:]-(losses@x[:n]-x[n])}]
            for sec,cap in (sector_caps or {}).items():
                ids=[i for i,t in enumerate(self.tickers) if (sector_map or {}).get(t)==sec]
                if ids: cons.append({'type':'ineq','fun':lambda x,ids=ids,cap=cap:cap-x[ids].sum()})
            res=minimize(obj,x0,method='SLSQP',bounds=bounds+[(None,None)]+[(0,None)]*len(R),constraints=cons,options={'maxiter':500,'ftol':1e-9})
            return res.x[:n] if res.success else w0
        def obj(w):
            cost=tc*np.abs(w-w0).sum()/2; v=np.sqrt(max(w@self.cov_annual@w,1e-12))
            if method=='max_sharpe': return -((w@self.mean_annual-self.rf)/v)+cost
            return float(w@self.cov_annual@w)+cost
        res=minimize(obj,w0,method='SLSQP',bounds=bounds,constraints=constraints_for(w0),options={'maxiter':500})
        return res.x if res.success else w0
    def _metrics(self,w):
        p=self.returns_df.to_numpy()@w; l=-p; v=float(np.quantile(l,.95)); tail=l[l>=v]
        vol=float(np.sqrt(max(w@self.cov_annual@w,0))); ret=float(w@self.mean_annual)
        return {'annual_return':ret,'annual_volatility':vol,'sharpe_ratio':(ret-self.rf)/max(vol,1e-12),'daily_var_95':v,'daily_es_95':float(tail.mean()) if len(tail) else v,'hhi_concentration':float(w@w)}
