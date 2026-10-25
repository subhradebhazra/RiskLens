"""Four-state Markov regime model with explicit fitted/fallback status."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import norm
try:
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
except ImportError:
    MarkovRegression = None

LABELS = ["Calm / Low Volatility", "Normal / Transitioning", "Elevated Volatility", "Crisis / Stress"]

class HamiltonRegimeSwitchingModel:
    def __init__(self, n_regimes: int = 4):
        if n_regimes != 4: raise ValueError("This implementation defines exactly four interpretable regimes.")
        self.n_regimes=4; self.transition_matrix=np.eye(4); self.filtered_probs=pd.DataFrame(); self.smoothed_probs=pd.DataFrame(); self.regime_params=[]; self.converged=False; self.method_used="unfitted"; self.warnings=[]
    def fit(self, returns: pd.Series) -> dict:
        r=pd.Series(returns).replace([np.inf,-np.inf],np.nan).dropna().astype(float)
        if len(r)<120: raise ValueError("At least 120 finite returns are required for four-state regime estimation.")
        if MarkovRegression is not None:
            try:
                res=MarkovRegression(r.to_numpy(), k_regimes=4, trend='c', switching_variance=True).fit(disp=False,maxiter=500)
                if bool(res.mle_retvals.get('converged',False)) and np.isfinite(res.llf):
                    filt=np.asarray(res.filtered_marginal_probabilities); smooth=np.asarray(res.smoothed_marginal_probabilities)
                    means=[]; stds=[]
                    for k in range(4):
                        w=np.clip(smooth[:,k],1e-12,None); m=np.average(r.to_numpy(),weights=w); means.append(m); stds.append(np.sqrt(np.average((r.to_numpy()-m)**2,weights=w)))
                    P=np.asarray(res.regime_transition)
                    if P.ndim==3: P=P[:,:,0]
                    if np.allclose(P.sum(axis=0),1,atol=1e-4): P=P.T
                    order=np.argsort(stds); self.transition_matrix=P[np.ix_(order,order)]; filt=filt[:,order]; smooth=smooth[:,order]
                    self._populate(r,np.asarray(means)[order],np.asarray(stds)[order],filt,smooth); self.converged=True; self.method_used='Hamilton_MLE_Statsmodels'; return self.get_diagnostics()
            except Exception as exc: self.warnings.append(f"Statsmodels fit failed ({type(exc).__name__}); transparent fallback used.")
        self._fit_gaussian_proxy(r); self.method_used='Gaussian-mixture proxy (not Hamilton EM)'; self.converged=False; self.warnings.append('Hamilton MLE did not converge or was unavailable; proxy probabilities are not Hamilton-filter estimates.'); return self.get_diagnostics()
    def _fit_gaussian_proxy(self,r):
        from sklearn.mixture import GaussianMixture
        x=np.c_[r.to_numpy(),np.abs(r.to_numpy())]
        g=GaussianMixture(n_components=4,random_state=42,n_init=5,reg_covar=1e-6,max_iter=500).fit(x); p=g.predict_proba(x)
        vols=np.array([np.average(np.abs(r.to_numpy()),weights=p[:,k]) for k in range(4)]); order=np.argsort(vols); p=p[:,order]
        means=np.array([np.average(r.to_numpy(),weights=p[:,k]) for k in range(4)]); stds=np.array([np.sqrt(np.average((r.to_numpy()-means[k])**2,weights=p[:,k])) for k in range(4)])
        states=p.argmax(1); P=np.ones((4,4))*0.5
        for a,b in zip(states[:-1],states[1:]): P[a,b]+=1
        self.transition_matrix=P/P.sum(1,keepdims=True); self._populate(r,means,stds,p,p)
    def _populate(self,r,means,stds,filt,smooth):
        self.filtered_probs=pd.DataFrame(filt,index=r.index,columns=LABELS); self.smoothed_probs=pd.DataFrame(smooth,index=r.index,columns=LABELS); self.regime_params=[]
        for k,label in enumerate(LABELS):
            p=float(np.clip(self.transition_matrix[k,k],0,.9999)); rr=r[self.smoothed_probs[label] >= self.smoothed_probs.max(axis=1)]
            if len(rr)<5: rr=r
            v=float(-np.quantile(rr,.05)); tail=rr[rr<=-v]; es=float(-tail.mean()) if len(tail) else v
            self.regime_params.append({'label':label,'daily_mean':float(means[k]),'daily_vol':float(stds[k]),'annualized_vol':float(stds[k]*np.sqrt(252)),'persistence_prob':p,'expected_duration_days':1/(1-p),'var_95':v,'es_95':es})
    def get_diagnostics(self):
        if self.filtered_probs.empty: return {'method_used':self.method_used,'converged':False,'warnings':self.warnings}
        latest=self.filtered_probs.iloc[-1].to_dict(); return {'method_used':self.method_used,'converged':self.converged,'current_regime':max(latest,key=latest.get),'current_probabilities':latest,'transition_matrix':self.transition_matrix.tolist(),'regime_statistics':self.regime_params,'warnings':self.warnings}
