"""Seeded portfolio Monte Carlo using one explicit jump-arrival process."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import norm, t as student_t
from .unified_event_generator import UnifiedEventGenerator

class UnifiedMonteCarloEngine:
    def __init__(self,n_paths=10000,horizon_days=5,seed=42,use_antithetic=True):
        if not 1000<=int(n_paths)<=50000: raise ValueError('n_paths must be between 1,000 and 50,000.')
        if not 1<=int(horizon_days)<=30: raise ValueError('horizon_days must be between 1 and 30.')
        self.n_paths=int(n_paths); self.horizon_days=int(horizon_days); self.seed=int(seed); self.use_antithetic=bool(use_antithetic)
    def run_portfolio_simulation(self,returns_df:pd.DataFrame,weights,portfolio_value:float,model_mode='full_bates_hawkes_copula',copula_df=5.,event_gen=None,regime_vol_multiplier=1.,sentiment_z_shock=0.,news_validated=False):
        r=returns_df.replace([np.inf,-np.inf],np.nan).dropna(how='any')
        if r.empty or r.shape[1]<1 or len(r)<20: raise ValueError('Need at least 20 complete observations and one asset.')
        if not np.isfinite(portfolio_value) or portfolio_value<=0: raise ValueError('portfolio_value must be finite and positive.')
        w=np.asarray(weights,dtype=float)
        if len(w)!=r.shape[1] or not np.isfinite(w).all() or w.sum()<=0: raise ValueError('Weights must be finite, positive-sum, and match assets.')
        w=w/w.sum(); rng=np.random.default_rng(self.seed); n=self.n_paths
        mu=r.mean().to_numpy(); vol=r.std(ddof=1).to_numpy()*float(regime_vol_multiplier); corr=r.corr().fillna(0).to_numpy(); np.fill_diagonal(corr,1)
        eig,vec=np.linalg.eigh(corr); corr=vec@np.diag(np.maximum(eig,1e-6))@vec.T; d=np.sqrt(np.diag(corr)); corr=corr/np.outer(d,d); L=np.linalg.cholesky(corr)
        half=(n+1)//2 if self.use_antithetic else n; z=rng.standard_normal((half,self.horizon_days,r.shape[1])); z=np.concatenate([z,-z],axis=0)[:n] if self.use_antithetic else z
        if model_mode=='full_bates_hawkes_copula':
            chi=rng.chisquare(max(float(copula_df),2.1),size=(n,self.horizon_days,1)); z=z@L.T*np.sqrt(copula_df/np.maximum(chi,1e-8)); u=np.clip(student_t.cdf(z,df=copula_df),1e-7,1-1e-7); z=norm.ppf(u)
        else: z=z@L.T
        jump_mode='hawkes_self_exciting' if model_mode=='full_bates_hawkes_copula' else ('merton_poisson' if model_mode in ('merton_only','bates_static_jumps') else 'none')
        eg=event_gen or UnifiedEventGenerator(seed=self.seed); jd=eg.simulate_jump_returns(n,self.horizon_days,jump_mode,sentiment_z_shock,news_validated,rng=rng)
        jump=jd['jump_returns'][:,:,None]; daily=np.zeros((n,self.horizon_days,r.shape[1])); variance=np.tile(vol**2,(n,1)); k=jd['compensator_k']; lam=jd.get('effective_mu',0.)
        for day in range(self.horizon_days):
            if model_mode in ('full_bates_hawkes_copula','bates_static_jumps','heston_only'):
                sigma=np.sqrt(np.maximum(variance,1e-10)); variance=np.maximum(variance+3.5*(vol**2-variance)/252+.35*sigma*rng.standard_normal((n,r.shape[1]))/np.sqrt(252),1e-10)
            else: sigma=np.tile(vol,(n,1))
            drift=mu-.5*sigma**2-(lam*k if jump_mode!='none' else 0.)
            daily[:,day,:]=drift+sigma*z[:,day,:]+jump[:,day,:]
        asset_simple=np.expm1(daily.sum(axis=1)); pr=asset_simple@w; pnl=portfolio_value*pr; losses=-pnl
        v95=float(np.quantile(losses,.95)); v99=float(np.quantile(losses,.99))
        return {'model_mode':model_mode,'n_paths':n,'horizon_days':self.horizon_days,'portfolio_returns':pr,'portfolio_pnl':pnl,'losses':losses,'var_95_currency':v95,'var_99_currency':v99,'es_95_currency':float(losses[losses>=v95].mean()),'es_99_currency':float(losses[losses>=v99].mean()),'jump_diagnostics':{'total_jumps_simulated':int(jd['jump_counts'].sum()),'branching_ratio':jd.get('branching_ratio',0.)},'warnings':['Simplified Euler variance dynamics; not a calibrated full Bates model unless model parameters were independently estimated.']}
