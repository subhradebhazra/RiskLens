"""Exactly-one-arrival-mechanism jump generator; Hawkes uses Ogata thinning."""
from __future__ import annotations
import numpy as np
from typing import Literal

class UnifiedEventGenerator:
    MODES=('none','merton_poisson','hawkes_self_exciting','historical_bootstrap')
    def __init__(self, mu_0=.05, alpha=.12, beta=.30, jump_mean=-.025, jump_std=.035, news_gamma=.25, seed=42):
        if mu_0<0 or alpha<0 or beta<=0 or alpha>=beta: raise ValueError('Require mu_0>=0, beta>0 and stationary Hawkes alpha<beta.')
        self.mu_0=float(mu_0); self.alpha=float(alpha); self.beta=float(beta); self.jump_mean=float(jump_mean); self.jump_std=max(float(jump_std),1e-8); self.news_gamma=float(news_gamma); self.rng=np.random.default_rng(seed)
    @property
    def branching_ratio(self): return self.alpha/self.beta
    def effective_baseline_intensity(self, sentiment_z_shock=0., validated=False): return self.mu_0*(1+self.news_gamma*max(0.,-float(sentiment_z_shock))) if validated else self.mu_0
    def simulate_jump_returns(self,n_paths,horizon_days,mode:Literal['none','merton_poisson','hawkes_self_exciting','historical_bootstrap']='hawkes_self_exciting',sentiment_z_shock=0.,news_validated=False,historical_jumps=None,rng=None):
        if mode not in self.MODES: raise ValueError(f'Unknown jump mode: {mode}')
        if n_paths<1 or horizon_days<1: raise ValueError('n_paths and horizon_days must be positive.')
        rng=rng or self.rng; out=np.zeros((n_paths,horizon_days)); counts=np.zeros((n_paths,horizon_days),dtype=int); mu=self.effective_baseline_intensity(sentiment_z_shock,news_validated); k=float(np.exp(self.jump_mean+.5*self.jump_std**2)-1)
        if mode=='none': return {'jump_returns':out,'jump_counts':counts,'compensator_k':0.,'effective_mu':0.,'branching_ratio':0.}
        for path in range(n_paths):
            if mode=='merton_poisson': times=rng.uniform(0,horizon_days,size=rng.poisson(mu*horizon_days))
            elif mode=='historical_bootstrap':
                pool=np.asarray(historical_jumps if historical_jumps is not None else [])
                if pool.size==0: raise ValueError('Historical bootstrap requires observed jump magnitudes.')
                times=rng.uniform(0,horizon_days,size=rng.poisson(mu*horizon_days))
            else:
                # Ogata thinning for exponential Hawkes intensity over continuous horizon.
                t=0.; events=[]; lam=mu
                while t<horizon_days:
                    upper=mu+sum(self.alpha*np.exp(-self.beta*(t-e)) for e in events)
                    if upper<=0: break
                    t+=rng.exponential(1/upper)
                    if t>=horizon_days: break
                    actual=mu+sum(self.alpha*np.exp(-self.beta*(t-e)) for e in events)
                    if rng.random()<=actual/upper: events.append(t)
                times=np.asarray(events)
            for tm in times:
                day=min(int(tm),horizon_days-1); counts[path,day]+=1
                mag=float(rng.choice(pool)) if mode=='historical_bootstrap' else float(rng.normal(self.jump_mean,self.jump_std))
                out[path,day]+=mag
        return {'jump_returns':out,'jump_counts':counts,'compensator_k':k,'effective_mu':mu,'branching_ratio':self.branching_ratio if mode=='hawkes_self_exciting' else 0.}
