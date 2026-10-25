"""Single-source event generator prevents duplicate jump-arrival processes."""
from __future__ import annotations
import numpy as np
from risklens.models.hawkes import HawkesFit, simulate_ogata

class UnifiedEventGenerator:
    MODES={"static_poisson","hawkes_clustered","historical_bootstrap"}
    def __init__(self, mode: str="static_poisson", seed: int=42):
        if mode not in self.MODES: raise ValueError(f"mode must be one of {sorted(self.MODES)}")
        self.mode=mode; self.rng=np.random.default_rng(seed)
    def generate(self, horizon: float, intensity: float=1.0, hawkes_fit: HawkesFit|None=None, historical_event_times=None) -> np.ndarray:
        """Return event times in [0,horizon); exactly one arrival mechanism is selected."""
        horizon=float(horizon)
        if horizon<=0: return np.array([],dtype=float)
        if self.mode=="static_poisson":
            count=self.rng.poisson(max(intensity,0)*horizon)
            return np.sort(self.rng.uniform(0,horizon,count))
        if self.mode=="hawkes_clustered":
            if hawkes_fit is None: raise ValueError("Hawkes mode requires a calibrated HawkesFit.")
            return simulate_ogata(hawkes_fit,horizon,seed=int(self.rng.integers(0,2**32-1)))
        historical=np.asarray([] if historical_event_times is None else historical_event_times,dtype=float)
        historical=historical[np.isfinite(historical)]
        if not len(historical): raise ValueError("Historical bootstrap mode requires observed event times.")
        # Sample with replacement and scale observed event-time support to the requested horizon.
        scale=horizon/max(float(historical.max()),1e-12)
        return np.sort(self.rng.choice(historical,size=len(historical),replace=True)*scale)
