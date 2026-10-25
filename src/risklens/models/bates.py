"""Bates model orchestration with explicit component/fallback reporting."""
from __future__ import annotations
from dataclasses import dataclass
from risklens.models.merton import fit_merton, MertonFit
from risklens.models.heston import fit_heston, HestonFit

@dataclass
class BatesFit:
    merton: MertonFit | None
    heston: HestonFit | None
    selected_model: str
    converged: bool
    warnings: list[str]

def fit_bates(returns):
    r=returns.dropna()
    warnings=[]
    m=fit_merton(r) if len(r)>=30 else None
    h=fit_heston(r) if len(r)>=60 else None
    if len(r)<252: warnings.append("Fewer than 252 observations; joint Bates calibration is not attempted.")
    warnings.append("This implementation does not claim joint Bates likelihood calibration; it combines independent component diagnostics only.")
    if m and m.converged and h and h.converged:
        return BatesFit(m,h,"component diagnostics only (not jointly calibrated)",False,warnings)
    if m and m.converged: return BatesFit(m,h,"Merton-only fallback",True,warnings+["Heston component unavailable or insufficiently reliable."])
    if h and h.converged: return BatesFit(m,h,"Heston-only fallback",True,warnings+["Merton component unavailable or did not converge."])
    return BatesFit(m,h,"Gaussian baseline fallback",False,warnings+["Neither component has a reliable fit."])
