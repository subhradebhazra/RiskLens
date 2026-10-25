"""Dependence estimation utilities with transparent sample alignment."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

def align_returns(returns: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    before = len(returns)
    aligned = returns.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    return aligned, before - len(aligned)

def kendall_to_pearson(returns: pd.DataFrame) -> pd.DataFrame:
    corr = returns.corr(method="kendall").clip(-1, 1)
    return np.sin(np.pi * corr / 2)

def nearest_correlation(matrix: np.ndarray, eigen_floor: float = 1e-8) -> np.ndarray:
    a = np.asarray(matrix, dtype=float)
    a = (a + a.T) / 2
    vals, vecs = np.linalg.eigh(a)
    vals = np.maximum(vals, eigen_floor)
    psd = vecs @ np.diag(vals) @ vecs.T
    d = np.sqrt(np.maximum(np.diag(psd), eigen_floor))
    out = psd / np.outer(d, d)
    np.fill_diagonal(out, 1.0)
    return (out + out.T) / 2
