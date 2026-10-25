"""Bundled multi-asset deterministic demo data; never labeled as live."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from risklens.config import ROOT

def load_demo_asset_prices() -> pd.DataFrame:
    path=ROOT/"data"/"demo"/"demo_asset_prices.csv"
    if not path.exists(): return pd.DataFrame()
    df=pd.read_csv(path,parse_dates=["Date"],index_col="Date")
    if df.index.tz is None: df.index=df.index.tz_localize("UTC")
    return df.sort_index()

def load_demo_portfolio() -> pd.DataFrame:
    path=ROOT/"data"/"demo"/"demo_portfolio.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["ticker","units","currency","sector","benchmark"])
