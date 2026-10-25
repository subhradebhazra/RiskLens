"""Reproducible risk report assembly."""
from __future__ import annotations
from datetime import datetime, timezone
import json
import numpy as np

def build_report(title: str, data: dict) -> dict:
    def clean(v):
        if isinstance(v,dict): return {str(k):clean(x) for k,x in v.items()}
        if isinstance(v,(list,tuple)): return [clean(x) for x in v]
        if hasattr(v,"item"):
            try: return clean(v.item())
            except Exception: return str(v)
        if isinstance(v,float) and not np.isfinite(v): return None
        return v
    return {"title":title,"created_at_utc":datetime.now(timezone.utc).isoformat(timespec="seconds"),"status":"ANALYTICAL OUTPUT — NOT A GUARANTEE OR PERSONALIZED FINANCIAL ADVICE","data":clean(data)}

def report_json(report: dict) -> str:
    return json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False,default=str)
