"""Sentiment scoring with optional FinBERT and a clearly labeled lexicon fallback."""
from __future__ import annotations
import re
import numpy as np
import pandas as pd

POSITIVE={"beat","growth","gain","gains","surge","surged","upgrade","upgraded","profit","profits","strong","bullish","rise","rises","rising","positive","recovery","recover","record","outperform","improve","improved","rally","rallied"}
NEGATIVE={"miss","decline","declined","loss","losses","fall","falls","falling","crash","crashed","downgrade","downgraded","weak","bearish","risk","risks","fraud","default","recession","selloff","sell-off","drop","dropped","negative","layoff","layoffs","warning","warns","uncertainty"}

def lexicon_score(text: str) -> dict:
    words=re.findall(r"[a-z]+(?:-[a-z]+)?",(text or "").lower())
    pos=sum(w in POSITIVE for w in words); neg=sum(w in NEGATIVE for w in words); total=pos+neg
    score=(pos-neg)/max(total,1)
    # Deterministic, intentionally low-confidence heuristic rather than a probability-calibrated classifier.
    p_pos=float(np.clip(0.33+0.33*score,0.05,0.85)); p_neg=float(np.clip(0.33-0.33*score,0.05,0.85)); p_neu=max(0.0,1-p_pos-p_neg)
    s=p_pos+p_neg+p_neu
    return {"sentiment":"positive" if score>0.15 else "negative" if score < -0.15 else "neutral","positive_probability":p_pos/s,"neutral_probability":p_neu/s,"negative_probability":p_neg/s,"confidence":float(min(0.65,0.35+0.06*total)),"method":"deterministic financial lexicon fallback"}

def score_news(articles: pd.DataFrame, use_finbert: bool=True) -> tuple[pd.DataFrame,str,list[str]]:
    df=articles.copy(); warnings=[]
    if df.empty:
        for c in ("sentiment","positive_probability","neutral_probability","negative_probability","confidence","sentiment_method"): df[c]=[]
        return df,"unavailable",["No article text to score."]
    if use_finbert:
        try:
            from transformers import pipeline
            model=pipeline("text-classification",model="ProsusAI/finbert",return_all_scores=True)
            texts=(df["title"].fillna("")+". "+df["text"].fillna("")).tolist()
            scores=model(texts,truncation=True)
            parsed=[]
            for arr in scores:
                d={x["label"].lower():float(x["score"]) for x in arr}
                parsed.append({"sentiment":max(d,key=d.get),"positive_probability":d.get("positive",0.),"neutral_probability":d.get("neutral",0.),"negative_probability":d.get("negative",0.),"confidence":max(d.values()),"sentiment_method":"ProsusAI/FinBERT"})
            for col in parsed[0]: df[col]=[x[col] for x in parsed]
            return df,"FinBERT",warnings
        except Exception as e: warnings.append(f"FinBERT unavailable; fallback used ({type(e).__name__}).")
    scores=[lexicon_score(f"{a} {b}") for a,b in zip(df["title"].fillna(""),df.get("text",pd.Series([""]*len(df))).fillna(""))]
    for col in ["sentiment","positive_probability","neutral_probability","negative_probability","confidence"]: df[col]=[x[col] for x in scores]
    df["sentiment_method"]="deterministic financial lexicon fallback"
    return df,"lexicon fallback",warnings
