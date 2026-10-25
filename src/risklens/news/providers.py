"""Multi-source news ingestion with source attribution and robust RSS fallback."""
from __future__ import annotations
from datetime import datetime, timezone
import json, os, xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd
import requests
from risklens.data.http import request_with_retry
from risklens.config import get_setting, ROOT, SETTINGS


def normalize_articles(items: list[dict], provider: str) -> pd.DataFrame:
    rows=[]
    for item in items:
        title=str(item.get("title") or item.get("headline") or "").strip()
        if not title: continue
        ts=item.get("published_at") or item.get("pubDate") or item.get("datetime") or item.get("time_published")
        try:
            date=pd.to_datetime(ts,utc=True,errors="raise")
        except Exception: date=pd.NaT
        rows.append({"title":title,"text":str(item.get("text") or item.get("description") or item.get("summary") or ""),"source":str(item.get("source") or item.get("source_id") or provider),"url":str(item.get("url") or item.get("link") or ""),"published_at":date,"provider":provider})
    df=pd.DataFrame(rows,columns=["title","text","source","url","published_at","provider"])
    if not df.empty:
        df=df.sort_values("published_at",ascending=False,na_position="last").reset_index(drop=True)
    return df

def fetch_manual_news(text_or_json: str) -> pd.DataFrame:
    try:
        parsed=json.loads(text_or_json)
        if isinstance(parsed,dict): parsed=parsed.get("articles",[parsed])
        if not isinstance(parsed,list): raise ValueError("JSON input must be an article object or array.")
        return normalize_articles(parsed,"manual upload")
    except json.JSONDecodeError:
        return normalize_articles([{"title":line.strip(),"text":line.strip(),"published_at":datetime.now(timezone.utc).isoformat()} for line in text_or_json.splitlines() if line.strip()],"manual text")

def fetch_rss(url: str, timeout: float=8.0) -> pd.DataFrame:
    if not (url.startswith("https://") or url.startswith("http://")): raise ValueError("RSS feed URL must start with http:// or https://")
    res=request_with_retry("GET",url,timeout=timeout,headers={"User-Agent":"RiskLens/2.0 (news reader)"})
    root=ET.fromstring(res.content); items=[]
    for node in root.findall(".//item"):
        get=lambda tag: (node.findtext(tag) or "").strip()
        items.append({"title":get("title"),"text":get("description"),"url":get("link"),"published_at":get("pubDate"),"source":get("source") or url})
    # Atom feeds use namespaces and entry elements.
    if not items:
        ns={"a":"http://www.w3.org/2005/Atom"}
        for node in root.findall(".//a:entry",ns):
            title=node.findtext("a:title",default="",namespaces=ns)
            updated=node.findtext("a:updated",default="",namespaces=ns)
            link=node.find("a:link",ns)
            items.append({"title":title,"published_at":updated,"url":link.attrib.get("href","") if link is not None else ""})
    return normalize_articles(items,"RSS")

def fetch_market_news(query: str="markets", limit: int=25) -> tuple[pd.DataFrame,list[str]]:
    frames=[]; warnings=[]
    priority=(get_setting("NEWS_PROVIDER_PRIORITY", "NewsData -> Currents -> Finnhub -> Alpha Vantage -> RSS -> Demo") or "").lower()
    manual_rss_only=priority.strip().startswith("manual/rss")
    rss_text=get_setting("RSS_FEED_URLS", "") or ""
    rss_urls=[u.strip() for u in rss_text.splitlines() if u.strip()][:10]

    def try_newsdata():
        key=get_setting("NEWSDATA_API_KEY")
        if not key: return
        r=request_with_retry("GET","https://newsdata.io/api/1/latest",params={"apikey":key,"q":query,"language":"en"},timeout=float(get_setting("REQUEST_TIMEOUT_SECONDS",SETTINGS.request_timeout_seconds))); r.raise_for_status()
        payload=r.json()
        if payload.get("status")=="error": raise RuntimeError("API error response")
        frames.append(normalize_articles(payload.get("results",[])[:limit],"NewsData.io"))
    def try_currents():
        key=get_setting("CURRENTS_API_KEY")
        if not key: return
        r=request_with_retry("GET","https://api.currentsapi.services/v1/search",params={"keywords":query,"apiKey":key},timeout=float(get_setting("REQUEST_TIMEOUT_SECONDS",SETTINGS.request_timeout_seconds))); r.raise_for_status()
        payload=r.json()
        if payload.get("status")=="error": raise RuntimeError("API error response")
        frames.append(normalize_articles(payload.get("news",[])[:limit],"Currents API"))
    def try_finnhub():
        key=get_setting("FINNHUB_API_KEY")
        if not key: return
        r=request_with_retry("GET","https://finnhub.io/api/v1/news",params={"category":"general","token":key},timeout=float(get_setting("REQUEST_TIMEOUT_SECONDS",SETTINGS.request_timeout_seconds))); r.raise_for_status()
        frames.append(normalize_articles(r.json()[:limit],"Finnhub"))
    def try_alpha():
        key=get_setting("ALPHA_VANTAGE_API_KEY")
        if not key: return
        r=request_with_retry("GET","https://www.alphavantage.co/query",params={"function":"NEWS_SENTIMENT","tickers":query,"limit":min(max(int(limit),1),1000),"apikey":key},timeout=float(get_setting("REQUEST_TIMEOUT_SECONDS",SETTINGS.request_timeout_seconds))); r.raise_for_status()
        payload=r.json()
        if "Note" in payload or "Information" in payload: raise RuntimeError("API rate-limit or entitlement message returned")
        frames.append(normalize_articles(payload.get("feed",[])[:limit],"Alpha Vantage News Sentiment"))
    def try_rss():
        for i,feed_url in enumerate(rss_urls,1):
            try: frames.append(fetch_rss(feed_url,timeout=float(get_setting("REQUEST_TIMEOUT_SECONDS",SETTINGS.request_timeout_seconds))))
            except Exception as e: warnings.append(f"RSS feed #{i} unavailable: {type(e).__name__}; URL details withheld.")

    funcs={"newsdata":try_newsdata,"currents":try_currents,"finnhub":try_finnhub,"alpha vantage":try_alpha,"rss":try_rss}
    if manual_rss_only:
        try_rss()
    else:
        tokens=[part.strip() for part in priority.split("->")]
        order=[]
        for token in tokens:
            match=next((k for k in funcs if k in token),None)
            if match and match not in order: order.append(match)
        if not order: order=["newsdata","currents","finnhub","alpha vantage","rss"]
        for source in order:
            try:
                before=len(frames); funcs[source]()
                # Empty results are not considered a successful connection.
                if len(frames)>before and frames[-1].empty: warnings.append(f"{source} returned no usable article records.")
            except Exception as e:
                warnings.append(f"{source} unavailable: {type(e).__name__}; credentials and response details withheld.")
    if not frames or all(f.empty for f in frames):
        path=ROOT/"data"/"demo"/"demo_news.json"
        if path.exists():
            raw=json.loads(path.read_text(encoding="utf-8")); frames.append(normalize_articles(raw,"DEMO fixture")); warnings.append("No live news source succeeded; showing historical bundled DEMO articles.")
    df=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()
    if not df.empty:
        df=df.drop_duplicates(subset=["url","title"],keep="first")
        return deduplicate_headlines(df,limit),warnings
    return pd.DataFrame(columns=["title","text","source","url","published_at","provider"]),warnings+["News unavailable; enter manual text/JSON or configure a provider."]

def deduplicate_headlines(articles: pd.DataFrame, limit: int=50, threshold: float=0.80) -> pd.DataFrame:
    if articles.empty or len(articles)==1: return articles.head(limit).copy()
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    df=articles.copy().reset_index(drop=True); titles=df["title"].fillna("").astype(str).tolist()
    try:
        mat=TfidfVectorizer(ngram_range=(1,2),stop_words="english").fit_transform(titles)
        sim=cosine_similarity(mat)
    except ValueError: return df.head(limit)
    keep=[]; groups=[]
    for i in range(len(df)):
        found=None
        for g,rep in enumerate(keep):
            if sim[i,rep]>=threshold: found=g; break
        if found is None: keep.append(i); groups.append([i])
        else: groups[found].append(i)
    rows=[]
    for group in groups:
        row=df.iloc[group[0]].to_dict()
        row["source_attributions"]="; ".join(dict.fromkeys(df.iloc[group]["source"].astype(str).tolist()))
        row["duplicate_count"]=len(group)-1
        rows.append(row)
    return pd.DataFrame(rows).sort_values("published_at",ascending=False,na_position="last").head(limit).reset_index(drop=True)
