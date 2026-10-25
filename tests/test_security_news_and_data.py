import json
import pandas as pd
from risklens.security import mask_secret,sanitize_text,validate_http_url
from risklens.news.providers import normalize_articles,fetch_manual_news,deduplicate_headlines
from risklens.news.finbert import lexicon_score,score_news
from risklens.news.sentiment_coupling import evaluate_sentiment_coupling
from risklens.data.market_data import load_demo_prices

def test_secret_masking_and_sanitization():
    secret='my-super-secret-token-123'
    assert secret not in mask_secret(secret)
    s=sanitize_text(f"request failed token={secret}",[secret])
    assert secret not in s and '[REDACTED]' in s
    assert validate_http_url('https://example.com') and not validate_http_url('file:///tmp/a')

def test_news_manual_parse_and_dedup_retains_attribution():
    frame=fetch_manual_news(json.dumps([{'title':'Company raises guidance','source':'A','published_at':'2026-01-01T10:00:00Z','url':'https://a.invalid'},{'title':'Company raises guidance strongly','source':'B','published_at':'2026-01-01T11:00:00Z','url':'https://b.invalid'}]))
    assert len(frame)==2
    dedup=deduplicate_headlines(frame,threshold=.5)
    assert len(dedup)<=len(frame)
    assert 'source_attributions' in dedup

def test_sentiment_fallback_is_not_finbert():
    s=lexicon_score('Markets crash as losses rise and risk grows')
    assert s['sentiment']=='negative'
    frame=pd.DataFrame([{'title':'Stocks rally after strong growth','text':'profits increase','source':'test','published_at':pd.Timestamp('2025-01-01',tz='UTC')}])
    out,method,w=score_news(frame,use_finbert=False)
    assert method=='lexicon fallback'
    assert out.iloc[0]['sentiment_method']=='deterministic financial lexicon fallback'

def test_news_coupling_insufficient_sample_fails_closed():
    idx=pd.date_range('2024-01-01',periods=20,tz='UTC')
    res=evaluate_sentiment_coupling(pd.Series([-.5]*20,index=idx),pd.Series([.01]*20,index=idx),min_observations=60)
    assert not res['validated'] and res['correlation'] is None

def test_demo_data_is_available_and_historical():
    prices=load_demo_prices()
    assert not prices.empty
    assert {'Open','High','Low','Close'}.issubset(prices.columns)
