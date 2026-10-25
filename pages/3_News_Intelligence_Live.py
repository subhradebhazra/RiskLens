from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st
import pandas as pd
import plotly.express as px

from risklens.news.providers import fetch_market_news, fetch_manual_news, deduplicate_headlines
from risklens.news.finbert import score_news
from risklens.news.sentiment_coupling import evaluate_sentiment_coupling
from risklens.data.demo_assets import load_demo_asset_prices
from risklens.features.returns import calculate_returns
from risklens.ui import header, dark_plot

header(
    st,
    "Live News & Sentiment Intelligence",
    "Multi-source news ingestion, TF-IDF cosine deduplication, FinBERT/lexicon scoring, and lagged volatility coupling validation.",
)

c1, c2 = st.columns([3, 1])
query = c1.text_input("Search Term (used by configured news providers)", "markets")
dedup_thresh = c2.slider("Cosine Deduplication Threshold", 0.50, 0.95, 0.80, step=0.05)

with st.expander("Add manual news (text lines or JSON array)"):
    manual = st.text_area(
        "Manual News Input",
        placeholder='[{"title":"Company raises guidance","published_at":"2026-10-22T10:00:00Z","source":"Example","url":"https://example.com"}]',
        height=110,
    )

if st.button("Fetch, deduplicate and score news", type="primary"):
    st.session_state["risklens_news_query"] = query
    st.session_state["risklens_manual_news"] = manual

manual = st.session_state.get("risklens_manual_news", manual)
query = st.session_state.get("risklens_news_query", query)

news, warnings = fetch_market_news(query=query, limit=50)
if manual.strip():
    try:
        add = fetch_manual_news(manual)
        news = deduplicate_headlines(pd.concat([news, add], ignore_index=True), limit=100, threshold=dedup_thresh)
    except Exception as e:
        st.error(f"Manual input rejected: {type(e).__name__}: {str(e)[:200]}")
elif not news.empty:
    news = deduplicate_headlines(news, limit=50, threshold=dedup_thresh)

scored, method, score_warnings = score_news(news, use_finbert=True)
st.caption(f"**Active Sentiment Engine:** `{method}` · FinBERT lexicon fallback is explicitly labeled and never misrepresented as calibrated neural probabilities.")

for w in warnings + score_warnings:
    st.warning(w)

if not scored.empty and any(str(x).upper().startswith("DEMO") for x in scored.get("provider", [])):
    st.warning("DEMO MODE — Bundled historical sample headlines are displayed because configured live news sources did not return usable articles.")

if scored.empty:
    st.info("No articles available. Configure an API provider or paste manual news.")
else:
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Unique Articles", str(len(scored)))
    k2.metric("Mean Negative Score", f"{scored['negative_probability'].mean():.2f}")
    k3.metric("Mean Positive Score", f"{scored['positive_probability'].mean():.2f}")
    k4.metric("Scoring Method", method)

    tab_charts, tab_table, tab_coupling = st.tabs([
        "Sentiment breakdown",
        "Article feed",
        "Lagged sentiment / volatility test",
    ])

    with tab_charts:
        col_a, col_b = st.columns(2)
        with col_a:
            sentiment_counts = scored["sentiment"].value_counts().rename_axis("Sentiment").reset_index(name="Articles")
            st.plotly_chart(dark_plot(px.bar(sentiment_counts, x="Sentiment", y="Articles", color="Sentiment", title="Article Sentiment Classification")), use_container_width=True)
        with col_b:
            fig_sc = px.scatter(
                scored,
                x="negative_probability",
                y="positive_probability",
                size="confidence",
                color="sentiment",
                hover_name="title",
                title="Positive vs. Negative Score Space (Size = Confidence)",
            )
            st.plotly_chart(dark_plot(fig_sc), use_container_width=True)

    with tab_table:
        cols = [
            c for c in [
                "published_at", "title", "sentiment", "positive_probability",
                "neutral_probability", "negative_probability", "confidence",
                "sentiment_method", "source_attributions", "source", "provider", "url"
            ] if c in scored
        ]
        st.dataframe(scored[cols], use_container_width=True, hide_index=True)
        st.download_button("Download Analyzed News CSV", scored.to_csv(index=False).encode(), file_name="risklens_news_analysis.csv", mime="text/csv")

    with tab_coupling:
        st.markdown("#### Lagged Sentiment → Realized Volatility Gate (`sentiment_coupling.py`)")
        st.caption("Tests whether lagged daily negative sentiment correlates with subsequent absolute returns without same-day forward lookahead.")
        demo_px = load_demo_asset_prices()
        if not demo_px.empty and "published_at" in scored:
            valid_dates = scored.dropna(subset=["published_at"]).copy()
            if not valid_dates.empty:
                daily_neg = valid_dates.set_index("published_at")["negative_probability"].resample("1D").mean().dropna()
                spy_ret = calculate_returns(demo_px["SPY"]).dropna()
                gate = evaluate_sentiment_coupling(daily_neg, spy_ret, min_observations=60)
                st.json(gate)
            else:
                st.info("Articles lack parseable timestamps for daily coupling alignment.")
