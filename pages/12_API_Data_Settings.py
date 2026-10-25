from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st
import requests
from datetime import datetime, timezone
from risklens.config import SECRET_NAMES, get_setting
from risklens.security import mask_secret, sanitize_text
from risklens.data.caching import cache_clear
from risklens.ui import header, ai_provider_selector

header(
    st,
    "API & Data Settings",
    "Session-only credential management, live endpoint verification tests, provider priority chains, and cache controls.",
)

st.markdown(
    """**Security Policy:** Secrets entered here are kept in this active Streamlit session only and are volatile.
    They are never written to `.env`, SQLite, browser storage, or logs."""
)

labels = {
    "GEMINI_API_KEY": "Google Gemini",
    "AGENT37_API_KEY": "Agent 37",
    "NEWSDATA_API_KEY": "NewsData.io",
    "CURRENTS_API_KEY": "Currents API",
    "FINNHUB_API_KEY": "Finnhub",
    "ALPHA_VANTAGE_API_KEY": "Alpha Vantage",
}

st.subheader("Provider Status Overview")
status_cols = st.columns(3)
for index, name in enumerate(SECRET_NAMES):
    configured = bool(get_setting(name, "") or "")
    test_info = st.session_state.get(f"test_status_{name}")
    state_label = "VERIFIED" if (test_info and test_info.get("connected")) else ("CONFIGURED" if configured else "UNSET")
    status_cols[index % 3].metric(labels[name], state_label, mask_secret(get_setting(name, "")))

with st.expander("Enter or update API keys", expanded=True):
    values = {}
    for name in SECRET_NAMES:
        existing = get_setting(name, "") or ""
        show_key = st.checkbox(f"Show {labels[name]} key while entering", key=f"show_{name}")
        values[name] = st.text_input(
            labels[name] + " API key",
            type="default" if show_key else "password",
            placeholder=(mask_secret(existing) if existing else "Optional — leave blank to keep current runtime setting"),
            key=f"entry_{name}",
            help="New value is stored only in this active Streamlit session when saved.",
        )
    existing_agent_url = str(get_setting("AGENT37_INSTANCE_URL", "") or "")
    agent_url_entry = st.text_input(
        "Agent 37 running instance URL (required for Agent 37 AI)",
        key="entry_AGENT37_INSTANCE_URL",
        value=existing_agent_url,
        placeholder="https://your-instance-id.agent37.app",
        help="Use the HTTPS instance URL shown in Agent 37 Cloud. An API key by itself does not identify which hosted agent to message.",
    )
    if st.button("Save keys to this session", type="primary"):
        for name in SECRET_NAMES:
            value = str(st.session_state.get(f"entry_{name}", "") or "")
            if value.strip():
                st.session_state[f"runtime_{name}"] = value.strip()
                st.session_state.pop(f"test_status_{name}", None)
                st.session_state.pop(f"last_success_{name}", None)
        if agent_url_entry.strip():
            st.session_state["runtime_AGENT37_INSTANCE_URL"] = agent_url_entry.strip().rstrip("/")
            st.session_state.pop("test_status_AGENT37_API_KEY", None)
        st.success("Non-empty keys and the Agent 37 instance URL were saved to this active session only. Run a connection test below to verify.")

st.subheader("Live Minimal Connection Tests")
provider = st.selectbox("Provider to test", list(labels), format_func=lambda n: labels[n])
if st.button("Run connection test", type="primary"):
    key = get_setting(provider)
    status = {"provider": labels[provider], "connected": False, "message": "Not tested"}
    try:
        if not key:
            raise RuntimeError("No key is configured.")
        if provider == "AGENT37_API_KEY":
            from urllib.parse import urlparse
            instance_url = str(get_setting("AGENT37_INSTANCE_URL", "") or "").strip().rstrip("/")
            parsed = urlparse(instance_url)
            hostname = (parsed.hostname or "").lower()
            if parsed.scheme != "https" or not hostname.endswith(".agent37.app"):
                raise RuntimeError("Enter an HTTPS instance URL ending in .agent37.app in the Agent 37 settings.")
            r = requests.get(
                f"{instance_url}/v1/health",
                headers={"X-Agent37-Key": key},
                timeout=max(10.0, float(get_setting("AI_REQUEST_TIMEOUT_SECONDS", "45") or 45)),
            )
            r.raise_for_status()
            payload = r.json()
            if isinstance(payload, dict) and (payload.get("ok") is False or payload.get("healthy") is False):
                raise RuntimeError("Agent 37 instance health check reported unhealthy.")
            status = {"provider": labels[provider], "connected": True, "message": "Agent 37 instance health endpoint returned a successful response."}
        elif provider == "GEMINI_API_KEY":
            from google import genai
            client = genai.Client(api_key=key)
            client.models.get(model=get_setting("GEMINI_MODEL", "gemini-2.5-flash"))
            status = {"provider": labels[provider], "connected": True, "message": "Model metadata request succeeded."}
        elif provider == "NEWSDATA_API_KEY":
            r = requests.get("https://newsdata.io/api/1/latest", params={"apikey": key, "q": "markets", "language": "en"}, timeout=float(get_setting("REQUEST_TIMEOUT_SECONDS", 8)))
            r.raise_for_status()
            payload = r.json()
            if payload.get("status") == "error":
                raise RuntimeError("Provider returned an API error; check key, quota or endpoint access.")
            status = {"provider": labels[provider], "connected": True, "message": "NewsData.io endpoint returned a valid response."}
        elif provider == "CURRENTS_API_KEY":
            r = requests.get("https://api.currentsapi.services/v1/latest-news", params={"apiKey": key, "language": "en"}, timeout=float(get_setting("REQUEST_TIMEOUT_SECONDS", 8)))
            r.raise_for_status()
            payload = r.json()
            if payload.get("status") == "error":
                raise RuntimeError("Provider returned an API error; check key or quota.")
            status = {"provider": labels[provider], "connected": True, "message": "Currents endpoint returned a valid response."}
        elif provider == "FINNHUB_API_KEY":
            r = requests.get("https://finnhub.io/api/v1/quote", params={"symbol": "AAPL", "token": key}, timeout=float(get_setting("REQUEST_TIMEOUT_SECONDS", 8)))
            r.raise_for_status()
            payload = r.json()
            if not payload or not any(k in payload for k in ("c", "error")):
                raise RuntimeError("Finnhub response was not recognized.")
            if "error" in payload:
                raise RuntimeError("Finnhub rejected the request or quota was exceeded.")
            status = {"provider": labels[provider], "connected": True, "message": "Finnhub quote endpoint returned the expected response shape."}
        else:
            r = requests.get("https://www.alphavantage.co/query", params={"function": "GLOBAL_QUOTE", "symbol": "IBM", "apikey": key}, timeout=float(get_setting("REQUEST_TIMEOUT_SECONDS", 8)))
            r.raise_for_status()
            payload = r.json()
            if "Error Message" in payload or "Note" in payload:
                raise RuntimeError("Alpha Vantage returned an error or quota message.")
            if "Global Quote" not in payload:
                raise RuntimeError("Alpha Vantage response did not include Global Quote; check key/quota.")
            status = {"provider": labels[provider], "connected": True, "message": "Alpha Vantage global-quote endpoint returned the expected response shape."}
    except Exception as e:
        status = {"provider": labels[provider], "connected": False, "message": sanitize_text(f"{type(e).__name__}: {str(e)}", [key] if key else [])}
    status["tested_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if status["connected"]:
        st.session_state[f"last_success_{provider}"] = status["tested_at_utc"]
    st.session_state[f"test_status_{provider}"] = status

status = st.session_state.get(f"test_status_{provider}")
if status:
    if status["connected"]:
        st.success("CONNECTED — " + status["message"])
    else:
        st.error("NOT CONNECTED — " + status["message"])
    st.caption("Last test: " + status.get("tested_at_utc", "unknown") + " UTC.")

st.subheader("AI Provider Selection")
ai_provider_selector(
    st,
    label="Default AI provider for the RiskLens copilot",
    help_text="This session-only choice controls both the AI Risk Officer briefing and the custom scenario compiler. It is remembered as you navigate between pages during this session.",
)
agent_key_present = bool(get_setting("AGENT37_API_KEY", ""))
agent_url_present = bool(get_setting("AGENT37_INSTANCE_URL", ""))
if agent_key_present and agent_url_present:
    st.caption("Agent 37 credentials are configured for this session. Connection testing verifies that the configured instance responds.")
else:
    st.caption("Agent 37 requires an API key and an already-running Agent 37 instance URL. Configuring the key alone does not provision an instance.")

st.subheader("Provider Priorities & Runtime Controls")
c1, c2 = st.columns(2)
with c1:
    st.selectbox("Market-data priority", ["yfinance -> Finnhub -> Alpha Vantage -> Demo", "Finnhub -> Alpha Vantage -> yfinance -> Demo", "Demo only"], key="runtime_MARKET_DATA_PRIORITY")
with c2:
    st.selectbox("News priority", ["NewsData -> Currents -> Finnhub -> Alpha Vantage -> RSS -> Demo", "Currents -> NewsData -> Finnhub -> Alpha Vantage -> RSS -> Demo", "Manual/RSS -> Demo"], key="runtime_NEWS_PROVIDER_PRIORITY")

st.selectbox("Gemini model", ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"], key="runtime_GEMINI_MODEL")
st.text_area("Optional RSS feed URLs (one URL per line; session only)", key="runtime_RSS_FEED_URLS")
s1, s2, s3 = st.columns(3)
s1.slider("Data-provider timeout (seconds)", 2, 30, 8, key="runtime_REQUEST_TIMEOUT_SECONDS")
s2.slider("AI-provider timeout (seconds)", 10, 120, 45, key="runtime_AI_REQUEST_TIMEOUT_SECONDS")
s3.slider("Cache duration (seconds)", 0, 3600, 300, key="runtime_CACHE_TTL_SECONDS")

col_clear, col_rem = st.columns(2)
with col_clear:
    if st.button("Clear market-data cache"):
        cache_clear()
        st.success("Market-data cache cleared for this running process.")
with col_rem:
    remove = st.selectbox("Credential to remove", SECRET_NAMES, format_func=lambda n: labels[n])
    if st.button("Remove selected session key"):
        st.session_state.pop(f"runtime_{remove}", None)
        st.session_state.pop(f"entry_{remove}", None)
        st.session_state.pop(f"test_status_{remove}", None)
        st.session_state.pop(f"last_success_{remove}", None)
        st.success("Session key removed.")
