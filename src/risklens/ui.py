"""Shared RiskLens UI helpers and a restrained, editorial dashboard design system."""
from __future__ import annotations

from html import escape
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from risklens.data.demo_assets import load_demo_asset_prices
from risklens.data.market_data import fetch_historical_prices, align_trading_calendars
from risklens.features.returns import calculate_returns, annualized_volatility, drawdown_series
from risklens.risk.var import historical_var_es
from risklens.risk.contagion import market_weather
from risklens.alerts.engine import ConfigurableAlertEngine
from risklens.config import get_setting

# Palette is intentionally restrained: warm paper, ink, forest green and a small
# citrus accent. No gradients/glass effects are used for core data surfaces.
PLOT_LAYOUT = {
    "paper_bgcolor": "#FFFEFA",
    "plot_bgcolor": "#FFFEFA",
    "font": {"color": "#28332C", "family": "Aptos, Segoe UI, sans-serif", "size": 12},
    "margin": {"l": 28, "r": 24, "t": 52, "b": 30},
    "xaxis": {"gridcolor": "#E7E8DF", "zerolinecolor": "#CDD2C6", "linecolor": "#D9DDD2"},
    "yaxis": {"gridcolor": "#E7E8DF", "zerolinecolor": "#CDD2C6", "linecolor": "#D9DDD2"},
}

STATUS_HEX = {
    "LIVE": "#397A5B",
    "FETCHED": "#4B7880",
    "CACHED": "#A57732",
    "STALE": "#B66C43",
    "DEMO": "#777D67",
    "UNAVAILABLE": "#B6534B",
}

NAV_PAGES = [
    ("app.py", "Overview", "Executive Risk Overview"),
    ("pages/1_Overview_Market_Weather.py", "Market weather", "Overview & Market Weather"),
    ("pages/2_Stock_Outlook.py", "Market deep dive", "Stock & Index Deep Dive"),
    ("pages/3_News_Intelligence_Live.py", "News & sentiment", "News Intelligence & Sentiment"),
    ("pages/4_Historical_Risk_Analysis.py", "Historical risk", "Historical Risk & Regimes"),
    ("pages/5_Forward_Risk_Outlook.py", "Forward scenarios", "Forward Risk & Simulation"),
    ("pages/6_Portfolio_XRay.py", "Portfolio x-ray", "Portfolio X-Ray"),
    ("pages/7_Contagion_Systemic_Risk.py", "Systemic risk", "Contagion & Systemic Risk"),
    ("pages/8_Stress_Testing.py", "Stress tests", "Stress Testing & Crisis Replay"),
    ("pages/9_Risk_Reduction_Studio.py", "Risk reduction", "Risk Reduction Studio"),
    ("pages/10_Model_Validation_Backtesting.py", "Model validation", "Model Validation & Backtesting"),
    ("pages/11_Gemini_Quant_Copilot.py", "Research copilot", "AI Quant Copilot"),
    ("pages/12_API_Data_Settings.py", "Data settings", "API & Data Settings"),
]


def inject_institutional_theme(st) -> None:
    """Apply the shared warm-paper, editorial UI across all Streamlit pages."""
    st.markdown(
        """
        <style>
        :root {
            --rl-paper: #F3F2EC;
            --rl-surface: #FFFEFA;
            --rl-ink: #28332C;
            --rl-muted: #70786F;
            --rl-line: #DDE0D5;
            --rl-green: #315F49;
            --rl-green-hover: #244A38;
            --rl-citrus: #D9E78A;
            --rl-red: #B6534B;
            --rl-shadow: 0 2px 8px rgba(35, 48, 39, .035);
        }
        html, body, [data-testid="stAppViewContainer"], .stApp {
            background: var(--rl-paper) !important;
            color: var(--rl-ink) !important;
            font-family: Aptos, "Segoe UI", sans-serif;
        }
        [data-testid="stAppViewContainer"] > .main { background: var(--rl-paper); }
        .main .block-container {
            max-width: 1540px;
            padding-top: 1.65rem;
            padding-bottom: 3.25rem;
        }
        h1, h2, h3, h4, h5, h6 {
            color: var(--rl-ink) !important;
            letter-spacing: -0.025em;
        }
        h1 { font-family: Georgia, "Times New Roman", serif !important; font-size: 2.05rem !important; font-weight: 500 !important; }
        h2 { font-size: 1.34rem !important; font-weight: 650 !important; }
        h3 { font-size: 1.08rem !important; font-weight: 650 !important; }
        p, li, label, [data-testid="stMarkdownContainer"] { color: var(--rl-ink); }
        [data-testid="stCaptionContainer"], .stCaption, small { color: var(--rl-muted) !important; }

        /* Horizontal website navigation; the dashboard is content-first. */
        [data-testid="stSidebar"] { display:none !important; }
        .rl-topnav { display:flex; align-items:center; justify-content:space-between; gap:16px; padding:10px 0 13px; border-bottom:1px solid var(--rl-line); margin-bottom:20px; }
        .rl-brand { display:flex; align-items:center; gap:10px; min-width:150px; }
        .rl-brand-mark { width:36px; height:36px; display:flex; align-items:center; justify-content:center; background:#142C24; color:#F8F8EE; border-radius:9px; font-family:Georgia,serif; font-size:21px; }
        .rl-brand-name { color:#182A22; font-size:1.14rem; font-weight:800; letter-spacing:-.045em; line-height:1.05; }
        .rl-brand-caption { color:#747C70; font-size:.62rem; letter-spacing:.10em; text-transform:uppercase; margin-top:4px; }
        .rl-topnav-links { display:flex; align-items:center; justify-content:center; gap:3px; flex-wrap:wrap; }
        .rl-topnav-links a { color:#37483B !important; text-decoration:none !important; font-size:.80rem; font-weight:650; padding:8px 9px; border-radius:6px; white-space:nowrap; }
        .rl-topnav-links a:hover { background:#E2E7DD; color:#203D2D !important; }
        .rl-topnav-note { color:#6F796E; font-size:.66rem; white-space:nowrap; border:1px solid #D9DED3; border-radius:5px; padding:7px 9px; background:#F8F8F2; }
        @media (max-width: 900px) { .rl-topnav { align-items:flex-start; flex-direction:column; gap:10px; } .rl-topnav-links { justify-content:flex-start; } .rl-topnav-note { display:none; } }

        /* Page heading: title and description do the work, no redundant nav ribbon. */
        .rl-page-head { border-bottom:1px solid var(--rl-line); padding:0 0 1.15rem 0; margin-bottom:1.25rem; }
        .rl-page-kicker { color:#687667; font-size:.69rem; font-weight:750; letter-spacing:.14em; text-transform:uppercase; margin-bottom:.6rem; }
        .rl-page-row { display:flex; align-items:flex-end; justify-content:space-between; gap:24px; flex-wrap:wrap; }
        .rl-page-title { margin:0; color:var(--rl-ink); font-family:Georgia, "Times New Roman", serif; font-size:2.15rem; font-weight:500; line-height:1.15; letter-spacing:-.04em; }
        .rl-page-subtitle { color:#737B70; font-size:.94rem; line-height:1.55; margin-top:.55rem; max-width:820px; }
        .rl-version-note { color:#747D72; border:1px solid #D9DED3; border-radius:5px; padding:7px 10px; white-space:nowrap; font-size:.69rem; letter-spacing:.08em; text-transform:uppercase; background:#F8F8F2; }

        /* Use crisp borders and very subtle depth; avoid glassmorphism. */
        [data-testid="stMetric"] {
            background:var(--rl-surface) !important;
            border:1px solid var(--rl-line) !important;
            border-radius:10px !important;
            box-shadow:var(--rl-shadow) !important;
            padding:14px 16px !important;
            min-height:106px;
        }
        [data-testid="stMetricLabel"] p { color:#747B70 !important; font-size:.72rem !important; font-weight:700 !important; letter-spacing:.065em; text-transform:uppercase; }
        [data-testid="stMetricValue"] { color:#263D30 !important; font-weight:690 !important; letter-spacing:-.04em; }
        [data-testid="stMetricDelta"] { font-size:.78rem !important; }
        [data-testid="stVerticalBlockBorderWrapper"] { border-color:var(--rl-line) !important; border-radius:11px !important; background:var(--rl-surface) !important; box-shadow:var(--rl-shadow); }

        /* Inputs */
        [data-baseweb="select"] > div, [data-baseweb="input"] > div, [data-baseweb="textarea"] > div,
        [data-testid="stDateInput"] > div, [data-testid="stNumberInput"] > div {
            background:var(--rl-surface) !important; border-color:#D4D9CF !important; border-radius:7px !important; color:var(--rl-ink) !important;
        }
        input, textarea { color:var(--rl-ink) !important; }
        [data-baseweb="tag"] { background:#DDE8D8 !important; color:#284C37 !important; }
        [role="radiogroup"] label, [data-testid="stCheckbox"] label { color:var(--rl-ink) !important; }

        /* Tabs read like restrained section navigation. */
        .stTabs [data-baseweb="tab-list"] { gap:5px; background:transparent; border-bottom:1px solid var(--rl-line); padding:0; }
        .stTabs [data-baseweb="tab"] { border-radius:6px 6px 0 0; padding:9px 13px; color:#70786F; font-weight:650; background:transparent; }
        .stTabs [aria-selected="true"] { color:#2F6047 !important; background:#E7EBDF !important; border-bottom:2px solid #527A58 !important; }
        .stTabs [data-baseweb="tab-highlight"] { background-color:#527A58 !important; }

        /* Buttons have one consistent dark-green action style. */
        .stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] button {
            border-radius:7px !important; border:1px solid #315F49 !important; background:#315F49 !important; color:#FAFAF4 !important;
            font-weight:650 !important; box-shadow:none !important; transition:background .14s ease, border-color .14s ease;
        }
        .stButton > button:hover, .stDownloadButton > button:hover, [data-testid="stFormSubmitButton"] button:hover {
            background:#244A38 !important; border-color:#244A38 !important; color:#FFFFFF !important;
        }
        .stButton > button:focus, .stDownloadButton > button:focus { box-shadow:0 0 0 2px #D9E78A !important; }
        [data-testid="stPlotlyChart"] { background:var(--rl-surface); border:1px solid var(--rl-line); border-radius:10px; padding:5px 6px 1px 6px; }
        [data-testid="stDataFrame"], [data-testid="stTable"] { border:1px solid var(--rl-line); border-radius:9px; overflow:hidden; }
        [data-testid="stExpander"] { border:1px solid var(--rl-line) !important; border-radius:9px !important; background:rgba(255,254,250,.7) !important; }
        [data-testid="stAlert"] { border-radius:8px !important; }
        hr { border-color:var(--rl-line) !important; }
        code { color:#355D47 !important; background:#E9EDE4 !important; border-radius:4px; }
        a { color:#315F49; }
        .rl-status { border:1px solid #DDE0D5; border-left:3px solid #777D67; background:#FBFBF6; border-radius:6px; padding:11px 13px; margin:0 0 1rem 0; }
        .rl-status-head { color:#344337; font-weight:750; font-size:.83rem; }
        .rl-status-meta { color:#737A70; font-size:.77rem; margin-top:5px; line-height:1.5; }
        .rl-status-message { color:#737A70; font-size:.79rem; margin-top:5px; line-height:1.5; }
        .rl-section-label { color:#657365; font-size:.7rem; font-weight:750; letter-spacing:.12em; text-transform:uppercase; }
        @media (max-width: 760px) {
            .main .block-container { padding-left:1rem; padding-right:1rem; padding-top:1rem; }
            .rl-page-title { font-size:1.75rem; }
            [data-testid="stMetric"] { min-height:90px; padding:11px !important; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_navigation_bar(st) -> None:
    """Render a familiar horizontal website navigation bar."""
    st.markdown(
        '<div class="rl-topnav"><div class="rl-brand"><div class="rl-brand-mark">R</div>'
        '<div><div class="rl-brand-name">RiskLens</div><div class="rl-brand-caption">Market risk intelligence</div></div></div>'
        '<div class="rl-topnav-note">V2.0 · RESEARCH STUDIO</div></div>',
        unsafe_allow_html=True,
    )
    nav_cols = st.columns([0.8, 0.7, 0.55, 1.05, 0.8, 1.15, 0.9])
    primary = [(NAV_PAGES[0][0], "Overview"), (NAV_PAGES[1][0], "Markets"), (NAV_PAGES[3][0], "News"),
               (NAV_PAGES[4][0], "Risk analysis"), (NAV_PAGES[6][0], "Portfolio"),
               (NAV_PAGES[7][0], "Systemic risk"), (NAV_PAGES[8][0], "Stress tests")]
    for col, (path, label) in zip(nav_cols, primary):
        with col:
            st.page_link(path, label=label, use_container_width=True)
    with st.expander("More workspace pages and data source", expanded=False):
        cols = st.columns(3)
        extra = [NAV_PAGES[2], NAV_PAGES[5], NAV_PAGES[9], NAV_PAGES[10], NAV_PAGES[11], NAV_PAGES[12]]
        for i, (path, short_label, full_label) in enumerate(extra):
            with cols[i % 3]:
                st.page_link(path, label=full_label, use_container_width=True)
        st.selectbox(
            "Workspace data source",
            ["Bundled Deterministic Demo", "Try Configured Providers (Fallback to Demo)"],
            key="rl_workspace_data_mode",
            help="Demo mode uses bundled fixtures. Provider mode tries configured sources first and falls back to demo data if usable data cannot be assembled.",
        )


def ai_provider_selector(st, label: str = "AI provider", help_text: str | None = None) -> str:
    """Choose Gemini or Agent 37 and preserve that choice across Streamlit pages."""
    if "runtime_AI_PROVIDER" not in st.session_state:
        saved = st.session_state.get("risklens_selected_ai_provider")
        if not saved:
            saved = get_setting("AI_PROVIDER", "Gemini")
        normalized = str(saved or "Gemini").strip().lower()
        st.session_state["runtime_AI_PROVIDER"] = "Agent 37" if normalized in {"agent 37", "agent37"} else "Gemini"

    def persist_selection() -> None:
        st.session_state["risklens_selected_ai_provider"] = st.session_state.get("runtime_AI_PROVIDER", "Gemini")

    return st.selectbox(
        label,
        ["Gemini", "Agent 37"],
        key="runtime_AI_PROVIDER",
        on_change=persist_selection,
        help=help_text or "Choose which AI provider RiskLens uses for its copilot and scenario compiler.",
    )


def header(st, title: str, subtitle: str) -> None:
    """Configure page, apply common styling, and render a clean editorial page header."""
    try:
        st.set_page_config(page_title=f"{title} · RiskLens v2.0", page_icon="R", layout="wide", initial_sidebar_state="expanded")
    except Exception:
        # Streamlit raises if configuration has already been set; the page can still render.
        pass
    inject_institutional_theme(st)
    render_navigation_bar(st)
    from risklens.ai.floating_chat import render_floating_chat
    render_floating_chat(st)
    safe_title = escape(str(title))
    safe_subtitle = escape(str(subtitle))
    st.markdown(
        f"""
        <section class="rl-page-head">
          <div class="rl-page-kicker">Risk &amp; portfolio intelligence <span style="color:#B1B8AA;padding:0 5px;">/</span> Workspace</div>
          <div class="rl-page-row">
            <div>
              <h1 class="rl-page-title">{safe_title}</h1>
              <div class="rl-page-subtitle">{safe_subtitle}</div>
            </div>
            <div class="rl-version-note">RiskLens &nbsp;·&nbsp; v2.0</div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def status_banner(st, status: str, provider: str, timestamp: str, message: str = "") -> None:
    """Show a provenance banner with a status-specific accent and readable metadata."""
    status_up = (status or "UNKNOWN").upper()
    color_hex = STATUS_HEX.get(status_up, "#B6534B")
    if status_up == "DEMO":
        headline = "Demo data · bundled historical fixtures are driving these outputs, not live quotes."
    elif status_up == "CACHED":
        headline = "Cached data · results were retrieved from local cache, not a real-time quote."
    elif status_up == "FETCHED":
        headline = "Provider data fetched · historical observations were retrieved from a configured provider."
    elif status_up == "LIVE":
        headline = "Live source · verify the timestamp and provider details before relying on the figures."
    else:
        headline = f"Data status · {status_up}"

    provider_text = escape(str(provider or "Unavailable"))
    timestamp_text = escape(str(timestamp or "N/A"))
    message_html = f'<div class="rl-status-message">{escape(str(message))}</div>' if message else ""
    st.markdown(
        f"""
        <div class="rl-status" style="border-left-color:{color_hex};">
          <div class="rl-status-head"><span style="color:{color_hex};">●</span>&nbsp; {escape(headline)}</div>
          <div class="rl-status-meta">Source: <strong>{provider_text}</strong> &nbsp;·&nbsp; Last timestamp: <code>{timestamp_text}</code> &nbsp;·&nbsp; Status: <strong>{escape(status_up)}</strong></div>
          {message_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def dark_plot(fig, show_rangeslider: bool = False):
    """Apply RiskLens' shared light plotting style (legacy name retained for compatibility)."""
    fig.update_layout(
        **PLOT_LAYOUT,
        template="plotly_white",
        title_font=dict(size=15, color="#28332C", family="Aptos, Segoe UI, sans-serif"),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#28332C", bordercolor="#526B57", font=dict(color="#FAFAF4", size=12)),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,254,250,0.88)",
            bordercolor="#E0E3D9",
            borderwidth=1,
        ),
        colorway=["#315F49", "#78958A", "#C58B4E", "#B85E52", "#687FA0", "#95985E", "#8D7E9E"],
    )
    fig.update_xaxes(
        gridcolor="#E7E8DF",
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikecolor="#7C927B",
        spikethickness=1,
        rangeslider=dict(visible=show_rangeslider, bgcolor="#F0F1E9", bordercolor="#DDE0D5", thickness=0.08),
        zerolinecolor="#CDD2C6",
        tickfont=dict(color="#70786F", size=11),
    )
    fig.update_yaxes(
        gridcolor="#E7E8DF",
        showspikes=True,
        spikecolor="#7C927B",
        spikethickness=1,
        zerolinecolor="#CDD2C6",
        tickfont=dict(color="#70786F", size=11),
    )
    return fig


def metric_value(value, digits: int = 2, suffix: str = "") -> str:
    try:
        v = float(value)
        if not np.isfinite(v):
            return "N/A"
        return f"{v:,.{digits}f}{suffix}"
    except Exception:
        return "N/A"


def pct(value, digits: int = 2) -> str:
    try:
        return metric_value(float(value) * 100.0, digits, "%")
    except Exception:
        return "N/A"


def get_demo_return_panel(prices=None) -> pd.DataFrame:
    pxdata = prices if prices is not None else load_demo_asset_prices()
    if pxdata.empty:
        return pd.DataFrame()
    return pxdata.drop(columns=["USDINR=X"], errors="ignore").pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).dropna(how="any")


def plotly_line(series, title: str, y_title: str = ""):
    if isinstance(series, pd.DataFrame):
        fig = px.line(series, title=title)
    else:
        fig = px.line(x=series.index, y=series.values, title=title, labels={"x": "Date", "y": y_title or series.name or "Value"})
    return dark_plot(fig)


def load_workspace_prices(st, symbols: list[str] | None = None) -> tuple[pd.DataFrame, str, str, str]:
    """Load bundled demo prices or attempt configured providers when selected in sidebar."""
    demo_df = load_demo_asset_prices()
    mode = st.session_state.get("rl_workspace_data_mode", "Bundled Deterministic Demo")
    if mode == "Bundled Deterministic Demo" or demo_df.empty:
        ts = str(demo_df.index.max()) if not demo_df.empty else ""
        return demo_df, "DEMO", "bundled deterministic fixture (data/demo/demo_asset_prices.csv)", ts

    target_symbols = symbols or ["SPY", "QQQ", "AAPL", "RELIANCE.NS", "TCS.NS", "USDINR=X"]
    frames = {}
    providers_used = set()
    statuses = set()
    for sym in target_symbols:
        res = fetch_historical_prices(sym, start="2023-01-01", allow_demo=False)
        if not res.data.empty:
            frames[sym] = res.data
            providers_used.add(res.provider)
            statuses.add(res.status)
    aligned = align_trading_calendars(frames, join="inner") if len(frames) == len(target_symbols) else pd.DataFrame()
    if not aligned.empty and len(aligned) >= 60:
        st_label = "FETCHED" if "FETCHED" in statuses else ("CACHED" if "CACHED" in statuses else "STALE")
        return aligned, st_label, ", ".join(sorted(providers_used)), str(aligned.index.max())

    ts = str(demo_df.index.max()) if not demo_df.empty else ""
    return demo_df, "DEMO", "bundled deterministic fixture (provider fallback)", ts


def current_telemetry(status: str = "DEMO", provider: str = "bundled deterministic fixture", timestamp: str = "", prices=None) -> dict[str, Any]:
    prices = prices if prices is not None else load_demo_asset_prices()
    returns = calculate_returns(prices["SPY"]) if not prices.empty and "SPY" in prices else pd.Series(dtype=float)
    port = returns * 100_000
    rmet = historical_var_es(port.to_numpy(), 0.95) if len(port) else None
    vol = annualized_volatility(returns) if len(returns) > 2 else None
    dd = drawdown_series(returns, input_is_returns=True).min() if len(returns) else None
    weather = market_weather(None, vol, dd)[0] if vol is not None and dd is not None else "Unavailable"
    return {
        "data_status": status,
        "provider": provider,
        "last_updated": timestamp or (str(prices.index.max()) if len(prices) else None),
        "market_weather": weather,
        "portfolio_value": 100000,
        "var_95": rmet.var if rmet else None,
        "es_95": rmet.es if rmet else None,
        "annualized_volatility": vol,
        "max_drawdown": float(dd) if dd is not None else None,
        "sample_size": len(returns),
        "model_warnings": ["Demo fixture only; not live market data."] if status == "DEMO" else [],
    }


def evaluate_and_render_alerts(st, telemetry_payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Evaluate ConfigurableAlertEngine against current telemetry and render alerts."""
    engine = ConfigurableAlertEngine()
    alerts = engine.evaluate_telemetry(telemetry_payload)
    with st.expander(f"Risk alerts · {len(alerts)} triggered", expanded=bool(alerts)):
        if not alerts:
            st.success("No configured risk thresholds were breached by the current telemetry.")
        else:
            for item in alerts:
                sev = item["severity"]
                prefix = "Critical" if sev == "CRITICAL" else "Warning"
                st.warning(
                    f"**{prefix} · {item['label']}** — Observed: `{item['observed_value']:.4f}` "
                    f"(Threshold: `{item['threshold']}` · Source: `{item['source']}`)"
                )
    return alerts
