"""Configuration loading with safe, optional environment-based credentials."""
from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env", override=False)

SECRET_NAMES = ("GEMINI_API_KEY", "AGENT37_API_KEY", "NEWSDATA_API_KEY", "CURRENTS_API_KEY", "FINNHUB_API_KEY", "ALPHA_VANTAGE_API_KEY")

@dataclass(frozen=True)
class Settings:
    cache_ttl_seconds: int = 300
    request_timeout_seconds: float = 8.0
    default_base_currency: str = "USD"
    default_simulation_paths: int = 10_000
    random_seed: int = 42


def get_setting(name: str, default: str | None = None) -> str | None:
    """Read session-only Streamlit settings first, then environment defaults.

    The function returns secrets only to server-side callers; do not log or display
    the returned value. Outside Streamlit the environment path works normally.
    """
    try:
        import streamlit as st
        value = st.session_state.get(f"runtime_{name}")
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, (int, float)):
            return str(value)
        try:
            value = st.secrets.get(name)
            if isinstance(value, str) and value.strip():
                return value.strip()
        except Exception:
            value = None
    except Exception:
        value = None
    return os.getenv(name, default)


def configured_secrets() -> dict[str, bool]:
    """Return availability flags only, not secret values."""
    return {name: bool(os.getenv(name, "").strip()) for name in SECRET_NAMES}

SETTINGS = Settings()
