"""Provider-selectable AI copilot: Gemini direct API or an Agent 37 hosted agent.

Agent 37 is an agent-hosting/runtime service, not a drop-in model API. Its path
requires both an Agent 37 API key and the HTTPS URL of a running agent instance.
All AI-generated stress scenarios are parsed and validated before being offered
for manual review; no generated output is automatically applied to application state.
"""
from __future__ import annotations

import json
from urllib.parse import urlparse

import requests

from risklens.config import get_setting
from risklens.ai.prompt_templates import SYSTEM_INSTRUCTIONS, briefing_prompt
from risklens.security import sanitize_text

MAX_RESPONSE_CHARS = 12_000
ALLOWED_AGENT37_DOMAIN = "agent37.app"


def deterministic_interpretation(telemetry: dict, reason: str = "AI provider not used") -> str:
    """Return a transparent local explanation without calling an external AI service."""
    parts = [f"Rule-based RiskLens interpretation ({reason}):"]
    status = telemetry.get("data_status", "unknown")
    parts.append(
        f"Data status: {status}. Treat conclusions cautiously if values are demo, cached, stale or unavailable."
    )
    var = telemetry.get("var_95")
    es = telemetry.get("es_95")
    if var is not None and es is not None:
        parts.append(
            f"Reported 95% VaR is {var}; expected shortfall is {es}. "
            "These are engine outputs, not loss guarantees."
        )
    weather = telemetry.get("market_weather")
    if weather:
        parts.append(
            f"Market-weather classification: {weather}; it is a heuristic observation classification, not a forecast."
        )
    missing = [key for key, value in telemetry.items() if value is None]
    if missing:
        parts.append("Missing telemetry: " + ", ".join(missing[:12]) + ". No values are inferred for these fields.")
    return " ".join(parts)


def _bounded_telemetry(telemetry: dict) -> dict:
    """Allow only relevant risk fields to cross the AI-provider boundary."""
    allowed = {
        "data_status", "provider", "last_updated", "market_weather", "portfolio_value",
        "var_95", "es_95", "annualized_volatility", "max_drawdown", "model_warnings",
        "sample_size", "headlines",
    }
    return {key: value for key, value in telemetry.items() if key in allowed}


def _agent37_base_url() -> str:
    raw_url = str(get_setting("AGENT37_INSTANCE_URL", "") or "").strip().rstrip("/")
    parsed = urlparse(raw_url)
    # This is a user-configured outbound request. Restrict it to the official HTTPS
    # instance domain to avoid accidentally turning this setting into an SSRF vector.
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or not host.endswith("." + ALLOWED_AGENT37_DOMAIN):
        raise ValueError("Enter an HTTPS Agent 37 instance URL ending in .agent37.app.")
    if parsed.path not in {"", "/"}:
        raise ValueError("Enter the base Agent 37 instance URL without a path suffix.")
    if parsed.port or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("The Agent 37 instance URL must not contain credentials, query parameters or fragments.")
    return raw_url


def _extract_agent37_text(payload: dict) -> str | None:
    """Extract text from Responses-style JSON without assuming one harness shape."""
    for field in ("output_text", "text", "response"):
        value = payload.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()

    output = payload.get("output")
    if isinstance(output, list):
        snippets: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            direct = item.get("text")
            if isinstance(direct, str) and direct.strip():
                snippets.append(direct.strip())
            content = item.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        part_text = part.get("text")
                        if isinstance(part_text, str) and part_text.strip():
                            snippets.append(part_text.strip())
        if snippets:
            return "\n\n".join(snippets)

    data = payload.get("data")
    if isinstance(data, dict):
        return _extract_agent37_text(data)
    return None


def _agent37_generate(contents: str) -> str:
    key = str(get_setting("AGENT37_API_KEY", "") or "").strip()
    if not key:
        raise ValueError("Agent 37 API key is not configured.")
    base_url = _agent37_base_url()
    timeout = max(10.0, float(get_setting("AI_REQUEST_TIMEOUT_SECONDS", "45") or 45))
    response = requests.post(
        f"{base_url}/v1/responses",
        headers={"X-Agent37-Key": key, "Content-Type": "application/json"},
        json={"input": contents, "stream": False},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("Agent 37 returned an unexpected response format.")
    text = _extract_agent37_text(payload)
    if not text:
        raise ValueError("Agent 37 response did not contain readable text.")
    return text[:MAX_RESPONSE_CHARS]


def _gemini_generate(contents: str) -> str:
    key = str(get_setting("GEMINI_API_KEY", "") or "").strip()
    if not key:
        raise ValueError("Gemini API key is not configured.")
    from google import genai

    client = genai.Client(api_key=key)
    model = str(get_setting("GEMINI_MODEL", "gemini-2.5-flash") or "gemini-2.5-flash")
    response = client.models.generate_content(model=model, contents=contents)
    text = getattr(response, "text", None)
    if not text or not str(text).strip():
        raise ValueError("Gemini returned an empty response.")
    return str(text).strip()[:MAX_RESPONSE_CHARS]


def ask_ai(question: str, telemetry: dict, provider: str | None = None) -> tuple[str, str]:
    """Send a bounded risk briefing request to the selected provider.

    The selection is explicit; missing credentials or network failures return a
    rule-based interpretation with a provider status label rather than silently
    switching to another paid provider.
    """
    selected = (provider or get_setting("AI_PROVIDER", "Gemini") or "Gemini").strip().lower()
    bounded = _bounded_telemetry(telemetry)
    contents = (
        SYSTEM_INSTRUCTIONS
        + "\n\n"
        + briefing_prompt(bounded)
        + "\nUSER QUESTION:\n"
        + sanitize_text(question)[:2000]
        + "\n\nOperational limits: Answer only from the supplied snapshot. Do not use tools, browse, execute commands, modify files, or take external actions. Identify uncertainty explicitly."
    )

    if selected in {"gemini", "google gemini"}:
        label = "GEMINI"
        reason = "Gemini provider not connected"
        generate = _gemini_generate
    elif selected in {"agent37", "agent 37"}:
        label = "AGENT 37"
        reason = "Agent 37 provider not connected"
        generate = _agent37_generate
    else:
        return deterministic_interpretation(telemetry, f"unknown provider selection: {selected}"), "RULE-BASED FALLBACK"

    try:
        result = generate(contents)
        return result, label
    except Exception as exc:
        # The type is useful to diagnose configuration without exposing exception
        # bodies, request headers, prompts, or credential-bearing URLs.
        return (
            deterministic_interpretation(telemetry, f"{reason} ({type(exc).__name__})"),
            f"RULE-BASED FALLBACK ({label} unavailable)",
        )


def ask_gemini(question: str, telemetry: dict) -> tuple[str, str]:
    """Backward-compatible alias for pages or scripts that still call ask_gemini."""
    return ask_ai(question, telemetry, provider="Gemini")


def compile_stress_scenario(description: str, provider: str | None = None) -> tuple[dict | None, str]:
    """Generate a scenario proposal with the selected provider, then validate it."""
    selected = (provider or get_setting("AI_PROVIDER", "Gemini") or "Gemini").strip().lower()
    if selected in {"gemini", "google gemini"}:
        label = "Gemini"
        try:
            key = str(get_setting("GEMINI_API_KEY", "") or "").strip()
            if not key:
                return None, "Gemini is not configured. Add its API key in API & Data Settings."
            from google import genai
            client = genai.Client(api_key=key)
            response = client.models.generate_content(
                model=str(get_setting("GEMINI_MODEL", "gemini-2.5-flash") or "gemini-2.5-flash"),
                contents=_scenario_prompt(description),
            )
            raw_text = str(getattr(response, "text", "") or "").strip()
        except Exception as exc:
            return None, f"{label} request failed ({type(exc).__name__}); no configuration applied."
    elif selected in {"agent37", "agent 37"}:
        label = "Agent 37"
        try:
            raw_text = _agent37_generate(_scenario_prompt(description))
        except Exception as exc:
            return None, f"{label} request failed ({type(exc).__name__}); no configuration applied."
    else:
        return None, "Unknown AI provider selection. Choose Gemini or Agent 37."

    try:
        # Allow a provider to wrap valid JSON in a Markdown code fence, but nothing
        # beyond one JSON object is passed to Pydantic.
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
            if cleaned.rstrip().endswith("```"):
                cleaned = cleaned.rstrip()[:-3]
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("No JSON object was returned.")
        data = json.loads(cleaned[start : end + 1])
        from risklens.simulation.scenarios import StressConfig
        valid = StressConfig.model_validate(data)
        return valid.model_dump(), f"Validated {label} proposal. Review all assumptions and explicitly approve before applying."
    except Exception as exc:
        return None, f"{label} response failed scenario validation ({type(exc).__name__}); no configuration applied."


def _scenario_prompt(description: str) -> str:
    return (
        SYSTEM_INSTRUCTIONS
        + "\nReturn ONLY one JSON object with fields name, market_shock (-0.95 to 2), "
        "volatility_multiplier (0.1 to 10), correlation_target (-0.99 to 0.99), "
        "jump_intensity_multiplier (0 to 20). Do not execute tools or take actions. "
        "This is only a proposal and requires human review. User description: "
        + sanitize_text(description)[:1000]
    )
