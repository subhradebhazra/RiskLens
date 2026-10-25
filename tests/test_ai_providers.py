"""Tests for the provider switch and Agent 37 request adapter."""
from __future__ import annotations

import risklens.ai.gemini_assistant as assistant


class FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError("HTTP failure")

    def json(self):
        return self._payload


def test_agent37_briefing_routes_to_selected_instance(monkeypatch):
    settings = {
        "AGENT37_API_KEY": "sk_live_test_key",
        "AGENT37_INSTANCE_URL": "https://demo123.agent37.app",
        "AI_REQUEST_TIMEOUT_SECONDS": "45",
    }
    monkeypatch.setattr(assistant, "get_setting", lambda name, default=None: settings.get(name, default))
    captured = {}

    def fake_post(url, headers, json, timeout):
        captured.update(url=url, headers=headers, json=json, timeout=timeout)
        return FakeResponse({"output_text": "Risk summary from Agent 37."})

    monkeypatch.setattr(assistant.requests, "post", fake_post)
    answer, provider = assistant.ask_ai(
        "Explain current risk.",
        {"data_status": "DEMO", "var_95": 120, "es_95": 170, "unrelated_secret": "do-not-send"},
        provider="Agent 37",
    )

    assert answer == "Risk summary from Agent 37."
    assert provider == "AGENT 37"
    assert captured["url"] == "https://demo123.agent37.app/v1/responses"
    assert captured["headers"]["X-Agent37-Key"] == "sk_live_test_key"
    assert captured["json"]["stream"] is False
    assert "DEMO" in captured["json"]["input"]
    assert "do-not-send" not in captured["json"]["input"]


def test_agent37_missing_config_returns_rule_based_fallback(monkeypatch):
    settings = {"AGENT37_API_KEY": "", "AGENT37_INSTANCE_URL": ""}
    monkeypatch.setattr(assistant, "get_setting", lambda name, default=None: settings.get(name, default))
    answer, provider = assistant.ask_ai("Explain risk.", {"data_status": "DEMO"}, provider="Agent 37")

    assert provider == "RULE-BASED FALLBACK (AGENT 37 unavailable)"
    assert "Rule-based RiskLens interpretation" in answer
    assert "Agent 37 provider not connected" in answer


def test_agent37_scenario_proposal_is_schema_validated(monkeypatch):
    settings = {
        "AGENT37_API_KEY": "sk_live_test_key",
        "AGENT37_INSTANCE_URL": "https://demo123.agent37.app",
        "AI_REQUEST_TIMEOUT_SECONDS": "45",
    }
    monkeypatch.setattr(assistant, "get_setting", lambda name, default=None: settings.get(name, default))

    def fake_post(url, headers, json, timeout):
        return FakeResponse({
            "output": [{
                "type": "message",
                "content": [{
                    "type": "output_text",
                    "text": '{"name":"Market shock","market_shock":-0.12,"volatility_multiplier":2.0,"correlation_target":0.8,"jump_intensity_multiplier":2.0}',
                }],
            }]
        })

    monkeypatch.setattr(assistant.requests, "post", fake_post)
    proposal, message = assistant.compile_stress_scenario("Market falls 12%.", provider="Agent 37")

    assert proposal is not None
    assert proposal["name"] == "Market shock"
    assert proposal["market_shock"] == -0.12
    assert "Validated Agent 37 proposal" in message


def test_agent37_rejects_non_agent37_or_plain_http_url(monkeypatch):
    settings = {"AGENT37_API_KEY": "sk_live_test_key", "AGENT37_INSTANCE_URL": "http://127.0.0.1:8501"}
    monkeypatch.setattr(assistant, "get_setting", lambda name, default=None: settings.get(name, default))
    answer, provider = assistant.ask_ai("Explain risk.", {"data_status": "DEMO"}, provider="Agent 37")

    assert provider == "RULE-BASED FALLBACK (AGENT 37 unavailable)"
    assert "ValueError" in answer
