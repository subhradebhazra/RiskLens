from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st
from risklens.ui import header, status_banner, current_telemetry, load_workspace_prices, ai_provider_selector
from risklens.config import get_setting
from risklens.ai.gemini_assistant import ask_ai, compile_stress_scenario
from risklens.reporting.reports import build_report, report_json

header(
    st,
    "AI Quantitative Risk Officer",
    "Telemetry-grounded risk briefing with a selectable Gemini or Agent 37 provider and a transparent rule-based fallback.",
)

prices, data_status, data_provider, data_ts = load_workspace_prices(st)
telemetry = current_telemetry(data_status, data_provider, data_ts, prices)
telemetry["headlines"] = []
telemetry["model_warnings"] = [
    f"Data status: {data_status}.",
    "This copilot snapshot uses reference SPY returns and an illustrative 100,000 currency-unit position.",
]

status_banner(st, data_status, data_provider, data_ts, "Any AI or rule-based analysis is bounded strictly to the telemetry JSON below; it is not personalized financial advice.")

st.subheader("AI Provider")
selected_provider = ai_provider_selector(
    st,
    label="Choose which AI service RiskLens should use",
    help_text="The selected provider is used for both the risk briefing and scenario proposal. If it is unavailable, RiskLens reports a rule-based fallback instead of silently switching providers.",
)
if selected_provider == "Gemini":
    if get_setting("GEMINI_API_KEY", ""):
        st.success("Gemini key is configured for this session.")
    else:
        st.warning("Gemini is selected, but no Gemini API key is configured. Add it in API & Data Settings.")
else:
    agent_key_ready = bool(get_setting("AGENT37_API_KEY", ""))
    agent_url_ready = bool(get_setting("AGENT37_INSTANCE_URL", ""))
    if agent_key_ready and agent_url_ready:
        st.success("Agent 37 key and instance URL are configured. A running instance is still required.")
    else:
        st.warning("Agent 37 needs both an Agent 37 API key and the HTTPS URL of a running Agent 37 instance. Configure both in API & Data Settings.")
    st.caption("Agent 37 hosts an agent runtime; it is not a drop-in Gemini model endpoint. Get the instance URL from your Agent 37 dashboard. Its model/provider is configured inside that agent instance; the Gemini model setting applies only to Gemini.")

with st.expander("Telemetry provided to the copilot", expanded=False):
    st.json(telemetry)

preset_questions = [
    "Summarize the main measured risks, what is uncertain, and which conclusions are not supported by the available data.",
    "Explain the difference between the reported 95% VaR and Expected Shortfall in this snapshot.",
    "What model limitations should an institutional risk committee note before acting on these numbers?",
]
selected_preset = st.selectbox("Quick Prompt Templates", preset_questions)
question = st.text_area("Ask about the current RiskLens risk snapshot", value=selected_preset, height=90)

if st.button("Generate risk briefing", type="primary"):
    answer, provider = ask_ai(question, telemetry, selected_provider)
    st.session_state["copilot_answer"] = (answer, provider)

if "copilot_answer" in st.session_state:
    answer, provider = st.session_state["copilot_answer"]
    st.caption(f"Provider mode: `{provider}`")
    with st.container(border=True):
        st.markdown(answer)
    rep = build_report("AI Risk Officer Briefing", {"provider": provider, "question": question, "answer": answer, "telemetry": telemetry})
    st.download_button("Download Briefing Report JSON", report_json(rep).encode(), file_name="risklens_copilot_briefing.json", mime="application/json")

st.divider()
st.subheader("Custom Scenario Compiler (Pydantic Schema Validated)")
scenario = st.text_area("Describe a hypothetical stress scenario", placeholder="Example: market declines 12%, volatility doubles, and correlations rise to 0.8.")
if st.button("Compile scenario proposal"):
    proposal, message = compile_stress_scenario(scenario, selected_provider)
    st.session_state["scenario_proposal"] = (proposal, message)

if "scenario_proposal" in st.session_state:
    proposal, message = st.session_state["scenario_proposal"]
    st.info(message)
    if proposal:
        st.json(proposal)
        approved = st.checkbox("I reviewed and approve this hypothetical configuration for manual use", key="approve_compiled_scenario")
        if approved:
            st.download_button(
                "Download reviewed scenario JSON",
                __import__("json").dumps(proposal, indent=2),
                file_name="risklens_stress_scenario.json",
                mime="application/json",
            )
