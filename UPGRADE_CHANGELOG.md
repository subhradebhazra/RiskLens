# RiskLens v2.0 — Upgrade Change Log

This package includes the supplied professional upgrade bundle plus a separate UI/UX refinement focused on a more natural, deliberate, human-designed feel.

## Visual redesign

- Replaced the purple/blue neon gradients and heavy shadows with a warm-paper background, forest-green controls, muted botanical colors, thin borders, and minimal depth.
- Replaced the repeated two-row, 13-link page ribbon with grouped sidebar navigation.
- Simplified the page header to an editorial title, a short description, and restrained version metadata.
- Restyled KPI tiles, tabs, inputs, alerts, expanders, tables, and Plotly charts to share a consistent visual system.
- Removed decorative emoji from many tab and action labels and made button copy more direct.
- Updated the special 3D contagion visualization and key chart annotations to use the same palette.
- Switched `.streamlit/config.toml` to a matching light theme.
- Added a short design note to `README.md`.

## Underlying professional upgrade

The project retains the supplied upgrade bundle's changes to market-data mode and provenance banners, regime model selection, simulation UI integration, structural-model diagnostics, sentiment coupling, telemetry alerts, report exports, and the historical-risk NumPy import fix. Existing quantitative modules, demo data, and tests were retained.

## Verification

- `python -m compileall -q app.py pages src`: passed.
- `python -m pytest -q`: 36 passed.
- Interactive Streamlit browser rendering and external provider connections were not verified in this build environment because Streamlit is not installed here. Run the provided PyCharm instructions in the target environment to verify the actual browser UI.

## Important limitations

- The redesign is a frontend styling/navigation pass; it does not make the app a trading system or guarantee forecasts.
- Provider data requires network access and may require API credentials. Bundled fixtures are demo data, not live quotes.
- Model approximations and fallbacks remain; see `README.md` and page-level warnings.

## Selectable AI provider integration

- Added a runtime selector on the AI Quant Copilot and API & Data Settings pages for Gemini or Agent 37. The same selected provider is used for risk briefings and AI-generated hypothetical scenario proposals.
- Implemented an Agent 37 adapter using the instance's `/v1/responses` endpoint and `X-Agent37-Key` authentication. Requires `AGENT37_API_KEY` and an already-running `AGENT37_INSTANCE_URL`; the app does not create cloud instances or incur provisioning charges automatically.
- Added an Agent 37 instance-health connection test, environment examples, bounded telemetry forwarding, and tests for response parsing, selection, missing configuration, and Pydantic validation.
- Provider failures produce a clearly labeled deterministic fallback; the app does not silently switch to the other provider. Scenario proposals remain manually reviewed and are not applied directly.
