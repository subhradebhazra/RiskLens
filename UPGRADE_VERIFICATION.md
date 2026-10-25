# RiskLens v2.0 — Upgrade Verification

## UI/UX changes applied

- Warm-light, editorial design system with forest-green actions, soft borders, serif page titles, restrained contrast, and less visual noise.
- Grouped sidebar navigation replaces the redundant navigation ribbon.
- Shared theme for metric cards, controls, alerts, tabs, tables, and all ordinary Plotly charts.
- 3D systemic-risk chart and key explicit line/fill colors updated to align with the palette.
- Streamlit theme config changed to light mode.

## Checks run in the build environment

- Python runtime available in the build environment: 3.13.5.
- `python -m compileall -q app.py pages src`: **passed**.
- `python -m pytest -q`: **36 passed**.
- Plotly 3D figure test checks that the updated figure can be constructed.

## Not verified in this build environment

- The interactive Streamlit app could not be launched here because Streamlit is not installed in the build environment. Therefore exact browser rendering, interactive page flow, and theme appearance have not been visually smoke-tested here.
- External market/news provider connections were not exercised.
- Optional Gemini and Agent 37 provider calls were not end-to-end verified; Agent 37 request flow is covered by mocked unit tests. FinBERT was not end-to-end verified.

## Target-machine instructions

Use Python 3.11.9, install from `requirements.txt`, run the tests, and then `python -m streamlit run app.py`. See `RUN_IN_PYCHARM.md`.

## Operating notes

- Demo data are deterministic example fixtures, not live market data.
- Some advanced features remain model approximations or use explicit fallbacks.
- Verify provider status and timestamps before a demonstration; do not claim a live provider worked until a connection succeeds.
