# RiskLens v2.0

**Quantitative market, news, contagion and portfolio risk dashboard**

RiskLens is a modular Streamlit application with explicit source/freshness labels, reproducible calculations, deterministic offline fixtures, provider fallbacks and independent unit tests. It is an analytical tool, not a trading system, a guarantee of risk outcomes, or personalized investment advice.

> **Demo mode is not live data.** The included price and news files are deterministic example fixtures. All pages that use them mark the result as `DEMO`. The included prices exist to exercise calculations and user flows; they are not a record of actual trades or the market history of arbitrary tickers.


## Interface design

The shared interface uses a warm-paper canvas, ink typography, forest-green actions, quiet borders, and a small natural accent palette. Navigation lives in a grouped sidebar rather than a repeated two-row ribbon; charts share a light plotting style and thin gridlines. The aim is to make data provenance and analysis easier to scan, with less visual noise and no glassmorphism/neon-gradient treatment.

The redesign takes broad layout inspiration from contemporary finance-dashboard work on [Dribbble's finance analytics dashboard collection](https://dribbble.com/search/finance-analytics-dashboard); it does not copy a specific shot or include third-party design imagery.

## Quick start

Python 3.11 or newer is the project target. The source was syntax-checked and its unit tests were run on Python 3.13.5. A complete Streamlit runtime smoke test could not be run in the build environment because Streamlit was not installed and package installation was blocked by unavailable network/DNS access.

### Windows PowerShell

```powershell
cd RiskLens
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python -m pytest -q
python -m compileall -q app.py pages src
streamlit run app.py
```

### macOS / Linux

```bash
cd RiskLens
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
python -m pytest -q
python -m compileall -q app.py pages src
streamlit run app.py
```

Open the local URL printed by Streamlit. The application is designed to start without API keys; data sources that are unavailable fall back to a clearly marked demo fixture or display an unavailable state. Demo fixtures are included under `data/demo/`.

### Optional extras

- The `yfinance` and `feedparser` adapters are included in `requirements.txt` for market history and RSS workflows.
- The AI Quant Copilot can select **Gemini** or **Agent 37** at runtime. Gemini uses the official `google-genai` SDK; install the optional AI extra when online package installation is available:

  ```bash
  python -m pip install -e ".[ai]"
  ```

- Agent 37 requires an **Agent 37 API key** plus the **HTTPS URL of an already-running Agent 37 instance** (for example, `https://<instance-id>.agent37.app`). It is an agent-hosting/runtime service, not a drop-in model API. Configure both fields under **API & Data Settings**, select **Agent 37** on the AI Quant Copilot page, and use the connection test before demoing it. This integration sends only the explicitly allowlisted RiskLens telemetry fields and user prompt to the configured instance. It does not provision an instance automatically.

- FinBERT requires `transformers`, `torch`, and accessible `ProsusAI/finbert` model weights. Without them, sentiment uses a deterministic financial lexicon and is labeled accordingly. The lexicon is **not** a calibrated ML model.

## Running in PyCharm (Windows, Python 3.11.9)

1. Extract the ZIP to a normal writable folder, for example `C:\Users\<you>\Downloads\RiskLens_v2.0`. Open that extracted `RiskLens` project folder in PyCharm (the folder containing `app.py`, `requirements.txt`, `pages`, and `src`).
2. Open **File → Settings → Project → Python Interpreter**. Select **Add Interpreter → Add Local Interpreter → Virtualenv**. Choose **Python 3.11.9** as the base interpreter and create the environment at `<project>\.venv`. If `.venv` already exists but uses another Python version, create a new environment rather than merely changing the display selection.
3. Open **View → Tool Windows → Terminal** in PyCharm. Make sure the terminal is in the project root (the folder containing `app.py`). Verify the interpreter:

   ```powershell
   python --version
   ```

   It should print `Python 3.11.9` when that interpreter is selected.

4. Install requirements (internet access is required for the initial installation):

   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

5. Optional: create a local environment file for API configuration. The app can run in deterministic demo mode without API keys.

   ```powershell
   Copy-Item .env.example .env
   ```

6. Run tests and compile checks before the demo:

   ```powershell
   python -m pytest -q
   python -m compileall -q app.py pages src
   ```

7. Start the application from the same terminal:

   ```powershell
   python -m streamlit run app.py
   ```

8. Open the local URL printed by Streamlit, normally `http://localhost:8501`. Keep the terminal open while using the app. Stop the server with `Ctrl+C`.

If port 8501 is busy, use `python -m streamlit run app.py --server.port 8502` and open `http://localhost:8502`. If installation fails, save the complete terminal error and resolve it before launching. A successful test run does not itself prove external API connectivity or a successful interactive UI smoke test.

See [`RUN_IN_PYCHARM.md`](RUN_IN_PYCHARM.md) for a troubleshooting checklist.

## Configuration and credentials

1. Copy `.env.example` to `.env` for local development, or use Streamlit's native `.streamlit/secrets.toml` / deployment secret manager.
2. Add any keys for `GEMINI_API_KEY`, `AGENT37_API_KEY`, `NEWSDATA_API_KEY`, `CURRENTS_API_KEY`, `FINNHUB_API_KEY`, and `ALPHA_VANTAGE_API_KEY`. For Agent 37, also set `AGENT37_INSTANCE_URL` to an existing running instance URL. `AI_PROVIDER` can set the default (`Gemini` or `Agent 37`).
3. Open **API & Data Settings** and run a real minimal connection test. Agent 37's test checks the health endpoint on the configured instance; a provider is shown as connected only after that call succeeds.
4. Keys entered into the Streamlit form are kept in the active session only. They are not written to disk by the app. Environment/native platform secrets remain under the operator's management.
5. Never commit `.env`, `.streamlit/secrets.toml`, real credentials, logs or caches. They are ignored by `.gitignore`.

Provider APIs can change, impose quotas, or restrict endpoints by plan. An error or quota response is not treated as a successful connection. Some endpoints pass API tokens in the query parameters required by the vendor API; RiskLens does not print these request URLs or raw provider exceptions to the UI.

## Pages

| Page | Implemented workflow | Status / boundary |
|---|---|---|
| Overview & Market Weather | Fixture market paths, observed volatility/drawdown, historical VaR, regime proxy, news status | Heuristic weather classification; fixture is not live |
| Stock & Index Deep Dive | Ticker normalization, provider request path, price/return/drawdown analysis, OHLC range volatility when available | Arbitrary live symbols require a working provider; the generic OHLC demo fixture is explicitly not symbol-specific |
| Live News & Sentiment | NewsData.io, Currents, Finnhub, Alpha Vantage, RSS, manual text/JSON, headline similarity deduplication | Provider calls require keys/network; results and fallbacks preserve attribution and labeling |
| Historical Risk | Historical/Gaussian/Cornish–Fisher VaR, empirical ES, EWMA volatility, drawdown, regime proxy | Cornish–Fisher may be unstable; warnings are emitted |
| Monte Carlo & Forward Risk | Seeded Gaussian and historical bootstrap; single-asset Merton/Heston paths; terminal P&L, VaR/ES | Full integrated Bates/Hawkes/copula simulation is not implemented; integrated mode visibly falls back to the Gaussian baseline |
| Portfolio X-Ray | Editable/uploaded holdings, USD/INR demo conversion, allocation/HHI, historical risk and correlation, volatility contributions | Demo holdings/prices are examples; other currencies are blocked rather than silently converted |
| Contagion & Systemic Risk | Student-t copula, tail dependence, Ledoit–Wolf absorption ratio, Hawkes event process, four-input regime proxy, weighted composite, and an interactive 3D dependence-based cascade | Cascade links are dependence estimates and simulated propagation, not causal contagion probabilities or calibrated forecasts |
| Stress Testing | Deterministic scenario vectors, asset-level hypothetical P&L, historical-window replay when exact observations exist, and clearly labeled proxy replay otherwise | The fallback is a hypothetical sector-beta proxy, not a historical market replay; no calibrated joint AI shock is claimed |
| Risk Reduction Studio | Long-only minimum variance, max-Sharpe and a target-return proxy; turnover bound; current-vs-suggested outputs | CVaR optimizer, HRP, sector caps, transaction costs, regime overlays and fractional Kelly are not implemented |
| Walk-Forward Validation | Strictly lagged historical VaR forecast, Kupiec POF, Christoffersen independence, ES calibration moment diagnostic | Multi-model ablation, GARCH/Merton/Heston/Bates comparative backtests and a full production-grade Acerbi–Szekely test battery are not implemented |
| AI Quant Copilot | Selectable Gemini or Agent 37 telemetry briefing and validated Pydantic stress proposal; deterministic rule fallback | Agent 37 requires a configured running instance; external providers need network access and valid credentials |
| API & Data Settings | Masked fields, session-only Gemini/Agent 37 and data keys, Agent 37 instance URL, provider selector, minimal connection checks, key removal | Session-entry credentials are volatile; Agent 37 instance must already exist; no secure persistent secret vault is claimed |

## Quantitative implementation notes

- **Returns and drawdowns:** adjusted close is preferred when the provider supplies it; OHLC columns are retained for range estimators. Invalid/missing data creates warnings; it is not silently interpolated.
- **Merton jump diffusion:** daily log-return truncated Poisson mixture-of-normals likelihood with bounded optimization, candidate jump threshold, AIC and convergence reporting. Fits are historical physical-measure estimates, not risk-neutral pricing parameters.
- **Heston:** method-of-moments volatility proxy for parameters, with the Feller condition reported and nonnegative full-truncation-style simulation. It is **not** a full maximum-likelihood calibration.
- **Bates:** reports independently fit component diagnostics and fallback status; it does **not** claim joint Bates likelihood calibration or a fully integrated event generator.
- **Regimes:** a deterministic-seed four-component Gaussian-mixture volatility classifier is supplied as a fallback proxy. It is not a Hamilton Markov-switching fit. Current classification is probabilistic, not deterministic forecasting.
- **Student-t copula:** aligned return ranks, Kendall-to-correlation transform, positive-definite projection, bounded degrees-of-freedom profile and asymptotic lower-tail dependence under the fitted copula assumptions.
- **Hawkes:** exponential point-process likelihood with a stationarity bound and Ogata thinning for stationary fitted parameters. Events are defined from documented return thresholds.
- **PCA absorption:** Ledoit–Wolf covariance, top-k eigenvalue absorption ratio and rolling analysis. A general Marchenko–Pastur noise-bound test is not claimed.
- **VaR and expected shortfall:** portfolio loss is negative P&L; historical ES averages losses at or beyond the empirical VaR threshold. Gaussian and Cornish–Fisher approximations have explicit distributional limitations.
- **Optimization:** long-only, normalized weights are optimized with SciPy SLSQP. Failed/infeasible solves retain current weights; the UI does not invent optimal weights.
- **Backtesting:** every VaR training window ends strictly before the realized next-period return. A simple ES calibration moment diagnostic is included; it is not a certification that forecasts are correct.
- **News:** timestamps are normalized to UTC when parseable; deduplication preserves source attributions. Sentiment does not modify a Hawkes baseline in this build; coupling is a timestamp-safe analysis-only gate.

## Data sources, caveats and licensing

Optional market sources are Yahoo Finance via `yfinance`, Finnhub, and Alpha Vantage. News sources include NewsData.io, Currents API, Finnhub, Alpha Vantage, user-provided RSS URLs, and manual text/JSON. Use each provider according to its terms, exchange-data licensing restrictions, attribution policy and rate limits. Endpoints and free quotas may change. Historical and live/quote data have different timestamps and must not be conflated. The demo CSV/JSON fixtures are supplied solely for software testing and demonstrations.

## Tests and checks

Run from the project root after installing requirements:

```bash
python -m pytest -q
python -m compileall -q app.py pages src
streamlit run app.py
```

The source test suite covers return/drawdown formulas, OHLC validation, ticker normalization, Merton candidate/fit reporting, Heston path behavior, Bates fallback reporting, regime probability normalization, copula positive-definiteness/tail dependence, Hawkes stationarity behavior, PCA absorption, VaR/ES sign convention, portfolio optimization constraints, seeded Monte Carlo, scenario schema validation, walk-forward chronology, secret masking, manual news parsing/deduplication, sentiment fallback labeling, fail-closed sentiment coupling, 3D cascade generation, reverse-stress feasibility, aggregate LTCG threshold handling, liquidity-adjusted VaR validation, and headline-heuristic labeling.

## Known limitations

This is a **functional engineering baseline, not production-ready**. The build environment could not download Streamlit or optional provider packages, had no API credentials, and had no external network/DNS. The Streamlit frontend was syntax-compiled but not launched interactively, and all external provider calls remain unverified here. The feature matrix above lists the advanced requirements that remain partial or unimplemented; no calibration or test is claimed unless it actually ran.

For production use, add provider integration tests, authentication/authorization and deployment controls, a richer audit trail, robust observability, stronger currency/calendar conventions, model-specific uncertainty estimates, full joint Bates/Hawkes/copula calibration and ablation, complete optimization constraints/costs, alert delivery, load testing, and independent quantitative review.

## Advanced risk modules added

The `src/risklens` tree also includes optional advanced modules:

- `models/hamilton_regime.py`: attempts a four-state Statsmodels Markov regression and explicitly labels the Gaussian-mixture fallback.
- `simulation/unified_event_generator.py` and `simulation/unified_engine.py`: one selected jump-arrival mechanism per run, Ogata thinning for exponential Hawkes arrivals, seeded portfolio simulation, and Monte Carlo tail metrics.
- `portfolio/institutional_optimizer.py`: mean-variance, minimum-variance, Rockafellar–Uryasev CVaR, HRP, and a conservative regime-aware allocation proxy with turnover/cost constraints.
- `risk/crisis_replay.py`: exact historical-window replay only when actual observations cover the window; otherwise an explicitly labeled hypothetical sector-beta proxy.
- `backtesting/walk_forward_suite.py`: strictly past-window forecasts with Kupiec, Christoffersen, and an ES diagnostic across labeled baselines/proxies.
- `alerts/engine.py`: configurable in-app telemetry thresholds; external webhook dispatch is deliberately not performed by this core engine.

These are research-grade implementations, not blanket claims of production readiness. In particular, the regime model reports whether Hamilton MLE converged, the integrated Monte Carlo uses simplified variance dynamics unless separately calibrated parameters are supplied, and proxy ablation outputs must not be represented as empirical validation of structural models.

Run tests from the project root with `python -m pytest -q` and syntax checks with `python -m compileall -q app.py pages src`.


## Institutional Command Suite: added differentiators

Page **7 — Contagion & Systemic Risk** now includes four interactive tabs backed by `src/risklens/risk/hackathon_differentiators.py`:

- **3D Contagion Cascade:** Student-t lower-tail dependence matrix, epicenter selection, hypothetical shock propagation through three edges/steps, and a Plotly 3D network. Spherical node positions are a deterministic visual layout. Tail dependence measures dependence under the copula assumptions; it is not causal evidence.
- **Reverse Stress Solver:** solves for a constrained joint-return vector that reaches a loss target while minimizing a regularized Mahalanobis-distance objective. It reports solver status and explicitly identifies infeasible targets. The displayed 0–100 plausibility index is a heuristic, not a probability.
- **Indian Liquidity and Rebalance Friction:** estimates delivery-equity STT, buyer-side stamp duty, an explicitly configurable exchange/SEBI fee assumption, GST on that modeled fee, and an illustrative capital-gains estimate. The ₹125,000 Section 112A LTCG threshold is applied as one aggregate allowance after user-entered prior usage, not separately per asset. Holding-period inputs are a day-count approximation. The estimate excludes cess, surcharge, loss set-off, other annual gains, several exceptions, and individualized treatment; it is not tax advice.
- **Judge Chaos Mode:** accepts a user-entered hypothetical headline and maps selected keywords to a transparent severity score, then applies a scenario multiplier to supplied VaR/ES values. It does not fetch a live breaking-news feed, run FinBERT, recalibrate a Hawkes model, or recalculate the portfolio distribution. All such output is labeled as a heuristic scenario.

The liquidity-adjusted VaR calculation uses a simple spread-plus-square-root-impact approximation and days-to-liquidate estimate. It is **not described as a universal SEBI-prescribed L-VaR formula**. The transaction-rate inputs are visible/configurable assumptions and must be checked against the trade date, segment, exchange, broker charges and applicable tax position.

### Official reference links for rate assumptions

These official sources were checked during implementation; rate rules can change and should be revalidated before real use:

- [NSE — STT and other levies](https://www.nseindia.com/static/invest/first-time-investor-sebi-turnover-fees-stt-other-levies)
- [NSE — stamp duty](https://www.nseindia.com/static/invest/first-time-investor-stamp-duty-charges-taxes)
- [Income Tax Department — tax on sale of shares](https://www.incometaxindia.gov.in/en/sale-of-shares)

The reported statutory capital-gain rates apply only to eligible transactions and taxpayers under the relevant provisions. This module is a simplified scenario estimator, not a tax return computation or legal opinion.
