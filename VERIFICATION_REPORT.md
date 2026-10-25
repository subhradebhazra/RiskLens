# RiskLens v2.0 Verification Report

## Checks run after differentiator integration

- `PYTHONPATH=src python -m compileall -q app.py pages src` — passed.
- `PYTHONPATH=src python -m pytest -q` — **36 passed**.
- Added six deterministic tests for tail-dependence matrix validity, 3-hop cascade and Plotly figure construction, reverse-stress target constraint/infeasibility handling, aggregate LTCG exemption calculation, liquidity-adjusted VaR input validation, and hypothetical headline heuristic labeling/Hawkes stationarity.
- Exercised the differentiator functions against the bundled demo asset-price fixture. Tail matrix, network plot, reverse-stress solution, rebalance-friction estimate, liquidity add-on, and headline scenario returned finite outputs for the tested example.

## Important limitations

- Streamlit was not installed in the build environment, and package installation could not be completed due to network/DNS restrictions. A browser/UI smoke test and live external-provider tests were not run. Page code was syntax-compiled.
- The 3D network uses Student-t lower-tail dependence as an edge weight and a deterministic spherical visual layout. The cascade is a hypothetical propagation rule, not an empirically calibrated causal contagion model.
- Reverse stress minimizes a regularized covariance Mahalanobis objective under per-asset return bounds. Its 0–100 plausibility index is an uncalibrated heuristic, not a probability. Infeasibility/optimization failures are surfaced instead of silently claiming a solution.
- Indian transaction and tax calculations are illustrative. STT/stamp assumptions are cash-equity delivery rates; the combined exchange/SEBI fee input is an explicit modeling assumption, not a universal broker bill. The aggregate LTCG threshold is modeled once based on user-entered prior usage. Cess, surcharge, loss set-off, annual aggregation beyond the provided assumption, and individualized tax rules are excluded.
- The liquidity-adjusted VaR is a spread-plus-square-root-impact add-on estimate, not a universal SEBI-prescribed formula. ADTV and impact assumptions must be accurate for the actual asset and order size.
- Judge Chaos Mode is interactive but hypothetical. It uses a keyword severity heuristic—not FinBERT or another fitted NLP model—and adjusts supplied VaR/ES numbers by deterministic multipliers. It does not connect to live news, recalibrate Hawkes, or rerun the full portfolio simulation.
- The Hamilton module attempts Statsmodels MarkovRegression; if convergence fails, it returns an explicitly labeled Gaussian-mixture proxy, not a custom Hamilton EM/Kim smoother.
- The unified Monte Carlo has simplified Euler variance dynamics; it must not be described as fully calibrated Bates merely because the selected mode includes jumps and dependence.
- The walk-forward ablation labels jump/heavy-tail methods as proxies; it does not fit every structural model independently at each forecast origin.
- Exact historical crisis replay is used only when supplied prices cover the window. Otherwise, it returns a hypothetical sector-beta proxy.
- Alert engine evaluates in-app telemetry only; external webhook dispatch is not implemented in this module.
- Live data and Gemini/news-provider connections remain unverified.

No verified empirical benchmark output was included in the supplied text, so none has been fabricated or added to documentation.
