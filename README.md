# RiskLens v2.0
### AI-Powered Market & Portfolio Risk Intelligence Platform

**Live Application:** https://risklensv2.streamlit.app

**GitHub Repository:** https://github.com/subhradebhazra/RiskLens

RiskLens is a Python-based quantitative financial risk analytics platform built with Streamlit. It combines market analysis, portfolio risk assessment, statistical modeling, stress testing, systemic risk research, and an integrated Agent37-powered AI chatbot within a unified interactive dashboard.

The platform is designed for financial research, education, risk exploration, and demonstration of quantitative modeling techniques.

> **Disclaimer:** RiskLens is a research and educational tool, not a trading system or personalized investment advisory service. Demo data is illustrative and should not be interpreted as current market information.

---

## Live Demo

**[Launch RiskLens](https://risklensv2.streamlit.app)**

Explore the application's market intelligence dashboards, quantitative risk models, portfolio analysis tools, and AI-powered assistance directly from your browser.

Some external data features depend on API availability, provider limits, and configuration. Unavailable sources may fall back to clearly labeled demonstration data.

---

## Key Features

### 1. Market Intelligence
- Market conditions and volatility analysis
- Stock and index exploration
- Historical return and drawdown visualization
- Market weather and regime indicators
- Optional external market data sources

### 2. Portfolio Risk Analytics
- Portfolio allocation analysis
- Value at Risk (VaR)
- Expected Shortfall (ES / CVaR)
- Annualized volatility
- Asset correlations and concentration
- Portfolio risk contributions

### 3. Advanced Quantitative Models
- Historical and parametric VaR
- Monte Carlo simulations
- Gaussian and historical-bootstrap scenarios
- Merton jump-diffusion modeling
- Heston volatility modeling and simplified Bates-related diagnostics
- Regime analysis
- Student-t copula dependence analysis
- Hawkes-process event modeling
- PCA-based absorption analysis

Advanced model outputs include assumptions, warnings, and fallbacks where appropriate. Not every model represents a fully calibrated institutional production implementation.

### 4. Stress Testing and Crisis Analysis
- Custom stress scenarios
- Hypothetical market shocks
- Historical crisis-window analysis where supported by the data
- Portfolio loss estimation
- Reverse stress testing
- Scenario comparison and research workflows

### 5. Systemic Risk and Contagion
- Tail-dependence analysis
- Systemic risk indicators
- 3D hypothetical contagion visualization
- Risk transmission scenarios
- Dependence-based financial network exploration

Visualized contagion and shock propagation are model-based scenarios, not proof of causal relationships.

### 6. Portfolio Optimization
- Minimum-variance optimization
- Mean-variance portfolio research
- Risk-adjusted portfolio comparisons
- Turnover and other configurable constraints in selected modules
- Suggested allocation analysis

Optimization results depend on modeling assumptions and are not investment recommendations.

### 7. News and Sentiment Intelligence
- Financial news aggregation
- Optional API-based news retrieval
- Financial text sentiment analysis
- Duplicate headline detection
- Fallback sentiment methods when optional models are unavailable

### 8. Agent37 AI Assistant

RiskLens includes a floating AI chatbot accessible from the bottom-right corner of the application.

The assistant is designed to help users explore:

- Financial risk concepts
- Value at Risk and Expected Shortfall
- Portfolio diversification
- Market volatility
- Quantitative models
- Risk management terminology

The chatbot integrates with an Agent37-hosted instance. The application operator can configure credentials using server-side Streamlit Secrets so visitors do not need to supply their own Agent37 key.

AI responses may be inaccurate and should be independently verified. The chatbot's availability depends on the Agent37 service, the configured instance, and usage limits.

RiskLens also retains an AI quantitative research copilot with provider-selectable briefing and scenario-proposal functionality.

---

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Primary programming language |
| Streamlit | Interactive web application |
| Pandas | Data analysis and manipulation |
| NumPy | Numerical computing |
| SciPy | Statistical modeling and optimization |
| Plotly | Interactive financial visualizations |
| Statsmodels | Statistical and econometric analysis |
| Scikit-learn | Machine learning and covariance estimation |
| Pydantic | Data and scenario validation |
| Agent37 | Hosted AI assistant integration |
| GitHub | Source code version control |
| Streamlit Community Cloud | Public application deployment |

Additional dependencies are documented in `requirements.txt` and `pyproject.toml`.

---

## Project Structure

```text
RiskLens/
├── app.py
├── pages/
│   └── ... application pages
├── src/
│   └── risklens/
│       ├── ai/
│       │   ├── gemini_assistant.py
│       │   └── floating_chat.py
│       ├── alerts/
│       ├── backtesting/
│       ├── data/
│       ├── features/
│       ├── models/
│       ├── news/
│       ├── portfolio/
│       ├── reporting/
│       ├── risk/
│       ├── simulation/
│       ├── config.py
│       └── ui.py
├── data/
│   └── demo/
├── tests/
├── .streamlit/
│   └── config.toml
├── .env.example
├── .gitignore
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Installation and Local Development

### Requirements

- Python 3.11 recommended
- Git
- Internet connection for installing dependencies
- VS Code, PyCharm, or another Python editor

### 1. Clone the repository

```bash
git clone https://github.com/subhradebhazra/RiskLens.git
cd RiskLens
```

### 2. Create a virtual environment

On Windows:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Start the application

```bash
python -m streamlit run app.py
```

Open the local address:

http://localhost:8501

RiskLens can run with bundled demo data without external API credentials. Features that require live providers or hosted AI depend on valid configuration.

---

## API Configuration

RiskLens supports optional external data and AI providers.

Possible integrations include:

- Agent37
- Google Gemini
- NewsData.io
- Currents API
- Finnhub
- Alpha Vantage
- Yahoo Finance-based market data adapters

### Agent37 Configuration

For local development, create a private file named `.streamlit/secrets.toml`:

```toml
AGENT37_API_KEY = "YOUR_PRIVATE_API_KEY"
AGENT37_INSTANCE_URL = "YOUR_HTTPS_AGENT37_INSTANCE_URL"
```

For deployment, configure these values through **Streamlit Community Cloud → App Settings → Secrets**.

**Security guidelines:**
- Never commit actual API keys to GitHub.
- Never publish `.streamlit/secrets.toml`.
- Never embed credentials in frontend HTML or JavaScript.
- Restrict AI usage appropriately when making a chatbot publicly accessible.
- Keep conversations and user-specific data isolated.
- Monitor API quotas and operational costs.

The Agent37 integration requires a configured, reachable instance. An API key alone is not sufficient.

---

## Deployment

RiskLens is deployed using Streamlit Community Cloud.

**Live URL:** https://risklensv2.streamlit.app

Deployment configuration:

```text
Repository: subhradebhazra/RiskLens
Branch: main
Main file: app.py
Python: 3.11
```

Updates pushed to the `main` branch are picked up by Streamlit Community Cloud.

For example:

```bash
git add .
git commit -m "Update RiskLens"
git push origin main
```

Environment secrets are managed separately from the public repository.

---

## Testing

The project contains automated tests covering mathematical calculations, data validation, portfolio analytics, financial models, scenarios, and application utilities.

Run:

```bash
python -m pytest -q
```

Check Python syntax:

```bash
python -m compileall -q app.py pages src
```

The availability of the deployed application does not itself certify the correctness of all numerical models or third-party integrations.

---

## Important Limitations

RiskLens includes advanced quantitative modules, but several workflows use simplified models, approximations, deterministic demonstrations, or fallback implementations.

Important limitations include:

- Demo datasets are not live market data.
- Historical estimates do not guarantee future outcomes.
- Monte Carlo results depend on model assumptions.
- Contagion estimates do not establish financial causality.
- News sentiment predictions may be inaccurate.
- AI-generated explanations are not guaranteed to be correct.
- External API features depend on provider connectivity and service availability.
- Some advanced models require further calibration and independent validation.
- Authentication, global API abuse prevention, operational monitoring, and other production controls may require further development.

The application should not be used as the sole basis for financial or investment decisions.

---

## Future Improvements

Potential development priorities include:

- Stronger chatbot abuse prevention and usage quotas
- Improved financial data coverage
- More robust real-time market integrations
- User authentication and individual workspaces
- Better portfolio import workflows
- Expanded quantitative model validation
- Additional historical crisis datasets
- Improved mobile responsiveness
- Enhanced AI-assisted risk explanations
- Stronger production monitoring and testing

---

## License

This project is distributed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Explore RiskLens

**[Open the Live RiskLens Dashboard](https://risklensv2.streamlit.app)**

**[View the GitHub Repository](https://github.com/subhradebhazra/RiskLens)**

*RiskLens v2.0 — Understand the risk behind the numbers.*
