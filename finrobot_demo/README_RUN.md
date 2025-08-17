Quick Start: FinRobot Market Forecaster (MSFT, NVDA)

Requirements
- Python 3.13.x (Linux or Windows)
- Git

Setup
1) Create venv and activate
   Linux/macOS:
     python3 -m venv .venv
     source .venv/bin/activate
   Windows (PowerShell):
     py -3.13 -m venv .venv
     .venv\\Scripts\\Activate.ps1

2) Install from source
     pip install -e .

3) Configure API keys
   - Copy `OAI_CONFIG_LIST_sample` to `OAI_CONFIG_LIST`, remove comments, and fill with your own model/api_key/base_url if needed
   - Copy `config_api_keys_sample` to `config_api_keys` and fill: `FINNHUB_API_KEY`, `FMP_API_KEY`, `SEC_API_KEY`

Run the demo
   python simple_market_forecaster.py

Outputs
   - report/forecast_MSFT.txt
   - report/forecast_NVDA.txt

Notes
- The script will try OpenAI chat.completions first. If your endpoint blocks or rate-limits, it will fall back to a structured report generated from Finnhub news and yfinance price momentum.
- For advanced agent workflows (AutoGen style), Python 3.13 requires newer `autogen-agentchat`/`autogen-core`. The included compatibility layer is best-effort and not guaranteed. The simple runner is recommended.
