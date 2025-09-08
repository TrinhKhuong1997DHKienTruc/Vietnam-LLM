# StockAgent RT (Python 3.13)

Real-time fetch + 14-day forecast + LLM report generation for NVDA/AAPL.

## Quickstart

### 1) Create environment (Windows PowerShell / Linux bash)
```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
pip install -r stockagent_rt/requirements.txt
```

### 2) Configure API keys
Copy `stockagent_rt/.env.example` to `.env` at repo root and fill:
- OPENAI_API_KEY and optional OPENAI_BASE_URL (for OpenAI-compatible endpoints)
- GOOGLE_API_KEY (Gemini)
- FINNHUB_API_KEY or FMP_API_KEY (market data)

### 3) Run
```bash
python -m stockagent_rt.run --symbols NVDA AAPL --history-days 365
```
Outputs are saved under `demo_reports/<SYMBOL>/`:
- `<SYMBOL>_historical.csv`
- `<SYMBOL>_forecast.csv`
- `<SYMBOL>_forecast_plot.png`
- `<SYMBOL>_report.txt`

## Notes
- If both OpenAI and Gemini keys are present, OpenAI-compatible is preferred.
- If Finnhub fails or is missing, FMP is used.
- Forecast uses Holt-Winters with ARIMA fallback.
