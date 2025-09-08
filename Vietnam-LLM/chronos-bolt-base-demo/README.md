# Chronos-Bolt Base Demo (NVDA & AAPL 14-Day Forecast)

This demo uses Amazon's `amazon/chronos-bolt-base` to forecast NVDA and AAPL close prices for the next 14 days and saves CSV/JSON/PNG reports under `demo_reports/chronos_bolt_base/`.

## Quick Start

- Python: 3.13 recommended
- OS: Windows or Linux

### 1) Install

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\pip install -r Vietnam-LLM/chronos-bolt-base-demo/requirements.txt
# Linux/macOS
source .venv/bin/activate && pip install -r Vietnam-LLM/chronos-bolt-base-demo/requirements.txt
```

If you have a CUDA GPU, install a CUDA-enabled torch per https://pytorch.org/get-started/locally/ before running.

### 2) Run forecast

```bash
python Vietnam-LLM/chronos-bolt-base-demo/forecast_stocks.py --tickers NVDA,AAPL --prediction_length 14 --history_days 1095
```

Artifacts will be written to `demo_reports/chronos_bolt_base/`:
- `<TICKER>_14day_forecast.csv`
- `<TICKER>_14day_report.json`
- `<TICKER>_14day_forecast.png`
- `combined_14day_forecasts.csv`
- `run_report.json`

### Notes
- Uses `yfinance` to fetch adjusted close history in real time.
- Defaults to GPU if available; otherwise CPU. Override with `--device cpu` or `--device cuda`.
- The `chronos-forecasting` package provides `ChronosPipeline` to run `amazon/chronos-bolt-base` directly.
