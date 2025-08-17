# Vietnam LSTM Stock Forecasts

This repo contains a ready-to-run LSTM demo to forecast next-week daily close prices for MSFT and NVDA.

## Quick start

- Create a Python 3.13 virtual environment
- Install dependencies
- Run the CLI to generate HTML reports under `reports/`

### Linux/macOS

```
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.cli --tickers MSFT,NVDA
```

### Windows (PowerShell)

```
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.cli --tickers MSFT,NVDA
```

## Configuration

Set optional environment variables in `.env` (not committed):

- `FINNHUB_API_KEY`, `FMP_API_KEY`, `SEC_API_KEY` (not required for Yahoo Finance)
- `LLM_MODEL`, `LLM_API_KEY`, `LLM_BASE_URL` (optional, not required)
- `TICKERS` to override default tickers

## Notes

- Dependencies pinned for Python 3.13.7 (CPU). GPU builds are out of scope.
- Reports are generated in `reports/forecast_<TICKER>.html`.
