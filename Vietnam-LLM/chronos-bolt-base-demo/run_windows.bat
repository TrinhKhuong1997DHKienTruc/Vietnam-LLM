@echo off
setlocal

REM Navigate to repo root from this script location
cd /d "%~dp0..\..\"

if not exist .venv (
	python -m venv .venv
)

call .venv\Scripts\activate
pip install -r Vietnam-LLM\chronos-bolt-base-demo\requirements.txt
python Vietnam-LLM\chronos-bolt-base-demo\forecast_stocks.py --tickers NVDA,AAPL --prediction_length 14 --history_days 1095

endlocal
pause
