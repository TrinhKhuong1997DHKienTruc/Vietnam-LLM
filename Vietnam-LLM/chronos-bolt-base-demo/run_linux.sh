#!/usr/bin/env bash
set -euo pipefail

# Move to repo root relative to this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

if [ ! -d .venv ]; then
	python3 -m venv .venv
fi

source .venv/bin/activate
pip install -r Vietnam-LLM/chronos-bolt-base-demo/requirements.txt
python Vietnam-LLM/chronos-bolt-base-demo/forecast_stocks.py --tickers NVDA,AAPL --prediction_length 14 --history_days 1095
