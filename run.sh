#!/usr/bin/env bash
set -euo pipefail

if [ ! -d .venv ]; then
	python3.13 -m venv .venv
fi
source .venv/bin/activate
pip install -r requirements.txt
python -m src.cli --tickers "${TICKERS:-MSFT,NVDA}"