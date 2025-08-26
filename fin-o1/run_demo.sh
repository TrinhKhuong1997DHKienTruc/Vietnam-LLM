#!/usr/bin/env bash
set -euo pipefail

# Create and activate venv if missing
if [[ ! -d .venv ]]; then
	if command -v python3.13 >/dev/null 2>&1; then
		python3.13 -m venv .venv
	else
		python3 -m venv .venv
	fi
fi
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt

python fin_o1_demo.py "$@"
