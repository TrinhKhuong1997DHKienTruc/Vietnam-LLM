#!/usr/bin/env bash
set -euo pipefail

if command -v python3 >/dev/null 2>&1; then
	PY=python3
else
	PY=python
fi

$PY -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt

python run_fin_o1_demo.py "$@"