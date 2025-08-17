#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

# Create venv if missing
if [ ! -d .venv ]; then
	python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install -U pip
# Install CPU Torch by default for portability
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
python -m pip install -r requirements.txt

export HF_HUB_ENABLE_HF_TRANSFER=1

PROMPT=${1:-"List 3 short key takeaways from AAPL's latest earnings."}

python fin_o1_demo.py --prompt "$PROMPT" --max-new-tokens 128