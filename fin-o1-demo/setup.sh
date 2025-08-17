#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install -U pip
# Install CPU Torch by default for maximum compatibility
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
python -m pip install -r requirements.txt

# Optional: pre-download the model
if [ "${1:-}" = "--download" ]; then
	export HF_HUB_ENABLE_HF_TRANSFER=1
	python download_model.py --model-id TheFinAI/Fin-o1-8B
fi

echo "\nSetup complete. Activate with:"
echo "source .venv/bin/activate"