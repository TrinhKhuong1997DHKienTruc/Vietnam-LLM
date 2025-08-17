#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

# shellcheck disable=SC1091
source .venv/bin/activate
export HF_HUB_ENABLE_HF_TRANSFER=1

python gradio_app.py