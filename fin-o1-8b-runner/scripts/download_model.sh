#!/usr/bin/env bash
set -euo pipefail

here_dir="$(cd "$(dirname "$0")" && pwd)"
proj_dir="$(cd "$here_dir/.." && pwd)"
cd "$proj_dir"

MODEL_ID=${MODEL_ID:-TheFinAI/Fin-o1-8B}

if [ -f .venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

python - <<PY
import os
from huggingface_hub import snapshot_download

model_id = os.environ.get("MODEL_ID", "TheFinAI/Fin-o1-8B")
token = os.environ.get("HF_TOKEN")

print(f"[download] Downloading {model_id}...")
local_dir = snapshot_download(repo_id=model_id, token=token)
print(f"[download] Saved to: {local_dir}")
PY

