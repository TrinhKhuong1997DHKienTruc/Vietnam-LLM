#!/usr/bin/env bash
set -euo pipefail

GPU=${1:-cpu}

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

if [[ "$GPU" == "gpu" ]]; then
  echo "Installing PyTorch GPU (adjust per selector if needed)"
  python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
else
  echo "Installing PyTorch CPU"
  python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
fi

python -m pip install -r requirements.txt
echo "Done. Activate with: source .venv/bin/activate"

