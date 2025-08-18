#!/usr/bin/env bash
set -euo pipefail

# Cross-platform setup for Python 3.14 (works for 3.10–3.13 too)
# - Creates .venv
# - Installs base deps
# - Installs torch with best-effort strategy (GPU → CPU → nightly)

here_dir="$(cd "$(dirname "$0")" && pwd)"
proj_dir="$(cd "$here_dir/.." && pwd)"

cd "$proj_dir"

python_bin="${PYTHON_BIN:-python3}"

echo "[setup] Using Python: $($python_bin --version 2>/dev/null || echo 'not found')"

if ! command -v "$python_bin" >/dev/null 2>&1; then
    echo "[setup] Python not found. Please install Python 3.14 (or 3.10–3.13) and re-run."
    exit 1
fi

# Create venv
need_new_venv=0
if [ -d .venv ]; then
    if [ ! -f .venv/bin/activate ]; then
        echo "[setup] Existing venv is incomplete. Recreating..."
        rm -rf .venv
        need_new_venv=1
    fi
else
    need_new_venv=1
fi

if [ "$need_new_venv" = "1" ]; then
    set +e
    "$python_bin" -m venv .venv
    rc=$?
    set -e
    if [ $rc -ne 0 ]; then
        echo "[setup] python -m venv failed (likely ensurepip missing). Installing virtualenv..."
        set +e
        "$python_bin" -m pip install --user --upgrade virtualenv
        rc_install=$?
        if [ $rc_install -ne 0 ]; then
            echo "[setup] pip refused due to externally managed environment; retrying with --break-system-packages"
            "$python_bin" -m pip install --user --break-system-packages --upgrade virtualenv
        fi
        set -e
        "$python_bin" -m virtualenv .venv
    fi
fi

if [ -f .venv/bin/activate ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
else
    echo "[setup] Failed to locate venv activation script"
    exit 1
fi

python -m pip install --upgrade pip wheel setuptools

echo "[setup] Installing base Python dependencies..."
python -m pip install -r requirements.txt

if [ "${SKIP_TORCH_INSTALL:-0}" = "1" ]; then
    echo "[setup] Skipping torch install because SKIP_TORCH_INSTALL=1"
    exit 0
fi

echo "[setup] Installing torch (best effort for Python 3.14)..."

cuda_ver=""
if command -v nvidia-smi >/dev/null 2>&1; then
    # Detect CUDA runtime version (best-effort)
    cuda_str=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || true)
    # Prefer latest CUDA wheels URL. Adjust as pytorch publishes.
    # Try CUDA 12.4 wheels first, then CPU.
    echo "[setup] NVIDIA driver: ${cuda_str:-unknown}. Trying CUDA wheels first."
    set +e
    python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
    rc=$?
    set -e
    if [ $rc -ne 0 ]; then
        echo "[setup] CUDA wheel failed; trying CPU wheel..."
        set +e
        python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
        rc=$?
        set -e
        if [ $rc -ne 0 ]; then
            echo "[setup] Stable CPU torch failed; trying nightly CPU wheel (may be needed for Python 3.14)..."
            set +e
            python -m pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu
            rc=$?
            set -e
        fi
    fi
else
    echo "[setup] No NVIDIA GPU detected. Installing CPU torch..."
    set +e
    python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    rc=$?
    set -e
    if [ $rc -ne 0 ]; then
        echo "[setup] Stable CPU torch failed; trying nightly CPU wheel (may be needed for Python 3.14)..."
        set +e
        python -m pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu
        rc=$?
        set -e
    fi
fi

python - <<'PY'
import sys
try:
    import torch
    print(f"[setup] Torch ready: {torch.__version__}")
except Exception as e:
    print("[setup] Torch import failed. You can retry later or install a supported Python (3.12/3.13).\n" \
          "If you plan to use the Inference API only, you can skip torch.")
    sys.exit(1)
PY

echo "[setup] Done. Activate with: source .venv/bin/activate"

