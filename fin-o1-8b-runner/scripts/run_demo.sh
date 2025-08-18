#!/usr/bin/env bash
set -euo pipefail

here_dir="$(cd "$(dirname "$0")" && pwd)"
proj_dir="$(cd "$here_dir/.." && pwd)"

cd "$proj_dir"

if [ -f .venv/bin/activate ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

export PYTHONUNBUFFERED=1

python demo.py "$@"

