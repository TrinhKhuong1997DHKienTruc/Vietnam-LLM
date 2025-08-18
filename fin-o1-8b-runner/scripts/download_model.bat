@echo off
setlocal

set PROJ_DIR=%~dp0..\
pushd %PROJ_DIR%

if exist .venv\Scripts\activate.bat (
  call .venv\Scripts\activate.bat
)

python - <<PY
import os
from huggingface_hub import snapshot_download

model_id = os.environ.get("MODEL_ID", "TheFinAI/Fin-o1-8B")
token = os.environ.get("HF_TOKEN")

print(f"[download] Downloading {model_id}...")
local_dir = snapshot_download(repo_id=model_id, token=token)
print(f"[download] Saved to: {local_dir}")
PY

popd

