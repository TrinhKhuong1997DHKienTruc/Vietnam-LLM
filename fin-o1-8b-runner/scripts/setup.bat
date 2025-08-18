@echo off
setlocal ENABLEDELAYEDEXECUTION

REM Cross-platform-ish setup for Windows (Python 3.14 ready)
REM - Creates .venv
REM - Installs base deps
REM - Installs torch with best-effort strategy (GPU -> CPU -> nightly)

set PROJ_DIR=%~dp0..\
pushd %PROJ_DIR%

where py >NUL 2>&1
if %ERRORLEVEL% NEQ 0 (
  echo [setup] Python launcher not found. Install Python 3.14 (or 3.10–3.13) from python.org.
  exit /b 1
)

REM Prefer py -3.14, fall back to py -3
for /f "tokens=*" %%a in ('py -3.14 -c "import sys; print(sys.version)" 2^>NUL') do set PYV=%%a
if defined PYV (
  set PY=py -3.14
) else (
  set PY=py -3
)

echo [setup] Using %PY%

if not exist .venv (
  %PY% -m venv .venv
  if %ERRORLEVEL% NEQ 0 (
    echo [setup] python -m venv failed. Installing virtualenv...
    %PY% -m pip install --user virtualenv
    %PY% -m virtualenv .venv
  )
)

if not exist .venv\Scripts\activate.bat (
  echo [setup] Could not find venv activation script
  exit /b 1
)

call .venv\Scripts\activate.bat

python -m pip install --upgrade pip wheel setuptools
echo [setup] Installing base Python dependencies...
python -m pip install -r requirements.txt

if "%SKIP_TORCH_INSTALL%"=="1" (
  echo [setup] Skipping torch install because SKIP_TORCH_INSTALL=1
  exit /b 0
)

echo [setup] Installing torch (best effort for Python 3.14)...

REM Try CUDA first; if fail, CPU; then nightly CPU
python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
if %ERRORLEVEL% NEQ 0 (
  echo [setup] CUDA wheel failed; trying CPU wheel...
  python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
)
if %ERRORLEVEL% NEQ 0 (
  echo [setup] Stable CPU torch failed; trying nightly CPU wheel (may be needed for Python 3.14)...
  python -m pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cpu
)

python -c "import sys; import importlib; \
torch = importlib.import_module('torch'); \
print(f'[setup] Torch ready: {torch.__version__}')" || (
  echo [setup] Torch import failed. You can retry later or install a supported Python (3.12/3.13). If you plan to use the Inference API only, you can skip torch.
  exit /b 1
)

echo [setup] Done. Activate with: .venv\Scripts\activate.bat
popd

