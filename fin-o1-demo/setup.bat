@echo off
setlocal ENABLEDELAYEDEXPANSION
cd /d %~dp0

where py >nul 2>nul
if %errorlevel%==0 (
	set PYTHON=py -3
) else (
	set PYTHON=python
)

%PYTHON% -m venv .venv

set PIP=.venv\Scripts\python -m pip
%PIP% install -U pip
REM Install default torch (CPU-only by default if no CUDA)
%PIP% install torch --index-url https://download.pytorch.org/whl/cpu
%PIP% install -r requirements.txt

if "%1"=="--download" (
	set HF_HUB_ENABLE_HF_TRANSFER=1
	.venv\Scripts\python download_model.py --model-id TheFinAI/Fin-o1-8B
)

echo.
echo Setup complete. Activate with:
echo .venv\Scripts\activate
endlocal