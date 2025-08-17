@echo off
setlocal ENABLEDELAYEDEXPANSION
cd /d %~dp0

set PROMPT=%*
if "%PROMPT%"=="" set PROMPT=List 3 short key takeaways from AAPL's latest earnings.

set HF_HUB_ENABLE_HF_TRANSFER=1
.venv\Scripts\python fin_o1_demo.py --prompt "%PROMPT%" --max-new-tokens 128
endlocal