@echo off
setlocal
cd /d %~dp0

set HF_HUB_ENABLE_HF_TRANSFER=1
.venv\Scripts\python gradio_app.py
endlocal