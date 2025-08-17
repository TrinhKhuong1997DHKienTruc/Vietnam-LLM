@echo off
setlocal
cd /d %~dp0

set HF_HUB_ENABLE_HF_TRANSFER=1
.venv\Scripts\python -m streamlit run streamlit_app.py --server.headless true
endlocal