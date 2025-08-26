@echo off
setlocal

REM Create and activate venv if missing
if not exist .venv (
	py -3.13 -m venv .venv || py -3.12 -m venv .venv
)
call .venv\Scripts\activate.bat

python -m pip install --upgrade pip
pip install -r requirements.txt

python fin_o1_demo.py %*

endlocal
