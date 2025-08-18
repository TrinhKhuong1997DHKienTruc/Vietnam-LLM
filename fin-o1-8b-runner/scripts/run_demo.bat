@echo off
setlocal

set PROJ_DIR=%~dp0..\
pushd %PROJ_DIR%

if exist .venv\Scripts\activate.bat (
  call .venv\Scripts\activate.bat
)

set PYTHONUNBUFFERED=1
python demo.py %*

popd

