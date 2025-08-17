$ErrorActionPreference = "Stop"

$py = Get-Command py -ErrorAction SilentlyContinue
if (-not $py) { $py = Get-Command python -ErrorAction SilentlyContinue }
if (-not $py) { throw "Python not found. Install Python 3.13 or 3.12 and try again." }

& $py.Path -m venv .venv
& .\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt

python run_fin_o1_demo.py @args