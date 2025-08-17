param(
	[string]$Tickers = "MSFT,NVDA"
)

if (!(Test-Path ".venv")) {
	py -3.13 -m venv .venv
}
. .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.cli --tickers $Tickers