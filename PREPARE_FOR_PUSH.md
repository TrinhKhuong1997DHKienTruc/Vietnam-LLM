Preparing this branch for public sharing

- Ensure no secrets are committed:
  - Remove `.env` (already gitignored).
  - Verify `LLM_API_KEY` and `LLM_BASE_URL` are NOT present in the repo.

- User instructions:
  1) Create venv and install requirements
     - python3 -m venv .venv
     - source .venv/bin/activate
     - pip install -r requirements.txt
  2) Copy env example and set keys (FMP is required):
     - cp .env.example .env
     - Edit `.env` with your keys
  3) Run forecasts:
     - python demo_forecast_next_week.py --symbols MSFT,NVDA --out_dir reports

- Cross-platform notes:
  - Windows PowerShell:
    - py -3 -m venv .venv
    - .venv\Scripts\Activate.ps1
    - pip install -r requirements.txt
    - python demo_forecast_next_week.py --symbols MSFT,NVDA --out_dir reports

- Artifacts:
  - `reports/MSFT_forecast_next_week.csv`
  - `reports/NVDA_forecast_next_week.csv`
  - `reports/summary.json`