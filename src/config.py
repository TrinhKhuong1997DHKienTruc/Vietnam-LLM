import os
from dataclasses import dataclass


@dataclass
class APIKeys:
	finnhub_api_key: str | None
	fmp_api_key: str | None
	sec_api_key: str | None


@dataclass
class LLMProvider:
	model: str | None
	api_key: str | None
	base_url: str | None


@dataclass
class Settings:
	api_keys: APIKeys
	llm: LLMProvider
	data_dir: str
	reports_dir: str
	models_dir: str
	ticker_symbols: list[str]


DEFAULT_TICKERS = ["MSFT", "NVDA"]


def load_settings() -> Settings:
	api_keys = APIKeys(
		finnhub_api_key=os.getenv("FINNHUB_API_KEY"),
		fmp_api_key=os.getenv("FMP_API_KEY"),
		sec_api_key=os.getenv("SEC_API_KEY"),
	)
	llm = LLMProvider(
		model=os.getenv("LLM_MODEL"),
		api_key=os.getenv("LLM_API_KEY"),
		base_url=os.getenv("LLM_BASE_URL"),
	)
	project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
	data_dir = os.path.join(project_root, "data")
	reports_dir = os.path.join(project_root, "reports")
	models_dir = os.path.join(project_root, "models")
	os.makedirs(data_dir, exist_ok=True)
	os.makedirs(reports_dir, exist_ok=True)
	os.makedirs(models_dir, exist_ok=True)
	return Settings(
		api_keys=api_keys,
		llm=llm,
		data_dir=data_dir,
		reports_dir=reports_dir,
		models_dir=models_dir,
		ticker_symbols=os.getenv("TICKERS", ",".join(DEFAULT_TICKERS)).split(","),
	)