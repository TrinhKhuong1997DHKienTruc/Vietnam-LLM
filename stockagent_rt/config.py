import os
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

class LLMConfig(BaseModel):
	openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
	openai_base_url: str | None = os.getenv("OPENAI_BASE_URL")
	openai_model: str = os.getenv("OPENAI_MODEL", "gpt-5-mini-2025-08-07")

	google_api_key: str | None = os.getenv("GOOGLE_API_KEY")
	gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")

class MarketKeys(BaseModel):
	finnhub_api_key: str | None = os.getenv("FINNHUB_API_KEY")
	fmp_api_key: str | None = os.getenv("FMP_API_KEY")
	sec_api_key: str | None = os.getenv("SEC_API_KEY")

class Paths(BaseModel):
	root_reports_dir: str = os.path.abspath(os.path.join(os.getcwd(), "demo_reports"))

llm_config = LLMConfig()
market_keys = MarketKeys()
paths = Paths()

os.makedirs(paths.root_reports_dir, exist_ok=True)
