import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "FinRobot"))
import autogen
from finrobot.utils import get_current_date, register_keys_from_json
from finrobot.agents.workflow import SingleAssistant

def load_llm_config():
    return {
        "config_list": autogen.config_list_from_json(
            os.path.join(os.path.dirname(__file__), "FinRobot", "OAI_CONFIG_LIST"),
            filter_dict={"model": ["gpt-5-mini-2025-08-07"]},
        ),
        "timeout": 120,
        "temperature": 0,
    }

def setup_api_keys():
    register_keys_from_json(os.path.join(os.path.dirname(__file__), "FinRobot", "config_api_keys"))

def generate_forecast(ticker: str, llm_config: dict, output_dir: str = "report"):
    os.makedirs(output_dir, exist_ok=True)
    assistant = SingleAssistant(
        "Market_Analyst",
        llm_config,
        human_input_mode="NEVER",
    )
    prompt = (
        f"Use all the tools provided to retrieve information available for {ticker} upon {get_current_date()}. "
        "Analyze the positive developments and potential concerns of {ticker} with 2-4 most important factors respectively and keep them concise. "
        "Most factors should be inferred from company-related news. "
        f"Then make a rough prediction (e.g., up/down by 2-3%) of the {ticker} stock price movement for next week. "
        "Provide a summary analysis to support your prediction."
    )
    response = assistant.chat(prompt)
    report_path = os.path.join(output_dir, f"{ticker}_forecast_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(str(response))
    print(f"[INFO] Generated report: {report_path}")


def main():
    llm_config = load_llm_config()
    setup_api_keys()
    for ticker in ["MSFT", "NVDA"]:
        generate_forecast(ticker, llm_config)

if __name__ == "__main__":
    main()