import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "FinRobot"))

import autogen
from finrobot.utils import get_current_date, register_keys_from_json
from finrobot.agents.workflow import SingleAssistant


def build_llm_config():
    """Return llm_config using the modified OAI_CONFIG_LIST file."""
    return {
        "config_list": autogen.config_list_from_json(
            "FinRobot/OAI_CONFIG_LIST",  # path relative to script
            filter_dict={"model": ["gpt-5-mini-2025-08-07"]},
        ),
        "timeout": 120,
        "temperature": 0,
    }


def forecast(ticker: str, output_path: str):
    """Run the Market Forecaster Agent for a given ticker and write output to file."""
    llm_config = build_llm_config()

    # Register API keys for data sources
    register_keys_from_json("FinRobot/config_api_keys")

    assistant = SingleAssistant(
        "Market_Analyst",
        llm_config,
        human_input_mode="NEVER",
    )

    prompt = (
        f"Use all the tools provided to retrieve information available for {ticker} upon {get_current_date()}. "
        f"Analyze the positive developments and potential concerns of {ticker} with 2-4 most important factors respectively and keep them concise. "
        "Most factors should be inferred from company related news. "
        f"Then make a rough prediction (e.g. up/down by 2-3%) of the {ticker} stock price movement for next week. "
        "Provide a summary analysis to support your prediction."
    )

    # Capture stdout
    from contextlib import redirect_stdout
    import io

    buf = io.StringIO()
    with redirect_stdout(buf):
        assistant.chat(prompt, use_cache=False)

    report_content = buf.getvalue()

    with open(output_path, "w") as f:
        f.write(report_content)

    print(f"Report for {ticker} saved to {output_path}")


if __name__ == "__main__":
    forecast("MSFT", "MSFT_forecast_report.txt")
    forecast("NVDA", "NVDA_forecast_report.txt")