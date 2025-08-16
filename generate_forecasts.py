#!/usr/bin/env python3

"""Generate market forecast reports for MSFT and NVDA for next week using FinRobot SingleAssistant.
This script assumes FinRobot repo is present under ./FinRobot and that OAI_CONFIG_LIST and config_api_keys
have been filled with valid credentials. The outputs are saved under ./FinRobot/report directory.
"""

import json
import os
import sys
from datetime import datetime

# Ensure FinRobot package path is available
WORKSPACE_DIR = os.path.abspath(os.path.dirname(__file__))
FINROBOT_SRC = os.path.join(WORKSPACE_DIR, "FinRobot")
if FINROBOT_SRC not in sys.path:
    sys.path.insert(0, FINROBOT_SRC)

import autogen  # type: ignore
from finrobot.utils import get_current_date, register_keys_from_json
from finrobot.agents.workflow import SingleAssistant


def forecast_ticker(ticker: str, output_dir: str) -> str:
    """Generate forecast for a single ticker and save result to text file.

    Returns path to generated report file.
    """
    # Prepare llm config
    llm_config = {
        "config_list": autogen.config_list_from_json(
            os.path.join(FINROBOT_SRC, "OAI_CONFIG_LIST"),
            filter_dict={"model": ["gpt-5-mini-2025-08-07"]},
        ),
        "timeout": 120,
        "temperature": 0,
    }

    # Register API keys for data sources
    register_keys_from_json(os.path.join(FINROBOT_SRC, "config_api_keys"))

    assistant = SingleAssistant(
        "Market_Analyst",
        llm_config,
        human_input_mode="NEVER",  # no interactive
    )

    prompt = (
        f"Use all the tools provided to retrieve information available for {ticker} upon {get_current_date()}. "
        "Analyze the positive developments and potential concerns of {ticker} with 2-4 most important factors respectively and keep them concise. "
        "Most factors should be inferred from company related news. "
        f"Then make a rough prediction (e.g. up/down by 2-3%) of the {ticker} stock price movement for next week. "
        "Provide a summary analysis to support your prediction."
    )

    result = assistant.chat(prompt)

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    file_path = os.path.join(output_dir, f"{ticker}_forecast_{timestamp}.txt")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(result)

    print(f"Generated forecast for {ticker}: {file_path}")
    return file_path


def main():
    output_dir = os.path.join(FINROBOT_SRC, "report")
    tickers = ["MSFT", "NVDA"]
    report_files = []
    for ticker in tickers:
        try:
            report_files.append(forecast_ticker(ticker, output_dir))
        except Exception as e:
            print(f"Failed to generate report for {ticker}: {e}")

    # Summarize
    print("\nGenerated reports:")
    for p in report_files:
        print(" -", p)


if __name__ == "__main__":
    main()