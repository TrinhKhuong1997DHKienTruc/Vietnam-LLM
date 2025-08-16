import os
import sys

# Ensure FinRobot package path is in sys.path
repo_path = os.path.join(os.path.dirname(__file__), 'FinRobot')
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)

import autogen
from finrobot.utils import get_current_date, register_keys_from_json
from finrobot.agents.workflow import SingleAssistant


def generate_report(ticker: str, output_dir: str = 'reports'):
    """Generate a market forecast report for a given ticker and save to a txt file."""
    os.makedirs(output_dir, exist_ok=True)

    # Load OpenAI configuration
    llm_config = {
        "config_list": autogen.config_list_from_json(
            os.path.join(repo_path, 'OAI_CONFIG_LIST'),
            filter_dict={"model": ["gpt-5-mini-2025-08-07"]},
        ),
        "timeout": 120,
        "temperature": 0,
    }

    # Register external API keys
    register_keys_from_json(os.path.join(repo_path, 'config_api_keys'))

    assistant = SingleAssistant(
        "Market_Analyst",
        llm_config,
        human_input_mode="NEVER",
    )

    prompt = (
        f"Use all the tools provided to retrieve information available for {ticker} upon {get_current_date()}. "
        f"Analyze the positive developments and potential concerns of {ticker} with 2-4 most important factors respectively and keep them concise. "
        "Most factors should be inferred from company related news. "
        f"Then make a rough prediction (e.g. up/down by 2-3%) of the {ticker} stock price movement for next week. Provide a summary analysis to support your prediction."
    )

    # Capture stdout to write to file
    from io import StringIO
    import contextlib

    buffer = StringIO()
    with contextlib.redirect_stdout(buffer):
        assistant.chat(prompt)

    report_content = buffer.getvalue()
    file_path = os.path.join(output_dir, f"forecast_{ticker}.txt")
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    print(f"Saved report for {ticker} to {file_path}")


if __name__ == "__main__":
    for symbol in ["MSFT", "NVDA"]:
        generate_report(symbol)