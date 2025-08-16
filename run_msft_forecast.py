import sys, os
sys.path.append('/workspace/FinRobot')

# Alias autogen package name
import importlib
try:
    import autogen  # type: ignore
except ModuleNotFoundError:
    import autogen_agentchat as _ag
    sys.modules['autogen'] = _ag
    import autogen  # now resolved

from finrobot.utils import get_current_date, register_keys_from_json
from finrobot.agents.workflow import SingleAssistant

# Configure LLM
llm_config = {
    "config_list": autogen.config_list_from_json(
        "/workspace/FinRobot/OAI_CONFIG_LIST",
        filter_dict={"model": ["gpt-5-mini-2025-08-07"]},
    ),
    "timeout": 120,
    "temperature": 0,
}

# Register data source API keys
register_keys_from_json("/workspace/FinRobot/config_api_keys")

company = "MSFT"

assistant = SingleAssistant(
    "Market_Analyst",
    llm_config,
    human_input_mode="NEVER",  # set to "ALWAYS" if you want to chat interactively
)

response = assistant.chat(
    f"Use all the tools provided to retrieve information available for {company} upon {get_current_date()}. "
    "Analyze the positive developments and potential concerns of {company} with 2-4 most important factors "
    "respectively and keep them concise. Most factors should be inferred from company related news. "
    f"Then make a rough prediction (e.g. up/down by 2-3%) of the {company} stock price movement for next week. Provide a summary analysis to support your prediction."
)

print("\n=== Market Forecaster Output ===\n")
print(response)