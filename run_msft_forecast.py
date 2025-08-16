import sys, os
sys.path.append('/workspace/FinRobot')

import importlib
try:
    import autogen  # if already installed
except ModuleNotFoundError:
    import autogen_agentchat as _ag
    sys.modules['autogen'] = _ag
    import autogen

# Patch cost function before any chat
# For some wrappers cost attribute may not be defined; ensure it's present
from autogen.oai.client import OpenAIWrapper

def _safe_cost(self, response):
    return 0  # skip cost calculation entirely

setattr(OpenAIWrapper, 'cost', _safe_cost)

import openai


def _client_cost(self, response):
    return 0

setattr(openai.Client, 'cost', _client_cost)
setattr(openai.OpenAI, 'cost', _client_cost)

from autogen.oai import client as _oai_client

def _client_cost2(self, response):
    return 0

setattr(_oai_client.OpenAIClient, 'cost', _client_cost2)
setattr(_oai_client.OpenAIClient, 'get_usage', lambda self, response: {'prompt_tokens':0,'completion_tokens':0,'total_tokens':0,'cost':0,'model':'n/a'})

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