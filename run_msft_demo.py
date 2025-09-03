#!/usr/bin/env python3
"""
Market Forecaster Agent Demo for MSFT
This script runs the FinRobot Market Forecaster Agent to predict MSFT stock movements
"""

import autogen
from finrobot.utils import get_current_date, register_keys_from_json
from finrobot.agents.workflow import SingleAssistant

def main():
    print("=== FinRobot Market Forecaster Agent Demo for MSFT ===")
    print(f"Current date: {get_current_date()}")
    
    # Read OpenAI API keys from a JSON file
    llm_config = {
        "config_list": autogen.config_list_from_json(
            "OAI_CONFIG_LIST",
            filter_dict={"model": ["gpt-5-mini-2025-08-07"]},
        ),
        "timeout": 120,
        "temperature": 0,
    }

    # Register FINNHUB API keys
    register_keys_from_json("config_api_keys")
    
    print("Configuration loaded successfully")
    
    # Define the company
    company = "MSFT"
    
    print(f"Creating Market Analyst assistant for {company}...")
    
    # Create the assistant
    assistant = SingleAssistant(
        "Market_Analyst",
        llm_config,
        # set to "ALWAYS" if you want to chat instead of simply receiving the prediction
        human_input_mode="NEVER",
    )
    
    print(f"Starting analysis for {company}...")
    
    # Start the chat
    message = f"Use all the tools provided to retrieve information available for {company} upon {get_current_date()}. Analyze the positive developments and potential concerns of {company} " \
              "with 2-4 most important factors respectively and keep them concise. Most factors should be inferred from company related news. " \
              f"Then make a rough prediction (e.g. up/down by 2-3%) of the {company} stock price movement for next week. Provide a summary analysis to support your prediction."
    
    print("Sending analysis request...")
    print("-" * 50)
    
    try:
        result = assistant.chat(message)
        print("-" * 50)
        print("Analysis completed successfully!")
        return result
    except Exception as e:
        print(f"Error during analysis: {e}")
        return None

if __name__ == "__main__":
    main()