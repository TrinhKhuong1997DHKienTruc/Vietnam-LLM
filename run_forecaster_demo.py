#!/usr/bin/env python3
"""
FinRobot Market Forecaster Agent Demo
This script runs the Market Forecaster Agent to predict stock movements for MSFT and NVDA.
"""

import os
import sys
import autogen
from datetime import datetime
from finrobot.utils import get_current_date, register_keys_from_json
from finrobot.agents.workflow import SingleAssistant

def run_forecaster_for_stock(company_symbol, company_name):
    """
    Run the Market Forecaster Agent for a specific stock.
    
    Args:
        company_symbol (str): Stock symbol (e.g., 'MSFT', 'NVDA')
        company_name (str): Company name for display
    """
    print(f"\n{'='*80}")
    print(f"RUNNING MARKET FORECASTER FOR {company_name} ({company_symbol})")
    print(f"{'='*80}")
    print(f"Date: {get_current_date()}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    # Configure LLM
    llm_config = {
        "config_list": autogen.config_list_from_json(
            "OAI_CONFIG_LIST",
            filter_dict={"model": ["gpt-5-mini-2025-08-07"]},
        ),
        "timeout": 120,
        "temperature": 0,
    }
    
    # Register API keys
    register_keys_from_json("config_api_keys")
    
    # Create assistant
    assistant = SingleAssistant(
        "Market_Analyst",
        llm_config,
        human_input_mode="NEVER",
    )
    
    # Create the analysis message
    message = (
        f"Use all the tools provided to retrieve information available for {company_name} ({company_symbol}) "
        f"upon {get_current_date()}. Analyze the positive developments and potential concerns of {company_name} "
        "with 2-4 most important factors respectively and keep them concise. Most factors should be inferred "
        f"from company related news. Then make a rough prediction (e.g. up/down by 2-3%) of the {company_name} "
        "stock price movement for next week. Provide a summary analysis to support your prediction."
    )
    
    # Run the analysis
    try:
        result = assistant.chat(message)
        print(f"\n{'='*80}")
        print(f"ANALYSIS COMPLETED FOR {company_name} ({company_symbol})")
        print(f"{'='*80}")
        return result
    except Exception as e:
        print(f"Error analyzing {company_name}: {str(e)}")
        return None

def main():
    """Main function to run the Market Forecaster Agent demo."""
    print("FinRobot Market Forecaster Agent Demo")
    print("=" * 50)
    print(f"Starting analysis at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    
    # List of companies to analyze
    companies = [
        ("MSFT", "Microsoft Corporation"),
        ("NVDA", "NVIDIA Corporation")
    ]
    
    results = {}
    
    # Run analysis for each company
    for symbol, name in companies:
        result = run_forecaster_for_stock(symbol, name)
        results[symbol] = result
    
    # Print summary
    print(f"\n{'='*80}")
    print("DEMO SUMMARY")
    print(f"{'='*80}")
    print(f"Completed analysis for {len(companies)} companies:")
    for symbol, name in companies:
        status = "✓ Completed" if results[symbol] else "✗ Failed"
        print(f"  - {name} ({symbol}): {status}")
    print(f"{'='*80}")
    
    return results

if __name__ == "__main__":
    main()
