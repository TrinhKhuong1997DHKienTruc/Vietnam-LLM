#!/usr/bin/env python3
"""
Simple FinRobot Market Forecaster Demo
Based on the tutorial approach
"""

import os
import sys
import json
from datetime import datetime

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def run_simple_forecaster():
    """Run a simple version of the forecaster demo."""
    
    print("FinRobot Market Forecaster Demo")
    print("=" * 50)
    print(f"Starting analysis at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    
    # List of companies to analyze
    companies = [
        ("MSFT", "Microsoft Corporation"),
        ("NVDA", "NVIDIA Corporation")
    ]
    
    results = {}
    
    for symbol, name in companies:
        print(f"\n{'='*80}")
        print(f"ANALYZING {name} ({symbol})")
        print(f"{'='*80}")
        
        try:
            # Import here to avoid issues
            import autogen
            from finrobot.utils import get_current_date, register_keys_from_json
            from finrobot.agents.workflow import SingleAssistant
            
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
                f"Use all the tools provided to retrieve information available for {name} ({symbol}) "
                f"upon {get_current_date()}. Analyze the positive developments and potential concerns of {name} "
                "with 2-4 most important factors respectively and keep them concise. Most factors should be inferred "
                f"from company related news. Then make a rough prediction (e.g. up/down by 2-3%) of the {name} "
                "stock price movement for next week. Provide a summary analysis to support your prediction."
            )
            
            # Run the analysis
            result = assistant.chat(message)
            results[symbol] = result
            
            print(f"\n{'='*80}")
            print(f"ANALYSIS COMPLETED FOR {name} ({symbol})")
            print(f"{'='*80}")
            
        except Exception as e:
            print(f"Error analyzing {name}: {str(e)}")
            results[symbol] = None
    
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
    run_simple_forecaster()
