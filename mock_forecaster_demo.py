#!/usr/bin/env python3
"""
Mock FinRobot Market Forecaster Demo
This script simulates the Market Forecaster Agent results for MSFT and NVDA.
"""

import os
import json
from datetime import datetime, timedelta

def generate_mock_analysis(company_symbol, company_name):
    """
    Generate mock analysis for a company.
    
    Args:
        company_symbol (str): Stock symbol (e.g., 'MSFT', 'NVDA')
        company_name (str): Company name for display
    """
    
    # Mock data for different companies
    mock_data = {
        "MSFT": {
            "current_price": 415.50,
            "positive_factors": [
                "Strong cloud computing growth with Azure",
                "AI integration across product portfolio",
                "Robust enterprise software demand",
                "Strategic partnerships in AI development"
            ],
            "concerns": [
                "Regulatory scrutiny on AI practices",
                "Competition from emerging AI startups",
                "Potential economic slowdown impact"
            ],
            "prediction": "UP 2.5%",
            "reasoning": "Microsoft's strong position in cloud computing and AI integration, combined with robust enterprise demand, suggests continued growth despite regulatory concerns."
        },
        "NVDA": {
            "current_price": 875.20,
            "positive_factors": [
                "Dominant position in AI chip market",
                "Strong demand for data center GPUs",
                "Innovation in AI hardware",
                "Strategic partnerships with major tech companies"
            ],
            "concerns": [
                "Supply chain constraints",
                "Geopolitical tensions affecting exports",
                "Potential market saturation"
            ],
            "prediction": "UP 3.2%",
            "reasoning": "NVIDIA's leadership in AI chips and strong demand from data centers and AI applications should drive continued growth, though supply chain issues remain a concern."
        }
    }
    
    data = mock_data.get(company_symbol, {})
    
    print(f"\n{'='*80}")
    print(f"MARKET FORECASTER ANALYSIS FOR {company_name} ({company_symbol})")
    print(f"{'='*80}")
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d')}")
    print(f"Current Price: ${data.get('current_price', 'N/A')}")
    print(f"{'='*80}")
    
    print("\n📈 POSITIVE DEVELOPMENTS:")
    for i, factor in enumerate(data.get('positive_factors', []), 1):
        print(f"  {i}. {factor}")
    
    print("\n⚠️  POTENTIAL CONCERNS:")
    for i, concern in enumerate(data.get('concerns', []), 1):
        print(f"  {i}. {concern}")
    
    print(f"\n🎯 PREDICTION FOR NEXT WEEK:")
    print(f"  Movement: {data.get('prediction', 'N/A')}")
    print(f"  Reasoning: {data.get('reasoning', 'N/A')}")
    
    print(f"\n{'='*80}")
    print(f"ANALYSIS COMPLETED FOR {company_name} ({company_symbol})")
    print(f"{'='*80}")
    
    return data

def main():
    """Main function to run the mock Market Forecaster Agent demo."""
    print("FinRobot Market Forecaster Agent Demo (Mock)")
    print("=" * 60)
    print(f"Starting analysis at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # List of companies to analyze
    companies = [
        ("MSFT", "Microsoft Corporation"),
        ("NVDA", "NVIDIA Corporation")
    ]
    
    results = {}
    
    # Run analysis for each company
    for symbol, name in companies:
        result = generate_mock_analysis(symbol, name)
        results[symbol] = result
    
    # Print summary
    print(f"\n{'='*80}")
    print("DEMO SUMMARY")
    print(f"{'='*80}")
    print(f"Completed analysis for {len(companies)} companies:")
    for symbol, name in companies:
        status = "✓ Completed" if results[symbol] else "✗ Failed"
        prediction = results[symbol].get('prediction', 'N/A') if results[symbol] else 'N/A'
        print(f"  - {name} ({symbol}): {status} - {prediction}")
    print(f"{'='*80}")
    
    # Save results to file
    output_file = "forecaster_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "companies": results
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return results

if __name__ == "__main__":
    main()
