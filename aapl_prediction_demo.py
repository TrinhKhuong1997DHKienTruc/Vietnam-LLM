#!/usr/bin/env python3
"""
AAPL Stock Price Prediction Demo using FinRobot
This script demonstrates the Market Forecaster Agent for predicting AAPL stock movements
over the next 14 days with comprehensive analysis and reporting.
"""

import os
import json
import datetime
import autogen
from finrobot.utils import get_current_date, register_keys_from_json
from finrobot.agents.workflow import SingleAssistant
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

def create_demo_folder():
    """Create demo folder structure for storing results"""
    demo_dir = "demo"
    os.makedirs(demo_dir, exist_ok=True)
    os.makedirs(f"{demo_dir}/logs", exist_ok=True)
    os.makedirs(f"{demo_dir}/outputs", exist_ok=True)
    os.makedirs(f"{demo_dir}/reports", exist_ok=True)
    return demo_dir

def get_aapl_data():
    """Fetch recent AAPL stock data for analysis"""
    try:
        ticker = yf.Ticker("AAPL")
        hist = ticker.history(period="1y")
        
        # Get recent data
        recent_data = hist.tail(30)
        
        # Calculate basic metrics
        current_price = recent_data['Close'].iloc[-1]
        price_change = recent_data['Close'].iloc[-1] - recent_data['Close'].iloc[-2]
        price_change_pct = (price_change / recent_data['Close'].iloc[-2]) * 100
        
        # Get company info
        info = ticker.info
        company_name = info.get('longName', 'Apple Inc.')
        
        return {
            'company_name': company_name,
            'current_price': current_price,
            'price_change': price_change,
            'price_change_pct': price_change_pct,
            'recent_data': recent_data,
            'volume': recent_data['Volume'].iloc[-1]
        }
    except Exception as e:
        print(f"Error fetching AAPL data: {e}")
        return None

def create_price_chart(data, demo_dir):
    """Create a price chart for AAPL"""
    try:
        plt.figure(figsize=(12, 8))
        plt.plot(data['recent_data'].index, data['recent_data']['Close'], linewidth=2, color='blue')
        plt.title(f"{data['company_name']} (AAPL) - 30-Day Price Chart", fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price (USD)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        chart_path = f"{demo_dir}/outputs/aapl_price_chart.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Price chart saved to: {chart_path}")
        return chart_path
    except Exception as e:
        print(f"Error creating price chart: {e}")
        return None

def run_aapl_prediction():
    """Run the main AAPL prediction analysis"""
    print("=" * 80)
    print("AAPL STOCK PRICE PREDICTION DEMO")
    print("Using FinRobot Market Forecaster Agent")
    print("=" * 80)
    
    # Create demo folder
    demo_dir = create_demo_folder()
    print(f"Demo folder created: {demo_dir}")
    
    # Get AAPL data
    print("\nFetching AAPL stock data...")
    aapl_data = get_aapl_data()
    
    if not aapl_data:
        print("Failed to fetch AAPL data. Exiting...")
        return
    
    print(f"Company: {aapl_data['company_name']}")
    print(f"Current Price: ${aapl_data['current_price']:.2f}")
    print(f"Price Change: ${aapl_data['price_change']:.2f} ({aapl_data['price_change_pct']:.2f}%)")
    print(f"Volume: {aapl_data['volume']:,}")
    
    # Create price chart
    print("\nCreating price chart...")
    chart_path = create_price_chart(aapl_data, demo_dir)
    
    # Configure LLM
    print("\nConfiguring AI models...")
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
    
    # Create Market Analyst Agent
    print("\nInitializing Market Analyst Agent...")
    assistant = SingleAssistant(
        "Market_Analyst",
        llm_config,
        human_input_mode="NEVER",
    )
    
    # Prepare comprehensive analysis prompt
    current_date = get_current_date()
    analysis_prompt = f"""
    Use all the tools provided to retrieve comprehensive information available for AAPL (Apple Inc.) upon {current_date}. 
    
    Please perform a detailed analysis including:
    
    1. **Current Market Data Analysis:**
       - Current price: ${aapl_data['current_price']:.2f}
       - Recent price change: ${aapl_data['price_change']:.2f} ({aapl_data['price_change_pct']:.2f}%)
       - Recent trading volume: {aapl_data['volume']:,}
    
    2. **Fundamental Analysis:**
       - Analyze Apple's financial health, revenue trends, and profitability
       - Review recent earnings reports and guidance
       - Assess competitive position and market share
    
    3. **Technical Analysis:**
       - Analyze price trends, support/resistance levels
       - Review technical indicators and momentum
       - Identify key chart patterns
    
    4. **Market Sentiment & News Analysis:**
       - Review recent news and analyst reports
       - Assess market sentiment and investor confidence
       - Identify any significant events or announcements
    
    5. **Risk Assessment:**
       - Identify potential risks and challenges
       - Consider macroeconomic factors
       - Evaluate regulatory and competitive threats
    
    6. **14-Day Price Prediction:**
       - Provide a detailed prediction for AAPL stock price movement over the next 14 days
       - Include specific price targets and confidence levels
       - Explain the reasoning behind your prediction
       - Consider both bullish and bearish scenarios
    
    Please provide a comprehensive summary analysis to support your 14-day prediction with specific price targets and reasoning.
    """
    
    print("\nRunning comprehensive market analysis...")
    print("This may take several minutes as the agent gathers and analyzes data...")
    
    # Run the analysis
    try:
        result = assistant.chat(analysis_prompt)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save text report
        report_path = f"{demo_dir}/reports/aapl_analysis_report_{timestamp}.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("AAPL STOCK ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated on: {current_date}\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Current AAPL Price: ${aapl_data['current_price']:.2f}\n")
            f.write(f"Price Change: ${aapl_data['price_change']:.2f} ({aapl_data['price_change_pct']:.2f}%)\n\n")
            f.write("ANALYSIS RESULTS:\n")
            f.write("-" * 30 + "\n")
            f.write(str(result))
        
        # Save JSON summary
        summary_data = {
            "timestamp": timestamp,
            "analysis_date": current_date,
            "company": "Apple Inc. (AAPL)",
            "current_price": aapl_data['current_price'],
            "price_change": aapl_data['price_change'],
            "price_change_pct": aapl_data['price_change_pct'],
            "volume": aapl_data['volume'],
            "analysis_result": str(result),
            "chart_path": chart_path if chart_path else None
        }
        
        summary_path = f"{demo_dir}/reports/aapl_analysis_summary_{timestamp}.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n" + "=" * 80)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"Report saved to: {report_path}")
        print(f"Summary saved to: {summary_path}")
        if chart_path:
            print(f"Chart saved to: {chart_path}")
        
        print(f"\nDemo folder contents:")
        print(f"📁 {demo_dir}/")
        print(f"  📁 logs/")
        print(f"  📁 outputs/")
        print(f"  📁 reports/")
        print(f"    📄 aapl_analysis_report_{timestamp}.txt")
        print(f"    📄 aapl_analysis_summary_{timestamp}.json")
        if chart_path:
            print(f"    📊 aapl_price_chart.png")
        
        return summary_data
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        error_path = f"{demo_dir}/logs/error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(error_path, 'w') as f:
            f.write(f"Error occurred during AAPL analysis:\n{str(e)}")
        print(f"Error logged to: {error_path}")
        return None

if __name__ == "__main__":
    print("Starting AAPL Prediction Demo...")
    print("This demo will analyze AAPL stock and predict price movements for the next 14 days.")
    print("Please ensure you have configured your API keys in OAI_CONFIG_LIST and config_api_keys files.")
    print()
    
    result = run_aapl_prediction()
    
    if result:
        print("\n✅ Demo completed successfully!")
        print("Check the demo/reports/ folder for detailed analysis results.")
    else:
        print("\n❌ Demo failed. Check the demo/logs/ folder for error details.")

