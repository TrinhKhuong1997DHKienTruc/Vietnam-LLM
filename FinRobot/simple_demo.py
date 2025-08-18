#!/usr/bin/env python3
"""
Simple FinRobot Demo - Direct API Usage
This script demonstrates basic FinRobot functionality without complex workflows
"""

import json
import requests
from datetime import datetime, timedelta

def get_finnhub_data(symbol, api_key):
    """Get basic company data from Finnhub"""
    try:
        # Company profile
        profile_url = f"https://finnhub.io/api/v1/stock/profile2?symbol={symbol}&token={api_key}"
        profile_response = requests.get(profile_url)
        profile_data = profile_response.json()
        
        # Company news
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        news_url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from={start_date.strftime('%Y-%m-%d')}&to={end_date.strftime('%Y-%m-%d')}&token={api_key}"
        news_response = requests.get(news_url)
        news_data = news_response.json()
        
        # Stock quote
        quote_url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={api_key}"
        quote_response = requests.get(quote_url)
        quote_data = quote_response.json()
        
        return {
            'profile': profile_data,
            'news': news_data[:5] if news_data else [],  # Last 5 news items
            'quote': quote_data
        }
    except Exception as e:
        print(f"Error fetching data from Finnhub: {e}")
        return None

def get_fmp_data(symbol, api_key):
    """Get financial data from FMP"""
    try:
        # Basic financial metrics
        metrics_url = f"https://financialmodelingprep.com/api/v3/quote/{symbol}?apikey={api_key}"
        metrics_response = requests.get(metrics_url)
        metrics_data = metrics_response.json()
        
        return metrics_data[0] if metrics_data else None
    except Exception as e:
        print(f"Error fetching data from FMP: {e}")
        return None

def analyze_stock(symbol, finnhub_key, fmp_key):
    """Analyze stock data and provide insights"""
    print(f"=== Stock Analysis for {symbol} ===")
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 50)
    
    # Get data from APIs
    finnhub_data = get_finnhub_data(symbol, finnhub_key)
    fmp_data = get_fmp_data(symbol, fmp_key)
    
    if not finnhub_data:
        print("Failed to fetch data from Finnhub")
        return
    
    # Display company profile
    profile = finnhub_data['profile']
    if profile:
        print(f"Company: {profile.get('name', 'N/A')}")
        print(f"Industry: {profile.get('finnhubIndustry', 'N/A')}")
        print(f"Market Cap: ${profile.get('marketCapitalization', 0):,.0f}")
        print(f"Country: {profile.get('country', 'N/A')}")
        print()
    
    # Display current stock info
    quote = finnhub_data['quote']
    if quote:
        print(f"Current Price: ${quote.get('c', 0):.2f}")
        print(f"Previous Close: ${quote.get('pc', 0):.2f}")
        change = quote.get('c', 0) - quote.get('pc', 0)
        change_pct = (change / quote.get('pc', 1)) * 100 if quote.get('pc', 0) != 0 else 0
        print(f"Change: ${change:.2f} ({change_pct:+.2f}%)")
        print(f"High: ${quote.get('h', 0):.2f}")
        print(f"Low: ${quote.get('l', 0):.2f}")
        print()
    
    # Display financial metrics from FMP
    if fmp_data:
        print("Financial Metrics:")
        print(f"PE Ratio: {fmp_data.get('pe', 'N/A')}")
        print(f"Forward PE: {fmp_data.get('forwardPE', 'N/A')}")
        print(f"PEG Ratio: {fmp_data.get('pegRatio', 'N/A')}")
        print(f"Price to Book: {fmp_data.get('priceToBook', 'N/A')}")
        print(f"Enterprise Value: ${fmp_data.get('enterpriseValue', 0):,.0f}")
        print()
    
    # Display recent news
    news = finnhub_data['news']
    if news:
        print("Recent News:")
        for i, article in enumerate(news[:3], 1):
            print(f"{i}. {article.get('headline', 'No headline')}")
            print(f"   Date: {article.get('datetime', 'N/A')}")
            print(f"   Source: {article.get('source', 'N/A')}")
            print()
    
    # Simple analysis and prediction
    print("=== Analysis & Prediction ===")
    
    # Analyze price movement
    if quote and quote.get('pc', 0) > 0:
        current_price = quote.get('c', 0)
        prev_close = quote.get('pc', 0)
        daily_change = current_price - prev_close
        daily_change_pct = (daily_change / prev_close) * 100
        
        print(f"Today's Performance: {daily_change_pct:+.2f}%")
        
        # Simple trend analysis
        if daily_change_pct > 2:
            trend = "bullish"
            prediction = "likely to continue upward momentum"
        elif daily_change_pct < -2:
            trend = "bearish"
            prediction = "may face continued downward pressure"
        else:
            trend = "neutral"
            prediction = "likely to remain stable with moderate movement"
        
        print(f"Trend: {trend}")
        print(f"Next Week Prediction: {prediction}")
        
        # Generate percentage prediction
        if trend == "bullish":
            weekly_prediction = f"+{min(5, daily_change_pct + 2):.1f}% to +{min(8, daily_change_pct + 4):.1f}%"
        elif trend == "bearish":
            weekly_prediction = f"-{min(5, abs(daily_change_pct) + 2):.1f}% to -{min(8, abs(daily_change_pct) + 4):.1f}%"
        else:
            weekly_prediction = f"-2% to +3%"
        
        print(f"Expected Weekly Movement: {weekly_prediction}")
    
    print("-" * 50)

def main():
    """Main function to run the demo"""
    print("=== FinRobot Simple Stock Analysis Demo ===")
    print("This demo shows basic stock analysis capabilities")
    print()
    
    # Load API keys
    try:
        with open('config_api_keys', 'r') as f:
            config = json.load(f)
        
        finnhub_key = config.get('FINNHUB_API_KEY')
        fmp_key = config.get('FMP_API_KEY')
        
        if not finnhub_key or not fmp_key:
            print("Error: Missing API keys in config_api_keys file")
            return
        
    except Exception as e:
        print(f"Error loading config: {e}")
        return
    
    # Analyze MSFT
    print("Running MSFT analysis...")
    analyze_stock('MSFT', finnhub_key, fmp_key)
    
    print("\n" + "="*60 + "\n")
    
    # Analyze NVDA
    print("Running NVDA analysis...")
    analyze_stock('NVDA', finnhub_key, fmp_key)
    
    print("\nDemo completed successfully!")

if __name__ == "__main__":
    main()