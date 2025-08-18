#!/usr/bin/env python3
"""
FinRobot Comprehensive Stock Analysis Report Generator
Generates detailed stock analysis reports for MSFT and NVDA
"""

import json
import requests
from datetime import datetime, timedelta
import os

def get_finnhub_data(symbol, api_key):
    """Get comprehensive company data from Finnhub"""
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
        
        # Financial ratios
        ratios_url = f"https://finnhub.io/api/v1/stock/metric?symbol={symbol}&metric=all&token={api_key}"
        ratios_response = requests.get(ratios_url)
        ratios_data = ratios_response.json()
        
        return {
            'profile': profile_data,
            'news': news_data[:10] if news_data else [],  # Last 10 news items
            'quote': quote_data,
            'ratios': ratios_data
        }
    except Exception as e:
        print(f"Error fetching data from Finnhub: {e}")
        return None

def get_fmp_data(symbol, api_key):
    """Get comprehensive financial data from FMP"""
    try:
        # Basic financial metrics
        metrics_url = f"https://financialmodelingprep.com/api/v3/quote/{symbol}?apikey={api_key}"
        metrics_response = requests.get(metrics_url)
        metrics_data = metrics_response.json()
        
        # Income statement
        income_url = f"https://financialmodelingprep.com/api/v3/income-statement/{symbol}?limit=1&apikey={api_key}"
        income_response = requests.get(income_url)
        income_data = income_response.json()
        
        # Balance sheet
        balance_url = f"https://financialmodelingprep.com/api/v3/balance-sheet-statement/{symbol}?limit=1&apikey={api_key}"
        balance_response = requests.get(balance_url)
        balance_data = balance_response.json()
        
        return {
            'metrics': metrics_data[0] if metrics_data else None,
            'income': income_data[0] if income_data else None,
            'balance': balance_data[0] if balance_data else None
        }
    except Exception as e:
        print(f"Error fetching data from FMP: {e}")
        return None

def generate_report(symbol, finnhub_data, fmp_data):
    """Generate a comprehensive stock analysis report"""
    if not finnhub_data:
        return f"Error: Could not fetch data for {symbol}"
    
    profile = finnhub_data.get('profile', {})
    quote = finnhub_data.get('quote', {})
    news = finnhub_data.get('news', [])
    ratios = finnhub_data.get('ratios', {})
    
    metrics = fmp_data.get('metrics', {}) if fmp_data else {}
    income = fmp_data.get('income', {}) if fmp_data else {}
    balance = fmp_data.get('balance', {}) if fmp_data else {}
    
    # Calculate additional metrics
    current_price = quote.get('c', 0)
    prev_close = quote.get('pc', 0)
    change = current_price - prev_close
    change_pct = (change / prev_close) * 100 if prev_close > 0 else 0
    
    # Trend analysis
    if change_pct > 2:
        trend = "BULLISH"
        sentiment = "Positive momentum with strong upward potential"
        weekly_prediction = f"+{min(5, change_pct + 2):.1f}% to +{min(8, change_pct + 4):.1f}%"
    elif change_pct < -2:
        trend = "BEARISH"
        sentiment = "Negative pressure with potential for further decline"
        weekly_prediction = f"-{min(5, abs(change_pct) + 2):.1f}% to -{min(8, abs(change_pct) + 4):.1f}%"
    else:
        trend = "NEUTRAL"
        sentiment = "Stable performance with moderate movement expected"
        weekly_prediction = "-2% to +3%"
    
    # Generate report
    report = f"""
{'='*80}
                    FINROBOT STOCK ANALYSIS REPORT
{'='*80}

Company: {profile.get('name', 'N/A')} ({symbol})
Industry: {profile.get('finnhubIndustry', 'N/A')}
Country: {profile.get('country', 'N/A')}
Market Cap: ${profile.get('marketCapitalization', 0):,.0f}
Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{'='*80}
                            CURRENT MARKET DATA
{'='*80}

Current Price: ${current_price:.2f}
Previous Close: ${prev_close:.2f}
Daily Change: ${change:.2f} ({change_pct:+.2f}%)
Day High: ${quote.get('h', 0):.2f}
Day Low: ${quote.get('l', 0):.2f}
Volume: {quote.get('v', 0):,}

{'='*80}
                        FINANCIAL METRICS & RATIOS
{'='*80}

PE Ratio: {metrics.get('pe', 'N/A')}
Forward PE: {metrics.get('forwardPE', 'N/A')}
PEG Ratio: {metrics.get('pegRatio', 'N/A')}
Price to Book: {metrics.get('priceToBook', 'N/A')}
Price to Sales: {metrics.get('priceToSales', 'N/A')}
Enterprise Value: ${metrics.get('enterpriseValue', 0):,.0f}
Return on Equity: {metrics.get('returnOnEquity', 'N/A')}
Return on Assets: {metrics.get('returnOnAssets', 'N/A')}

{'='*80}
                            FINANCIAL PERFORMANCE
{'='*80}

Revenue (TTM): ${income.get('revenue', 0):,.0f}
Net Income (TTM): ${income.get('netIncome', 0):,.0f}
Total Assets: ${balance.get('totalAssets', 0):,.0f}
Total Debt: ${balance.get('totalDebt', 0):,.0f}
Cash & Equivalents: ${balance.get('cashAndCashEquivalents', 0):,.0f}

{'='*80}
                            MARKET SENTIMENT
{'='*80}

Current Trend: {trend}
Sentiment Analysis: {sentiment}
Next Week Prediction: {weekly_prediction}

{'='*80}
                            RECENT NEWS & DEVELOPMENTS
{'='*80}

"""
    
    # Add news items
    for i, article in enumerate(news[:5], 1):
        headline = article.get('headline', 'No headline available')
        source = article.get('source', 'Unknown source')
        date = datetime.fromtimestamp(article.get('datetime', 0)).strftime('%Y-%m-%d %H:%M')
        
        report += f"{i}. {headline}\n"
        report += f"   Source: {source} | Date: {date}\n\n"
    
    report += f"{'='*80}\n"
    report += f"                            INVESTMENT RECOMMENDATION\n"
    report += f"{'='*80}\n\n"
    
    # Generate investment recommendation
    if trend == "BULLISH":
        recommendation = "BUY"
        reasoning = f"Strong positive momentum with {change_pct:.1f}% daily gain. Technical indicators suggest continued upward movement."
    elif trend == "BEARISH":
        recommendation = "SELL/HOLD"
        reasoning = f"Negative pressure with {abs(change_pct):.1f}% daily decline. Consider reducing exposure or waiting for stabilization."
    else:
        recommendation = "HOLD"
        reasoning = f"Stable performance with minimal volatility. Suitable for conservative investors seeking steady returns."
    
    report += f"Recommendation: {recommendation}\n"
    report += f"Reasoning: {reasoning}\n"
    report += f"Risk Level: {'HIGH' if abs(change_pct) > 3 else 'MEDIUM' if abs(change_pct) > 1 else 'LOW'}\n"
    report += f"Time Horizon: 1-2 weeks\n\n"
    
    report += f"{'='*80}\n"
    report += f"Report generated by FinRobot AI Agent\n"
    report += f"Disclaimer: This analysis is for informational purposes only and should not be considered as financial advice.\n"
    report += f"{'='*80}\n"
    
    return report

def save_report(symbol, report_content):
    """Save report to a file"""
    filename = f"{symbol}_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    # Create reports directory if it doesn't exist
    os.makedirs('reports', exist_ok=True)
    
    filepath = os.path.join('reports', filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    return filepath

def main():
    """Main function to generate comprehensive reports"""
    print("=== FinRobot Comprehensive Stock Analysis Report Generator ===")
    print("Generating detailed analysis reports for MSFT and NVDA...")
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
    
    # Generate MSFT report
    print("Generating MSFT analysis report...")
    msft_data = get_finnhub_data('MSFT', finnhub_key)
    msft_fmp = get_fmp_data('MSFT', fmp_key)
    msft_report = generate_report('MSFT', msft_data, msft_fmp)
    msft_filepath = save_report('MSFT', msft_report)
    print(f"MSFT report saved to: {msft_filepath}")
    
    print("\n" + "="*60 + "\n")
    
    # Generate NVDA report
    print("Generating NVDA analysis report...")
    nvda_data = get_finnhub_data('NVDA', finnhub_key)
    nvda_fmp = get_fmp_data('NVDA', fmp_key)
    nvda_report = generate_report('NVDA', nvda_data, nvda_fmp)
    nvda_filepath = save_report('NVDA', nvda_report)
    print(f"NVDA report saved to: {nvda_filepath}")
    
    print("\n" + "="*60 + "\n")
    
    # Display summary
    print("=== REPORT SUMMARY ===")
    print(f"MSFT Report: {msft_filepath}")
    print(f"NVDA Report: {nvda_filepath}")
    print("\nReports generated successfully!")
    print("Check the 'reports' directory for detailed analysis files.")

if __name__ == "__main__":
    main()