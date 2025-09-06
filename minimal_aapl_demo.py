#!/usr/bin/env python3
"""
Minimal AAPL Stock Price Prediction Demo
This script demonstrates basic stock analysis and prediction without complex dependencies.
"""

import os
import json
import datetime
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
        
        # Calculate additional metrics
        sma_20 = recent_data['Close'].rolling(window=20).mean().iloc[-1]
        sma_50 = recent_data['Close'].rolling(window=50).mean().iloc[-1] if len(recent_data) >= 50 else None
        
        # Get company info
        info = ticker.info
        company_name = info.get('longName', 'Apple Inc.')
        market_cap = info.get('marketCap', 0)
        pe_ratio = info.get('trailingPE', 0)
        
        return {
            'company_name': company_name,
            'current_price': current_price,
            'price_change': price_change,
            'price_change_pct': price_change_pct,
            'recent_data': recent_data,
            'volume': recent_data['Volume'].iloc[-1],
            'sma_20': sma_20,
            'sma_50': sma_50,
            'market_cap': market_cap,
            'pe_ratio': pe_ratio
        }
    except Exception as e:
        print(f"Error fetching AAPL data: {e}")
        return None

def create_price_chart(data, demo_dir):
    """Create a comprehensive price chart for AAPL"""
    try:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Price chart with moving averages
        ax1.plot(data['recent_data'].index, data['recent_data']['Close'], linewidth=2, color='blue', label='Close Price')
        ax1.plot(data['recent_data'].index, data['recent_data']['Close'].rolling(window=20).mean(), 
                linewidth=1.5, color='orange', label='SMA 20', alpha=0.8)
        if data['sma_50']:
            ax1.plot(data['recent_data'].index, data['recent_data']['Close'].rolling(window=50).mean(), 
                    linewidth=1.5, color='red', label='SMA 50', alpha=0.8)
        
        ax1.set_title(f"{data['company_name']} (AAPL) - 30-Day Price Chart", fontsize=16, fontweight='bold')
        ax1.set_ylabel('Price (USD)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.tick_params(axis='x', rotation=45)
        
        # Volume chart
        ax2.bar(data['recent_data'].index, data['recent_data']['Volume'], alpha=0.7, color='green')
        ax2.set_title('Trading Volume', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Volume', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        chart_path = f"{demo_dir}/outputs/aapl_comprehensive_chart.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Comprehensive chart saved to: {chart_path}")
        return chart_path
    except Exception as e:
        print(f"Error creating price chart: {e}")
        return None

def generate_technical_analysis(data):
    """Generate basic technical analysis"""
    try:
        recent_data = data['recent_data']
        
        # Calculate RSI
        delta = recent_data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]
        
        # Calculate MACD
        exp1 = recent_data['Close'].ewm(span=12).mean()
        exp2 = recent_data['Close'].ewm(span=26).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9).mean()
        current_macd = macd.iloc[-1]
        current_signal = signal.iloc[-1]
        
        # Price trend analysis
        price_trend = "Bullish" if data['current_price'] > data['sma_20'] else "Bearish"
        
        # Volume analysis
        avg_volume = recent_data['Volume'].mean()
        volume_trend = "High" if data['volume'] > avg_volume * 1.5 else "Normal" if data['volume'] > avg_volume else "Low"
        
        return {
            'rsi': current_rsi,
            'macd': current_macd,
            'macd_signal': current_signal,
            'price_trend': price_trend,
            'volume_trend': volume_trend,
            'sma_20': data['sma_20'],
            'sma_50': data['sma_50']
        }
    except Exception as e:
        print(f"Error in technical analysis: {e}")
        return None

def generate_prediction(tech_analysis, data):
    """Generate a simple prediction based on technical analysis"""
    try:
        # Simple prediction logic based on technical indicators
        rsi = tech_analysis['rsi']
        macd = tech_analysis['macd']
        macd_signal = tech_analysis['macd_signal']
        price_trend = tech_analysis['price_trend']
        volume_trend = tech_analysis['volume_trend']
        
        # Prediction factors
        factors = []
        confidence = 0
        
        # RSI analysis
        if rsi < 30:
            factors.append("RSI indicates oversold condition - potential buying opportunity")
            confidence += 20
        elif rsi > 70:
            factors.append("RSI indicates overbought condition - potential selling pressure")
            confidence -= 20
        else:
            factors.append("RSI in neutral zone")
            confidence += 5
        
        # MACD analysis
        if macd > macd_signal:
            factors.append("MACD bullish crossover - positive momentum")
            confidence += 15
        else:
            factors.append("MACD bearish - negative momentum")
            confidence -= 15
        
        # Price trend analysis
        if price_trend == "Bullish":
            factors.append("Price above 20-day SMA - bullish trend")
            confidence += 10
        else:
            factors.append("Price below 20-day SMA - bearish trend")
            confidence -= 10
        
        # Volume analysis
        if volume_trend == "High":
            factors.append("High trading volume - strong conviction")
            confidence += 10
        elif volume_trend == "Low":
            factors.append("Low trading volume - weak conviction")
            confidence -= 5
        
        # Generate prediction
        if confidence > 20:
            prediction = "BULLISH"
            price_change_pct = 2 + (confidence - 20) * 0.1
        elif confidence < -20:
            prediction = "BEARISH"
            price_change_pct = -2 + (confidence + 20) * 0.1
        else:
            prediction = "NEUTRAL"
            price_change_pct = 0
        
        # Calculate target price
        target_price = data['current_price'] * (1 + price_change_pct / 100)
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'price_change_pct': price_change_pct,
            'target_price': target_price,
            'factors': factors
        }
    except Exception as e:
        print(f"Error generating prediction: {e}")
        return None

def run_minimal_aapl_prediction():
    """Run the minimal AAPL prediction analysis"""
    print("=" * 80)
    print("AAPL STOCK PRICE PREDICTION DEMO")
    print("Minimal Technical Analysis Version")
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
    print(f"Market Cap: ${aapl_data['market_cap']:,}")
    print(f"P/E Ratio: {aapl_data['pe_ratio']:.2f}")
    
    # Create price chart
    print("\nCreating comprehensive price chart...")
    chart_path = create_price_chart(aapl_data, demo_dir)
    
    # Generate technical analysis
    print("\nPerforming technical analysis...")
    tech_analysis = generate_technical_analysis(aapl_data)
    
    if tech_analysis:
        print(f"RSI (14): {tech_analysis['rsi']:.2f}")
        print(f"MACD: {tech_analysis['macd']:.4f}")
        print(f"Price Trend: {tech_analysis['price_trend']}")
        print(f"Volume Trend: {tech_analysis['volume_trend']}")
    
    # Generate prediction
    print("\nGenerating 14-day prediction...")
    prediction = generate_prediction(tech_analysis, aapl_data)
    
    if prediction:
        print(f"\nPREDICTION: {prediction['prediction']}")
        print(f"Confidence Score: {prediction['confidence']}")
        print(f"Expected Price Change: {prediction['price_change_pct']:.2f}%")
        print(f"Target Price: ${prediction['target_price']:.2f}")
        print("\nKey Factors:")
        for factor in prediction['factors']:
            print(f"  • {factor}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save comprehensive report
    report_path = f"{demo_dir}/reports/aapl_analysis_report_{timestamp}.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("AAPL STOCK ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("COMPANY INFORMATION:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Company: {aapl_data['company_name']}\n")
        f.write(f"Current Price: ${aapl_data['current_price']:.2f}\n")
        f.write(f"Price Change: ${aapl_data['price_change']:.2f} ({aapl_data['price_change_pct']:.2f}%)\n")
        f.write(f"Volume: {aapl_data['volume']:,}\n")
        f.write(f"Market Cap: ${aapl_data['market_cap']:,}\n")
        f.write(f"P/E Ratio: {aapl_data['pe_ratio']:.2f}\n\n")
        
        if tech_analysis:
            f.write("TECHNICAL ANALYSIS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"RSI (14): {tech_analysis['rsi']:.2f}\n")
            f.write(f"MACD: {tech_analysis['macd']:.4f}\n")
            f.write(f"MACD Signal: {tech_analysis['macd_signal']:.4f}\n")
            f.write(f"20-Day SMA: ${tech_analysis['sma_20']:.2f}\n")
            if tech_analysis['sma_50']:
                f.write(f"50-Day SMA: ${tech_analysis['sma_50']:.2f}\n")
            f.write(f"Price Trend: {tech_analysis['price_trend']}\n")
            f.write(f"Volume Trend: {tech_analysis['volume_trend']}\n\n")
        
        if prediction:
            f.write("14-DAY PREDICTION:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Prediction: {prediction['prediction']}\n")
            f.write(f"Confidence Score: {prediction['confidence']}\n")
            f.write(f"Expected Price Change: {prediction['price_change_pct']:.2f}%\n")
            f.write(f"Target Price: ${prediction['target_price']:.2f}\n\n")
            f.write("Key Factors:\n")
            for factor in prediction['factors']:
                f.write(f"  • {factor}\n")
    
    # Save JSON summary
    summary_data = {
        "timestamp": timestamp,
        "analysis_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "company": aapl_data['company_name'],
        "current_price": float(aapl_data['current_price']),
        "price_change": float(aapl_data['price_change']),
        "price_change_pct": float(aapl_data['price_change_pct']),
        "volume": int(aapl_data['volume']),
        "market_cap": int(aapl_data['market_cap']),
        "pe_ratio": float(aapl_data['pe_ratio']),
        "technical_analysis": tech_analysis,
        "prediction": prediction,
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
        print(f"    📊 aapl_comprehensive_chart.png")
    
    return summary_data

if __name__ == "__main__":
    print("Starting Minimal AAPL Prediction Demo...")
    print("This demo will analyze AAPL stock using technical analysis and predict price movements for the next 14 days.")
    print()
    
    result = run_minimal_aapl_prediction()
    
    if result:
        print("\n✅ Demo completed successfully!")
        print("Check the demo/reports/ folder for detailed analysis results.")
    else:
        print("\n❌ Demo failed. Check the demo/logs/ folder for error details.")
