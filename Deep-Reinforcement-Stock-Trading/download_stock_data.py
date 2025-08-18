#!/usr/bin/env python3
"""
Script to download recent stock data for MSFT and NVDA
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os

def download_stock_data(symbol, start_date, end_date, output_dir="data"):
    """
    Download stock data for a given symbol and date range
    
    Args:
        symbol (str): Stock symbol (e.g., 'MSFT', 'NVDA')
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        output_dir (str): Directory to save the CSV file
    
    Returns:
        str: Path to the saved CSV file
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Download data
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date)
        
        if data.empty:
            print(f"No data found for {symbol}")
            return None
        
        # Reset index to get date as a column
        data = data.reset_index()
        
        # Check the actual columns and handle them properly
        print(f"Columns in {symbol} data: {list(data.columns)}")
        
        # Ensure we have the expected columns
        expected_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        
        # If we have more columns, select only the ones we need
        if len(data.columns) > 6:
            # Select the first 6 columns (Date, OHLCV)
            data = data.iloc[:, :6]
            data.columns = expected_columns
        elif len(data.columns) == 6:
            data.columns = expected_columns
        else:
            print(f"Unexpected number of columns for {symbol}: {len(data.columns)}")
            return None
        
        # Format date
        data['Date'] = pd.to_datetime(data['Date']).dt.strftime('%Y-%m-%d')
        
        # Save to CSV
        filename = f"{symbol}_{start_date}_{end_date}.csv"
        filepath = os.path.join(output_dir, filename)
        data.to_csv(filepath, index=False)
        
        print(f"Successfully downloaded {symbol} data: {len(data)} records")
        print(f"Saved to: {filepath}")
        
        return filepath
        
    except Exception as e:
        print(f"Error downloading {symbol} data: {e}")
        return None

def main():
    """Main function to download MSFT and NVDA data"""
    
    # Calculate date range for next week forecast
    today = datetime.now()
    end_date = today + timedelta(days=7)
    
    # Format dates
    start_date = "2023-01-01"  # Start from beginning of 2023
    end_date_str = end_date.strftime("%Y-%m-%d")
    
    print("Downloading stock data for next week forecast...")
    print(f"Date range: {start_date} to {end_date_str}")
    print()
    
    # Download MSFT data
    print("Downloading MSFT data...")
    msft_file = download_stock_data("MSFT", start_date, end_date_str)
    
    print()
    
    # Download NVDA data
    print("Downloading NVDA data...")
    nvda_file = download_stock_data("NVDA", start_date, end_date_str)
    
    print()
    
    if msft_file and nvda_file:
        print("✅ All stock data downloaded successfully!")
        print("Ready to run forecasting models.")
    else:
        print("❌ Some data downloads failed. Please check the errors above.")

if __name__ == "__main__":
    main()