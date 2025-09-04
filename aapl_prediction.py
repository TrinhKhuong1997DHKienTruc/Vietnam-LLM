"""
AAPL Hourly Price Prediction using Chronos-Bolt Base Model
=========================================================

This script demonstrates how to use the Chronos-Bolt base model to predict
AAPL (Apple Inc.) hourly stock prices for the next 14 days.

Author: Trinh Khuong
Date: 2025
"""

import pandas as pd
import numpy as np
import torch
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    from chronos import BaseChronosPipeline
    print("✅ Chronos-forecasting library imported successfully")
except ImportError:
    print("❌ chronos-forecasting library not found. Installing...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "chronos-forecasting"])
    from chronos import BaseChronosPipeline
    print("✅ Chronos-forecasting library installed and imported successfully")

def download_aapl_data(days_back=90):
    """
    Download AAPL hourly data from Yahoo Finance
    
    Args:
        days_back (int): Number of days to look back for historical data
        
    Returns:
        pd.DataFrame: AAPL hourly data
    """
    print(f"📈 Downloading AAPL hourly data for the last {days_back} days...")
    
    # Calculate start date
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    # Download data
    ticker = yf.Ticker("AAPL")
    data = ticker.history(
        start=start_date.strftime('%Y-%m-%d'),
        end=end_date.strftime('%Y-%m-%d'),
        interval='1h'
    )
    
    if data.empty:
        raise ValueError("No data downloaded. Please check your internet connection.")
    
    # Clean and prepare data
    data = data.reset_index()
    data = data[['Datetime', 'Close']].copy()
    data.columns = ['timestamp', 'price']
    data = data.dropna()
    
    print(f"✅ Downloaded {len(data)} hourly data points")
    print(f"📅 Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
    print(f"💰 Price range: ${data['price'].min():.2f} - ${data['price'].max():.2f}")
    
    return data

def prepare_model():
    """
    Initialize and prepare the Chronos-Bolt base model
    
    Returns:
        BaseChronosPipeline: Loaded Chronos-Bolt model
    """
    print("🤖 Loading Chronos-Bolt base model...")
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Using device: {device}")
    
    try:
        # Load the model
        pipeline = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-bolt-base",
            device_map=device,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        )
        print("✅ Chronos-Bolt base model loaded successfully")
        return pipeline
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("🔄 Trying with CPU fallback...")
        pipeline = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-bolt-base",
            device_map="cpu",
            torch_dtype=torch.float32,
        )
        print("✅ Chronos-Bolt base model loaded successfully on CPU")
        return pipeline

def make_predictions(pipeline, data, prediction_hours=336):  # 14 days * 24 hours
    """
    Make predictions using the Chronos-Bolt model
    
    Args:
        pipeline: Loaded Chronos-Bolt model
        data (pd.DataFrame): Historical price data
        prediction_hours (int): Number of hours to predict (14 days = 336 hours)
        
    Returns:
        dict: Prediction results with quantiles
    """
    print(f"🔮 Making predictions for the next {prediction_hours} hours (14 days)...")
    
    # Prepare context data (last 168 hours = 7 days for context)
    context_length = min(168, len(data))
    context_data = data['price'].tail(context_length).values
    
    print(f"📊 Using {context_length} hours of historical data as context")
    print(f"📈 Context price range: ${context_data.min():.2f} - ${context_data.max():.2f}")
    
    try:
        # Make predictions
        forecast = pipeline.predict(
            context=torch.tensor(context_data, dtype=torch.float32),
            prediction_length=prediction_hours
        )
        
        # Extract quantiles (Chronos-Bolt provides quantile forecasts)
        # Shape: [num_series, num_quantiles, prediction_length]
        quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        predictions = {}
        for i, q in enumerate(quantiles):
            predictions[f'q{int(q*100)}'] = forecast[0, i, :].cpu().numpy()
        
        print("✅ Predictions generated successfully")
        return predictions, context_data
        
    except Exception as e:
        print(f"❌ Error making predictions: {e}")
        raise

def create_prediction_timeline(predictions, last_timestamp, prediction_hours=336):
    """
    Create timeline for predictions
    
    Args:
        predictions (dict): Prediction results
        last_timestamp: Last timestamp from historical data
        prediction_hours (int): Number of hours to predict
        
    Returns:
        pd.DataFrame: Prediction timeline
    """
    # Create future timestamps (hourly)
    future_timestamps = []
    current_time = last_timestamp
    
    for i in range(prediction_hours):
        # Add 1 hour
        current_time += timedelta(hours=1)
        # Skip weekends (Saturday=5, Sunday=6)
        if current_time.weekday() < 5:  # Monday=0, Friday=4
            future_timestamps.append(current_time)
    
    # Create DataFrame
    timeline_df = pd.DataFrame({
        'timestamp': future_timestamps,
        'hour': range(len(future_timestamps))
    })
    
    # Add predictions
    for quantile, values in predictions.items():
        timeline_df[quantile] = values[:len(future_timestamps)]
    
    return timeline_df

def visualize_predictions(historical_data, predictions, timeline_df):
    """
    Create comprehensive visualization of predictions
    
    Args:
        historical_data (pd.DataFrame): Historical price data
        predictions (dict): Prediction results
        timeline_df (pd.DataFrame): Prediction timeline
    """
    print("📊 Creating visualizations...")
    
    # Set up the plot style
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle('AAPL Hourly Price Prediction - Next 14 Days\nUsing Chronos-Bolt Base Model', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Historical data + predictions
    ax1 = axes[0, 0]
    
    # Plot historical data (last 7 days)
    hist_data = historical_data.tail(168)  # Last 7 days
    ax1.plot(hist_data['timestamp'], hist_data['price'], 
             label='Historical Data (Last 7 Days)', color='blue', linewidth=2)
    
    # Plot predictions with confidence intervals
    ax1.fill_between(timeline_df['timestamp'], 
                     timeline_df['q10'], timeline_df['q90'],
                     alpha=0.3, color='red', label='80% Confidence Interval')
    ax1.fill_between(timeline_df['timestamp'], 
                     timeline_df['q20'], timeline_df['q80'],
                     alpha=0.5, color='orange', label='60% Confidence Interval')
    ax1.plot(timeline_df['timestamp'], timeline_df['q50'], 
             label='Median Prediction', color='red', linewidth=2)
    
    ax1.set_title('AAPL Price Prediction with Confidence Intervals')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Prediction quantiles
    ax2 = axes[0, 1]
    quantiles = ['q10', 'q20', 'q30', 'q40', 'q50', 'q60', 'q70', 'q80', 'q90']
    colors = plt.cm.viridis(np.linspace(0, 1, len(quantiles)))
    
    for i, (q, color) in enumerate(zip(quantiles, colors)):
        ax2.plot(timeline_df['hour'], timeline_df[q], 
                label=f'{q} ({int(q[1:])}%)', color=color, alpha=0.7)
    
    ax2.set_title('Prediction Quantiles Over Time')
    ax2.set_xlabel('Hours from Now')
    ax2.set_ylabel('Predicted Price ($)')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Price distribution at different time points
    ax3 = axes[1, 0]
    time_points = [0, 24, 48, 72, 96, 120, 144, 168]  # Every 24 hours
    time_labels = ['Now', '1 Day', '2 Days', '3 Days', '4 Days', '5 Days', '6 Days', '7 Days']
    
    for i, (tp, label) in enumerate(zip(time_points, time_labels)):
        if tp < len(timeline_df):
            prices = [timeline_df.iloc[tp][q] for q in quantiles]
            ax3.plot(prices, [i] * len(prices), 'o-', label=label, markersize=6)
    
    ax3.set_title('Price Distribution at Different Time Points')
    ax3.set_xlabel('Predicted Price ($)')
    ax3.set_ylabel('Time Point')
    ax3.set_yticks(range(len(time_labels)))
    ax3.set_yticklabels(time_labels)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Calculate summary statistics
    current_price = historical_data['price'].iloc[-1]
    final_price_median = timeline_df['q50'].iloc[-1]
    price_change = final_price_median - current_price
    price_change_pct = (price_change / current_price) * 100
    
    # Create summary text
    summary_text = f"""
    📊 PREDICTION SUMMARY
    
    Current Price: ${current_price:.2f}
    Predicted Price (14 days): ${final_price_median:.2f}
    Expected Change: ${price_change:.2f} ({price_change_pct:+.2f}%)
    
    📈 CONFIDENCE INTERVALS (14 days)
    80% Range: ${timeline_df['q10'].iloc[-1]:.2f} - ${timeline_df['q90'].iloc[-1]:.2f}
    60% Range: ${timeline_df['q20'].iloc[-1]:.2f} - ${timeline_df['q80'].iloc[-1]:.2f}
    
    📅 PREDICTION PERIOD
    Start: {timeline_df['timestamp'].iloc[0].strftime('%Y-%m-%d %H:%M')}
    End: {timeline_df['timestamp'].iloc[-1].strftime('%Y-%m-%d %H:%M')}
    
    🤖 MODEL INFO
    Model: Chronos-Bolt Base (205M parameters)
    Context: {len(historical_data.tail(168))} hours
    Prediction: {len(timeline_df)} hours
    """
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=12,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('aapl_prediction_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Visualizations saved as 'aapl_prediction_results.png'")

def save_predictions_to_csv(timeline_df, historical_data):
    """
    Save predictions to CSV file
    
    Args:
        timeline_df (pd.DataFrame): Prediction timeline
        historical_data (pd.DataFrame): Historical price data
    """
    print("💾 Saving predictions to CSV...")
    
    # Save predictions
    timeline_df.to_csv('aapl_predictions_14days.csv', index=False)
    
    # Save historical data
    historical_data.to_csv('aapl_historical_data.csv', index=False)
    
    print("✅ Predictions saved to 'aapl_predictions_14days.csv'")
    print("✅ Historical data saved to 'aapl_historical_data.csv'")

def main():
    """
    Main function to run the AAPL prediction pipeline
    """
    print("🚀 Starting AAPL Hourly Price Prediction using Chronos-Bolt Base Model")
    print("=" * 80)
    
    try:
        # Step 1: Download historical data
        historical_data = download_aapl_data(days_back=90)
        
        # Step 2: Prepare model
        pipeline = prepare_model()
        
        # Step 3: Make predictions
        predictions, context_data = make_predictions(pipeline, historical_data)
        
        # Step 4: Create prediction timeline
        last_timestamp = historical_data['timestamp'].iloc[-1]
        timeline_df = create_prediction_timeline(predictions, last_timestamp)
        
        # Step 5: Visualize results
        visualize_predictions(historical_data, predictions, timeline_df)
        
        # Step 6: Save results
        save_predictions_to_csv(timeline_df, historical_data)
        
        print("\n" + "=" * 80)
        print("🎉 AAPL prediction completed successfully!")
        print("📁 Files generated:")
        print("   - aapl_prediction_results.png (visualizations)")
        print("   - aapl_predictions_14days.csv (predictions)")
        print("   - aapl_historical_data.csv (historical data)")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Error in main execution: {e}")
        print("Please check your internet connection and try again.")
        raise

if __name__ == "__main__":
    main()
