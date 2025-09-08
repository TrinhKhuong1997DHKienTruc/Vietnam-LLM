"""
NVDA Stock Price Prediction using Amazon Chronos-Bolt Full Model
This script downloads NVDA stock data and predicts the next 14 days using Chronos-Bolt full model.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Try to import ChronosPipeline, install if not available
try:
    from chronos import ChronosPipeline
except ImportError:
    print("Installing Chronos library...")
    os.system("pip install git+https://github.com/amazon-science/chronos-forecasting.git")
    from chronos import ChronosPipeline

def download_nvda_data():
    """Download NVDA stock data from Yahoo Finance"""
    print("Downloading NVDA stock data...")
    
    # Download data for the last 2 years to have enough historical data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)  # 2 years
    
    ticker = yf.Ticker("NVDA")
    data = ticker.history(start=start_date, end=end_date)
    
    if data.empty:
        raise ValueError("Failed to download NVDA data. Please check your internet connection.")
    
    # Reset index to make Date a column
    data.reset_index(inplace=True)
    
    # Save to CSV
    data.to_csv("NVDA.csv", index=False)
    print(f"Downloaded {len(data)} days of NVDA data")
    return data

def load_and_preprocess_data():
    """Load and preprocess the NVDA data"""
    print("Loading and preprocessing data...")
    
    # Load the dataset
    nvda_dataset = pd.read_csv("NVDA.csv")
    
    # Display dataset info
    print(f"Dataset shape: {nvda_dataset.shape}")
    print(f"Date range: {nvda_dataset['Date'].min()} to {nvda_dataset['Date'].max()}")
    print("\nFirst few rows:")
    print(nvda_dataset.head())
    
    # Define test size (14 days for prediction)
    test_size = 14
    split_index = len(nvda_dataset) - test_size
    
    # Split data into training and test sets
    if split_index < 0:
        print("Warning: Not enough data for 14-day prediction. Using all available data for training.")
        train_data = nvda_dataset
        test_data = pd.DataFrame()  # Empty test data
    else:
        train_data = nvda_dataset.iloc[:split_index]
        test_data = nvda_dataset.iloc[split_index:]
    
    # Use 'Close' price for prediction (more stable than 'Open')
    train_close = train_data['Close'].values
    test_close = test_data['Close'].values if not test_data.empty else np.array([])
    
    return train_close, test_close, train_data, test_data

def initialize_chronos_model():
    """Initialize the Chronos full model"""
    print("Initializing Chronos full model...")
    
    try:
        # Load the Chronos-T5 large model (full model)
        pipeline = ChronosPipeline.from_pretrained(
            "amazon/chronos-t5-large",
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        print(f"Chronos-T5 Large model loaded successfully on {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        return pipeline
    except Exception as e:
        print(f"Error loading large model: {e}")
        print("Trying base model as fallback...")
        try:
            pipeline = ChronosPipeline.from_pretrained(
                "amazon/chronos-t5-base",
                device_map="cuda" if torch.cuda.is_available() else "cpu",
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            )
            print(f"Chronos-T5 Base model loaded successfully on {'CUDA' if torch.cuda.is_available() else 'CPU'}")
            return pipeline
        except Exception as e2:
            print(f"Error loading base model: {e2}")
            print("Trying small model as fallback...")
            try:
                pipeline = ChronosPipeline.from_pretrained(
                    "amazon/chronos-t5-small",
                    device_map="cuda" if torch.cuda.is_available() else "cpu",
                    torch_dtype=torch.float32,
                )
                print(f"Chronos-T5 Small model loaded successfully on {'CUDA' if torch.cuda.is_available() else 'CPU'}")
                return pipeline
            except Exception as e3:
                print(f"Error loading small model: {e3}")
                print("Trying tiny model as final fallback...")
                try:
                    pipeline = ChronosPipeline.from_pretrained(
                        "amazon/chronos-t5-tiny",
                        device_map="auto",
                        torch_dtype=torch.float32,
                    )
                    print("Chronos-T5 Tiny model loaded with auto device mapping")
                    return pipeline
                except Exception as e4:
                    print(f"Failed to load any Chronos model: {e4}")
                    raise

def make_predictions(pipeline, train_data, prediction_length=14):
    """Make predictions using the Chronos model"""
    print(f"Making predictions for next {prediction_length} days...")
    
    # Convert training data to PyTorch tensor
    context = torch.tensor(train_data, dtype=torch.float32)
    
    # Generate forecasts
    forecast = pipeline.predict(context, prediction_length)
    
    # Extract median forecast (index 5 in the quantiles)
    median_forecast = forecast[:, 5].detach().cpu().numpy()
    
    # Also extract other quantiles for uncertainty estimation
    lower_bound = forecast[:, 2].detach().cpu().numpy()  # 10th percentile
    upper_bound = forecast[:, 8].detach().cpu().numpy()  # 90th percentile
    
    # Ensure all arrays are 1-dimensional
    if median_forecast.ndim > 1:
        median_forecast = median_forecast.flatten()
    if lower_bound.ndim > 1:
        lower_bound = lower_bound.flatten()
    if upper_bound.ndim > 1:
        upper_bound = upper_bound.flatten()
    
    return median_forecast, lower_bound, upper_bound

def create_visualization(train_data, test_data, predictions, lower_bound, upper_bound, save_path="demo_reports"):
    """Create and save prediction visualization"""
    print("Creating visualization...")
    
    # Create demo_reports directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    plt.figure(figsize=(15, 8))
    
    # Plot training data
    train_indices = range(len(train_data))
    plt.plot(train_indices, train_data, label="Historical Data", color='blue', alpha=0.7)
    
    # Plot predictions
    pred_indices = range(len(train_data), len(train_data) + len(predictions))
    plt.plot(pred_indices, predictions, label="Predicted", color='red', linewidth=2)
    
    # Plot confidence intervals
    plt.fill_between(pred_indices, lower_bound, upper_bound, 
                    alpha=0.3, color='red', label="90% Confidence Interval")
    
    # Plot actual test data if available
    if len(test_data) > 0:
        test_indices = range(len(train_data), len(train_data) + len(test_data))
        plt.plot(test_indices, test_data, label="Actual (Test)", color='green', linewidth=2)
    
    plt.xlabel("Days")
    plt.ylabel("NVDA Stock Price ($)")
    plt.title("NVDA Stock Price Prediction using Chronos-T5 Large Model\nNext 14 Days Forecast")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plot_path = os.path.join(save_path, "NVDA_Prediction_Plot.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Prediction plot saved to: {plot_path}")
    
    plt.show()

def evaluate_predictions(test_data, predictions):
    """Evaluate model performance if test data is available"""
    if len(test_data) == 0:
        print("No test data available for evaluation.")
        return None
    
    print("Evaluating model performance...")
    
    # Calculate evaluation metrics
    mae = mean_absolute_error(test_data, predictions)
    mse = mean_squared_error(test_data, predictions)
    rmse = np.sqrt(mse)
    
    # Calculate MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((test_data - predictions) / test_data)) * 100
    
    metrics = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape
    }
    
    print(f"Evaluation Metrics:")
    print(f"Mean Absolute Error (MAE): ${mae:.2f}")
    print(f"Mean Squared Error (MSE): ${mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    
    return metrics

def save_evaluation_report(metrics, predictions, save_path="demo_reports"):
    """Save evaluation report to file"""
    report_path = os.path.join(save_path, "Evaluation_Report.txt")
    
    with open(report_path, "w") as f:
        f.write("NVDA Stock Price Prediction - Evaluation Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: Amazon Chronos-T5 Large (Full Model)\n")
        f.write(f"Prediction Period: 14 days\n\n")
        
        if metrics:
            f.write("Performance Metrics:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Mean Absolute Error (MAE): ${metrics['MAE']:.2f}\n")
            f.write(f"Mean Squared Error (MSE): ${metrics['MSE']:.2f}\n")
            f.write(f"Root Mean Squared Error (RMSE): ${metrics['RMSE']:.2f}\n")
            f.write(f"Mean Absolute Percentage Error (MAPE): {metrics['MAPE']:.2f}%\n\n")
        else:
            f.write("No test data available for evaluation.\n\n")
        
        f.write("Predicted Prices (Next 14 Days):\n")
        f.write("-" * 30 + "\n")
        for i, price in enumerate(predictions, 1):
            f.write(f"Day {i:2d}: ${price:.2f}\n")
        
        f.write(f"\nAverage Predicted Price: ${np.mean(predictions):.2f}\n")
        f.write(f"Price Range: ${np.min(predictions):.2f} - ${np.max(predictions):.2f}\n")
    
    print(f"Evaluation report saved to: {report_path}")

def save_predictions_csv(predictions, save_path="demo_reports"):
    """Save predictions to CSV file"""
    csv_path = os.path.join(save_path, "NVDA_Predictions.csv")
    
    # Create DataFrame with predictions
    future_dates = pd.date_range(start=datetime.now(), periods=len(predictions), freq='D')
    pred_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Price': predictions
    })
    
    pred_df.to_csv(csv_path, index=False)
    print(f"Predictions saved to: {csv_path}")

def main():
    """Main function to run the NVDA prediction pipeline"""
    print("=" * 60)
    print("NVDA Stock Price Prediction using Amazon Chronos-T5 Large Model")
    print("=" * 60)
    
    try:
        # Step 1: Download data
        data = download_nvda_data()
        
        # Step 2: Load and preprocess data
        train_close, test_close, train_data, test_data = load_and_preprocess_data()
        
        # Step 3: Initialize model
        pipeline = initialize_chronos_model()
        
        # Step 4: Make predictions
        predictions, lower_bound, upper_bound = make_predictions(pipeline, train_close, 14)
        
        # Step 5: Create visualization
        create_visualization(train_close, test_close, predictions, lower_bound, upper_bound)
        
        # Step 6: Evaluate predictions
        metrics = evaluate_predictions(test_close, predictions)
        
        # Step 7: Save reports
        save_evaluation_report(metrics, predictions)
        save_predictions_csv(predictions)
        
        print("\n" + "=" * 60)
        print("Prediction completed successfully!")
        print("Check the 'demo_reports' folder for all generated files.")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

