#!/usr/bin/env python3
"""
AAPL Price Prediction Demo with FinBERT Sentiment Analysis
Predicts AAPL hourly prices for the next 14 days using historical data and sentiment analysis
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# FinBERT imports
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Machine Learning imports
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

class AAPLPricePredictor:
    """AAPL Price Prediction with FinBERT Sentiment Analysis"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.scaler = StandardScaler()
        self.price_model = None
        self.sentiment_model = None
        self.historical_data = None
        self.news_data = []
        
    def load_finbert_model(self):
        """Load FinBERT model for sentiment analysis"""
        print("📥 Loading FinBERT model...")
        try:
            model_name = "ProsusAI/finbert"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            print("✅ FinBERT model loaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Error loading FinBERT model: {str(e)}")
            return False
    
    def get_historical_data(self, period="2y"):
        """Fetch historical AAPL data"""
        print(f"📊 Fetching AAPL historical data for {period}...")
        try:
            ticker = yf.Ticker("AAPL")
            self.historical_data = ticker.history(period=period, interval="1h")
            
            if self.historical_data.empty:
                print("❌ No data retrieved")
                return False
                
            print(f"✅ Retrieved {len(self.historical_data)} data points")
            print(f"📅 Date range: {self.historical_data.index[0]} to {self.historical_data.index[-1]}")
            return True
        except Exception as e:
            print(f"❌ Error fetching data: {str(e)}")
            return False
    
    def analyze_sentiment(self, text):
        """Analyze sentiment of financial text using FinBERT"""
        if not self.model or not self.tokenizer:
            return 0.0, "neutral"
        
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            probs = predictions[0].numpy()
            labels = ['negative', 'neutral', 'positive']
            
            predicted_label = labels[np.argmax(probs)]
            sentiment_score = probs[2] - probs[0]  # positive - negative
            
            return sentiment_score, predicted_label
        except Exception as e:
            print(f"⚠️ Error in sentiment analysis: {str(e)}")
            return 0.0, "neutral"
    
    def create_features(self, data):
        """Create technical features for price prediction"""
        df = data.copy()
        
        # Technical indicators
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        df['EMA_5'] = df['Close'].ewm(span=5).mean()
        df['EMA_20'] = df['Close'].ewm(span=20).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
        df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
        df['BB_width'] = df['BB_upper'] - df['BB_lower']
        df['BB_position'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Price changes
        df['Price_change'] = df['Close'].pct_change()
        df['Price_change_5'] = df['Close'].pct_change(5)
        df['Price_change_20'] = df['Close'].pct_change(20)
        
        # Volatility
        df['Volatility'] = df['Price_change'].rolling(window=20).std()
        
        # Time features
        df['Hour'] = df.index.hour if hasattr(df.index, 'hour') else 12
        df['Day_of_week'] = df.index.dayofweek if hasattr(df.index, 'dayofweek') else 0
        df['Month'] = df.index.month if hasattr(df.index, 'month') else 1
        
        # Lag features
        for lag in [1, 2, 3, 5, 10]:
            df[f'Close_lag_{lag}'] = df['Close'].shift(lag)
            df[f'Volume_lag_{lag}'] = df['Volume'].shift(lag)
        
        return df
    
    def prepare_training_data(self):
        """Prepare data for model training"""
        print("🔧 Preparing training data...")
        
        # Create features
        df = self.create_features(self.historical_data)
        
        # Remove rows with NaN values
        df = df.dropna()
        
        # Define feature columns
        feature_columns = [
            'SMA_5', 'SMA_20', 'SMA_50', 'EMA_5', 'EMA_20', 'RSI',
            'BB_width', 'BB_position', 'Volume_ratio', 'Price_change',
            'Price_change_5', 'Price_change_20', 'Volatility',
            'Hour', 'Day_of_week', 'Month'
        ]
        
        # Add lag features
        for lag in [1, 2, 3, 5, 10]:
            feature_columns.extend([f'Close_lag_{lag}', f'Volume_lag_{lag}'])
        
        # Create target variable (next hour's close price)
        df['Target'] = df['Close'].shift(-1)
        
        # Remove the last row (no target)
        df = df[:-1]
        
        # Select features and target
        X = df[feature_columns]
        y = df['Target']
        
        # Split data (80% train, 20% test)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"✅ Training data prepared: {X_train_scaled.shape[0]} train, {X_test_scaled.shape[0]} test samples")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, feature_columns
    
    def train_price_model(self):
        """Train the price prediction model"""
        print("🤖 Training price prediction model...")
        
        X_train, X_test, y_train, y_test, feature_columns = self.prepare_training_data()
        
        # Train Random Forest model
        self.price_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        self.price_model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.price_model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"📊 Model Performance:")
        print(f"   MSE: {mse:.4f}")
        print(f"   MAE: {mae:.4f}")
        print(f"   R²: {r2:.4f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': self.price_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n🔝 Top 10 Most Important Features:")
        for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
            print(f"   {i+1}. {row['feature']}: {row['importance']:.4f}")
        
        return True
    
    def generate_sample_news(self):
        """Generate sample financial news for demonstration"""
        sample_news = [
            "Apple reports strong Q4 earnings with iPhone sales exceeding expectations",
            "AAPL stock rises on positive analyst upgrades and strong market sentiment",
            "Apple announces new product launch scheduled for next month",
            "Market volatility affects tech stocks including Apple",
            "Apple's services revenue shows continued growth momentum",
            "Analysts raise price targets for AAPL following strong guidance",
            "Apple faces supply chain challenges affecting production",
            "Strong demand for Apple products drives revenue growth",
            "Market uncertainty impacts Apple stock performance",
            "Apple's innovation pipeline shows promising developments"
        ]
        return sample_news
    
    def predict_future_prices(self, days=14):
        """Predict future prices for the specified number of days"""
        print(f"🔮 Predicting AAPL prices for next {days} days...")
        
        if not self.price_model:
            print("❌ Price model not trained yet")
            return None
        
        # Get the last available data point
        last_data = self.historical_data.iloc[-1:].copy()
        
        # Generate sample news for demonstration
        sample_news = self.generate_sample_news()
        
        predictions = []
        current_data = last_data.copy()
        
        for day in range(days):
            for hour in range(24):  # 24 hours per day
                # Create features for current prediction
                df_features = self.create_features(current_data)
                latest_features = df_features.iloc[-1:]
                
                # Select the same features used in training
                feature_columns = [
                    'SMA_5', 'SMA_20', 'SMA_50', 'EMA_5', 'EMA_20', 'RSI',
                    'BB_width', 'BB_position', 'Volume_ratio', 'Price_change',
                    'Price_change_5', 'Price_change_20', 'Volatility',
                    'Hour', 'Day_of_week', 'Month'
                ]
                
                for lag in [1, 2, 3, 5, 10]:
                    feature_columns.extend([f'Close_lag_{lag}', f'Volume_lag_{lag}'])
                
                # Prepare features
                X_pred = latest_features[feature_columns].values
                X_pred_scaled = self.scaler.transform(X_pred)
                
                # Predict price
                predicted_price = self.price_model.predict(X_pred_scaled)[0]
                
                # Add sentiment influence (simulate news impact)
                if hour % 6 == 0:  # Every 6 hours, simulate news impact
                    news_text = sample_news[day % len(sample_news)]
                    sentiment_score, sentiment_label = self.analyze_sentiment(news_text)
                    
                    # Apply sentiment influence (small impact)
                    sentiment_impact = sentiment_score * 0.01  # 1% max impact
                    predicted_price *= (1 + sentiment_impact)
                
                # Create timestamp
                current_time = datetime.now() + timedelta(days=day, hours=hour)
                
                predictions.append({
                    'timestamp': current_time,
                    'predicted_price': predicted_price,
                    'day': day + 1,
                    'hour': hour
                })
                
                # Update current data for next prediction (simplified)
                new_row = current_data.iloc[-1:].copy()
                new_row.index = pd.DatetimeIndex([current_time])
                new_row['Close'] = predicted_price
                new_row['High'] = predicted_price * 1.01
                new_row['Low'] = predicted_price * 0.99
                new_row['Open'] = predicted_price
                new_row['Volume'] = current_data['Volume'].iloc[-1] * np.random.uniform(0.8, 1.2)
                
                current_data = pd.concat([current_data, new_row])
        
        return pd.DataFrame(predictions)
    
    def create_visualizations(self, predictions_df):
        """Create visualizations for the predictions"""
        print("📊 Creating visualizations...")
        
        # Historical data plot
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('AAPL Historical Prices', 'AAPL Price Predictions (Next 14 Days)'),
            vertical_spacing=0.1
        )
        
        # Historical data
        fig.add_trace(
            go.Scatter(
                x=self.historical_data.index,
                y=self.historical_data['Close'],
                mode='lines',
                name='Historical Close',
                line=dict(color='blue', width=1)
            ),
            row=1, col=1
        )
        
        # Predictions
        fig.add_trace(
            go.Scatter(
                x=predictions_df['timestamp'],
                y=predictions_df['predicted_price'],
                mode='lines+markers',
                name='Predicted Price',
                line=dict(color='red', width=2),
                marker=dict(size=4)
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title='AAPL Price Prediction with FinBERT Sentiment Analysis',
            height=800,
            showlegend=True
        )
        
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Predicted Price ($)", row=2, col=1)
        
        # Save plot
        fig.write_html("aapl_price_prediction.html")
        try:
            fig.write_image("aapl_price_prediction.png", width=1200, height=800)
            print("✅ PNG image saved successfully")
        except Exception as e:
            print(f"⚠️ Could not save PNG image: {str(e)}")
            print("📊 HTML visualization saved instead")
        
        # Daily summary plot
        daily_predictions = predictions_df.groupby('day').agg({
            'predicted_price': ['min', 'max', 'mean']
        }).round(2)
        
        daily_predictions.columns = ['Min_Price', 'Max_Price', 'Avg_Price']
        daily_predictions = daily_predictions.reset_index()
        
        fig2 = go.Figure()
        
        fig2.add_trace(go.Scatter(
            x=daily_predictions['day'],
            y=daily_predictions['Avg_Price'],
            mode='lines+markers',
            name='Average Price',
            line=dict(color='green', width=3)
        ))
        
        fig2.add_trace(go.Scatter(
            x=daily_predictions['day'],
            y=daily_predictions['Max_Price'],
            mode='lines',
            name='Max Price',
            line=dict(color='red', width=1, dash='dash'),
            fill=None
        ))
        
        fig2.add_trace(go.Scatter(
            x=daily_predictions['day'],
            y=daily_predictions['Min_Price'],
            mode='lines',
            name='Min Price',
            line=dict(color='blue', width=1, dash='dash'),
            fill='tonexty'
        ))
        
        fig2.update_layout(
            title='AAPL Daily Price Range Predictions (Next 14 Days)',
            xaxis_title='Day',
            yaxis_title='Price ($)',
            height=500
        )
        
        fig2.write_html("aapl_daily_predictions.html")
        try:
            fig2.write_image("aapl_daily_predictions.png", width=1200, height=500)
            print("✅ Daily predictions PNG saved successfully")
        except Exception as e:
            print(f"⚠️ Could not save daily predictions PNG: {str(e)}")
            print("📊 HTML visualization saved instead")
        
        print("✅ Visualizations created and saved!")
        
        return daily_predictions
    
    def save_predictions(self, predictions_df, daily_summary):
        """Save predictions to CSV files"""
        print("💾 Saving predictions...")
        
        # Save detailed predictions
        predictions_df.to_csv("aapl_hourly_predictions.csv", index=False)
        
        # Save daily summary
        daily_summary.to_csv("aapl_daily_summary.csv", index=False)
        
        # Save model
        joblib.dump(self.price_model, "aapl_price_model.pkl")
        joblib.dump(self.scaler, "aapl_scaler.pkl")
        
        print("✅ Predictions and model saved!")
    
    def run_demo(self):
        """Run the complete AAPL price prediction demo"""
        print("🚀 Starting AAPL Price Prediction Demo with FinBERT Sentiment Analysis")
        print("=" * 80)
        
        # Step 1: Load FinBERT model
        if not self.load_finbert_model():
            return False
        
        # Step 2: Get historical data
        if not self.get_historical_data():
            return False
        
        # Step 3: Train price prediction model
        if not self.train_price_model():
            return False
        
        # Step 4: Generate predictions
        predictions_df = self.predict_future_prices(days=14)
        if predictions_df is None:
            return False
        
        # Step 5: Create visualizations
        daily_summary = self.create_visualizations(predictions_df)
        
        # Step 6: Save results
        self.save_predictions(predictions_df, daily_summary)
        
        # Step 7: Display summary
        print("\n" + "=" * 80)
        print("📈 AAPL PRICE PREDICTION SUMMARY")
        print("=" * 80)
        
        current_price = self.historical_data['Close'].iloc[-1]
        avg_predicted_price = predictions_df['predicted_price'].mean()
        price_change = ((avg_predicted_price - current_price) / current_price) * 100
        
        print(f"📊 Current AAPL Price: ${current_price:.2f}")
        print(f"🔮 Average Predicted Price (14 days): ${avg_predicted_price:.2f}")
        print(f"📈 Expected Price Change: {price_change:+.2f}%")
        
        print(f"\n📅 Daily Price Range Predictions:")
        for _, row in daily_summary.iterrows():
            print(f"   Day {int(row['day']):2d}: ${row['Min_Price']:.2f} - ${row['Max_Price']:.2f} (Avg: ${row['Avg_Price']:.2f})")
        
        print(f"\n📁 Files Generated:")
        print(f"   - aapl_hourly_predictions.csv")
        print(f"   - aapl_daily_summary.csv")
        print(f"   - aapl_price_prediction.html")
        print(f"   - aapl_price_prediction.png")
        print(f"   - aapl_daily_predictions.html")
        print(f"   - aapl_daily_predictions.png")
        print(f"   - aapl_price_model.pkl")
        print(f"   - aapl_scaler.pkl")
        
        print("\n🎉 Demo completed successfully!")
        return True

def main():
    """Main function to run the demo"""
    predictor = AAPLPricePredictor()
    success = predictor.run_demo()
    
    if success:
        print("\n✅ AAPL Price Prediction Demo completed successfully!")
        print("📊 Check the generated files for detailed results and visualizations.")
    else:
        print("\n❌ Demo failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
