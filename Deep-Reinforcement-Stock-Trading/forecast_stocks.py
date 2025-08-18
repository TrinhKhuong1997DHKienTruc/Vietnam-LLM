#!/usr/bin/env python3
"""
Script to run Deep Reinforcement Learning models on MSFT and NVDA stocks
and generate comprehensive forecasting reports
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from utils import *
import importlib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StockForecaster:
    def __init__(self, stock_symbol, data_file, initial_balance=50000, window_size=10):
        self.stock_symbol = stock_symbol
        self.data_file = data_file
        self.initial_balance = initial_balance
        self.window_size = window_size
        self.stock_prices = None
        self.results = {}
        
    def load_data(self):
        """Load stock data from CSV file"""
        try:
            # Read the CSV file
            df = pd.read_csv(self.data_file)
            # Extract close prices (column 4 is Close price)
            self.stock_prices = df['Close'].values
            logger.info(f"Loaded {len(self.stock_prices)} data points for {self.stock_symbol}")
            return True
        except Exception as e:
            logger.error(f"Error loading data for {self.stock_symbol}: {e}")
            return False
    
    def run_model(self, model_name, num_episodes=3):
        """Run a specific model and return results"""
        try:
            # Import the model
            model = importlib.import_module(f'agents.{model_name}')
            agent = model.Agent(state_dim=self.window_size + 3, balance=self.initial_balance)
            
            trading_period = len(self.stock_prices) - 1
            returns_across_episodes = []
            
            for episode in range(num_episodes):
                agent.reset()
                
                # Debug state generation
                try:
                    state = generate_combined_state(0, self.window_size, self.stock_prices, 
                                                 agent.balance, len(agent.inventory))
                    logger.info(f"Initial state shape: {state.shape}")
                except Exception as e:
                    logger.error(f"Error generating initial state: {e}")
                    raise e
                
                episode_return = 0
                for t in range(1, trading_period + 1):
                    try:
                        next_state = generate_combined_state(t, self.window_size, self.stock_prices,
                                                           agent.balance, len(agent.inventory))
                    except Exception as e:
                        logger.error(f"Error generating state at step {t}: {e}")
                        raise e
                        
                    previous_portfolio_value = len(agent.inventory) * self.stock_prices[t] + agent.balance
                    
                    if model_name == 'DDPG':
                        actions = agent.act(state, t)
                        action = np.argmax(actions)
                    else:
                        actions = agent.model.predict(state, verbose=0)[0]
                        action = agent.act(state)
                    
                    # Execute action
                    if action == 1 and agent.balance > self.stock_prices[t]:  # Buy
                        agent.balance -= self.stock_prices[t]
                        agent.inventory.append(self.stock_prices[t])
                    elif action == 2 and len(agent.inventory) > 0:  # Sell
                        agent.balance += self.stock_prices[t]
                        bought_price = agent.inventory.pop(0)
                        episode_return += self.stock_prices[t] - bought_price
                    
                    current_portfolio_value = len(agent.inventory) * self.stock_prices[t] + agent.balance
                    agent.return_rates.append((current_portfolio_value - previous_portfolio_value) / previous_portfolio_value)
                    agent.portfolio_values.append(current_portfolio_value)
                    state = next_state
                
                returns_across_episodes.append(episode_return)
            
            # Calculate metrics
            final_portfolio_value = agent.portfolio_values[-1]
            total_return = final_portfolio_value - self.initial_balance
            avg_return = np.mean(returns_across_episodes)
            sharpe_ratio_val = sharpe_ratio(np.array(agent.return_rates))
            max_drawdown = maximum_drawdown(agent.portfolio_values)
            
            return {
                'model_name': model_name,
                'final_portfolio_value': final_portfolio_value,
                'total_return': total_return,
                'avg_return_per_episode': avg_return,
                'sharpe_ratio': sharpe_ratio_val,
                'max_drawdown': max_drawdown,
                'return_rates': agent.return_rates,
                'portfolio_values': agent.portfolio_values
            }
            
        except Exception as e:
            logger.error(f"Error running {model_name} for {self.stock_symbol}: {e}")
            return None
    
    def generate_forecast_report(self):
        """Generate comprehensive forecast report for the stock"""
        if not self.load_data():
            return None
        
        models = ['DQN', 'DDQN', 'DDPG']
        self.results = {}
        
        logger.info(f"Running forecasting models for {self.stock_symbol}...")
        
        for model_name in models:
            logger.info(f"Running {model_name}...")
            result = self.run_model(model_name)
            if result:
                self.results[model_name] = result
        
        return self.results
    
    def create_visualizations(self, output_dir="reports"):
        """Create visualizations for the forecasting results"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Portfolio performance comparison
        plt.figure(figsize=(15, 10))
        
        for model_name, result in self.results.items():
            if result and 'portfolio_values' in result:
                plt.plot(result['portfolio_values'], label=f'{model_name}', linewidth=2)
        
        plt.axhline(y=self.initial_balance, color='r', linestyle='--', label='Initial Balance')
        plt.title(f'{self.stock_symbol} Portfolio Performance Comparison')
        plt.xlabel('Trading Steps')
        plt.ylabel('Portfolio Value ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(output_dir, f'{self.stock_symbol}_portfolio_performance.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved portfolio performance plot to {plot_path}")
        
        return plot_path
    
    def generate_text_report(self, output_dir="reports"):
        """Generate text report for the forecasting results"""
        os.makedirs(output_dir, exist_ok=True)
        
        report_path = os.path.join(output_dir, f'{self.stock_symbol}_forecast_report.txt')
        
        with open(report_path, 'w') as f:
            f.write(f"STOCK FORECASTING REPORT - {self.stock_symbol}\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Data period: {len(self.stock_prices)} trading days\n")
            f.write(f"Initial balance: ${self.initial_balance:,.2f}\n")
            f.write(f"Window size: {self.window_size} days\n\n")
            
            f.write("MODEL PERFORMANCE COMPARISON\n")
            f.write("-" * 40 + "\n")
            
            for model_name, result in self.results.items():
                if result:
                    f.write(f"\n{model_name}:\n")
                    f.write(f"  Final Portfolio Value: ${result['final_portfolio_value']:,.2f}\n")
                    f.write(f"  Total Return: ${result['total_return']:,.2f}\n")
                    f.write(f"  Average Return per Episode: ${result['avg_return_per_episode']:,.2f}\n")
                    f.write(f"  Sharpe Ratio: {result['sharpe_ratio']:.4f}\n")
                    f.write(f"  Maximum Drawdown: {result['max_drawdown']*100:.2f}%\n")
            
            f.write("\n" + "=" * 60 + "\n")
            f.write("FORECAST FOR NEXT WEEK\n")
            f.write("-" * 40 + "\n")
            
            # Simple trend analysis for next week
            if len(self.stock_prices) >= 5:
                recent_prices = self.stock_prices[-5:]
                trend = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]
                
                f.write(f"Recent 5-day trend: {'UPWARD' if trend > 0 else 'DOWNWARD'}\n")
                f.write(f"Trend slope: {trend:.4f}\n")
                
                # Simple prediction for next week
                current_price = self.stock_prices[-1]
                predicted_change = trend * 5  # 5 trading days
                predicted_price = current_price + predicted_change
                
                f.write(f"Current price: ${current_price:.2f}\n")
                f.write(f"Predicted price in 5 days: ${predicted_price:.2f}\n")
                f.write(f"Predicted change: ${predicted_change:.2f} ({predicted_change/current_price*100:.2f}%)\n")
            
            f.write("\nRECOMMENDATIONS\n")
            f.write("-" * 40 + "\n")
            
            # Generate recommendations based on model performance
            best_model = max(self.results.keys(), 
                           key=lambda x: self.results[x]['total_return'] if self.results[x] else -float('inf'))
            
            if best_model and self.results[best_model]:
                best_return = self.results[best_model]['total_return']
                if best_return > 0:
                    f.write(f"✅ RECOMMENDATION: Consider {self.stock_symbol} based on {best_model} model performance\n")
                    f.write(f"   The {best_model} model achieved a positive return of ${best_return:,.2f}\n")
                else:
                    f.write(f"⚠️  CAUTION: All models show negative returns for {self.stock_symbol}\n")
                    f.write(f"   Best performing model ({best_model}) still had a loss of ${abs(best_return):,.2f}\n")
        
        logger.info(f"Saved text report to {report_path}")
        return report_path

def main():
    """Main function to run forecasting for MSFT and NVDA"""
    
    # Stock symbols to forecast
    stocks_to_forecast = [
        ('MSFT', 'data/MSFT_2023-01-01_2025-08-25.csv'),
        ('NVDA', 'data/NVDA_2023-01-01_2025-08-25.csv')
    ]
    
    logger.info("Starting Deep Reinforcement Learning Stock Forecasting...")
    
    for stock_symbol, data_file in stocks_to_forecast:
        if not os.path.exists(data_file):
            logger.warning(f"Data file {data_file} not found. Skipping {stock_symbol}.")
            continue
        
        logger.info(f"\n{'='*60}")
        logger.info(f"FORECASTING {stock_symbol}")
        logger.info(f"{'='*60}")
        
        # Create forecaster instance
        forecaster = StockForecaster(stock_symbol, data_file)
        
        # Generate forecast
        results = forecaster.generate_forecast_report()
        
        if results:
            # Create visualizations
            forecaster.create_visualizations()
            
            # Generate text report
            forecaster.generate_text_report()
            
            logger.info(f"✅ Completed forecasting for {stock_symbol}")
        else:
            logger.error(f"❌ Failed to generate forecast for {stock_symbol}")
    
    logger.info("\n🎉 All forecasting completed!")
    logger.info("Check the 'reports' directory for detailed results.")

if __name__ == "__main__":
    main()