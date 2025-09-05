#!/usr/bin/env python3
"""
Quick start script for FinBERT AAPL Price Prediction Demo
"""

import sys
import os
import subprocess

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'torch', 'transformers', 'pandas', 'numpy', 'scikit-learn',
        'matplotlib', 'seaborn', 'plotly', 'yfinance', 'requests'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n💡 Please install missing packages:")
        print("   pip install -r requirements.txt")
        return False
    
    return True

def main():
    """Main function to run the demo"""
    print("🚀 FinBERT AAPL Price Prediction Demo - Quick Start")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    print("✅ All dependencies found!")
    print("🎯 Starting AAPL price prediction demo...")
    print()
    
    # Run the main demo
    try:
        from aapl_price_prediction_demo import main as run_demo
        run_demo()
    except ImportError as e:
        print(f"❌ Error importing demo: {e}")
        print("💡 Make sure you're in the correct directory")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error running demo: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
