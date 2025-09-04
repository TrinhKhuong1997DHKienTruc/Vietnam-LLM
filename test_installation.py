"""
Test script to verify Chronos-Bolt installation and basic functionality
"""

import sys
import subprocess

def test_imports():
    """Test if all required packages can be imported"""
    print("🧪 Testing package imports...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} imported successfully")
        print(f"   CUDA available: {torch.cuda.is_available()}")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print(f"✅ Pandas {pd.__version__} imported successfully")
    except ImportError as e:
        print(f"❌ Pandas import failed: {e}")
        return False
    
    try:
        import yfinance as yf
        print("✅ yfinance imported successfully")
    except ImportError as e:
        print(f"❌ yfinance import failed: {e}")
        return False
    
    try:
        from chronos import BaseChronosPipeline
        print("✅ chronos-forecasting imported successfully")
    except ImportError as e:
        print(f"❌ chronos-forecasting import failed: {e}")
        print("   Installing chronos-forecasting...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "chronos-forecasting"])
            from chronos import BaseChronosPipeline
            print("✅ chronos-forecasting installed and imported successfully")
        except Exception as install_error:
            print(f"❌ Failed to install chronos-forecasting: {install_error}")
            return False
    
    return True

def test_data_download():
    """Test if we can download AAPL data"""
    print("\n🧪 Testing AAPL data download...")
    
    try:
        import yfinance as yf
        from datetime import datetime, timedelta
        
        # Download a small amount of data
        ticker = yf.Ticker("AAPL")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        data = ticker.history(
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            interval='1h'
        )
        
        if data.empty:
            print("❌ No data downloaded")
            return False
        
        print(f"✅ Downloaded {len(data)} data points")
        print(f"   Date range: {data.index.min()} to {data.index.max()}")
        print(f"   Price range: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
        return True
        
    except Exception as e:
        print(f"❌ Data download failed: {e}")
        return False

def test_model_loading():
    """Test if we can load the Chronos-Bolt model"""
    print("\n🧪 Testing Chronos-Bolt model loading...")
    
    try:
        from chronos import BaseChronosPipeline
        import torch
        
        print("   Loading model (this may take a few minutes on first run)...")
        pipeline = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-bolt-base",
            device_map="cpu",  # Use CPU for testing
            torch_dtype=torch.float32,
        )
        print("✅ Chronos-Bolt model loaded successfully")
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting Chronos-Bolt Installation Test")
    print("=" * 50)
    
    tests = [
        ("Package Imports", test_imports),
        ("Data Download", test_data_download),
        ("Model Loading", test_model_loading),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    
    all_passed = True
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if not result:
            all_passed = False
    
    print("=" * 50)
    if all_passed:
        print("🎉 All tests passed! You're ready to run the AAPL prediction.")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
    
    return all_passed

if __name__ == "__main__":
    main()
