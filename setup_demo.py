#!/usr/bin/env python3
"""
FinRobot AAPL Demo Setup Script
This script helps users set up the FinRobot demo environment quickly.
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def create_config_files():
    """Create configuration files with placeholders"""
    print("\n⚙️ Setting up configuration files...")
    
    # Create OAI_CONFIG_LIST if it doesn't exist
    if not os.path.exists("OAI_CONFIG_LIST"):
        oai_config = [
            {
                "model": "gpt-5-mini-2025-08-07",
                "api_key": "<your OpenAI API key here>",
                "base_url": "<your API base URL here>"
            },
            {
                "model": "gemini-2.5-pro",
                "api_key": "<your Gemini API key here>"
            }
        ]
        with open("OAI_CONFIG_LIST", "w") as f:
            json.dump(oai_config, f, indent=4)
        print("✅ Created OAI_CONFIG_LIST")
    
    # Create config_api_keys if it doesn't exist
    if not os.path.exists("config_api_keys"):
        api_keys = {
            "FINNHUB_API_KEY": "YOUR_FINNHUB_API_KEY",
            "FMP_API_KEY": "YOUR_FMP_API_KEY",
            "SEC_API_KEY": "YOUR_SEC_API_KEY",
            "REDDIT_CLIENT_ID": "YOUR_REDDIT_CLIENT_ID",
            "REDDIT_CLIENT_SECRET": "YOUR_REDDIT_CLIENT_SECRET",
            "TWITTER_BEARER_TOKEN": "YOUR_TWITTER_BEARER_TOKEN"
        }
        with open("config_api_keys", "w") as f:
            json.dump(api_keys, f, indent=4)
        print("✅ Created config_api_keys")
    
    return True

def create_demo_structure():
    """Create demo folder structure"""
    print("\n📁 Creating demo folder structure...")
    
    demo_dirs = ["demo", "demo/logs", "demo/outputs", "demo/reports"]
    for dir_path in demo_dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    print("✅ Demo folder structure created")
    return True

def create_run_script():
    """Create a simple run script"""
    run_script = """#!/usr/bin/env python3
# Quick run script for FinRobot AAPL Demo

import sys
import os

def main():
    print("🚀 Starting FinRobot AAPL Demo...")
    print("=" * 50)
    
    # Check if API keys are configured
    if not os.path.exists("OAI_CONFIG_LIST") or not os.path.exists("config_api_keys"):
        print("❌ Configuration files not found!")
        print("Please run setup_demo.py first")
        return
    
    # Check if API keys are still placeholders
    with open("OAI_CONFIG_LIST", "r") as f:
        oai_config = f.read()
        if "<your" in oai_config:
            print("⚠️  Please configure your API keys in OAI_CONFIG_LIST and config_api_keys")
            print("Edit the files and replace the placeholder values with your actual API keys")
            return
    
    # Run the demo
    try:
        import minimal_aapl_demo
        minimal_aapl_demo.run_minimal_aapl_prediction()
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please install dependencies: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Error running demo: {e}")

if __name__ == "__main__":
    main()
"""
    
    with open("run_demo.py", "w") as f:
        f.write(run_script)
    
    print("✅ Created run_demo.py")
    return True

def main():
    """Main setup function"""
    print("🎯 FinRobot AAPL Demo Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Install dependencies
    if not install_dependencies():
        return False
    
    # Create config files
    if not create_config_files():
        return False
    
    # Create demo structure
    if not create_demo_structure():
        return False
    
    # Create run script
    if not create_run_script():
        return False
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("=" * 50)
    print("\n📋 Next steps:")
    print("1. Configure your API keys:")
    print("   - Edit OAI_CONFIG_LIST with your OpenAI/Gemini API keys")
    print("   - Edit config_api_keys with your financial data API keys")
    print("\n2. Run the demo:")
    print("   python run_demo.py")
    print("   or")
    print("   python minimal_aapl_demo.py")
    print("\n3. Check the demo/ folder for results")
    print("\n📚 For more information, see README_FINROBOT_DEMO.md")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
