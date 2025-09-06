#!/usr/bin/env python3
"""
FinRobot Setup Script
Automates the setup process for the FinRobot project
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 13):
        print("❌ Error: Python 3.13+ is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def create_virtual_environment():
    """Create and activate virtual environment"""
    venv_name = "finrobot_env"
    
    if os.path.exists(venv_name):
        print(f"✅ Virtual environment '{venv_name}' already exists")
        return True
    
    try:
        print(f"🔧 Creating virtual environment '{venv_name}'...")
        subprocess.run([sys.executable, "-m", "venv", venv_name], check=True)
        print("✅ Virtual environment created successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error creating virtual environment: {e}")
        return False

def install_dependencies():
    """Install required dependencies"""
    venv_name = "finrobot_env"
    
    # Determine the pip path based on OS
    if os.name == 'nt':  # Windows
        pip_path = os.path.join(venv_name, "Scripts", "pip")
        activate_path = os.path.join(venv_name, "Scripts", "activate")
    else:  # Unix/Linux/macOS
        pip_path = os.path.join(venv_name, "bin", "pip")
        activate_path = os.path.join(venv_name, "bin", "activate")
    
    try:
        print("📦 Installing dependencies...")
        
        # Upgrade pip first
        subprocess.run([pip_path, "install", "--upgrade", "pip"], check=True)
        
        # Install requirements
        subprocess.run([pip_path, "install", "-r", "requirements_simple.txt"], check=True)
        
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def create_sample_configs():
    """Create sample configuration files if they don't exist"""
    
    # Create sample config_api_keys if it doesn't exist
    if not os.path.exists("config_api_keys_sample"):
        sample_config = {
            "FINNHUB_API_KEY": "YOUR_FINNHUB_API_KEY_HERE",
            "FMP_API_KEY": "YOUR_FMP_API_KEY_HERE",
            "SEC_API_KEY": "YOUR_SEC_API_KEY_HERE",
            "REDDIT_CLIENT_ID": "YOUR_REDDIT_CLIENT_ID",
            "REDDIT_CLIENT_SECRET": "YOUR_REDDIT_CLIENT_SECRET",
            "TWITTER_BEARER_TOKEN": "YOUR_TWITTER_BEARER_TOKEN"
        }
        
        with open("config_api_keys_sample", "w") as f:
            json.dump(sample_config, f, indent=4)
        print("✅ Created config_api_keys_sample")
    
    # Create sample OAI_CONFIG_LIST if it doesn't exist
    if not os.path.exists("OAI_CONFIG_LIST_sample"):
        sample_oai_config = [
            {
                "model": "gpt-5-mini-2025-08-07",
                "api_key": "YOUR_OPENAI_API_KEY_HERE"
            }
        ]
        
        with open("OAI_CONFIG_LIST_sample", "w") as f:
            json.dump(sample_oai_config, f, indent=4)
        print("✅ Created OAI_CONFIG_LIST_sample")

def run_test():
    """Run a simple test to verify installation"""
    try:
        print("🧪 Running test...")
        
        # Test basic imports
        import requests
        import pandas
        import numpy
        print("✅ Basic imports successful")
        
        # Test API configuration loading
        if os.path.exists("config_api_keys"):
            with open("config_api_keys", "r") as f:
                config = json.load(f)
            print("✅ Configuration loading successful")
        else:
            print("⚠️  Configuration file not found - please configure API keys")
        
        print("✅ Test completed successfully")
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\n📋 NEXT STEPS:")
    print("1. Configure your API keys:")
    print("   - Edit 'config_api_keys' with your API keys")
    print("   - Edit 'OAI_CONFIG_LIST' with your OpenAI configuration")
    print("\n2. Activate the virtual environment:")
    if os.name == 'nt':  # Windows
        print("   finrobot_env\\Scripts\\activate")
    else:  # Unix/Linux/macOS
        print("   source finrobot_env/bin/activate")
    print("\n3. Run the demo:")
    print("   python simple_demo.py")
    print("\n4. Generate comprehensive reports:")
    print("   python generate_reports.py")
    print("\n📚 For more information, see README.md")
    print("="*60)

def main():
    """Main setup function"""
    print("🚀 FinRobot Setup Script")
    print("="*40)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create virtual environment
    if not create_virtual_environment():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    # Create sample configs
    create_sample_configs()
    
    # Run test
    if not run_test():
        print("⚠️  Setup completed with warnings")
    
    # Print next steps
    print_next_steps()

if __name__ == "__main__":
    main()
