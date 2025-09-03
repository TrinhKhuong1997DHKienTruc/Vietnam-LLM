#!/usr/bin/env python3
"""
Quick Start Script for FinRobot
This script helps users get started with FinRobot quickly.
"""

import os
import sys
import json
import subprocess
from pathlib import Path

def print_banner():
    """Print the FinRobot banner."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║                    🚀 FinRobot Quick Start                    ║
║                                                              ║
║  AI Agent Platform for Financial Analysis                    ║
║  Made with ❤️ for the Vietnamese AI community               ║
╚══════════════════════════════════════════════════════════════╝
    """)

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Error: Python 3.8 or higher is required!")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def install_dependencies():
    """Install required dependencies."""
    print("\n📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Error installing dependencies!")
        return False

def setup_config_files():
    """Set up configuration files."""
    print("\n⚙️  Setting up configuration files...")
    
    # Check if config files exist
    config_files = ["OAI_CONFIG_LIST", "config_api_keys"]
    for file in config_files:
        if not os.path.exists(file):
            print(f"⚠️  Warning: {file} not found!")
            continue
        
        print(f"✅ {file} found")
    
    print("\n📝 Please configure your API keys:")
    print("1. Edit 'OAI_CONFIG_LIST' with your OpenAI API key")
    print("2. Edit 'config_api_keys' with your financial API keys")
    print("3. See README_FINROBOT.md for detailed instructions")

def run_demo():
    """Run the FinRobot demo."""
    print("\n🎯 Running FinRobot Market Forecaster Demo...")
    try:
        subprocess.check_call([sys.executable, "mock_forecaster_demo.py"])
        print("✅ Demo completed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Error running demo!")
        return False

def show_next_steps():
    """Show next steps for users."""
    print("""
🎉 Congratulations! FinRobot is now set up!

📚 Next Steps:
1. Configure your API keys in the config files
2. Run the demo: python mock_forecaster_demo.py
3. Explore tutorials in tutorials_beginner/
4. Read README_FINROBOT.md for detailed documentation

🔗 Useful Links:
- Repository: https://github.com/TrinhKhuong1997DHKienTruc/Vietnam-LLM
- Documentation: README_FINROBOT.md
- Tutorials: tutorials_beginner/

🤝 Need Help?
- Check the troubleshooting section in README_FINROBOT.md
- Open an issue on GitHub
- Contact the maintainer

Happy coding! 🚀
    """)

def main():
    """Main function for quick start."""
    print_banner()
    
    # Check Python version
    if not check_python_version():
        return
    
    # Install dependencies
    if not install_dependencies():
        return
    
    # Setup config files
    setup_config_files()
    
    # Run demo
    if run_demo():
        show_next_steps()
    else:
        print("\n❌ Demo failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
