"""
Setup script for NVDA Stock Prediction with Chronos-T5
This script helps users set up the environment and run the prediction.
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    print(f"Python version: {sys.version}")
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required!")
        return False
    print("✅ Python version is compatible!")
    return True

def create_demo_folder():
    """Create demo_reports folder if it doesn't exist"""
    if not os.path.exists("demo_reports"):
        os.makedirs("demo_reports")
        print("✅ Created demo_reports folder")
    else:
        print("✅ demo_reports folder already exists")

def run_prediction():
    """Run the NVDA prediction script"""
    print("\n🚀 Starting NVDA stock prediction...")
    try:
        subprocess.check_call([sys.executable, "chronos_nvda_prediction.py"])
        print("✅ Prediction completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running prediction: {e}")
        return False

def main():
    """Main setup function"""
    print("=" * 60)
    print("NVDA Stock Prediction Setup")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Create demo folder
    create_demo_folder()
    
    # Install requirements
    if not install_requirements():
        return False
    
    # Ask user if they want to run prediction
    response = input("\n🤔 Would you like to run the NVDA prediction now? (y/n): ").lower().strip()
    if response in ['y', 'yes']:
        return run_prediction()
    else:
        print("✅ Setup complete! Run 'python chronos_nvda_prediction.py' when ready.")
        return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Setup completed successfully!")
    else:
        print("\n💥 Setup failed. Please check the error messages above.")
        sys.exit(1)
