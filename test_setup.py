#!/usr/bin/env python3
"""
Test script to verify the Fin-o1-14B setup
"""

import sys
import subprocess

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required")
        return False
    else:
        print("✅ Python version is compatible")
        return True

def check_packages():
    """Check if required packages are available"""
    required_packages = [
        'torch', 'transformers', 'accelerate', 'bitsandbytes',
        'sentencepiece', 'protobuf', 'huggingface_hub'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} is available")
        except ImportError:
            print(f"❌ {package} is missing")
            missing_packages.append(package)
    
    return len(missing_packages) == 0, missing_packages

def check_cuda():
    """Check CUDA availability"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA is available")
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            return True
        else:
            print("⚠️  CUDA is not available (will run on CPU - very slow)")
            return False
    except ImportError:
        print("❌ PyTorch is not installed")
        return False

def check_disk_space():
    """Check available disk space"""
    try:
        import shutil
        total, used, free = shutil.disk_usage(".")
        free_gb = free / (1024**3)
        print(f"✅ Available disk space: {free_gb:.1f} GB")
        
        if free_gb < 30:
            print("⚠️  Warning: Less than 30GB available. The model requires ~28GB.")
            return False
        return True
    except Exception as e:
        print(f"❌ Could not check disk space: {e}")
        return False

def check_internet():
    """Check internet connectivity"""
    try:
        import urllib.request
        urllib.request.urlopen('https://huggingface.co', timeout=10)
        print("✅ Internet connectivity to Hugging Face is available")
        return True
    except Exception as e:
        print(f"❌ Internet connectivity check failed: {e}")
        return False

def main():
    print("=== Fin-o1-14B Environment Check ===\n")
    
    checks = [
        ("Python Version", check_python_version),
        ("Required Packages", lambda: check_packages()[0]),
        ("CUDA Availability", check_cuda),
        ("Disk Space", check_disk_space),
        ("Internet Connectivity", check_internet)
    ]
    
    results = []
    for check_name, check_func in checks:
        print(f"\n--- {check_name} ---")
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"❌ Error during {check_name} check: {e}")
            results.append((check_name, False))
    
    print("\n=== Summary ===")
    all_passed = True
    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{check_name}: {status}")
        if not result:
            all_passed = False
    
    print("\n" + "="*50)
    if all_passed:
        print("🎉 All checks passed! You're ready to run Fin-o1-14B.")
        print("\nNext steps:")
        print("1. Activate virtual environment: source fin_o1_env/bin/activate")
        print("2. Run the model: python run_fin_o1.py")
    else:
        print("⚠️  Some checks failed. Please address the issues above before proceeding.")
        print("\nCommon solutions:")
        print("- Run ./setup.sh to install dependencies")
        print("- Ensure you have sufficient disk space")
        print("- Check your internet connection")
        print("- Verify CUDA installation if using GPU")

if __name__ == "__main__":
    main()