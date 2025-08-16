#!/usr/bin/env python3
"""
Test Setup Script for Fin-o1-8B
Verifies that all dependencies are properly installed and the environment is ready.
"""

import sys
import importlib
import subprocess
import platform

def check_python_version():
    """Check Python version."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    print(f"   Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("   ❌ Python 3.8+ is required!")
        return False
    else:
        print("   ✅ Python version is compatible")
        return True

def check_package(package_name, import_name=None):
    """Check if a package is installed."""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        print(f"   ✅ {package_name} is installed")
        return True
    except ImportError:
        print(f"   ❌ {package_name} is NOT installed")
        return False

def check_cuda():
    """Check CUDA availability."""
    print("🎮 Checking CUDA availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"   ✅ CUDA is available")
            print(f"   🎯 GPU: {torch.cuda.get_device_name(0)}")
            print(f"   🧠 CUDA Version: {torch.version.cuda}")
            return True
        else:
            print("   ⚠️  CUDA is not available (will use CPU)")
            return False
    except ImportError:
        print("   ❌ PyTorch is not installed")
        return False

def check_system_info():
    """Check system information."""
    print("💻 System Information:")
    print(f"   OS: {platform.system()} {platform.release()}")
    print(f"   Architecture: {platform.machine()}")
    print(f"   Processor: {platform.processor()}")

def main():
    """Main test function."""
    print("🧪 Fin-o1-8B Setup Test")
    print("=" * 50)
    
    # Check Python version
    python_ok = check_python_version()
    
    # Check system info
    check_system_info()
    
    # Check CUDA
    cuda_ok = check_cuda()
    
    print("\n📦 Checking required packages...")
    
    # Core packages
    core_packages = [
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("accelerate", "accelerate"),
        ("sentencepiece", "sentencepiece"),
        ("protobuf", "google.protobuf"),
        ("huggingface-hub", "huggingface_hub"),
    ]
    
    core_ok = True
    for package, import_name in core_packages:
        if not check_package(package, import_name):
            core_ok = False
    
    print("\n🎨 Checking optional packages...")
    
    # Optional packages for web interfaces
    optional_packages = [
        ("streamlit", "streamlit"),
        ("gradio", "gradio"),
        ("pandas", "pandas"),
        ("plotly", "plotly"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
    ]
    
    optional_ok = True
    for package, import_name in optional_packages:
        if not check_package(package, import_name):
            optional_ok = False
    
    print("\n📊 Test Results:")
    print("=" * 50)
    
    if python_ok and core_ok:
        print("✅ Core setup is ready!")
        print("🚀 You can now run: python fin_o1_demo.py")
        
        if cuda_ok:
            print("🎮 GPU acceleration is available")
        else:
            print("🖥️  Will run on CPU (slower but functional)")
        
        if optional_ok:
            print("🎨 Web interfaces are available:")
            print("   - Streamlit: streamlit run streamlit_app.py")
            print("   - Gradio: python gradio_app.py")
        else:
            print("⚠️  Some web interface packages are missing")
            print("   Install with: pip install -r requirements.txt")
    else:
        print("❌ Setup is incomplete!")
        if not python_ok:
            print("   Please upgrade to Python 3.8+")
        if not core_ok:
            print("   Please install core packages: pip install -r requirements.txt")
    
    print("\n📚 Next steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Run command line demo: python fin_o1_demo.py")
    print("3. Run web interface: streamlit run streamlit_app.py")
    print("4. Check README.md for detailed instructions")

if __name__ == "__main__":
    main()