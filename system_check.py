#!/usr/bin/env python3
"""
System Check for Fin-o1-14B Model
This script checks system capabilities without requiring additional packages.
"""

import os
import sys
import subprocess
import platform

def check_python():
    """Check Python version and capabilities."""
    print("🐍 Python Environment")
    print("="*40)
    print(f"Python Version: {sys.version}")
    print(f"Python Executable: {sys.executable}")
    print(f"Platform: {platform.platform()}")
    
    # Check if we can import key modules
    modules_to_check = ['torch', 'transformers', 'numpy']
    print(f"\n📦 Available Modules:")
    for module in modules_to_check:
        try:
            __import__(module)
            print(f"   ✅ {module}")
        except ImportError:
            print(f"   ❌ {module}")

def check_memory():
    """Check available memory using system commands."""
    print(f"\n💾 Memory Information")
    print("="*40)
    
    try:
        # Try to read /proc/meminfo
        with open('/proc/meminfo', 'r') as f:
            meminfo = f.read()
        
        # Parse memory info
        total_mem = 0
        available_mem = 0
        
        for line in meminfo.split('\n'):
            if line.startswith('MemTotal:'):
                total_mem = int(line.split()[1]) // 1024  # Convert to MB
            elif line.startswith('MemAvailable:'):
                available_mem = int(line.split()[1]) // 1024  # Convert to MB
        
        print(f"Total Memory: {total_mem} MB ({total_mem/1024:.1f} GB)")
        print(f"Available Memory: {available_mem} MB ({available_mem/1024:.1f} GB)")
        
        # Assess memory adequacy
        if total_mem < 16384:  # Less than 16GB
            print("❌ Insufficient RAM for most LLMs")
            print("   Minimum recommended: 16GB")
        elif total_mem < 32768:  # Less than 32GB
            print("⚠️  Limited RAM - only small models recommended")
            print("   Recommended: 32GB+ for 7B+ models")
        else:
            print("✅ Sufficient RAM for larger models")
            
    except Exception as e:
        print(f"Could not read memory info: {e}")

def check_gpu():
    """Check GPU availability."""
    print(f"\n🎮 GPU Information")
    print("="*40)
    
    # Check for NVIDIA GPU
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ NVIDIA GPU detected")
            print("GPU output:")
            print(result.stdout[:500] + "..." if len(result.stdout) > 500 else result.stdout)
        else:
            print("❌ NVIDIA GPU not available")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ NVIDIA GPU not available")
    
    # Check for other GPU types
    try:
        result = subprocess.run(['lspci'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            gpu_lines = [line for line in result.stdout.split('\n') if 'VGA' in line or '3D' in line]
            if gpu_lines:
                print(f"\n🔍 Other graphics devices found:")
                for line in gpu_lines:
                    print(f"   {line}")
            else:
                print("No dedicated graphics devices found")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("Could not check for other graphics devices")

def check_storage():
    """Check available storage."""
    print(f"\n💿 Storage Information")
    print("="*40)
    
    try:
        result = subprocess.run(['df', '-h', '.'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:
                parts = lines[1].split()
                if len(parts) >= 4:
                    total = parts[1]
                    used = parts[2]
                    available = parts[3]
                    print(f"Current directory storage:")
                    print(f"   Total: {total}")
                    print(f"   Used: {used}")
                    print(f"   Available: {available}")
                    
                    # Check if we have enough space for model
                    if 'G' in available:
                        available_gb = float(available.replace('G', ''))
                        if available_gb < 50:
                            print("❌ Insufficient storage for 14B model")
                            print("   Need at least 50GB free space")
                        else:
                            print("✅ Sufficient storage for model download")
        else:
            print("Could not check storage")
    except Exception as e:
        print(f"Could not check storage: {e}")

def provide_recommendations():
    """Provide recommendations based on system capabilities."""
    print(f"\n📋 Recommendations")
    print("="*40)
    
    print("Based on your system analysis:")
    print("\n🚀 For Fin-o1-14B (14B parameters):")
    print("   • Requires 32GB+ RAM")
    print("   • Requires 50GB+ storage")
    print("   • GPU recommended (24GB+ VRAM)")
    
    print("\n🟡 Alternative approaches:")
    print("   1. Use cloud-based solutions (Google Colab, AWS, etc.)")
    print("   2. Try smaller models (7B, 3B, or 1B parameters)")
    print("   3. Use quantized versions (4-bit, 8-bit)")
    print("   4. Upgrade system resources if possible")
    
    print("\n💡 Next steps:")
    print("   1. Install Python dependencies in virtual environment")
    print("   2. Try the CPU-optimized script")
    print("   3. Consider cloud alternatives for full model")

def main():
    """Main function."""
    print("🔍 System Check for Fin-o1-14B Model")
    print("="*50)
    
    check_python()
    check_memory()
    check_gpu()
    check_storage()
    provide_recommendations()
    
    print(f"\n" + "="*50)
    print("System check complete!")
    print("Use this information to decide on the best approach.")

if __name__ == "__main__":
    main()