#!/usr/bin/env python3
"""
Quick test script to verify Fin-o1-8B model setup
This script tests if the model can be imported and loaded without errors.
"""

import sys
import traceback

def test_imports():
    """Test if all required packages can be imported."""
    print("🧪 Testing package imports...")
    
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        
        from transformers import AutoTokenizer, AutoModelForCausalLM
        print("✅ Transformers: Imported successfully")
        
        import accelerate
        print("✅ Accelerate: Imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_model_download():
    """Test if the model can be downloaded (this will take time on first run)."""
    print("\n🧪 Testing model download capability...")
    
    try:
        from transformers import AutoTokenizer
        
        print("📥 Attempting to download tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("TheFinAI/Fin-o1-8B")
        print("✅ Tokenizer downloaded successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Model download error: {e}")
        print("💡 This might be due to network issues or insufficient disk space")
        return False

def main():
    """Main test function."""
    print("🚀 Fin-o1-8B Model Setup Test")
    print("=" * 40)
    
    # Test imports
    if not test_imports():
        print("\n❌ Package import test failed!")
        print("💡 Please run: pip install -r requirements.txt")
        sys.exit(1)
    
    # Test model download
    if not test_model_download():
        print("\n❌ Model download test failed!")
        print("💡 Please check your internet connection and disk space")
        sys.exit(1)
    
    print("\n🎉 All tests passed!")
    print("✅ Your Fin-o1-8B setup is ready!")
    print("\n🚀 You can now run:")
    print("   python fin_o1_simple.py")
    print("   python fin_o1_demo.py")

if __name__ == "__main__":
    main()