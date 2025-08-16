#!/usr/bin/env python3
"""
Test Setup Script for Fin-o1-14B
This script verifies that all dependencies are properly installed.
"""

import sys

def test_imports():
    """Test if all required packages can be imported."""
    
    print("🧪 Testing package imports...")
    print("=" * 40)
    
    packages = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("accelerate", "Accelerate"),
        ("sentencepiece", "SentencePiece"),
        ("google.protobuf", "Protobuf"),
        ("huggingface_hub", "Hugging Face Hub"),
        ("numpy", "NumPy")
    ]
    
    all_good = True
    
    for package, name in packages:
        try:
            __import__(package)
            print(f"✅ {name} ({package}) - OK")
        except ImportError as e:
            print(f"❌ {name} ({package}) - FAILED: {e}")
            all_good = False
    
    return all_good

def test_torch():
    """Test PyTorch functionality."""
    
    print("\n🔥 Testing PyTorch functionality...")
    print("=" * 40)
    
    try:
        import torch
        
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA version: {torch.version.cuda}")
            print(f"✅ GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("ℹ️  Running on CPU")
        
        # Test basic tensor operations
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        z = torch.mm(x, y)
        print(f"✅ Basic tensor operations: OK (result shape: {z.shape})")
        
        return True
        
    except Exception as e:
        print(f"❌ PyTorch test failed: {e}")
        return False

def test_transformers():
    """Test Transformers library."""
    
    print("\n🤗 Testing Transformers library...")
    print("=" * 40)
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        print("✅ AutoTokenizer import: OK")
        print("✅ AutoModelForCausalLM import: OK")
        
        # Test tokenizer creation (without downloading)
        try:
            tokenizer = AutoTokenizer.from_pretrained("gpt2", trust_remote_code=True)
            print("✅ Tokenizer creation test: OK")
        except Exception as e:
            print(f"⚠️  Tokenizer creation test: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Transformers test failed: {e}")
        return False

def test_memory():
    """Test available memory."""
    
    print("\n💾 Testing system memory...")
    print("=" * 40)
    
    try:
        import psutil
        
        memory = psutil.virtual_memory()
        print(f"✅ Total RAM: {memory.total // (1024**3):.1f} GB")
        print(f"✅ Available RAM: {memory.available // (1024**3):.1f} GB")
        print(f"✅ RAM usage: {memory.percent:.1f}%")
        
        if memory.available < 8 * (1024**3):  # Less than 8GB
            print("⚠️  Warning: Less than 8GB RAM available")
            print("   The 14B model may not load or run very slowly")
        else:
            print("✅ Sufficient RAM available")
        
        return True
        
    except ImportError:
        print("⚠️  psutil not available - cannot check memory")
        return True
    except Exception as e:
        print(f"❌ Memory check failed: {e}")
        return False

def test_disk_space():
    """Test available disk space."""
    
    print("\n💿 Testing disk space...")
    print("=" * 40)
    
    try:
        import shutil
        
        total, used, free = shutil.disk_usage(".")
        total_gb = total // (1024**3)
        free_gb = free // (1024**3)
        
        print(f"✅ Total disk space: {total_gb:.1f} GB")
        print(f"✅ Free disk space: {free_gb:.1f} GB")
        
        if free_gb < 30:
            print("⚠️  Warning: Less than 30GB free space")
            print("   The model requires ~30GB for download and storage")
        else:
            print("✅ Sufficient disk space available")
        
        return True
        
    except Exception as e:
        print(f"❌ Disk space check failed: {e}")
        return False

def main():
    """Run all tests."""
    
    print("🚀 Fin-o1-14B Setup Test")
    print("=" * 50)
    
    tests = [
        ("Package Imports", test_imports),
        ("PyTorch Functionality", test_torch),
        ("Transformers Library", test_transformers),
        ("System Memory", test_memory),
        ("Disk Space", test_disk_space)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your environment is ready.")
        print("\n💡 Next steps:")
        print("   1. Run: python3 lightweight_run.py")
        print("   2. Or run: python3 simple_run.py")
        print("   3. Or run: python3 run_model.py")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed.")
        print("   Please check the errors above and fix them before proceeding.")
        
        if passed < 3:  # Critical failures
            print("\n🔧 Critical issues detected:")
            print("   - Missing core dependencies")
            print("   - PyTorch not working")
            print("   - Insufficient system resources")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)