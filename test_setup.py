#!/usr/bin/env python3
"""
Simple Setup Test for Fin-o1-8B
Tests the environment setup without loading the full model
"""

import sys
import importlib

def test_imports():
    """Test if all required packages can be imported"""
    print("🧪 Testing Package Imports")
    print("-" * 30)
    
    required_packages = [
        'torch',
        'transformers',
        'accelerate',
        'sentencepiece',
        'google.protobuf',
        'numpy',
        'scipy',
        'datasets',
        'huggingface_hub',
        'einops',
        'xformers',
        'bitsandbytes',
        'optimum'
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            module = importlib.import_module(package)
            version = getattr(module, '__version__', 'Unknown')
            print(f"✅ {package}: {version}")
        except ImportError as e:
            print(f"❌ {package}: {e}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        return False
    else:
        print("\n✅ All packages imported successfully!")
        return True

def test_torch():
    """Test PyTorch functionality"""
    print("\n🔥 Testing PyTorch")
    print("-" * 20)
    
    try:
        import torch
        
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("CUDA not available - will use CPU")
        
        # Test basic tensor operations
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        z = torch.mm(x, y)
        print(f"✅ Basic tensor operations: {z.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ PyTorch test failed: {e}")
        return False

def test_transformers():
    """Test Transformers library"""
    print("\n🤖 Testing Transformers")
    print("-" * 25)
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        print("✅ Transformers classes imported successfully")
        
        # Test tokenizer loading (this will download a small test model)
        print("🔧 Testing tokenizer download...")
        tokenizer = AutoTokenizer.from_pretrained("gpt2", trust_remote_code=True)
        print(f"✅ Tokenizer loaded: {tokenizer.__class__.__name__}")
        
        # Test basic tokenization
        text = "Hello, world!"
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        print(f"✅ Tokenization test: '{text}' -> {len(tokens)} tokens -> '{decoded}'")
        
        return True
        
    except Exception as e:
        print(f"❌ Transformers test failed: {e}")
        return False

def test_huggingface_hub():
    """Test Hugging Face Hub access"""
    print("\n🌐 Testing Hugging Face Hub")
    print("-" * 30)
    
    try:
        from huggingface_hub import HfApi
        
        api = HfApi()
        
        # Test model info access
        print("🔍 Testing model info access...")
        model_info = api.model_info("TheFinAI/Fin-o1-8B")
        print(f"✅ Model found: {model_info.modelId}")
        print(f"   Downloads: {model_info.downloads}")
        print(f"   Likes: {model_info.likes}")
        
        return True
        
    except Exception as e:
        print(f"❌ Hugging Face Hub test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Fin-o1-8B Setup Test")
    print("=" * 40)
    
    tests = [
        ("Package Imports", test_imports),
        ("PyTorch", test_torch),
        ("Transformers", test_transformers),
        ("Hugging Face Hub", test_huggingface_hub)
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
    print("\n" + "=" * 40)
    print("📊 Test Summary")
    print("=" * 40)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your environment is ready for Fin-o1-8B.")
        print("\n🚀 Next steps:")
        print("  1. Run: python run_fin_o1.py --four-bit --interactive")
        print("  2. Or try: python financial_demo.py")
        print("\n💡 Note: The full model is ~16GB, so ensure you have sufficient memory.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())