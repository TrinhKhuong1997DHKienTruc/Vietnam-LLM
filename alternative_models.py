#!/usr/bin/env python3
"""
Alternative Models for Limited Resources
This script suggests smaller models that work better with limited RAM and no GPU.
"""

def suggest_alternatives():
    """Suggest alternative models based on system capabilities."""
    print("🔍 System Analysis")
    print("="*50)
    
    # Check system resources
    import psutil
    import torch
    
    ram_gb = psutil.virtual_memory().total / 1024**3
    cuda_available = torch.cuda.is_available()
    
    print(f"Total RAM: {ram_gb:.1f} GB")
    print(f"CUDA Available: {cuda_available}")
    
    print("\n📊 Model Recommendations")
    print("="*50)
    
    if ram_gb < 16:
        print("❌ Your system has insufficient RAM for most LLMs")
        print("   Consider upgrading to at least 16GB RAM")
        return
    
    elif ram_gb < 32:
        print("⚠️  Limited RAM detected. Recommended models:")
        print("\n🟡 3B Parameter Models (8-12GB RAM):")
        print("   • TheBloke/Llama-2-7B-Chat-GGUF")
        print("   • microsoft/DialoGPT-medium")
        print("   • gpt2-medium")
        
        print("\n🟢 1B Parameter Models (4-8GB RAM):")
        print("   • microsoft/DialoGPT-small")
        print("   • gpt2")
        print("   • distilgpt2")
        
    elif ram_gb < 64:
        print("🟡 Moderate RAM. Recommended models:")
        print("\n🟡 7B Parameter Models (16-24GB RAM):")
        print("   • TheBloke/Llama-2-7B-Chat-GGUF")
        print("   • microsoft/DialoGPT-large")
        print("   • facebook/opt-6.7b")
        
        print("\n🟢 3B Parameter Models (8-12GB RAM):")
        print("   • microsoft/DialoGPT-medium")
        print("   • gpt2-medium")
        
    else:
        print("🟢 Sufficient RAM for larger models:")
        print("\n🟢 14B Parameter Models (32-48GB RAM):")
        print("   • TheFinAI/Fin-o1-14B (your original choice)")
        print("   • TheBloke/Llama-2-13B-Chat-GGUF")
        
        print("\n🟡 7B Parameter Models (16-24GB RAM):")
        print("   • TheBloke/Llama-2-7B-Chat-GGUF")
    
    print("\n💡 Optimization Tips")
    print("="*50)
    print("1. Use quantization (4-bit or 8-bit)")
    print("2. Enable low_cpu_mem_usage=True")
    print("3. Use smaller max_length for generation")
    print("4. Close other applications to free RAM")
    
    if not cuda_available:
        print("\n⚠️  No GPU detected:")
        print("   • CPU inference will be very slow")
        print("   • Consider cloud-based solutions")
        print("   • Or use much smaller models")

def run_small_model_example():
    """Example of running a small model that fits in limited RAM."""
    print("\n🚀 Quick Start with Small Model")
    print("="*50)
    
    try:
        from transformers import pipeline
        
        print("Loading a small model (GPT-2)...")
        generator = pipeline(
            "text-generation",
            model="gpt2",
            device_map="cpu"
        )
        
        print("✅ Model loaded successfully!")
        
        # Test generation
        prompt = "The future of artificial intelligence"
        print(f"\nTesting with prompt: '{prompt}'")
        
        outputs = generator(
            prompt,
            max_length=100,
            num_return_sequences=1,
            temperature=0.7
        )
        
        print(f"Generated text:\n{outputs[0]['generated_text']}")
        
        print("\n💡 This small model:")
        print("   • Loads quickly")
        print("   • Uses minimal RAM")
        print("   • Works on CPU")
        print("   • Provides decent text generation")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Try installing dependencies first: pip install transformers torch")

def main():
    """Main function."""
    print("🤖 Alternative Model Suggestions")
    print("="*50)
    
    suggest_alternatives()
    
    print("\n" + "="*50)
    response = input("Would you like to try a small model example? (y/N): ").strip().lower()
    
    if response == 'y':
        run_small_model_example()
    
    print("\n📚 Next Steps:")
    print("1. Choose a model from the recommendations above")
    print("2. Install dependencies: pip install transformers torch")
    print("3. Use the model with appropriate quantization")
    print("4. Consider cloud-based solutions for larger models")

if __name__ == "__main__":
    main()