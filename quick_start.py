#!/usr/bin/env python3
"""
Quick Start Script for Fin-o1-14B
This script provides a simple way to test the model with predefined prompts.
"""

from transformers import pipeline
import torch

def quick_test():
    """Run a quick test of the Fin-o1-14B model."""
    
    model_name = "TheFinAI/Fin-o1-14B"
    
    print("🚀 Quick Start: Fin-o1-14B Model")
    print("=" * 50)
    
    # Check device
    device = 0 if torch.cuda.is_available() else -1
    device_name = "GPU" if device >= 0 else "CPU"
    print(f"📱 Using device: {device_name}")
    
    if device == -1:
        print("⚠️  Warning: Running on CPU will be very slow!")
        print("   Consider using a GPU for better performance.")
    
    try:
        print(f"\n📥 Loading model: {model_name}")
        print("   This may take several minutes...")
        
        # Load model
        generator = pipeline(
            "text-generation",
            model=model_name,
            device=device,
            torch_dtype=torch.float16 if device >= 0 else torch.float32,
            trust_remote_code=True
        )
        
        print("✅ Model loaded successfully!")
        
        # Test prompts
        test_prompts = [
            "What is the stock market?",
            "Explain inflation in simple terms.",
            "What are the benefits of diversification in investing?",
            "How do interest rates affect the economy?"
        ]
        
        print(f"\n🧪 Running {len(test_prompts)} test prompts...")
        print("-" * 50)
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n📝 Test {i}: {prompt}")
            print("⏳ Generating response...")
            
            try:
                outputs = generator(
                    prompt,
                    max_length=256,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=generator.tokenizer.eos_token_id
                )
                
                response = outputs[0]['generated_text']
                
                # Clean up response
                if response.startswith(prompt):
                    response = response[len(prompt):].strip()
                
                print(f"💬 Response: {response}")
                
            except Exception as e:
                print(f"❌ Error generating response: {e}")
            
            print("-" * 30)
        
        print("\n🎉 Quick test completed!")
        print("\n💡 To run the full interactive mode:")
        print("   python3 simple_run.py")
        print("\n💡 To run the advanced version with quantization:")
        print("   python3 run_model.py")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("\n🔧 Troubleshooting tips:")
        print("   1. Check if you have sufficient memory")
        print("   2. Ensure all dependencies are installed")
        print("   3. Check your internet connection")
        print("   4. Verify you have enough disk space (~30GB)")

if __name__ == "__main__":
    quick_test()