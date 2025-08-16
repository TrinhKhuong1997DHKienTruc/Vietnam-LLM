#!/usr/bin/env python3
"""
CPU-Optimized Fin-o1-14B Runner
This version is designed for systems with limited RAM and no GPU.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import gc
import os

def check_memory():
    """Check available memory and warn if insufficient."""
    import psutil
    ram_gb = psutil.virtual_memory().available / 1024**3
    print(f"Available RAM: {ram_gb:.1f} GB")
    
    if ram_gb < 20:
        print("⚠️  WARNING: Less than 20GB RAM available.")
        print("   The model may not load or may run very slowly.")
        print("   Consider using a smaller model or a system with more RAM.")
        return False
    return True

def load_model_cpu_optimized():
    """Load the model with CPU-optimized settings."""
    model_name = "TheFinAI/Fin-o1-14B"
    
    print("Loading Fin-o1-14B model with CPU optimization...")
    print("This will take several minutes and requires significant RAM...")
    
    try:
        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir="./models"
        )
        
        # Use 8-bit quantization for CPU
        print("Loading model with 8-bit quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False
        )
        
        # Load model with CPU optimizations
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="cpu",
            trust_remote_code=True,
            cache_dir="./models",
            torch_dtype=torch.float32,  # Use float32 for CPU
            low_cpu_mem_usage=True
        )
        
        print("✅ Model loaded successfully!")
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("\nPossible solutions:")
        print("1. Ensure you have at least 20GB RAM available")
        print("2. Close other applications to free up memory")
        print("3. Consider using a smaller model")
        raise

def generate_text_cpu(model, tokenizer, prompt, max_length=100):
    """Generate text using CPU-optimized settings."""
    try:
        print("Generating text (this may be slow on CPU)...")
        
        # Encode input
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        
        # Generate with CPU-optimized settings
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
                length_penalty=1.0
            )
        
        # Decode output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up memory
        del outputs, inputs
        gc.collect()
        
        return generated_text
        
    except Exception as e:
        print(f"Error generating text: {e}")
        raise

def interactive_chat_cpu():
    """Run interactive chat with CPU-optimized settings."""
    print("\n" + "="*60)
    print("Fin-o1-14B CPU-Optimized Chat")
    print("⚠️  Running on CPU - responses will be slow!")
    print("Type 'quit' to exit")
    print("="*60 + "\n")
    
    # Check memory before loading
    if not check_memory():
        response = input("Continue anyway? (y/N): ").strip().lower()
        if response != 'y':
            print("Exiting...")
            return
    
    try:
        # Load model
        model, tokenizer = load_model_cpu_optimized()
        
        # Test with simple prompt
        print("\nTesting model...")
        test_response = generate_text_cpu(model, tokenizer, "Hello", max_length=50)
        print(f"Test response: {test_response}")
        
        # Interactive chat
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                # Generate response
                response = generate_text_cpu(model, tokenizer, user_input, max_length=150)
                
                # Extract only the generated part
                if user_input in response:
                    response = response.replace(user_input, "").strip()
                
                print(f"Assistant: {response}")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
                
    except Exception as e:
        print(f"Fatal error: {e}")
        print("\nRecommendations:")
        print("1. Use a system with more RAM (32GB+)")
        print("2. Use a smaller model (7B or 3B parameters)")
        print("3. Use a system with GPU support")

def main():
    """Main function."""
    print("Fin-o1-14B CPU-Optimized Runner")
    print("="*40)
    
    # Check system
    print(f"Python version: {torch.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print("⚠️  CUDA detected but using CPU mode for compatibility")
    
    # Start interactive chat
    interactive_chat_cpu()

if __name__ == "__main__":
    main()