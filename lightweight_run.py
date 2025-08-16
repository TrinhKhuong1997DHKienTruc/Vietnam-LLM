#!/usr/bin/env python3
"""
Lightweight Fin-o1-14B Model Runner for CPU/Limited Memory
This version is optimized for systems with limited RAM and CPU-only inference.
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import gc
import os

def setup_lightweight_model():
    """Setup the Fin-o1-14B model with memory optimization for CPU."""
    
    model_name = "TheFinAI/Fin-o1-14B"
    
    print(f"Loading model: {model_name}")
    print("⚠️  Running on CPU - this will be very slow for a 14B parameter model!")
    print("📱 Memory optimization enabled for limited RAM environments")
    
    try:
        # Force CPU usage and memory optimization
        torch.set_num_threads(4)  # Limit CPU threads to prevent memory issues
        
        # Load tokenizer first
        print("📥 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True,
            use_fast=False  # Use slower but more memory-efficient tokenizer
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        print("📥 Loading model (this may take 10+ minutes on CPU)...")
        
        # Load model with maximum memory optimization
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Use float32 for CPU
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map=None,  # Force CPU
            offload_folder="offload"  # Enable disk offloading
        )
        
        # Move to CPU explicitly
        model = model.cpu()
        
        # Enable memory-efficient attention if available
        if hasattr(model, 'config') and hasattr(model.config, 'attention_mode'):
            model.config.attention_mode = 'flash_attention_2'
        
        print("✅ Model loaded successfully!")
        print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("\n🔧 This model requires significant memory resources.")
        print("   Consider:")
        print("   - Using a machine with more RAM (32GB+)")
        print("   - Using a GPU with sufficient VRAM")
        print("   - Using a smaller model variant")
        return None, None

def generate_lightweight_response(model, tokenizer, prompt, max_length=128):
    """Generate a response with memory optimization."""
    
    try:
        # Clear memory before generation
        gc.collect()
        
        # Tokenize with limited length
        inputs = tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=512  # Limit input length
        )
        
        print("⏳ Generating response...")
        
        # Generate with memory constraints
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
                length_penalty=1.0,
                early_stopping=True
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up response
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        
        # Clear memory after generation
        del outputs
        gc.collect()
        
        return response
        
    except Exception as e:
        print(f"❌ Error generating response: {e}")
        return f"Error: {str(e)}"

def interactive_lightweight_mode(model, tokenizer):
    """Run the model in lightweight interactive mode."""
    
    print("\n" + "="*50)
    print("Fin-o1-14B Model - Lightweight Interactive Mode")
    print("Type 'quit' to exit, 'clear' to clear conversation")
    print("⚠️  Responses may be slow due to CPU inference")
    print("="*50)
    
    conversation_history = ""
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() == 'quit':
                print("Goodbye!")
                break
            elif user_input.lower() == 'clear':
                conversation_history = ""
                print("Conversation history cleared.")
                continue
            elif not user_input:
                continue
            
            # Limit conversation history to prevent memory issues
            if len(conversation_history) > 1000:
                conversation_history = conversation_history[-500:]
                print("💾 Conversation history trimmed for memory efficiency")
            
            # Build prompt
            full_prompt = conversation_history + f"\nUser: {user_input}\nAssistant:"
            
            # Generate response
            response = generate_lightweight_response(model, tokenizer, full_prompt, max_length=128)
            print(f"Assistant: {response}")
            
            # Update conversation history
            conversation_history = full_prompt + f" {response}"
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue

def main():
    """Main function for lightweight model runner."""
    
    print("🚀 Fin-o1-14B Lightweight Runner")
    print("=" * 50)
    
    # Check available memory
    try:
        with open('/proc/meminfo', 'r') as f:
            meminfo = f.read()
            total_mem = int([line for line in meminfo.split('\n') if 'MemTotal:' in line][0].split()[1]) // 1024 // 1024
            print(f"📱 Available system memory: {total_mem}GB")
            
            if total_mem < 16:
                print("⚠️  Warning: Less than 16GB RAM detected!")
                print("   The model may not load or run very slowly.")
                print("   Consider using a machine with more memory.")
    except:
        print("📱 Memory information unavailable")
    
    # Setup model
    model, tokenizer = setup_lightweight_model()
    
    if model is None or tokenizer is None:
        print("❌ Failed to load model. Exiting.")
        return
    
    try:
        # Test with a simple prompt
        test_prompt = "Hello! What is finance?"
        print(f"\n🧪 Testing with prompt: {test_prompt}")
        
        response = generate_lightweight_response(model, tokenizer, test_prompt, max_length=64)
        print(f"💬 Response: {response}")
        
        # Ask user if they want interactive mode
        user_choice = input("\nWould you like to enter interactive mode? (y/n): ").strip().lower()
        
        if user_choice in ['y', 'yes']:
            interactive_lightweight_mode(model, tokenizer)
        
    except Exception as e:
        print(f"Error during execution: {e}")
        
    finally:
        # Clean up memory
        if 'model' in locals():
            del model
        if 'tokenizer' in locals():
            del tokenizer
        gc.collect()
        print("🧹 Memory cleaned up")

if __name__ == "__main__":
    main()