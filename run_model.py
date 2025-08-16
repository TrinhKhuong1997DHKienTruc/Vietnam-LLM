#!/usr/bin/env python3
"""
Fin-o1-14B Model Runner
This script loads and runs the Fin-o1-14B model from Hugging Face
with memory optimization and quantization support.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import gc
import os

def setup_model():
    """Setup the Fin-o1-14B model with quantization and memory optimization."""
    
    model_name = "TheFinAI/Fin-o1-14B"
    
    print(f"Loading model: {model_name}")
    print("This may take several minutes depending on your hardware...")
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if device == "cuda":
        # Memory optimization for GPU
        torch.cuda.empty_cache()
        
        # Quantization configuration for memory efficiency
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        # Load model with quantization
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
    else:
        # CPU fallback (will be very slow for 14B model)
        print("Warning: Running on CPU will be very slow for a 14B parameter model!")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_length=512):
    """Generate a response from the model."""
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    
    # Move to same device as model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Remove the input prompt from response
    if response.startswith(prompt):
        response = response[len(prompt):].strip()
    
    return response

def interactive_mode(model, tokenizer):
    """Run the model in interactive mode."""
    
    print("\n" + "="*50)
    print("Fin-o1-14B Model Interactive Mode")
    print("Type 'quit' to exit, 'clear' to clear conversation")
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
            
            # Build full prompt with conversation history
            full_prompt = conversation_history + f"\nUser: {user_input}\nAssistant:"
            
            print("Generating response...")
            response = generate_response(model, tokenizer, full_prompt)
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
    """Main function to run the model."""
    
    try:
        # Setup model and tokenizer
        model, tokenizer = setup_model()
        
        print("\nModel loaded successfully!")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test with a simple prompt
        test_prompt = "Hello! Can you tell me about financial markets?"
        print(f"\nTesting with prompt: {test_prompt}")
        
        response = generate_response(model, tokenizer, test_prompt)
        print(f"Response: {response}")
        
        # Ask user if they want interactive mode
        user_choice = input("\nWould you like to enter interactive mode? (y/n): ").strip().lower()
        
        if user_choice in ['y', 'yes']:
            interactive_mode(model, tokenizer)
        
    except Exception as e:
        print(f"Error setting up model: {e}")
        print("Make sure you have sufficient memory and the model is accessible.")
        
    finally:
        # Clean up memory
        if 'model' in locals():
            del model
        if 'tokenizer' in locals():
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()