#!/usr/bin/env python3
"""
Fin-o1-14B Model Runner
A script to load and run the Fin-o1-14B model from Hugging Face
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import argparse
import sys

def setup_model(model_name="TheFinAI/Fin-o1-14B", load_in_8bit=True, load_in_4bit=False):
    """
    Setup the Fin-o1-14B model with quantization for efficient inference
    """
    print(f"Loading model: {model_name}")
    
    # Configure quantization
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        print("Using 4-bit quantization")
    elif load_in_8bit:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        print("Using 8-bit quantization")
    else:
        quantization_config = None
        print("No quantization (requires significant GPU memory)")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Load model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_length=512, temperature=0.7, top_p=0.9):
    """
    Generate a response from the model
    """
    # Encode the prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Move inputs to the same device as the model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode and return the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def interactive_mode(model, tokenizer):
    """
    Run the model in interactive mode
    """
    print("\n=== Fin-o1-14B Interactive Mode ===")
    print("Type 'quit' to exit, 'clear' to clear conversation")
    print("Type your financial questions or prompts below:\n")
    
    conversation_history = ""
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() == 'quit':
                print("Goodbye!")
                break
            elif user_input.lower() == 'clear':
                conversation_history = ""
                print("Conversation cleared.\n")
                continue
            elif not user_input:
                continue
            
            # Build the full prompt with conversation history
            full_prompt = conversation_history + f"\nUser: {user_input}\nAssistant:"
            
            print("Generating response...")
            response = generate_response(model, tokenizer, full_prompt)
            
            # Extract just the new response part
            if "Assistant:" in response:
                new_response = response.split("Assistant:")[-1].strip()
            else:
                new_response = response
            
            print(f"Assistant: {new_response}\n")
            
            # Update conversation history
            conversation_history = full_prompt + " " + new_response
            
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue

def main():
    parser = argparse.ArgumentParser(description="Run Fin-o1-14B model")
    parser.add_argument("--model", default="TheFinAI/Fin-o1-14B", help="Model name or path")
    parser.add_argument("--prompt", help="Single prompt to run")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--no-8bit", action="store_true", help="Disable 8-bit quantization")
    parser.add_argument("--4bit", action="store_true", help="Use 4-bit quantization instead of 8-bit")
    
    args = parser.parse_args()
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("CUDA not available. Running on CPU (this will be very slow for a 14B model)")
    
    try:
        # Setup model
        load_in_8bit = not args.no_8bit and not args.4bit
        model, tokenizer = setup_model(
            model_name=args.model,
            load_in_8bit=load_in_8bit,
            load_in_4bit=args.4bit
        )
        
        if args.prompt:
            # Single prompt mode
            print(f"Prompt: {args.prompt}")
            response = generate_response(
                model, tokenizer, args.prompt, 
                max_length=args.max_length,
                temperature=args.temperature,
                top_p=args.top_p
            )
            print(f"Response: {response}")
        else:
            # Interactive mode
            interactive_mode(model, tokenizer)
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()