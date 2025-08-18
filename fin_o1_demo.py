#!/usr/bin/env python3
"""
Fin-o1-8B Model Demo
A demonstration script for the Fin-o1-8B language model from Hugging Face.
This script provides an interactive chat interface with the model.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import os

def load_model():
    """Load the Fin-o1-8B model and tokenizer."""
    print("🔄 Loading Fin-o1-8B model...")
    print("📥 This may take a few minutes on first run as the model downloads...")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("TheFinAI/Fin-o1-8B")
        
        # Load model with 8-bit quantization for memory efficiency
        model = AutoModelForCausalLM.from_pretrained(
            "TheFinAI/Fin-o1-8B",
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_8bit=True,
            trust_remote_code=True
        )
        
        print("✅ Model loaded successfully!")
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("💡 Try running: pip install bitsandbytes")
        return None, None

def generate_response(model, tokenizer, prompt, max_length=512):
    """Generate a response from the model."""
    try:
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the input prompt from response
        response = response[len(prompt):].strip()
        
        return response
        
    except Exception as e:
        return f"Error generating response: {e}"

def interactive_chat(model, tokenizer):
    """Run interactive chat with the model."""
    print("\n🤖 Fin-o1-8B Model Demo")
    print("=" * 50)
    print("💬 Start chatting with the model! (Type 'quit' to exit)")
    print("📝 The model is specialized in financial and general knowledge.")
    print("-" * 50)
    
    while True:
        try:
            # Get user input
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye! Thanks for trying Fin-o1-8B!")
                break
            
            if not user_input:
                continue
            
            print("\n🤖 Fin-o1-8B: ", end="", flush=True)
            
            # Generate response
            response = generate_response(model, tokenizer, user_input)
            print(response)
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! Thanks for trying Fin-o1-8B!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

def demo_examples(model, tokenizer):
    """Run some demo examples to showcase the model."""
    print("\n🚀 Running Demo Examples...")
    print("=" * 50)
    
    examples = [
        "What is the difference between stocks and bonds?",
        "Explain the concept of compound interest.",
        "What are the main types of investment portfolios?",
        "How does inflation affect investments?",
        "What is diversification in investing?"
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n📝 Example {i}: {example}")
        print("-" * 40)
        
        response = generate_response(model, tokenizer, example)
        print(f"🤖 Fin-o1-8B: {response}")
        print()

def main():
    """Main function to run the demo."""
    print("🚀 Fin-o1-8B Model Demo")
    print("=" * 50)
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"🎯 CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("💻 Running on CPU")
    
    # Load model
    model, tokenizer = load_model()
    
    if model is None or tokenizer is None:
        print("❌ Failed to load model. Exiting.")
        sys.exit(1)
    
    # Show menu
    while True:
        print("\n📋 Choose an option:")
        print("1. Interactive Chat")
        print("2. Run Demo Examples")
        print("3. Exit")
        
        choice = input("\n🎯 Enter your choice (1-3): ").strip()
        
        if choice == "1":
            interactive_chat(model, tokenizer)
        elif choice == "2":
            demo_examples(model, tokenizer)
        elif choice == "3":
            print("👋 Goodbye! Thanks for trying Fin-o1-8B!")
            break
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()