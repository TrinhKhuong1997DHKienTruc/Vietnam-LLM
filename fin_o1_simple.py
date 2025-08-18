#!/usr/bin/env python3
"""
Fin-o1-8B Model Demo (Simple Version)
A demonstration script for the Fin-o1-8B language model from Hugging Face.
This version is optimized for broader compatibility across different systems.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys

def load_model():
    """Load the Fin-o1-8B model and tokenizer."""
    print("🔄 Loading Fin-o1-8B model...")
    print("📥 This may take a few minutes on first run as the model downloads...")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("TheFinAI/Fin-o1-8B")
        
        # Load model with basic configuration for compatibility
        model = AutoModelForCausalLM.from_pretrained(
            "TheFinAI/Fin-o1-8B",
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        print("✅ Model loaded successfully!")
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None

def generate_response(model, tokenizer, prompt, max_length=256):
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

def run_demo():
    """Run a simple demo with predefined questions."""
    print("\n🚀 Fin-o1-8B Model Demo")
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
    
    # Demo questions
    questions = [
        "What is the difference between stocks and bonds?",
        "Explain compound interest in simple terms.",
        "What is diversification in investing?"
    ]
    
    print("\n📝 Running demo questions...")
    print("-" * 50)
    
    for i, question in enumerate(questions, 1):
        print(f"\n❓ Question {i}: {question}")
        print("🤖 Answer:", end=" ", flush=True)
        
        response = generate_response(model, tokenizer, question)
        print(response)
        print("-" * 50)
    
    print("\n✅ Demo completed successfully!")
    print("💡 You can now use the interactive version: python fin_o1_demo.py")

if __name__ == "__main__":
    run_demo()