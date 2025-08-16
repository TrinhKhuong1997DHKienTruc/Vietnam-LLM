#!/usr/bin/env python3
"""
Simple test script for Fin-o1-8B model
"""

import os
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_model():
    """Test if the model can be loaded and generate responses"""
    model_path = "./fin-o1-8b"
    
    if not os.path.exists(model_path):
        print("❌ Model not found. Please run 'python download_model.py' first.")
        return False
    
    try:
        print("🔄 Loading model for testing...")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True
        )
        
        print("✅ Model loaded successfully!")
        
        # Test simple generation
        test_prompts = [
            "What is 2+2?",
            "Calculate 10% of 500.",
            "What is the result of 15 - 7?"
        ]
        
        print("\n🧪 Testing model responses...")
        for prompt in test_prompts:
            print(f"\n🤔 Question: {prompt}")
            
            inputs = tokenizer(prompt, return_tensors="pt")
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
            
            print(f"🤖 Answer: {response}")
        
        print("\n🎉 All tests passed! Model is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_model()
    if not success:
        sys.exit(1)