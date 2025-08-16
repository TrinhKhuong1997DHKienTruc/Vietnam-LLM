#!/usr/bin/env python3
"""
Fin-o1-8B Demo Script
A demonstration of the Fin-o1-8B model for financial reasoning tasks.
Based on: https://huggingface.co/TheFinAI/Fin-o1-8B
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import json
from typing import List, Dict, Any
import warnings
warnings.filterwarnings("ignore")

class FinO1Model:
    def __init__(self, model_name: str = "TheFinAI/Fin-o1-8B", device: str = "auto"):
        """
        Initialize the Fin-o1-8B model.
        
        Args:
            model_name: Hugging Face model identifier
            device: Device to run the model on ('auto', 'cpu', 'cuda', 'mps')
        """
        self.model_name = model_name
        self.device = self._get_device(device)
        self.model = None
        self.tokenizer = None
        
    def _get_device(self, device: str) -> str:
        """Determine the best available device."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def download_and_load(self):
        """Download and load the model and tokenizer."""
        print(f"🚀 Initializing Fin-o1-8B model...")
        print(f"📱 Using device: {self.device}")
        
        try:
            print("📥 Downloading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            print("📥 Downloading model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            if self.device == "cuda":
                self.model = self.model.cuda()
            elif self.device == "mps":
                self.model = self.model.to("mps")
                
            print("✅ Model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def generate_response(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.7) -> str:
        """
        Generate a response from the model.
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (higher = more creative)
            
        Returns:
            Generated response text
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call download_and_load() first.")
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            
            if self.device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}
            elif self.device == "mps":
                inputs = {k: v.to("mps") for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the input prompt from the response
            response = response[len(prompt):].strip()
            
            return response
            
        except Exception as e:
            print(f"❌ Error generating response: {e}")
            return f"Error: {e}"
    
    def run_financial_examples(self):
        """Run various financial reasoning examples."""
        examples = [
            {
                "category": "Basic Math",
                "prompt": "What is the result of 3-5?",
                "description": "Simple arithmetic calculation"
            },
            {
                "category": "Financial Calculation",
                "prompt": "If I invest $1000 at 5% annual interest for 3 years, how much will I have?",
                "description": "Compound interest calculation"
            },
            {
                "category": "Financial Reasoning",
                "prompt": "A company has revenue of $1M and expenses of $800K. What is the profit margin percentage?",
                "description": "Profit margin calculation"
            },
            {
                "category": "Economic Logic",
                "prompt": "Explain the relationship between inflation and interest rates.",
                "description": "Economic concept explanation"
            },
            {
                "category": "Business Analysis",
                "prompt": "What are the key factors to consider when evaluating a company's financial health?",
                "description": "Financial analysis framework"
            }
        ]
        
        print("\n" + "="*80)
        print("🧠 FIN-O1-8B FINANCIAL REASONING DEMO")
        print("="*80)
        
        for i, example in enumerate(examples, 1):
            print(f"\n📊 Example {i}: {example['category']}")
            print(f"📝 Description: {example['description']}")
            print(f"❓ Prompt: {example['prompt']}")
            print("-" * 60)
            
            start_time = time.time()
            response = self.generate_response(example['prompt'])
            end_time = time.time()
            
            print(f"🤖 Response: {response}")
            print(f"⏱️  Generation time: {end_time - start_time:.2f} seconds")
            print("-" * 60)
            
            # Add a small delay between examples
            time.sleep(1)
    
    def interactive_mode(self):
        """Run interactive mode for user input."""
        print("\n" + "="*80)
        print("💬 INTERACTIVE MODE - Type 'quit' to exit")
        print("="*80)
        
        while True:
            try:
                user_input = input("\n❓ Enter your financial question: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not user_input:
                    continue
                
                print("🤖 Generating response...")
                start_time = time.time()
                response = self.generate_response(user_input)
                end_time = time.time()
                
                print(f"✅ Response ({end_time - start_time:.2f}s):")
                print(f"   {response}")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def main():
    """Main function to run the demo."""
    print("🚀 Fin-o1-8B Financial Reasoning Model Demo")
    print("📚 Based on: https://huggingface.co/TheFinAI/Fin-o1-8B")
    print("🔬 Paper: https://arxiv.org/abs/2502.08127")
    
    # Initialize model
    model = FinO1Model()
    
    try:
        # Download and load the model
        model.download_and_load()
        
        # Run predefined examples
        model.run_financial_examples()
        
        # Run interactive mode
        model.interactive_mode()
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())