#!/usr/bin/env python3
"""
Fin-o1-8B Model Runner
A fine-tuned version of Qwen3-8B for financial reasoning tasks
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import argparse
import time
from typing import Optional, List

class FinO1Runner:
    def __init__(self, model_name: str = "TheFinAI/Fin-o1-8B", 
                 device: str = "auto", 
                 load_in_8bit: bool = False,
                 load_in_4bit: bool = False):
        """
        Initialize the Fin-o1-8B model runner
        
        Args:
            model_name: Hugging Face model identifier
            device: Device to run on ('auto', 'cuda', 'cpu')
            load_in_8bit: Whether to load model in 8-bit precision
            load_in_4bit: Whether to load model in 4-bit precision
        """
        self.model_name = model_name
        self.device = device
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        
        # Auto-detect device if specified
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"🚀 Initializing Fin-o1-8B model on {self.device}")
        print(f"📦 Model: {model_name}")
        
        self.tokenizer = None
        self.model = None
        
    def load_model(self):
        """Load the model and tokenizer"""
        try:
            print("🔧 Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print("🔧 Loading model...")
            
            # Configure quantization if requested
            quantization_config = None
            if self.load_in_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            elif self.load_in_8bit:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            
            # Load model with appropriate configuration
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                quantization_config=quantization_config,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            print("✅ Model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def generate_response(self, prompt: str, 
                         max_new_tokens: int = 512,
                         temperature: float = 0.7,
                         top_p: float = 0.9,
                         do_sample: bool = True) -> str:
        """
        Generate a response from the model
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            do_sample: Whether to use sampling
            
        Returns:
            Generated response
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        try:
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
            
            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate response
            start_time = time.time()
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            generation_time = time.time() - start_time
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove input prompt from response
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
            
            print(f"⏱️  Generation time: {generation_time:.2f}s")
            
            return response
            
        except Exception as e:
            print(f"❌ Error generating response: {e}")
            raise
    
    def interactive_chat(self):
        """Start an interactive chat session"""
        print("\n💬 Starting interactive chat session...")
        print("Type 'quit' to exit, 'clear' to clear conversation")
        print("-" * 50)
        
        conversation = []
        
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if user_input.lower() == 'quit':
                    print("👋 Goodbye!")
                    break
                elif user_input.lower() == 'clear':
                    conversation = []
                    print("🧹 Conversation cleared")
                    continue
                elif not user_input:
                    continue
                
                # Build context from conversation
                if conversation:
                    context = "\n".join([f"{'User' if i % 2 == 0 else 'Assistant'}: {msg}" 
                                       for i, msg in enumerate(conversation[-6:])])  # Last 6 messages
                    full_prompt = f"{context}\nUser: {user_input}\nAssistant:"
                else:
                    full_prompt = f"User: {user_input}\nAssistant:"
                
                print("🤖 Assistant is thinking...")
                response = self.generate_response(full_prompt, max_new_tokens=256)
                
                print(f"🤖 Assistant: {response}")
                
                # Store conversation
                conversation.extend([user_input, response])
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="Run Fin-o1-8B model")
    parser.add_argument("--model", default="TheFinAI/Fin-o1-8B", 
                       help="Model name or path")
    parser.add_argument("--device", default="auto", 
                       choices=["auto", "cuda", "cpu"],
                       help="Device to run on")
    parser.add_argument("--eight-bit", action="store_true", 
                       help="Load model in 8-bit precision")
    parser.add_argument("--four-bit", action="store_true", 
                       help="Load model in 4-bit precision")
    parser.add_argument("--prompt", type=str, 
                       help="Single prompt to run")
    parser.add_argument("--interactive", action="store_true", 
                       help="Start interactive chat session")
    parser.add_argument("--max-tokens", type=int, default=512,
                       help="Maximum new tokens to generate")
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = FinO1Runner(
        model_name=args.model,
        device=args.device,
        load_in_8bit=args.eight_bit,
        load_in_4bit=args.four_bit
    )
    
    try:
        # Load model
        runner.load_model()
        
        # Run based on arguments
        if args.prompt:
            print(f"📝 Prompt: {args.prompt}")
            response = runner.generate_response(args.prompt, max_new_tokens=args.max_tokens)
            print(f"🤖 Response: {response}")
        elif args.interactive:
            runner.interactive_chat()
        else:
            # Default interactive mode
            runner.interactive_chat()
            
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())