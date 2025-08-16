#!/usr/bin/env python3
"""
Simple Fin-o1-14B Runner using Hugging Face Pipeline
"""

from transformers import pipeline
import torch

def main():
    print("Loading Fin-o1-14B model...")
    print("This may take several minutes on first run...")
    
    try:
        # Create the pipeline
        generator = pipeline(
            "text-generation",
            model="TheFinAI/Fin-o1-14B",
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        print("Model loaded successfully!")
        print("\n" + "="*50)
        print("Fin-o1-14B Text Generation")
        print("Type 'quit' to exit")
        print("="*50 + "\n")
        
        while True:
            user_input = input("Enter your prompt: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not user_input:
                continue
            
            try:
                # Generate text
                outputs = generator(
                    user_input,
                    max_length=200,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True
                )
                
                # Print the generated text
                generated_text = outputs[0]['generated_text']
                print(f"\nGenerated text:\n{generated_text}\n")
                
            except Exception as e:
                print(f"Error generating text: {e}\n")
                
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure you have enough GPU memory and the required dependencies installed.")

if __name__ == "__main__":
    main()