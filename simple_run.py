#!/usr/bin/env python3
"""
Simple Fin-o1-14B Model Runner using Hugging Face Pipeline
This is a simpler alternative to run_model.py
"""

from transformers import pipeline
import torch

def main():
    """Load and run the Fin-o1-14B model using pipeline."""
    
    model_name = "TheFinAI/Fin-o1-14B"
    
    print(f"Loading model: {model_name}")
    print("This may take several minutes depending on your hardware...")
    
    try:
        # Check device availability
        device = 0 if torch.cuda.is_available() else -1
        device_name = "GPU" if device >= 0 else "CPU"
        print(f"Using device: {device_name}")
        
        if device == -1:
            print("Warning: Running on CPU will be very slow for a 14B parameter model!")
        
        # Load the model using pipeline
        generator = pipeline(
            "text-generation",
            model=model_name,
            device=device,
            torch_dtype=torch.float16 if device >= 0 else torch.float32,
            trust_remote_code=True
        )
        
        print("Model loaded successfully!")
        
        # Interactive mode
        print("\n" + "="*50)
        print("Fin-o1-14B Model Interactive Mode")
        print("Type 'quit' to exit")
        print("="*50)
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() == 'quit':
                    print("Goodbye!")
                    break
                elif not user_input:
                    continue
                
                print("Generating response...")
                
                # Generate response
                outputs = generator(
                    user_input,
                    max_length=512,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=generator.tokenizer.eos_token_id
                )
                
                # Extract and display response
                response = outputs[0]['generated_text']
                
                # Remove the input prompt if it appears at the beginning
                if response.startswith(user_input):
                    response = response[len(user_input):].strip()
                
                print(f"Assistant: {response}")
                
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
                continue
                
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure you have sufficient memory and the model is accessible.")

if __name__ == "__main__":
    main()