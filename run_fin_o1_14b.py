#!/usr/bin/env python3
"""
Fin-o1-14B Model Runner
This script downloads and runs the Fin-o1-14B model from Hugging Face.
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_system_requirements():
    """Check if the system meets the requirements for running the model."""
    logger.info("Checking system requirements...")
    
    # Check CUDA availability
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_memory = []
        for i in range(gpu_count):
            gpu_memory.append(torch.cuda.get_device_properties(i).total_memory / 1024**3)
        logger.info(f"CUDA available with {gpu_count} GPU(s)")
        for i, mem in enumerate(gpu_memory):
            logger.info(f"GPU {i}: {mem:.1f} GB")
    else:
        logger.warning("CUDA not available. Running on CPU (this will be very slow for a 14B model)")
    
    # Check available RAM
    import psutil
    ram_gb = psutil.virtual_memory().total / 1024**3
    logger.info(f"Available RAM: {ram_gb:.1f} GB")
    
    if ram_gb < 32:
        logger.warning("Less than 32GB RAM available. The model may not load properly.")
    
    return torch.cuda.is_available()

def setup_model_config():
    """Set up the model configuration with quantization for memory efficiency."""
    logger.info("Setting up model configuration...")
    
    # Use 4-bit quantization to reduce memory usage
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    
    return bnb_config

def download_model():
    """Download the Fin-o1-14B model from Hugging Face."""
    model_name = "TheFinAI/Fin-o1-14B"
    logger.info(f"Downloading model: {model_name}")
    
    try:
        # Download tokenizer
        logger.info("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir="./models"
        )
        
        # Download model with quantization
        logger.info("Downloading model (this may take a while)...")
        bnb_config = setup_model_config()
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            cache_dir="./models",
            torch_dtype=torch.float16
        )
        
        logger.info("Model downloaded successfully!")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        raise

def generate_text(model, tokenizer, prompt, max_length=512):
    """Generate text using the model."""
    logger.info("Generating text...")
    
    try:
        # Encode the input
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # Move inputs to the same device as the model
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate text
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode the output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return generated_text
        
    except Exception as e:
        logger.error(f"Error generating text: {e}")
        raise

def interactive_chat(model, tokenizer):
    """Run an interactive chat session with the model."""
    logger.info("Starting interactive chat session...")
    print("\n" + "="*50)
    print("Fin-o1-14B Interactive Chat")
    print("Type 'quit' to exit")
    print("="*50 + "\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not user_input:
                continue
            
            # Generate response
            response = generate_text(model, tokenizer, user_input, max_length=256)
            
            # Extract only the generated part (remove the input)
            if user_input in response:
                response = response.replace(user_input, "").strip()
            
            print(f"Assistant: {response}\n")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            print(f"Error: {e}\n")

def main():
    """Main function to run the model."""
    logger.info("Starting Fin-o1-14B model runner...")
    
    # Check system requirements
    has_cuda = check_system_requirements()
    
    try:
        # Download model
        model, tokenizer = download_model()
        
        # Test the model with a simple prompt
        test_prompt = "Hello, how are you?"
        logger.info("Testing model with a simple prompt...")
        test_response = generate_text(model, tokenizer, test_prompt, max_length=100)
        logger.info(f"Test response: {test_response}")
        
        # Start interactive chat
        interactive_chat(model, tokenizer)
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())