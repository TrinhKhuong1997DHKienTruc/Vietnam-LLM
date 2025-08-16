#!/usr/bin/env python3
"""
Download Fin-o1-8B model from Hugging Face
"""

import os
import sys
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM

def download_model():
    """Download the Fin-o1-8B model and tokenizer"""
    model_name = "TheFinAI/Fin-o1-8B"
    
    print(f"🔄 Downloading {model_name}...")
    print("This may take a while depending on your internet connection...")
    
    try:
        # Download tokenizer
        print("📥 Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.save_pretrained("./fin-o1-8b")
        
        # Download model
        print("📥 Downloading model (8.19B parameters)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True
        )
        model.save_pretrained("./fin-o1-8b")
        
        print("✅ Model downloaded successfully!")
        print(f"📁 Model saved to: {os.path.abspath('./fin-o1-8b')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        return False

if __name__ == "__main__":
    success = download_model()
    if success:
        print("\n🎉 Model download completed!")
        print("You can now run the demo with: python demo.py")
    else:
        print("\n💥 Model download failed!")
        sys.exit(1)