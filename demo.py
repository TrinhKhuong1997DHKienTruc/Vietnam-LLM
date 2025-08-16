#!/usr/bin/env python3
"""
Demo script for Fin-o1-14B model
Shows various financial analysis capabilities
"""

import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from run_fin_o1 import setup_model, generate_response

def run_demo():
    """Run a series of financial analysis demos"""
    
    print("🚀 Fin-o1-14B Financial Analysis Demo")
    print("=" * 50)
    
    # Check if model is already loaded
    try:
        # Try to load the model
        print("Loading Fin-o1-14B model...")
        model, tokenizer = setup_model(load_in_8bit=True)
        print("✅ Model loaded successfully!")
        
        # Demo prompts
        demo_prompts = [
            {
                "title": "📊 Financial Statement Analysis",
                "prompt": "What are the key financial ratios I should look at when analyzing a company's balance sheet and income statement? Please explain each ratio and what it tells us about the company's financial health."
            },
            {
                "title": "💼 Investment Strategy",
                "prompt": "Explain the differences between value investing and growth investing strategies. What are the pros and cons of each approach, and when might an investor choose one over the other?"
            },
            {
                "title": "📈 Market Analysis",
                "prompt": "What are the main factors that drive stock market volatility? How can individual investors prepare for and potentially benefit from market volatility?"
            },
            {
                "title": "🛡️ Risk Management",
                "prompt": "Describe the different types of investment risk (market risk, credit risk, liquidity risk, etc.) and provide specific strategies for managing each type of risk in a portfolio."
            },
            {
                "title": "🏦 Banking & Finance",
                "prompt": "How do central banks influence interest rates and what impact does this have on different types of investments like bonds, stocks, and real estate?"
            }
        ]
        
        print(f"\nRunning {len(demo_prompts)} financial analysis examples...\n")
        
        for i, demo in enumerate(demo_prompts, 1):
            print(f"{'='*60}")
            print(f"Example {i}: {demo['title']}")
            print(f"{'='*60}")
            print(f"Prompt: {demo['prompt']}")
            print("\nGenerating response...")
            
            try:
                response = generate_response(
                    model, tokenizer, demo['prompt'],
                    max_length=1024,
                    temperature=0.7
                )
                
                # Clean up the response
                if "Assistant:" in response:
                    response = response.split("Assistant:")[-1].strip()
                
                print(f"\nResponse:\n{response}")
                print("\n" + "-"*60 + "\n")
                
            except Exception as e:
                print(f"❌ Error generating response: {e}")
                continue
        
        print("🎉 Demo completed!")
        print("\nYou can now run the model interactively with:")
        print("python run_fin_o1.py")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("\nPlease ensure you have:")
        print("1. Run ./setup.sh to install dependencies")
        print("2. Activated the virtual environment: source fin_o1_env/bin/activate")
        print("3. Have sufficient GPU memory (16GB+ recommended)")
        print("4. Stable internet connection for model download")

if __name__ == "__main__":
    run_demo()