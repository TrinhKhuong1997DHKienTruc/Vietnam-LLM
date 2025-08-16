#!/usr/bin/env python3
"""
Financial Reasoning Demo with Fin-o1-8B
Demonstrates the model's capabilities on financial tasks
"""

from run_fin_o1 import FinO1Runner
import time

def run_financial_demo():
    """Run a series of financial reasoning examples"""
    
    # Initialize the model
    print("🚀 Starting Financial Reasoning Demo with Fin-o1-8B")
    print("=" * 60)
    
    runner = FinO1Runner(load_in_4bit=True)  # Use 4-bit quantization for memory efficiency
    
    try:
        # Load the model
        runner.load_model()
        
        # Financial reasoning examples
        examples = [
            {
                "title": "Basic Financial Calculation",
                "prompt": "If a company has revenue of $1,000,000 and expenses of $750,000, what is the net profit margin as a percentage?",
                "expected": "Financial calculation involving profit margin"
            },
            {
                "title": "Investment Analysis",
                "prompt": "A stock is currently trading at $50 per share. If the company pays a quarterly dividend of $0.75 per share, what is the annual dividend yield?",
                "expected": "Dividend yield calculation"
            },
            {
                "title": "Compound Interest",
                "prompt": "If you invest $10,000 at an annual interest rate of 7% compounded annually, how much will you have after 10 years?",
                "expected": "Compound interest calculation"
            },
            {
                "title": "Financial Ratio Analysis",
                "prompt": "A company has current assets of $500,000 and current liabilities of $200,000. What is the current ratio and what does it indicate about the company's liquidity?",
                "expected": "Liquidity ratio analysis"
            },
            {
                "title": "Portfolio Diversification",
                "prompt": "Explain the concept of portfolio diversification and why it's important for risk management in investing.",
                "expected": "Portfolio theory explanation"
            }
        ]
        
        print("\n📊 Running Financial Reasoning Examples")
        print("-" * 40)
        
        for i, example in enumerate(examples, 1):
            print(f"\n🔢 Example {i}: {example['title']}")
            print(f"📝 Question: {example['prompt']}")
            print(f"🎯 Expected: {example['expected']}")
            print("🤖 Generating response...")
            
            start_time = time.time()
            response = runner.generate_response(
                example['prompt'], 
                max_new_tokens=300,
                temperature=0.3  # Lower temperature for more focused responses
            )
            generation_time = time.time() - start_time
            
            print(f"✅ Response ({generation_time:.2f}s):")
            print(f"   {response}")
            print("-" * 40)
            
            # Small delay between examples
            time.sleep(1)
        
        # Interactive financial chat
        print("\n💬 Interactive Financial Chat Session")
        print("Ask any financial questions or type 'quit' to exit")
        print("=" * 60)
        
        runner.interactive_chat()
        
    except Exception as e:
        print(f"❌ Error in financial demo: {e}")
        raise

def run_quick_test():
    """Run a quick test to verify the model is working"""
    
    print("🧪 Quick Test - Verifying Model Functionality")
    print("-" * 40)
    
    runner = FinO1Runner(load_in_4bit=True)
    
    try:
        runner.load_model()
        
        # Simple test
        test_prompt = "What is 2 + 2?"
        print(f"📝 Test prompt: {test_prompt}")
        
        response = runner.generate_response(test_prompt, max_new_tokens=50)
        print(f"✅ Test response: {response}")
        
        print("🎉 Quick test completed successfully!")
        
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Financial Reasoning Demo")
    parser.add_argument("--quick-test", action="store_true", 
                       help="Run quick test only")
    
    args = parser.parse_args()
    
    if args.quick_test:
        run_quick_test()
    else:
        run_financial_demo()