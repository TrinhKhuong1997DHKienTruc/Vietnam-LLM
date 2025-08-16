#!/usr/bin/env python3
"""
Quick Start Script for Fin-o1-8B
Provides easy access to common operations
"""

import sys
import os

def show_menu():
    """Display the main menu"""
    print("\n🚀 Fin-o1-8B Quick Start Menu")
    print("=" * 40)
    print("1. 🧪 Run Setup Test")
    print("2. 💬 Start Interactive Chat (4-bit)")
    print("3. 💬 Start Interactive Chat (8-bit)")
    print("4. 💬 Start Interactive Chat (CPU)")
    print("5. 📊 Run Financial Demo")
    print("6. 🔍 Single Prompt Test")
    print("7. 📚 Show Help")
    print("8. 🚪 Exit")
    print("-" * 40)

def run_command(command):
    """Run a system command"""
    print(f"🚀 Running: {command}")
    os.system(command)

def main():
    """Main menu loop"""
    while True:
        show_menu()
        
        try:
            choice = input("\nSelect an option (1-8): ").strip()
            
            if choice == "1":
                print("\n🧪 Running setup test...")
                run_command("python test_setup.py")
                
            elif choice == "2":
                print("\n💬 Starting interactive chat with 4-bit quantization...")
                print("💡 This is recommended for systems with limited memory")
                run_command("python run_fin_o1.py --four-bit --interactive")
                
            elif choice == "3":
                print("\n💬 Starting interactive chat with 8-bit quantization...")
                print("💡 This requires more memory but is faster")
                run_command("python run_fin_o1.py --eight-bit --interactive")
                
            elif choice == "4":
                print("\n💬 Starting interactive chat on CPU...")
                print("⚠️  This will be slower but uses less memory")
                run_command("python run_fin_o1.py --device cpu --interactive")
                
            elif choice == "5":
                print("\n📊 Running financial demo...")
                run_command("python financial_demo.py")
                
            elif choice == "6":
                prompt = input("\n📝 Enter your prompt: ").strip()
                if prompt:
                    print(f"\n🔍 Testing prompt: {prompt}")
                    run_command(f'python run_fin_o1.py --prompt "{prompt}" --four-bit')
                else:
                    print("❌ No prompt entered")
                    
            elif choice == "7":
                show_help()
                
            elif choice == "8":
                print("\n👋 Goodbye! Happy financial reasoning!")
                break
                
            else:
                print("❌ Invalid choice. Please select 1-8.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
        
        input("\nPress Enter to continue...")

def show_help():
    """Show help information"""
    print("\n📚 Fin-o1-8B Help")
    print("=" * 30)
    print("\n🎯 What is Fin-o1-8B?")
    print("   A fine-tuned version of Qwen3-8B specifically designed")
    print("   for financial reasoning tasks.")
    print("\n💾 Memory Requirements:")
    print("   - Full model: ~16GB RAM/VRAM")
    print("   - 8-bit quantization: ~8GB RAM/VRAM")
    print("   - 4-bit quantization: ~4GB RAM/VRAM")
    print("\n🚀 Quick Commands:")
    print("   - Test setup: python test_setup.py")
    print("   - Interactive chat: python run_fin_o1.py --four-bit --interactive")
    print("   - Financial demo: python financial_demo.py")
    print("\n🔧 Command Line Options:")
    print("   --four-bit: Use 4-bit quantization (saves memory)")
    print("   --eight-bit: Use 8-bit quantization (faster)")
    print("   --device cpu: Force CPU usage")
    print("   --prompt TEXT: Run single prompt")
    print("   --interactive: Start chat session")
    print("\n💡 Tips:")
    print("   - Start with 4-bit quantization if you have memory issues")
    print("   - Use CPU mode if GPU memory is insufficient")
    print("   - The model excels at financial calculations and analysis")
    print("\n📖 For more information, see README.md")

if __name__ == "__main__":
    main()