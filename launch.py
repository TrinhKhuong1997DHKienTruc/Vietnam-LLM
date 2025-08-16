#!/usr/bin/env python3
"""
Fin-o1-14B Model Launcher
This script provides a menu to choose how to run the model.
"""

import os
import sys
import subprocess

def print_banner():
    """Print the application banner."""
    print("🚀 Fin-o1-14B Model Launcher")
    print("=" * 50)
    print("Choose how you want to run the model:")
    print()

def print_menu():
    """Print the main menu."""
    print("1. 🧪 Test Setup - Verify environment is ready")
    print("2. 🚀 Quick Start - Run predefined test prompts")
    print("3. 💡 Simple Runner - Basic pipeline-based runner")
    print("4. ⚡ Lightweight Runner - Memory-optimized for CPU")
    print("5. 🔧 Advanced Runner - Full features with quantization")
    print("6. 📚 View README - Show usage instructions")
    print("7. 🚪 Exit")
    print()

def run_script(script_name, description):
    """Run a Python script."""
    print(f"\n🚀 Launching: {description}")
    print("=" * 50)
    
    try:
        # Check if script exists
        if not os.path.exists(script_name):
            print(f"❌ Error: {script_name} not found!")
            return False
        
        # Run the script
        result = subprocess.run([sys.executable, script_name], check=False)
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Error running {script_name}: {e}")
        return False

def show_readme():
    """Display the README content."""
    try:
        with open('README.md', 'r') as f:
            content = f.read()
            print("\n📚 README.md")
            print("=" * 50)
            print(content)
    except FileNotFoundError:
        print("❌ README.md not found!")
    except Exception as e:
        print(f"❌ Error reading README: {e}")

def main():
    """Main launcher function."""
    
    # Check if virtual environment is activated
    if not os.environ.get('VIRTUAL_ENV'):
        print("⚠️  Warning: Virtual environment not detected!")
        print("   It's recommended to activate the virtual environment first:")
        print("   source fin_o1_env/bin/activate")
        print()
    
    while True:
        print_banner()
        print_menu()
        
        try:
            choice = input("Enter your choice (1-7): ").strip()
            
            if choice == '1':
                success = run_script('test_setup.py', 'Environment Test')
                if success:
                    print("\n✅ Setup test completed successfully!")
                else:
                    print("\n❌ Setup test failed!")
                    
            elif choice == '2':
                success = run_script('quick_start.py', 'Quick Start Test')
                if success:
                    print("\n✅ Quick start completed!")
                else:
                    print("\n❌ Quick start failed!")
                    
            elif choice == '3':
                success = run_script('simple_run.py', 'Simple Pipeline Runner')
                if success:
                    print("\n✅ Simple runner completed!")
                else:
                    print("\n❌ Simple runner failed!")
                    
            elif choice == '4':
                success = run_script('lightweight_run.py', 'Lightweight Memory-Optimized Runner')
                if success:
                    print("\n✅ Lightweight runner completed!")
                else:
                    print("\n❌ Lightweight runner failed!")
                    
            elif choice == '5':
                success = run_script('run_model.py', 'Advanced Runner with Quantization')
                if success:
                    print("\n✅ Advanced runner completed!")
                else:
                    print("\n❌ Advanced runner failed!")
                    
            elif choice == '6':
                show_readme()
                
            elif choice == '7':
                print("\n👋 Goodbye! Happy modeling!")
                break
                
            else:
                print("\n❌ Invalid choice. Please enter a number between 1-7.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! Happy modeling!")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
        
        # Wait for user input before showing menu again
        if choice != '6' and choice != '7':
            input("\nPress Enter to continue...")
        
        print("\n" + "="*50)

if __name__ == "__main__":
    main()