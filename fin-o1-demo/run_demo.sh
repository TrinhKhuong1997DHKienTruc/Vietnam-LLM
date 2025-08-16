#!/bin/bash

# Fin-o1-8B Demo Runner Script
# A simple script to run different interfaces of the Fin-o1-8B model

echo "💰 Fin-o1-8B Financial AI Demo"
echo "================================"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed or not in PATH"
    echo "Please install Python 3.8+ and try again"
    exit 1
fi

# Check if requirements are installed
echo "🔍 Checking dependencies..."
if ! python3 -c "import torch, transformers" &> /dev/null; then
    echo "⚠️  Some dependencies are missing"
    echo "Installing requirements..."
    pip3 install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install requirements"
        echo "Please install manually: pip3 install -r requirements.txt"
        exit 1
    fi
fi

echo "✅ Dependencies are ready!"
echo ""

# Menu
echo "Choose an interface to run:"
echo "1. Command Line Interface (Recommended for first time)"
echo "2. Streamlit Web App (Full-featured web interface)"
echo "3. Gradio Web App (Simple web interface)"
echo "4. Test Setup (Verify environment)"
echo "5. Exit"
echo ""

read -p "Enter your choice (1-5): " choice

case $choice in
    1)
        echo "🚀 Starting Command Line Interface..."
        python3 fin_o1_demo.py
        ;;
    2)
        echo "🌐 Starting Streamlit Web App..."
        echo "The app will open in your browser at http://localhost:8501"
        echo "Press Ctrl+C to stop the server"
        streamlit run streamlit_app.py
        ;;
    3)
        echo "🌐 Starting Gradio Web App..."
        echo "The app will open in your browser at http://localhost:7860"
        echo "Press Ctrl+C to stop the server"
        python3 gradio_app.py
        ;;
    4)
        echo "🧪 Running Setup Test..."
        python3 test_setup.py
        ;;
    5)
        echo "👋 Goodbye!"
        exit 0
        ;;
    *)
        echo "❌ Invalid choice. Please run the script again."
        exit 1
        ;;
esac