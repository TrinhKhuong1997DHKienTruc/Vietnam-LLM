#!/bin/bash

echo "🦙 Fin-o1-8B Comprehensive Installation Script"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check system requirements
print_status "Checking system requirements..."

# Check Python
if ! command -v python3 &> /dev/null; then
    print_error "Python3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    print_error "Python version $python_version is too old. Please install Python 3.8+ first."
    exit 1
fi

print_success "Python $python_version detected"

# Check disk space
available_space=$(df . | awk 'NR==2 {print $4}')
required_space=20000000  # 20GB in KB

if [ $available_space -lt $required_space ]; then
    print_error "Insufficient disk space. Need at least 20GB, but only $(($available_space / 1000000))GB available."
    exit 1
fi

print_success "Sufficient disk space available"

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    print_success "NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | while IFS=, read -r gpu_name memory; do
        print_status "   GPU: $gpu_name with ${memory}MB VRAM"
    done
else
    print_warning "No NVIDIA GPU detected. Will use CPU (slower but functional)."
fi

# Try different installation methods
print_status "Attempting to install dependencies..."

# Method 1: Try standard pip install
print_status "Method 1: Standard pip install..."
if pip3 install -r requirements.txt 2>/dev/null; then
    print_success "Dependencies installed successfully with standard pip"
    INSTALL_METHOD="standard"
else
    print_warning "Standard pip install failed, trying alternative methods..."
    
    # Method 2: Try pip with --user flag
    print_status "Method 2: pip install --user..."
    if pip3 install --user -r requirements.txt 2>/dev/null; then
        print_success "Dependencies installed successfully with --user flag"
        INSTALL_METHOD="user"
        export PATH=$HOME/.local/bin:$PATH
    else
        print_warning "User install failed, trying virtual environment..."
        
        # Method 3: Try virtual environment
        print_status "Method 3: Creating virtual environment..."
        if python3 -c "import venv" &> /dev/null; then
            if python3 -m venv fin-o1-env 2>/dev/null; then
                print_success "Virtual environment created successfully"
                
                # Activate and install
                source fin-o1-env/bin/activate
                if pip install -r requirements.txt 2>/dev/null; then
                    print_success "Dependencies installed successfully in virtual environment"
                    INSTALL_METHOD="venv"
                else
                    print_error "Failed to install dependencies in virtual environment"
                    INSTALL_METHOD="failed"
                fi
                deactivate
            else
                print_warning "Failed to create virtual environment"
                INSTALL_METHOD="failed"
            fi
        else
            print_warning "Python venv module not available"
            INSTALL_METHOD="failed"
        fi
    fi
fi

# Check if any method succeeded
if [ "$INSTALL_METHOD" = "failed" ]; then
    print_error "All installation methods failed. Please try manual installation:"
    echo ""
    echo "1. Install python3-venv: sudo apt install python3-venv"
    echo "2. Create virtual environment: python3 -m venv fin-o1-env"
    echo "3. Activate it: source fin-o1-env/bin/activate"
    echo "4. Install dependencies: pip install -r requirements.txt"
    echo ""
    echo "Or use Docker:"
    echo "1. docker build -t fin-o1-8b ."
    echo "2. docker run -it --rm -p 7860:7860 fin-o1-8b"
    exit 1
fi

print_success "Installation completed successfully!"
echo ""

# Print next steps based on installation method
case $INSTALL_METHOD in
    "standard")
        print_status "Next steps:"
        echo "1. Download the model: python3 download_model.py"
        echo "2. Run the web demo: python3 demo.py"
        echo "3. Or run CLI demo: python3 demo.py --cli"
        ;;
    "user")
        print_status "Next steps:"
        echo "1. Download the model: python3 download_model.py"
        echo "2. Run the web demo: python3 demo.py"
        echo "3. Or run CLI demo: python3 demo.py --cli"
        echo ""
        print_warning "If you encounter import errors, add to your PATH:"
        echo "   export PATH=\$HOME/.local/bin:\$PATH"
        ;;
    "venv")
        print_status "Next steps:"
        echo "1. Activate virtual environment: source fin-o1-env/bin/activate"
        echo "2. Download the model: python download_model.py"
        echo "3. Run the web demo: python demo.py"
        echo "4. Or run CLI demo: python demo.py --cli"
        echo ""
        print_status "Quick activation and run:"
        echo "   source fin-o1-env/bin/activate && python download_model.py"
        ;;
esac

echo ""
print_status "Alternative: Use Docker (if available)"
echo "1. docker build -t fin-o1-8b ."
echo "2. docker run -it --rm -p 7860:7860 fin-o1-8b"
echo ""
print_status "Note: Model download will take some time and requires ~16GB of data."
echo ""
print_success "Happy Financial Reasoning! 🦙💰"