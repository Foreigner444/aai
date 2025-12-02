#!/bin/bash

echo "Smart Research Assistant - Setup Script"
echo "========================================"
echo ""

echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
required_version="3.10"

if (( $(echo "$python_version < $required_version" | bc -l) )); then
    echo "❌ Python 3.10+ required. You have Python $python_version"
    exit 1
fi

echo "✓ Python $python_version detected"
echo ""

echo "Creating virtual environment..."
python3 -m venv venv

echo "✓ Virtual environment created"
echo ""

echo "Activating virtual environment..."
source venv/bin/activate

echo "✓ Virtual environment activated"
echo ""

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "✓ Dependencies installed"
echo ""

echo "Checking for GEMINI_API_KEY..."
if [ -z "$GEMINI_API_KEY" ]; then
    echo "⚠️  GEMINI_API_KEY not set"
    echo ""
    echo "To set it, run:"
    echo "  export GEMINI_API_KEY='your-key-here'"
    echo ""
    echo "Get your key from: https://makersuite.google.com/app/apikey"
    echo ""
else
    echo "✓ GEMINI_API_KEY is set"
    echo ""
fi

echo "Running tests..."
pytest tests/test_servers.py -v

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Tests passed!"
else
    echo ""
    echo "⚠️  Some tests failed (this may be normal on first run)"
fi

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "To get started:"
echo "  1. Activate virtual environment: source venv/bin/activate"
echo "  2. Set API key: export GEMINI_API_KEY='your-key'"
echo "  3. Run the app: python main.py"
echo "  4. Or use CLI: python cli.py"
echo ""
echo "For more info, see GETTING_STARTED.md"
echo ""
