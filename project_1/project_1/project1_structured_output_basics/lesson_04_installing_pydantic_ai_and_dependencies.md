# 📘 Lesson 4: Installing Pydantic AI and Dependencies

With your environment ready, let's install the packages that will power your AI applications! 🎉

---

## A. Concept Overview

### What & Why

We need to install several packages:

| Package | Purpose |
|---------|---------|
| **pydantic-ai** | The main framework for type-safe AI agents |
| **pydantic** | Data validation (installed automatically with pydantic-ai) |
| **google-generativeai** | Google's Gemini API client |
| **python-dotenv** | Load API keys from .env files securely |

**Why these specific packages?**

- **pydantic-ai** provides the Agent class and structured output handling
- **pydantic** (v2) provides the BaseModel class and validation
- **google-generativeai** lets you communicate with Gemini models
- **python-dotenv** keeps your API keys out of your code (security best practice!)

### The Analogy 📦

Think of pip packages like LEGO sets:

- **pydantic-ai** = The instruction manual that tells you how to build
- **pydantic** = The base plates everything snaps onto
- **google-generativeai** = The special motor pieces that make things move
- **python-dotenv** = The storage box that keeps small pieces safe

Together, they let you build something amazing!

### Type Safety Benefit

Installing the right versions ensures:
- Pydantic v2's improved validation and performance
- Full type stub support for IDE autocomplete
- Compatible versions that work together seamlessly

---

## B. Code Implementation

### File Structure

After this lesson:
```
pydantic-ai-project/
├── venv/
├── .env                  # Your API keys (NEVER commit this!)
├── .gitignore
├── requirements.txt      # Package list for reproducibility
└── README.md
```

### Step 1: Activate Your Environment

If not already activated:

```bash
# macOS/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

You should see `(venv)` in your prompt.

### Step 2: Install Pydantic AI with Gemini Support

```bash
# Install pydantic-ai with Google Gemini support
pip install 'pydantic-ai[google]'
```

**What this installs:**
- `pydantic-ai` - The main framework
- `pydantic` - Data validation library
- `google-generativeai` - Gemini API client
- Various supporting packages

### Step 3: Install Additional Utilities

```bash
# For secure API key management
pip install python-dotenv

# For better async support (optional but recommended)
pip install httpx
```

### Step 4: Verify Installation

```python
# verify_install.py
# Run this to check everything installed correctly

def check_imports():
    """Verify all packages are importable."""
    print("Checking installations...\n")
    
    # Check pydantic-ai
    try:
        import pydantic_ai
        print(f"✅ pydantic-ai: {pydantic_ai.__version__}")
    except ImportError as e:
        print(f"❌ pydantic-ai: {e}")
    
    # Check pydantic
    try:
        import pydantic
        print(f"✅ pydantic: {pydantic.__version__}")
        if pydantic.__version__.startswith("1"):
            print("   ⚠️  Warning: You have Pydantic v1, but v2 is required!")
    except ImportError as e:
        print(f"❌ pydantic: {e}")
    
    # Check google-generativeai
    try:
        import google.generativeai
        print(f"✅ google-generativeai: installed")
    except ImportError as e:
        print(f"❌ google-generativeai: {e}")
    
    # Check python-dotenv
    try:
        import dotenv
        print(f"✅ python-dotenv: installed")
    except ImportError as e:
        print(f"❌ python-dotenv: {e}")
    
    print("\n🎉 All core packages installed!" if all else "❌ Some packages missing!")

if __name__ == "__main__":
    check_imports()
```

Run it:
```bash
python verify_install.py
```

**Expected output:**
```
Checking installations...

✅ pydantic-ai: 0.0.24
✅ pydantic: 2.5.2
✅ google-generativeai: installed
✅ python-dotenv: installed

🎉 All core packages installed!
```

### Step 5: Create requirements.txt

This file lets others (or future you) recreate your environment:

```bash
# Generate requirements file
pip freeze > requirements.txt
```

Your `requirements.txt` will look something like:
```
annotated-types==0.6.0
anyio==4.2.0
certifi==2023.11.17
google-ai-generativelanguage==0.4.0
google-api-core==2.15.0
google-auth==2.25.2
google-generativeai==0.3.2
grpcio==1.60.0
httpx==0.26.0
pydantic==2.5.2
pydantic-ai==0.0.24
pydantic_core==2.14.5
python-dotenv==1.0.0
...
```

### Step 6: Set Up Environment Variables

Create a `.env` file for your API keys:

```bash
# Create .env file (will be filled in next lesson)
touch .env
```

Add this template to `.env`:
```
# Google Gemini API Key
# Get yours at: https://makersuite.google.com/app/apikey
GEMINI_API_KEY=your_api_key_here
```

**⚠️ CRITICAL SECURITY NOTE:**
- NEVER commit `.env` to version control
- NEVER share your API keys
- NEVER put API keys directly in code

Verify `.env` is in your `.gitignore`:
```bash
cat .gitignore | grep ".env"
```

### Complete Installation Script

```bash
#!/bin/bash
# install.sh - Run after setting up virtual environment

# Ensure we're in a virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "❌ Please activate your virtual environment first!"
    echo "Run: source venv/bin/activate"
    exit 1
fi

echo "📦 Installing Pydantic AI with Google Gemini support..."
pip install 'pydantic-ai[google]'

echo "📦 Installing additional utilities..."
pip install python-dotenv httpx

echo "📦 Upgrading pip..."
pip install --upgrade pip

echo "📝 Generating requirements.txt..."
pip freeze > requirements.txt

echo "🔐 Creating .env template..."
cat > .env << EOF
# Google Gemini API Key
# Get yours at: https://makersuite.google.com/app/apikey
GEMINI_API_KEY=your_api_key_here
EOF

echo ""
echo "✅ Installation complete!"
echo ""
echo "Next steps:"
echo "1. Get your Gemini API key from: https://makersuite.google.com/app/apikey"
echo "2. Add it to your .env file"
echo "3. Continue to the next lesson!"
```

---

## C. Test & Apply

### Quick Installation Test

Create a file called `test_installation.py`:

```python
"""
Test that all packages are correctly installed and compatible.
"""
from pydantic import BaseModel
from pydantic_ai import Agent

class TestModel(BaseModel):
    message: str
    number: int

# This won't actually call Gemini (no API key yet)
# but it verifies the packages work together
agent = Agent('gemini-1.5-flash', result_type=TestModel)

print("✅ All packages are correctly installed and compatible!")
print(f"   Agent created with result_type: {agent.result_type}")
print("\nYou're ready to get your Gemini API key in the next lesson!")
```

Run it:
```bash
python test_installation.py
```

**Expected output:**
```
✅ All packages are correctly installed and compatible!
   Agent created with result_type: <class '__main__.TestModel'>

You're ready to get your Gemini API key in the next lesson!
```

### Understanding What You Installed

```bash
# See all installed packages
pip list

# See package details
pip show pydantic-ai
pip show pydantic
```

---

## D. Common Stumbling Blocks

### "ModuleNotFoundError: No module named 'pydantic_ai'"

You probably forgot to activate the virtual environment:
```bash
source venv/bin/activate  # Then try again
pip install 'pydantic-ai[google]'
```

### "ERROR: Could not find a version that satisfies the requirement"

Your Python version might be too old. Check:
```bash
python --version
```

If it's below 3.9, upgrade Python first.

### "Pydantic v1 vs v2 Confusion"

Pydantic AI requires Pydantic v2. Check your version:
```python
import pydantic
print(pydantic.__version__)  # Should start with "2."
```

If you have v1, uninstall and reinstall:
```bash
pip uninstall pydantic
pip install 'pydantic-ai[google]'  # This installs correct version
```

### "SSL Certificate Errors"

On some systems, you might see SSL errors. Try:
```bash
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org 'pydantic-ai[google]'
```

Or update certificates:
```bash
# macOS
pip install --upgrade certifi

# Linux
sudo apt install ca-certificates
```

### "Permission Denied During Install"

Never use `sudo pip install`! Make sure you're in a virtual environment:
```bash
# Verify you're in venv
echo $VIRTUAL_ENV  # Should print your venv path

# If empty, activate first
source venv/bin/activate
```

---

## ✅ Lesson 4 Complete!

### Key Takeaways

1. **Install with extras:** `pip install 'pydantic-ai[google]'` gets everything you need
2. **Always use virtual environments** for clean, isolated installs
3. **Create requirements.txt** for reproducibility
4. **Use .env files** for API keys - never hardcode them!
5. **Verify installations** before moving forward

### Your Package Checklist

- [ ] pydantic-ai installed
- [ ] pydantic v2 installed (check version!)
- [ ] google-generativeai installed
- [ ] python-dotenv installed
- [ ] requirements.txt created
- [ ] .env file template created (in .gitignore)

### What's Next?

In Lesson 5, you'll create your first Pydantic model - the building block of type-safe AI outputs!

---

*Packages installed? Let's build our first model in Lesson 5!* 🚀
