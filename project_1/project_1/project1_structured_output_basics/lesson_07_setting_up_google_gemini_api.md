# 📘 Lesson 7: Setting Up Google Gemini API

Time to connect to the AI! 🤖 In this lesson, you'll get your Gemini API key and configure it securely.

---

## A. Concept Overview

### What & Why

To use Google's Gemini AI models, you need an **API key** - a secret password that:
- Identifies your application to Google
- Tracks your usage and billing
- Protects against unauthorized access

We'll use **Google AI Studio** to get a free API key, then configure it securely using environment variables.

### The Analogy 🔑

An API key is like a library card:
- You need it to borrow books (make AI requests)
- It identifies you (tracks your usage)
- It has limits (free tier limits)
- You shouldn't share it (others could use your quota)

### Type Safety Benefit

Proper API configuration ensures:
- Keys never appear in code (security)
- Easy switching between development/production keys
- Clear error messages when keys are invalid
- Reproducible setup across environments

---

## B. Code Implementation

### File Structure
```
pydantic-ai-project/
├── .env                    # Your API key goes here (NEVER commit!)
├── .env.example           # Template for others (safe to commit)
├── config.py              # Configuration loader
├── .gitignore             # Must include .env
└── ...
```

### Step 1: Get Your Gemini API Key

1. **Go to Google AI Studio:**
   ```
   https://makersuite.google.com/app/apikey
   ```
   
   Or search "Google AI Studio" in your browser.

2. **Sign in** with your Google account

3. **Click "Create API Key"**
   - Select "Create API key in new project" (easiest option)
   - Or choose an existing Google Cloud project

4. **Copy your API key**
   - It looks like: `AIzaSy...` (39 characters)
   - Keep this window open - you'll need it!

### Step 2: Store Your API Key Securely

**Never put API keys directly in your code!** ❌

Instead, use a `.env` file:

```bash
# Create .env file
touch .env
```

Add your key to `.env`:
```
# .env - NEVER commit this file!
GEMINI_API_KEY=AIzaSyYourActualKeyHere
```

### Step 3: Verify .gitignore

Make sure `.env` is in your `.gitignore`:

```bash
cat .gitignore
```

Should include:
```
.env
```

If not, add it:
```bash
echo ".env" >> .gitignore
```

### Step 4: Create .env.example

Create a template that IS safe to commit:

```bash
# .env.example - Safe to commit, shows required variables
GEMINI_API_KEY=your_api_key_here
```

This tells other developers what environment variables they need.

### Step 5: Create Configuration Loader

Create `config.py`:

```python
"""
Configuration loader for the project.
Loads environment variables from .env file securely.
"""
import os
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()


def get_gemini_api_key() -> str:
    """
    Get the Gemini API key from environment variables.
    Raises an error if not found.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found!\n"
            "Please create a .env file with your API key:\n"
            "GEMINI_API_KEY=your_key_here\n\n"
            "Get your key at: https://makersuite.google.com/app/apikey"
        )
    
    if api_key == "your_api_key_here":
        raise ValueError(
            "Please replace 'your_api_key_here' with your actual API key!\n"
            "Get your key at: https://makersuite.google.com/app/apikey"
        )
    
    return api_key


# Configuration object
class Config:
    """Application configuration."""
    
    @property
    def gemini_api_key(self) -> str:
        return get_gemini_api_key()
    
    @property
    def gemini_model(self) -> str:
        """Default Gemini model to use."""
        return os.getenv("GEMINI_MODEL", "gemini-1.5-flash")


# Singleton instance
config = Config()


if __name__ == "__main__":
    # Test the configuration
    try:
        key = config.gemini_api_key
        # Only show first/last 4 characters for security
        masked = f"{key[:4]}...{key[-4:]}"
        print(f"✅ API Key loaded: {masked}")
        print(f"✅ Default model: {config.gemini_model}")
    except ValueError as e:
        print(f"❌ Configuration Error:\n{e}")
```

### Step 6: Test Your API Key

Create `test_api_key.py`:

```python
"""
Test that your Gemini API key is valid.
"""
import google.generativeai as genai
from config import config


def test_api_key():
    """Test the Gemini API key with a simple request."""
    print("Testing Gemini API connection...\n")
    
    # Configure the API
    genai.configure(api_key=config.gemini_api_key)
    
    # Create a model instance
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Make a simple request
    response = model.generate_content("Say 'Hello, Pydantic AI!' in exactly 3 words.")
    
    print(f"✅ API Key is valid!")
    print(f"✅ Response: {response.text}")


if __name__ == "__main__":
    try:
        test_api_key()
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check your API key is correct in .env")
        print("2. Ensure you have internet connection")
        print("3. Try generating a new key at:")
        print("   https://makersuite.google.com/app/apikey")
```

Run it:
```bash
python test_api_key.py
```

**Expected output:**
```
Testing Gemini API connection...

✅ API Key is valid!
✅ Response: Hello, Pydantic AI!
```

### Complete Setup Script

```bash
#!/bin/bash
# setup_api.sh - Set up Gemini API configuration

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cat > .env << EOF
# Google Gemini API Key
# Get yours at: https://makersuite.google.com/app/apikey
GEMINI_API_KEY=your_api_key_here

# Optional: Override default model
# GEMINI_MODEL=gemini-1.5-pro
EOF
    echo "✅ Created .env file"
else
    echo "ℹ️  .env file already exists"
fi

# Create .env.example
cat > .env.example << EOF
# Google Gemini API Key (required)
GEMINI_API_KEY=your_api_key_here

# Default model (optional)
GEMINI_MODEL=gemini-1.5-flash
EOF
echo "✅ Created .env.example"

# Ensure .env is in .gitignore
if ! grep -q "^\.env$" .gitignore 2>/dev/null; then
    echo ".env" >> .gitignore
    echo "✅ Added .env to .gitignore"
fi

echo ""
echo "Next steps:"
echo "1. Go to https://makersuite.google.com/app/apikey"
echo "2. Create an API key"
echo "3. Edit .env and replace 'your_api_key_here' with your key"
echo "4. Run 'python test_api_key.py' to verify"
```

---

## C. Test & Apply

### Verify Your Setup

1. **Check .env exists and has your key:**
   ```bash
   cat .env
   ```
   Should show your API key (not the placeholder)

2. **Run the test script:**
   ```bash
   python test_api_key.py
   ```
   Should show "API Key is valid!"

3. **Verify security:**
   ```bash
   git status
   ```
   The `.env` file should NOT appear in files to be committed

### Understanding API Limits

Google AI Studio provides a free tier:

| Limit | Free Tier |
|-------|-----------|
| Requests per minute | 60 |
| Requests per day | 1,500 |
| Tokens per minute | 1,000,000 |

For learning and development, this is plenty! If you need more, you can:
- Enable billing on Google Cloud
- Use Vertex AI for production workloads

### Environment Variable Priority

The `python-dotenv` library loads variables in this order:
1. System environment variables (highest priority)
2. `.env` file
3. Default values in code (lowest priority)

This means you can override `.env` values:
```bash
# Temporarily use a different key
GEMINI_API_KEY=AIzaDifferentKey python test_api_key.py
```

---

## D. Common Stumbling Blocks

### "GEMINI_API_KEY not found!"

The `.env` file isn't being loaded. Check:

1. `.env` file exists in project root:
   ```bash
   ls -la .env
   ```

2. `python-dotenv` is installed:
   ```bash
   pip install python-dotenv
   ```

3. You're running from the project directory:
   ```bash
   pwd  # Should be your project folder
   ```

### "API key not valid"

Your API key might be:
- Copied incorrectly (extra spaces?)
- From a different project that was deleted
- Restricted to certain APIs

**Fix:** Generate a new key at Google AI Studio.

### "Quota exceeded"

You've hit the free tier limits. Options:
- Wait until the limit resets (usually per minute or per day)
- Enable billing on Google Cloud
- Use a different Google account for testing

### "Permission denied"

Your API key might be restricted. In Google Cloud Console:
1. Go to APIs & Services → Credentials
2. Find your API key
3. Check "API restrictions"
4. Ensure "Generative Language API" is allowed

### "I accidentally committed my API key!"

**Don't panic, but act fast:**

1. **Regenerate your key immediately** at Google AI Studio
2. **Update your `.env`** with the new key
3. **Remove from git history** (optional but recommended):
   ```bash
   # This is complex - consider using git-filter-repo or BFG Repo-Cleaner
   # Or simply regenerate the key and move on
   ```

The old key should be considered compromised. Always regenerate.

---

## ✅ Lesson 7 Complete!

### Key Takeaways

1. **Get your API key** from Google AI Studio
2. **Never hardcode keys** - use `.env` files
3. **Add `.env` to `.gitignore** - never commit secrets
4. **Create `.env.example`** as a template for others
5. **Test your key** before building features
6. **Understand limits** - free tier is generous but not unlimited

### Your API Setup Checklist

- [ ] Google AI Studio account created
- [ ] API key generated
- [ ] `.env` file created with your key
- [ ] `.env` is in `.gitignore`
- [ ] `.env.example` created
- [ ] `config.py` created for loading config
- [ ] API key tested and working

### What's Next?

In Lesson 8, we'll explore the different Gemini model variants and when to use each one!

---

*API key working? Let's learn about Gemini models in Lesson 8!* 🚀
