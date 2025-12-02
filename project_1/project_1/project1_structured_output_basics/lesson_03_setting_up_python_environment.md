# 📘 Lesson 3: Setting Up Python Environment

Time to get your hands dirty! 🛠️ In this lesson, we'll set up a proper Python development environment that you'll use throughout this project and beyond.

---

## A. Concept Overview

### What & Why

A **Python environment** is an isolated workspace where you can install packages without affecting other projects or your system Python. This is crucial because:

- Different projects might need different versions of the same package
- You don't want to break your system by installing conflicting packages
- It makes your project reproducible on other machines

We'll use Python's built-in `venv` module to create a **virtual environment** - think of it as a fresh, clean Python installation just for this project.

### The Analogy 🏠

Think of virtual environments like apartments in a building:

- **System Python** = The building's shared facilities
- **Virtual Environment** = Your own apartment

You can decorate your apartment however you want (install any packages), and it won't affect your neighbors (other projects). If you make a mess, you can just move to a new apartment (create a new virtual environment) without affecting the building.

### Type Safety Benefit

A clean environment ensures:
- Consistent package versions for type checking
- No conflicts between Pydantic v1 and v2 (we need v2!)
- Reproducible setup for team members
- Clear dependencies for deployment

---

## B. Code Implementation

### File Structure

By the end of this lesson, you'll have:
```
pydantic-ai-project/
├── venv/                 # Virtual environment (auto-generated)
├── .gitignore           # Files to exclude from git
└── README.md            # Project documentation
```

### Step 1: Check Your Python Version

First, let's verify you have Python 3.9 or higher (required for Pydantic AI):

```bash
# On macOS/Linux:
python3 --version

# On Windows:
python --version
```

**Expected output:** `Python 3.9.x` or higher (3.10, 3.11, 3.12 are all great)

If you see Python 3.8 or lower, you'll need to install a newer version from [python.org](https://www.python.org/downloads/).

### Step 2: Create Your Project Directory

```bash
# Create a new directory for your project
mkdir pydantic-ai-project

# Navigate into it
cd pydantic-ai-project
```

### Step 3: Create a Virtual Environment

```bash
# On macOS/Linux:
python3 -m venv venv

# On Windows:
python -m venv venv
```

**What this does:**
- `python3 -m venv` = Run Python's venv module
- `venv` (second one) = Name of the directory to create

This creates a `venv` folder containing a complete Python installation.

### Step 4: Activate the Virtual Environment

This step is crucial - you must activate the environment before installing packages!

```bash
# On macOS/Linux:
source venv/bin/activate

# On Windows (Command Prompt):
venv\Scripts\activate.bat

# On Windows (PowerShell):
venv\Scripts\Activate.ps1
```

**How to know it worked:**
Your terminal prompt will change to show `(venv)` at the beginning:

```bash
# Before activation:
user@computer:~/pydantic-ai-project$

# After activation:
(venv) user@computer:~/pydantic-ai-project$
```

### Step 5: Verify the Environment

```bash
# Check that pip is using the virtual environment
which pip    # On macOS/Linux
where pip    # On Windows

# Should show something like:
# /home/user/pydantic-ai-project/venv/bin/pip
# NOT /usr/bin/pip or /usr/local/bin/pip
```

Also verify pip is up to date:

```bash
pip install --upgrade pip
```

### Step 6: Create Essential Project Files

Create a `.gitignore` file to exclude the virtual environment from version control:

```bash
# Create .gitignore
echo "venv/" > .gitignore
echo "__pycache__/" >> .gitignore
echo "*.pyc" >> .gitignore
echo ".env" >> .gitignore
```

Contents of `.gitignore`:
```
venv/
__pycache__/
*.pyc
.env
```

Create a basic `README.md`:

```markdown
# Pydantic AI Project

A type-safe AI application using Pydantic AI and Google Gemini.

## Setup

1. Create virtual environment: `python3 -m venv venv`
2. Activate it: `source venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`

## Usage

Coming soon!
```

### Complete Setup Script

Here's everything in one script you can run:

```bash
#!/bin/bash
# setup.sh - Run this to set up your project

# Create project directory
mkdir -p pydantic-ai-project
cd pydantic-ai-project

# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Create .gitignore
cat > .gitignore << EOF
venv/
__pycache__/
*.pyc
.env
.idea/
.vscode/
*.egg-info/
dist/
build/
EOF

# Create README
cat > README.md << EOF
# Pydantic AI Project

A type-safe AI application using Pydantic AI and Google Gemini.

## Setup

\`\`\`bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
\`\`\`
EOF

echo "✅ Environment setup complete!"
echo "Run 'source venv/bin/activate' to activate the environment."
```

---

## C. Test & Apply

### Verify Your Setup

Run these commands to confirm everything is working:

```bash
# 1. Check you're in the virtual environment
echo $VIRTUAL_ENV
# Should print the path to your venv folder

# 2. Check Python version
python --version
# Should be 3.9+

# 3. Check pip is local
pip --version
# Should show path inside your venv

# 4. List installed packages (should be minimal)
pip list
# Should show only pip and setuptools
```

### Expected Output

```
$ echo $VIRTUAL_ENV
/home/user/pydantic-ai-project/venv

$ python --version
Python 3.11.4

$ pip --version
pip 23.2.1 from /home/user/pydantic-ai-project/venv/lib/python3.11/site-packages/pip (python 3.11)

$ pip list
Package    Version
---------- -------
pip        23.2.1
setuptools 68.0.0
```

### Deactivating the Environment

When you're done working, you can deactivate the environment:

```bash
deactivate
```

The `(venv)` prefix will disappear from your prompt. Remember to activate it again next time you work on this project!

---

## D. Common Stumbling Blocks

### "Command not found: python3"

**On macOS:** Install Python via Homebrew:
```bash
brew install python
```

**On Windows:** Download from [python.org](https://www.python.org/downloads/) and make sure to check "Add Python to PATH" during installation.

**On Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install python3 python3-venv python3-pip
```

### "Permission denied when creating venv"

You might be trying to create the venv in a protected directory. Create it in your home folder or a projects directory:

```bash
cd ~
mkdir projects
cd projects
mkdir pydantic-ai-project
cd pydantic-ai-project
python3 -m venv venv
```

### "Activate script not found" on Windows

If using PowerShell, you might need to enable script execution:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then try activating again.

### "Still using system pip after activation"

Make sure you're using `pip` not `pip3` after activation, and verify with:

```bash
which pip  # macOS/Linux
where pip  # Windows
```

If it still shows system pip, try:
```bash
python -m pip install --upgrade pip
```

### "I accidentally installed packages globally!"

No worries! Just deactivate, delete the venv, and start over:

```bash
deactivate
rm -rf venv
python3 -m venv venv
source venv/bin/activate
```

---

## ✅ Lesson 3 Complete!

### Key Takeaways

1. **Virtual environments** isolate project dependencies
2. **Always activate** the environment before installing packages
3. **Check your Python version** - you need 3.9+
4. **Add venv to .gitignore** - never commit it to version control
5. **Deactivate** when switching to other projects

### Your Environment Checklist

- [ ] Python 3.9+ installed
- [ ] Project directory created
- [ ] Virtual environment created (`python3 -m venv venv`)
- [ ] Environment activated (you see `(venv)` in prompt)
- [ ] Pip upgraded (`pip install --upgrade pip`)
- [ ] `.gitignore` file created

### What's Next?

In Lesson 4, we'll install Pydantic AI and all the dependencies you need to build your first type-safe AI application!

---

*Environment ready? Let's install the packages in Lesson 4!* 🚀
