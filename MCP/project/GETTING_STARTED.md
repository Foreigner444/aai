# Getting Started with Smart Research Assistant

This guide will help you set up and run the Smart Research Assistant project.

## Prerequisites

- Python 3.10 or higher
- pip (Python package installer)
- Google Gemini API key (free)

## Step 1: Get Your Gemini API Key

1. Visit https://makersuite.google.com/app/apikey or https://aistudio.google.com/apikey
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the generated key

## Step 2: Set Up the Environment

### Clone/Navigate to Project

```bash
cd /path/to/project
```

### Create Virtual Environment (Recommended)

```bash
python -m venv venv

# Activate it:
# On Linux/Mac:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Set API Key

**Linux/Mac**:
```bash
export GEMINI_API_KEY='your-api-key-here'
```

**Windows (PowerShell)**:
```powershell
$env:GEMINI_API_KEY='your-api-key-here'
```

**Windows (CMD)**:
```cmd
set GEMINI_API_KEY=your-api-key-here
```

### Or Use a .env File

Create a file named `.env` in the project directory:

```
GEMINI_API_KEY=your-api-key-here
```

Then install python-dotenv (already in requirements.txt):
```bash
pip install python-dotenv
```

## Step 3: Configure the Application

Edit `config.json` if needed. Default configuration works for most cases:

```json
{
  "gemini_api_key": "",
  "model": "gemini-1.5-flash",
  "max_papers": 10,
  "servers": [...]
}
```

**Note**: If you set `GEMINI_API_KEY` environment variable, you don't need to put it in config.json.

## Step 4: Run the Application

### Option A: Run Basic Example

```bash
python main.py
```

This will:
- Connect to all MCP servers
- Run a sample research query
- Display results

### Option B: Interactive CLI

```bash
python cli.py
```

Then use commands like:
```
>>> research transformer architectures
>>> notes
>>> help
>>> exit
```

### Option C: Run Examples

Basic usage:
```bash
python examples/basic_usage.py
```

Advanced usage:
```bash
python examples/advanced_usage.py
```

## Step 5: Verify Everything Works

Run the tests:

```bash
pytest tests/ -v
```

All tests should pass. If they fail, check:
- Python version (must be 3.10+)
- All dependencies installed
- API key is set correctly

## Common Commands

### Research a Topic

Using CLI:
```bash
python cli.py
>>> research quantum computing applications
```

### List Saved Notes

```bash
python cli.py
>>> notes
```

### Run Specific Example

```bash
python examples/basic_usage.py
```

### Run Specific Test

```bash
pytest tests/test_servers.py -v
```

## Troubleshooting

### Error: "GEMINI_API_KEY not set"

Make sure you've set the environment variable:
```bash
echo $GEMINI_API_KEY  # Should print your key
```

If empty, set it again following Step 2.

### Error: "No module named 'mcp'"

Install dependencies:
```bash
pip install -r requirements.txt
```

### Error: "Failed to connect to servers"

Check that:
1. Python version is 3.10+
2. All dependencies are installed
3. You're in the project directory
4. Server files exist in `servers/` directory

### Servers Start But Agent Fails

Check:
1. GEMINI_API_KEY is valid
2. You have internet connection
3. Check rate limits (free tier: 15 requests/minute)

### Tests Fail

This is often normal for the first run. The agent tests require:
1. Valid Gemini API key
2. Internet connection
3. Servers to start successfully

You can skip agent tests:
```bash
pytest tests/test_servers.py -v
```

## Project Structure

```
project/
├── README.md              # Overview and documentation
├── GETTING_STARTED.md     # This file
├── requirements.txt       # Python dependencies
├── config.json           # Configuration
├── main.py               # Main application
├── cli.py                # Interactive CLI
├── servers/              # MCP Servers
│   ├── papers_server.py
│   ├── web_server.py
│   └── notes_server.py
├── agent/                # AI Agent
│   ├── research_agent.py
│   └── models.py
├── examples/             # Usage examples
│   ├── basic_usage.py
│   └── advanced_usage.py
└── tests/                # Tests
    ├── test_servers.py
    └── test_agent.py
```

## Next Steps

1. **Explore the Code**: Look at how the agent and servers work
2. **Modify Servers**: Add new tools or resources to the MCP servers
3. **Customize Agent**: Modify the agent's system prompt or add new tools
4. **Build Features**: Add new functionality like exporting reports, saving sessions, etc.
5. **Study Lessons**: Review the lessons in `/lessons` for deeper understanding

## Getting Help

- Check the lesson files in `/lessons` directory
- Review the code documentation
- Check MCP documentation: https://modelcontextprotocol.io
- Check Pydantic AI: https://ai.pydantic.dev
- Check Gemini API: https://ai.google.dev

## Tips for Success

1. **Start Simple**: Run the basic example first
2. **Check Logs**: Look at terminal output for errors
3. **Test Components**: Test servers individually before using with agent
4. **Read Lessons**: The lesson files explain concepts thoroughly
5. **Experiment**: Try different queries and modify the code

Happy researching! 🔬
