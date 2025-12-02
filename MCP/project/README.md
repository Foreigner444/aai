# Smart Research Assistant - Complete Project

This is the complete implementation of the Smart Research Assistant built with MCP, Pydantic AI, and Google Gemini.

## Project Structure

```
project/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── config.json              # Configuration file
├── servers/                 # MCP Servers
│   ├── papers_server.py     # Academic papers search
│   ├── web_server.py        # Web search
│   └── notes_server.py      # Notes storage
├── agent/                   # Pydantic AI Agent
│   ├── research_agent.py    # Main research agent
│   └── models.py            # Data models
├── main.py                  # Main application
├── cli.py                   # Interactive CLI
├── tests/                   # Tests
│   ├── test_servers.py      # Server tests
│   └── test_agent.py        # Agent tests
└── examples/                # Usage examples
    ├── basic_usage.py
    └── advanced_usage.py
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set API Key

```bash
export GEMINI_API_KEY='your-key-here'
```

### 3. Run the Application

```bash
python main.py
```

Or use the interactive CLI:

```bash
python cli.py
```

## Usage Examples

### Basic Research Query

```python
from main import run_research

result = await run_research("transformer architectures in NLP")
print(result.summary)
for paper in result.papers:
    print(f"- {paper.title}")
```

### Interactive CLI

```bash
$ python cli.py
🔬 Smart Research Assistant
==================================================
✓ Connected to papers server
✓ Connected to web server
✓ Connected to notes server

Commands:
  research <query> - Research a topic
  notes - List saved notes
  help - Show help
  exit - Exit

>>> research quantum computing
```

## MCP Servers

### Papers Server

Provides academic paper search functionality.

**Tools**:
- `search_papers(query, max_results)` - Search for papers
- `get_paper_details(paper_id)` - Get detailed info
- `get_citations(paper_id)` - Get paper citations

**Resources**:
- `paper:///{id}` - Individual papers

### Web Server

Provides web search capabilities.

**Tools**:
- `search(query, max_results)` - Web search
- `get_page_content(url)` - Fetch page content

### Notes Server

Stores research notes and findings.

**Tools**:
- `add_note(title, content, tags)` - Create note
- `search_notes(query)` - Search notes
- `list_all_notes()` - List all notes
- `delete_note(id)` - Delete note

**Resources**:
- `note:///{id}` - Individual notes

## Agent Capabilities

The research agent can:
- Search for relevant academic papers
- Analyze paper abstracts and relevance
- Search web for supplementary information
- Save research notes with tags
- Summarize findings
- Recommend reading order
- Identify key insights

## Configuration

Edit `config.json`:

```json
{
  "gemini_api_key": "your-key",
  "model": "gemini-1.5-flash",
  "max_papers": 10,
  "servers": [
    {
      "name": "papers",
      "command": "python",
      "args": ["servers/papers_server.py"],
      "enabled": true
    },
    {
      "name": "web",
      "command": "python",
      "args": ["servers/web_server.py"],
      "enabled": true
    },
    {
      "name": "notes",
      "command": "python",
      "args": ["servers/notes_server.py"],
      "enabled": true
    }
  ]
}
```

## Testing

Run all tests:

```bash
pytest tests/ -v
```

Test specific server:

```bash
pytest tests/test_servers.py::test_papers_server -v
```

## Extending the Project

### Add New MCP Server

1. Create server file in `servers/`:

```python
import asyncio
from mcp.server import Server
from mcp.types import Tool, TextContent
import mcp.server.stdio

server = Server("my-server")

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [...]

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    # Implement tool logic
    pass

async def main():
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
```

2. Add to config.json
3. Update agent to use new tools

### Add New Agent Tool

In `agent/research_agent.py`:

```python
@research_agent.tool
async def my_new_tool(ctx: RunContext[ResearchContext], param: str) -> str:
    # Implement tool
    return result
```

## Troubleshooting

### Server won't start

- Check Python version (3.10+)
- Ensure all dependencies installed
- Check server logs

### Agent not finding papers

- Verify papers server is running
- Check MCP connection in logs
- Test server manually with test script

### Gemini API errors

- Verify API key is set correctly
- Check rate limits
- Ensure billing is enabled

## License

MIT License

## Support

For issues and questions, see the lesson files in `/lessons` or refer to:
- MCP Documentation: https://modelcontextprotocol.io
- Pydantic AI: https://ai.pydantic.dev
- Gemini API: https://ai.google.dev
