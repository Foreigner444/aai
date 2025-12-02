# Lesson 8: Integrating Everything

## Building the Smart Research Assistant

In this final lesson, we'll integrate MCP servers with Pydantic AI agents and Gemini models to build a complete **Smart Research Assistant**.

## Project Architecture

```
Smart Research Assistant
├── MCP Servers
│   ├── Papers Server (search academic papers)
│   ├── Web Server (search web)
│   └── Notes Server (store research notes)
├── Pydantic AI Agent
│   ├── Gemini Model
│   ├── MCP Client Integration
│   └── Structured Outputs
└── CLI Interface
    └── User interactions
```

## Step 1: Understanding MCP Client Integration

To use MCP servers from a Pydantic AI agent, we need to:

1. Connect to MCP servers
2. Discover their tools and resources
3. Make them available to the agent
4. Handle responses

### MCP Client Setup

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def create_mcp_connection(command: str, args: list[str]):
    server_params = StdioServerParameters(
        command=command,
        args=args,
    )
    
    read, write = await stdio_client(server_params).__aenter__()
    session = await ClientSession(read, write).__aenter__()
    await session.initialize()
    
    return session
```

### Discovering MCP Tools

```python
async def get_mcp_tools(session: ClientSession):
    tools_response = await session.list_tools()
    
    mcp_tools = {}
    for tool in tools_response.tools:
        mcp_tools[tool.name] = {
            'description': tool.description,
            'schema': tool.inputSchema,
            'session': session
        }
    
    return mcp_tools
```

## Step 2: Creating the Research Agent

```python
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from typing import List, Literal
from dataclasses import dataclass

class Paper(BaseModel):
    title: str
    authors: List[str]
    abstract: str = Field(max_length=500)
    url: str
    year: int | None = None

class ResearchResult(BaseModel):
    query: str
    papers: List[Paper]
    summary: str
    key_findings: List[str]
    recommended_reading_order: List[int] = Field(description="Indices of papers in recommended order")

@dataclass
class ResearchContext:
    user_id: str
    research_area: str
    mcp_sessions: dict

research_agent = Agent(
    'gemini-1.5-flash',
    result_type=ResearchResult,
    deps_type=ResearchContext,
    system_prompt="""
    You are an expert research assistant. Help users find and understand academic papers.
    
    Workflow:
    1. Search for relevant papers using available tools
    2. Analyze abstracts and relevance
    3. Summarize key findings
    4. Suggest reading order based on prerequisites and importance
    
    Be thorough but concise. Focus on actionable insights.
    """
)

@research_agent.tool
async def search_papers(
    ctx: RunContext[ResearchContext],
    query: str,
    max_results: int = 5
) -> List[dict]:
    papers_session = ctx.deps.mcp_sessions.get('papers')
    if not papers_session:
        return [{"error": "Papers server not available"}]
    
    result = await papers_session.call_tool(
        "search_papers",
        {"query": query, "max_results": max_results}
    )
    
    import json
    return json.loads(result.content[0].text) if result.content else []

@research_agent.tool
async def save_note(
    ctx: RunContext[ResearchContext],
    title: str,
    content: str,
    tags: List[str]
) -> str:
    notes_session = ctx.deps.mcp_sessions.get('notes')
    if not notes_session:
        return "Notes server not available"
    
    result = await notes_session.call_tool(
        "add_note",
        {"title": title, "content": content, "tags": tags}
    )
    
    return result.content[0].text if result.content else "Failed to save note"

@research_agent.tool
async def search_web(
    ctx: RunContext[ResearchContext],
    query: str
) -> List[dict]:
    web_session = ctx.deps.mcp_sessions.get('web')
    if not web_session:
        return [{"error": "Web server not available"}]
    
    result = await web_session.call_tool(
        "search",
        {"query": query, "max_results": 3}
    )
    
    import json
    return json.loads(result.content[0].text) if result.content else []
```

## Step 3: Building MCP Servers

### Papers MCP Server

```python
import asyncio
import json
from mcp.server import Server
from mcp.types import Tool, TextContent
import mcp.server.stdio

server = Server("papers-server")

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="search_papers",
            description="Search for academic papers by keywords",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "max_results": {
                        "type": "integer",
                        "default": 5,
                        "description": "Maximum results"
                    }
                },
                "required": ["query"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "search_papers":
        query = arguments["query"]
        max_results = arguments.get("max_results", 5)
        
        papers = [
            {
                "title": f"Paper about {query}",
                "authors": ["Author A", "Author B"],
                "abstract": f"This paper explores {query} in depth...",
                "url": f"https://arxiv.org/paper{i}",
                "year": 2024
            }
            for i in range(1, max_results + 1)
        ]
        
        return [TextContent(type="text", text=json.dumps(papers))]
    
    return [TextContent(type="text", text="Unknown tool")]

async def main():
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())
```

### Web Search MCP Server

```python
import asyncio
import json
from mcp.server import Server
from mcp.types import Tool, TextContent
import mcp.server.stdio

server = Server("web-server")

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="search",
            description="Search the web for information",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "default": 3}
                },
                "required": ["query"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "search":
        query = arguments["query"]
        max_results = arguments.get("max_results", 3)
        
        results = [
            {
                "title": f"Result {i} for {query}",
                "url": f"https://example.com/result{i}",
                "snippet": f"Information about {query}..."
            }
            for i in range(1, max_results + 1)
        ]
        
        return [TextContent(type="text", text=json.dumps(results))]
    
    return [TextContent(type="text", text="Unknown tool")]

async def main():
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())
```

## Step 4: Main Application

```python
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def connect_to_servers():
    sessions = {}
    
    servers = {
        'papers': StdioServerParameters(command="python", args=["papers_server.py"]),
        'web': StdioServerParameters(command="python", args=["web_server.py"]),
        'notes': StdioServerParameters(command="python", args=["notes_server.py"]),
    }
    
    for name, params in servers.items():
        try:
            read, write = await stdio_client(params).__aenter__()
            session = ClientSession(read, write)
            await session.initialize()
            sessions[name] = session
            print(f"✓ Connected to {name} server")
        except Exception as e:
            print(f"✗ Failed to connect to {name} server: {e}")
    
    return sessions

async def main():
    print("Starting Research Assistant...\n")
    
    sessions = await connect_to_servers()
    
    if not sessions:
        print("No servers available. Exiting.")
        return
    
    context = ResearchContext(
        user_id="user123",
        research_area="machine learning",
        mcp_sessions=sessions
    )
    
    query = "transformer architectures in natural language processing"
    
    print(f"\nResearching: {query}\n")
    
    result = await research_agent.run(
        f"Find and analyze papers about: {query}",
        deps=context
    )
    
    print("=== Research Results ===\n")
    print(f"Query: {result.data.query}")
    print(f"\nSummary: {result.data.summary}")
    
    print(f"\n=== Found {len(result.data.papers)} Papers ===")
    for i, paper in enumerate(result.data.papers, 1):
        print(f"\n{i}. {paper.title}")
        print(f"   Authors: {', '.join(paper.authors)}")
        print(f"   Year: {paper.year}")
        print(f"   URL: {paper.url}")
        print(f"   Abstract: {paper.abstract[:200]}...")
    
    print(f"\n=== Key Findings ===")
    for finding in result.data.key_findings:
        print(f"• {finding}")
    
    print(f"\n=== Recommended Reading Order ===")
    for i, idx in enumerate(result.data.recommended_reading_order, 1):
        paper = result.data.papers[idx]
        print(f"{i}. {paper.title}")
    
    for session in sessions.values():
        try:
            await session.__aexit__(None, None, None)
        except:
            pass

if __name__ == "__main__":
    asyncio.run(main())
```

## Step 5: Interactive CLI

```python
import asyncio
from prompt_toolkit import prompt
from prompt_toolkit.history import InMemoryHistory

class ResearchCLI:
    def __init__(self):
        self.sessions = None
        self.history = InMemoryHistory()
        self.running = True
    
    async def start(self):
        print("🔬 Smart Research Assistant")
        print("=" * 50)
        
        self.sessions = await connect_to_servers()
        
        if not self.sessions:
            print("❌ No servers available. Exiting.")
            return
        
        print("\nCommands:")
        print("  research <query> - Research a topic")
        print("  notes - List saved notes")
        print("  help - Show help")
        print("  exit - Exit")
        print()
        
        while self.running:
            try:
                user_input = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: prompt(">>> ", history=self.history)
                )
                
                await self.handle_command(user_input.strip())
            
            except (KeyboardInterrupt, EOFError):
                break
        
        await self.cleanup()
    
    async def handle_command(self, command: str):
        if not command:
            return
        
        if command == "exit":
            self.running = False
            print("Goodbye!")
        
        elif command == "help":
            print("Available commands:")
            print("  research <query> - Research a topic")
            print("  notes - List saved notes")
            print("  exit - Exit")
        
        elif command.startswith("research "):
            query = command[9:]
            await self.do_research(query)
        
        elif command == "notes":
            await self.list_notes()
        
        else:
            print(f"Unknown command: {command}. Type 'help' for commands.")
    
    async def do_research(self, query: str):
        print(f"\n🔍 Researching: {query}\n")
        
        context = ResearchContext(
            user_id="user123",
            research_area=query,
            mcp_sessions=self.sessions
        )
        
        try:
            result = await research_agent.run(
                f"Research this topic: {query}",
                deps=context
            )
            
            self.display_results(result.data)
        
        except Exception as e:
            print(f"❌ Error: {e}")
    
    def display_results(self, data):
        print("=" * 50)
        print(f"📊 Summary: {data.summary}\n")
        
        print(f"📄 Found {len(data.papers)} papers:\n")
        for i, paper in enumerate(data.papers, 1):
            print(f"{i}. {paper.title}")
            print(f"   {', '.join(paper.authors)} ({paper.year})")
            print(f"   {paper.url}\n")
        
        print("💡 Key Findings:")
        for finding in data.key_findings:
            print(f"  • {finding}")
        
        print("\n" + "=" * 50 + "\n")
    
    async def list_notes(self):
        notes_session = self.sessions.get('notes')
        if not notes_session:
            print("❌ Notes server not available")
            return
        
        try:
            result = await notes_session.call_tool("list_all_notes", {})
            print(result.content[0].text)
        except Exception as e:
            print(f"❌ Error: {e}")
    
    async def cleanup(self):
        for session in self.sessions.values():
            try:
                await session.__aexit__(None, None, None)
            except:
                pass

if __name__ == "__main__":
    cli = ResearchCLI()
    asyncio.run(cli.start())
```

## Step 6: Configuration Management

```python
from pydantic import BaseModel
from typing import List
import os
import json

class ServerConfig(BaseModel):
    name: str
    command: str
    args: List[str]
    enabled: bool = True

class AppConfig(BaseModel):
    gemini_api_key: str
    servers: List[ServerConfig]
    model: str = "gemini-1.5-flash"
    max_papers: int = 10

def load_config(path: str = "config.json") -> AppConfig:
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        return AppConfig(**data)
    
    default_config = AppConfig(
        gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        servers=[
            ServerConfig(name="papers", command="python", args=["papers_server.py"]),
            ServerConfig(name="web", command="python", args=["web_server.py"]),
            ServerConfig(name="notes", command="python", args=["notes_server.py"]),
        ]
    )
    
    with open(path, 'w') as f:
        json.dump(default_config.model_dump(), f, indent=2)
    
    return default_config

config = load_config()
```

## Step 7: Testing

```python
import pytest
import asyncio

@pytest.mark.asyncio
async def test_papers_server():
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    
    server_params = StdioServerParameters(
        command="python",
        args=["papers_server.py"],
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            result = await session.call_tool(
                "search_papers",
                {"query": "machine learning", "max_results": 3}
            )
            
            assert result.content
            papers = json.loads(result.content[0].text)
            assert len(papers) == 3
            assert papers[0]["title"]

@pytest.mark.asyncio
async def test_research_agent():
    sessions = await connect_to_servers()
    
    context = ResearchContext(
        user_id="test_user",
        research_area="testing",
        mcp_sessions=sessions
    )
    
    result = await research_agent.run(
        "Find papers about testing",
        deps=context
    )
    
    assert result.data.query
    assert result.data.papers
    assert result.data.summary
    assert result.data.key_findings

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

## Best Practices

### 1. Error Handling

```python
async def safe_mcp_call(session, tool_name, arguments):
    try:
        result = await session.call_tool(tool_name, arguments)
        return result.content[0].text if result.content else None
    except Exception as e:
        logger.error(f"MCP call failed: {e}")
        return None
```

### 2. Connection Pooling

```python
class MCPConnectionPool:
    def __init__(self):
        self.sessions = {}
        self.locks = {}
    
    async def get_session(self, server_name: str):
        if server_name not in self.sessions:
            await self.connect(server_name)
        return self.sessions.get(server_name)
    
    async def connect(self, server_name: str):
        if server_name not in self.locks:
            self.locks[server_name] = asyncio.Lock()
        
        async with self.locks[server_name]:
            if server_name not in self.sessions:
                params = get_server_params(server_name)
                session = await create_mcp_connection(params.command, params.args)
                self.sessions[server_name] = session
```

### 3. Caching

```python
from functools import lru_cache
from datetime import datetime, timedelta

class CachedMCPClient:
    def __init__(self, session):
        self.session = session
        self.cache = {}
    
    async def call_tool(self, name: str, arguments: dict, ttl: int = 300):
        cache_key = f"{name}:{json.dumps(arguments, sort_keys=True)}"
        
        if cache_key in self.cache:
            cached_time, cached_result = self.cache[cache_key]
            if datetime.now() - cached_time < timedelta(seconds=ttl):
                return cached_result
        
        result = await self.session.call_tool(name, arguments)
        self.cache[cache_key] = (datetime.now(), result)
        
        return result
```

## Deployment

### Docker Setup

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV GEMINI_API_KEY=""

CMD ["python", "main.py"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  papers-server:
    build: ./servers/papers
    container_name: papers_server
  
  web-server:
    build: ./servers/web
    container_name: web_server
  
  notes-server:
    build: ./servers/notes
    container_name: notes_server
    volumes:
      - notes_data:/data
  
  app:
    build: .
    depends_on:
      - papers-server
      - web-server
      - notes-server
    environment:
      - GEMINI_API_KEY=${GEMINI_API_KEY}
    volumes:
      - ./config.json:/app/config.json

volumes:
  notes_data:
```

## Summary

You've now learned how to:
- Integrate MCP servers with Pydantic AI agents
- Connect to multiple MCP servers
- Build a complete research assistant
- Handle real-world complexity (errors, config, caching)
- Deploy with Docker

## Final Project

Enhance the Research Assistant with:

1. **More MCP Servers**:
   - PDF parser server
   - Citation formatter server
   - GitHub repo analyzer server

2. **Advanced Features**:
   - Save research sessions
   - Export to markdown/PDF
   - Collaborative research (multi-user)
   - Research graphs/visualization

3. **Better UX**:
   - Web interface (FastAPI + React)
   - Progress indicators
   - Bookmark papers
   - Smart recommendations

4. **Production Ready**:
   - Logging and monitoring
   - Rate limiting
   - Authentication
   - Database persistence

Congratulations! You've completed the course and built a production-ready MCP application with Pydantic AI and Gemini!

---

**Next**: Check out the `/project` folder for the complete implementation and explore additional examples.
