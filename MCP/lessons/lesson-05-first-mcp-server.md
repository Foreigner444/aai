# Lesson 5: Building Your First MCP Server

## Overview

In this lesson, you'll build a simple but functional MCP server from scratch. We'll create a **Note-taking Server** that:
- Stores notes in memory
- Exposes notes as resources
- Provides tools to add/search notes

## MCP Python SDK

First, install the MCP SDK:

```bash
pip install mcp
```

The SDK provides:
- Server framework
- Protocol handling
- Transport mechanisms
- Type definitions

## Project Structure

Create a new directory:

```bash
mkdir notes_mcp_server
cd notes_mcp_server
```

Files we'll create:
```
notes_mcp_server/
├── server.py          # Main server code
├── models.py          # Data models
└── test_client.py     # Test client
```

## Step 1: Data Models

Create `models.py`:

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import List

@dataclass
class Note:
    id: int
    title: str
    content: str
    created_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'title': self.title,
            'content': self.content,
            'created_at': self.created_at.isoformat(),
            'tags': self.tags
        }
    
    def matches_search(self, query: str) -> bool:
        query_lower = query.lower()
        return (
            query_lower in self.title.lower() or
            query_lower in self.content.lower() or
            any(query_lower in tag.lower() for tag in self.tags)
        )

@dataclass
class NoteStore:
    notes: List[Note] = field(default_factory=list)
    _next_id: int = 1
    
    def add_note(self, title: str, content: str, tags: List[str] = None) -> Note:
        note = Note(
            id=self._next_id,
            title=title,
            content=content,
            tags=tags or []
        )
        self.notes.append(note)
        self._next_id += 1
        return note
    
    def get_note(self, note_id: int) -> Note | None:
        for note in self.notes:
            if note.id == note_id:
                return note
        return None
    
    def search_notes(self, query: str) -> List[Note]:
        return [note for note in self.notes if note.matches_search(query)]
    
    def get_all_notes(self) -> List[Note]:
        return self.notes.copy()
```

This provides:
- `Note`: Individual note with metadata
- `NoteStore`: In-memory storage with CRUD operations

## Step 2: Basic MCP Server

Create `server.py`:

```python
import asyncio
import json
from mcp.server import Server
from mcp.types import (
    Resource,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
)
import mcp.server.stdio
from models import NoteStore

note_store = NoteStore()

server = Server("notes-server")

@server.list_resources()
async def list_resources() -> list[Resource]:
    resources = []
    
    for note in note_store.get_all_notes():
        resource = Resource(
            uri=f"note:///{note.id}",
            name=f"Note: {note.title}",
            mimeType="application/json",
            description=f"Note created at {note.created_at.strftime('%Y-%m-%d %H:%M')}"
        )
        resources.append(resource)
    
    return resources

@server.read_resource()
async def read_resource(uri: str) -> str:
    if not uri.startswith("note:///"):
        raise ValueError(f"Invalid URI: {uri}")
    
    note_id = int(uri.replace("note:///", ""))
    note = note_store.get_note(note_id)
    
    if not note:
        raise ValueError(f"Note {note_id} not found")
    
    return json.dumps(note.to_dict(), indent=2)

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="add_note",
            description="Add a new note",
            inputSchema={
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Note title"
                    },
                    "content": {
                        "type": "string",
                        "description": "Note content"
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional tags",
                        "default": []
                    }
                },
                "required": ["title", "content"]
            }
        ),
        Tool(
            name="search_notes",
            description="Search notes by keyword",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="list_all_notes",
            description="List all notes with brief info",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "add_note":
        note = note_store.add_note(
            title=arguments["title"],
            content=arguments["content"],
            tags=arguments.get("tags", [])
        )
        return [
            TextContent(
                type="text",
                text=f"Created note {note.id}: {note.title}"
            )
        ]
    
    elif name == "search_notes":
        query = arguments["query"]
        results = note_store.search_notes(query)
        
        if not results:
            return [TextContent(type="text", text=f"No notes found for '{query}'")]
        
        result_text = f"Found {len(results)} notes:\n\n"
        for note in results:
            result_text += f"[{note.id}] {note.title}\n{note.content[:100]}...\n\n"
        
        return [TextContent(type="text", text=result_text)]
    
    elif name == "list_all_notes":
        notes = note_store.get_all_notes()
        
        if not notes:
            return [TextContent(type="text", text="No notes yet")]
        
        result_text = f"All notes ({len(notes)}):\n\n"
        for note in notes:
            result_text += f"[{note.id}] {note.title} - {note.created_at.strftime('%Y-%m-%d')}\n"
            result_text += f"Tags: {', '.join(note.tags) if note.tags else 'none'}\n\n"
        
        return [TextContent(type="text", text=result_text)]
    
    else:
        raise ValueError(f"Unknown tool: {name}")

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

## Step 3: Understanding the Code

### Server Creation

```python
server = Server("notes-server")
```

Creates an MCP server with the name "notes-server".

### Resource Listing

```python
@server.list_resources()
async def list_resources() -> list[Resource]:
    ...
```

Returns all available resources (notes). Each note becomes a resource with a URI like `note:///1`.

### Resource Reading

```python
@server.read_resource()
async def read_resource(uri: str) -> str:
    ...
```

Returns the content of a specific resource when requested by URI.

### Tool Listing

```python
@server.list_tools()
async def list_tools() -> list[Tool]:
    ...
```

Returns available tools with their schemas. The AI uses this to know what tools exist and how to call them.

### Tool Execution

```python
@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    ...
```

Executes a tool when called by the client. Returns results as TextContent.

### Server Run

```python
async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
    await server.run(read_stream, write_stream, ...)
```

Runs the server using stdio transport (communication via stdin/stdout).

## Step 4: Testing the Server

### Manual Testing with MCP Inspector

Install MCP Inspector:

```bash
npm install -g @modelcontextprotocol/inspector
```

Run your server with inspector:

```bash
mcp-inspector python server.py
```

This opens a web interface where you can:
- See available resources
- Call tools manually
- Inspect responses

### Automated Testing

Create `test_client.py`:

```python
import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    server_params = StdioServerParameters(
        command="python",
        args=["server.py"],
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            print("=== Available Tools ===")
            tools = await session.list_tools()
            for tool in tools.tools:
                print(f"- {tool.name}: {tool.description}")
            
            print("\n=== Adding Notes ===")
            result = await session.call_tool(
                "add_note",
                {
                    "title": "Learn MCP",
                    "content": "Study Model Context Protocol basics",
                    "tags": ["learning", "mcp"]
                }
            )
            print(result.content[0].text)
            
            result = await session.call_tool(
                "add_note",
                {
                    "title": "Build Project",
                    "content": "Create a research assistant with Pydantic AI",
                    "tags": ["project", "pydantic-ai"]
                }
            )
            print(result.content[0].text)
            
            print("\n=== Listing All Notes ===")
            result = await session.call_tool("list_all_notes", {})
            print(result.content[0].text)
            
            print("\n=== Searching Notes ===")
            result = await session.call_tool("search_notes", {"query": "MCP"})
            print(result.content[0].text)
            
            print("\n=== Available Resources ===")
            resources = await session.list_resources()
            for resource in resources.resources:
                print(f"- {resource.uri}: {resource.name}")
            
            print("\n=== Reading Resource ===")
            if resources.resources:
                uri = resources.resources[0].uri
                content = await session.read_resource(uri)
                print(f"Resource {uri}:")
                print(content.contents[0].text)

if __name__ == "__main__":
    asyncio.run(main())
```

Run the test:

```bash
python test_client.py
```

Expected output:
```
=== Available Tools ===
- add_note: Add a new note
- search_notes: Search notes by keyword
- list_all_notes: List all notes with brief info

=== Adding Notes ===
Created note 1: Learn MCP
Created note 2: Build Project

=== Listing All Notes ===
All notes (2):

[1] Learn MCP - 2024-12-02
Tags: learning, mcp

[2] Build Project - 2024-12-02
Tags: project, pydantic-ai

...
```

## Step 5: Adding Error Handling

Improve `server.py` with better error handling:

```python
@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    try:
        if name == "add_note":
            title = arguments.get("title", "").strip()
            content = arguments.get("content", "").strip()
            
            if not title:
                return [TextContent(type="text", text="Error: Title cannot be empty")]
            if not content:
                return [TextContent(type="text", text="Error: Content cannot be empty")]
            if len(title) > 200:
                return [TextContent(type="text", text="Error: Title too long (max 200 chars)")]
            
            note = note_store.add_note(
                title=title,
                content=content,
                tags=arguments.get("tags", [])
            )
            return [TextContent(
                type="text",
                text=f"✓ Created note {note.id}: {note.title}"
            )]
        
        elif name == "search_notes":
            query = arguments.get("query", "").strip()
            
            if not query:
                return [TextContent(type="text", text="Error: Search query cannot be empty")]
            
            results = note_store.search_notes(query)
            
            if not results:
                return [TextContent(type="text", text=f"No notes found for '{query}'")]
            
            result_text = f"Found {len(results)} note(s) matching '{query}':\n\n"
            for note in results:
                result_text += f"[{note.id}] {note.title}\n"
                result_text += f"{note.content[:150]}{'...' if len(note.content) > 150 else ''}\n"
                result_text += f"Tags: {', '.join(note.tags) if note.tags else 'none'}\n\n"
            
            return [TextContent(type="text", text=result_text)]
        
        elif name == "list_all_notes":
            notes = note_store.get_all_notes()
            
            if not notes:
                return [TextContent(type="text", text="No notes yet. Create your first note with add_note!")]
            
            result_text = f"📝 All notes ({len(notes)}):\n\n"
            for note in notes:
                result_text += f"[{note.id}] {note.title}\n"
                result_text += f"Created: {note.created_at.strftime('%Y-%m-%d %H:%M')}\n"
                result_text += f"Tags: {', '.join(note.tags) if note.tags else 'none'}\n\n"
            
            return [TextContent(type="text", text=result_text)]
        
        else:
            return [TextContent(type="text", text=f"Error: Unknown tool '{name}'")]
    
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]
```

## Step 6: Adding Logging

Add logging for debugging:

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("notes-server")

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    logger.info(f"Tool called: {name} with arguments: {arguments}")
    try:
        ...
        logger.info(f"Tool {name} executed successfully")
        return result
    except Exception as e:
        logger.error(f"Tool {name} failed: {e}")
        return [TextContent(type="text", text=f"Error: {str(e)}")]
```

## Common Issues and Solutions

### Issue 1: Server doesn't start

**Error**: `ModuleNotFoundError: No module named 'mcp'`

**Solution**: Install MCP SDK:
```bash
pip install mcp
```

### Issue 2: Client can't connect

**Error**: `Connection refused` or timeout

**Solution**: Ensure server is running and using stdio transport correctly.

### Issue 3: Tool not found

**Error**: `Unknown tool: xyz`

**Solution**: Check tool name matches exactly in list_tools() and call_tool().

### Issue 4: Invalid JSON response

**Error**: `JSON decode error`

**Solution**: Ensure all tool responses return valid TextContent objects.

## Practice Exercise

Enhance the notes server with:

1. **Delete Tool**: Add ability to delete notes by ID
   - Input: `note_id` (integer)
   - Validate note exists
   - Return success/error message

2. **Update Tool**: Add ability to edit existing notes
   - Input: `note_id`, `title` (optional), `content` (optional), `tags` (optional)
   - Update only provided fields
   - Return updated note info

3. **Filter by Tag**: Add tool to filter notes by tag
   - Input: `tag` (string)
   - Return all notes with that tag

4. **Export Tool**: Add tool to export all notes as JSON
   - Return formatted JSON of all notes

5. **Statistics Resource**: Add a resource that shows stats
   - URI: `stats:///overview`
   - Content: Total notes, average content length, most common tags

Test each feature thoroughly!

## Summary

- MCP servers are built with the `mcp` Python SDK
- Use decorators to register handlers (@server.list_tools, etc.)
- Resources expose read-only data with URIs
- Tools enable actions with JSON schemas
- Stdio transport enables local client-server communication
- Error handling and validation are critical
- Testing can be done with MCP Inspector or custom clients

---

**Next**: [Lesson 6: Tools and Resources](lesson-06-tools-and-resources.md)
