import asyncio
import json
from mcp.server import Server
from mcp.types import Resource, Tool, TextContent
import mcp.server.stdio
from datetime import datetime
from dataclasses import dataclass, field
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
    
    def delete_note(self, note_id: int) -> bool:
        for i, note in enumerate(self.notes):
            if note.id == note_id:
                self.notes.pop(i)
                return True
        return False
    
    def search_notes(self, query: str) -> List[Note]:
        return [note for note in self.notes if note.matches_search(query)]
    
    def get_all_notes(self) -> List[Note]:
        return self.notes.copy()

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
            description=f"Created: {note.created_at.strftime('%Y-%m-%d %H:%M')} | Tags: {', '.join(note.tags) if note.tags else 'none'}"
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
            description="Add a new research note with title, content, and optional tags",
            inputSchema={
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Note title"
                    },
                    "content": {
                        "type": "string",
                        "description": "Note content/body"
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional tags for categorization",
                        "default": []
                    }
                },
                "required": ["title", "content"]
            }
        ),
        Tool(
            name="search_notes",
            description="Search notes by keyword in title, content, or tags",
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
            description="List all notes with brief information",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="delete_note",
            description="Delete a note by ID",
            inputSchema={
                "type": "object",
                "properties": {
                    "note_id": {
                        "type": "integer",
                        "description": "ID of note to delete"
                    }
                },
                "required": ["note_id"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "add_note":
        title = arguments.get("title", "").strip()
        content = arguments.get("content", "").strip()
        tags = arguments.get("tags", [])
        
        if not title:
            return [TextContent(type="text", text="Error: Title cannot be empty")]
        if not content:
            return [TextContent(type="text", text="Error: Content cannot be empty")]
        
        note = note_store.add_note(title=title, content=content, tags=tags)
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
    
    elif name == "delete_note":
        note_id = arguments.get("note_id")
        
        if note_store.delete_note(note_id):
            return [TextContent(type="text", text=f"✓ Deleted note {note_id}")]
        else:
            return [TextContent(type="text", text=f"Error: Note {note_id} not found")]
    
    return [TextContent(type="text", text="Error: Unknown tool")]

async def main():
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())
