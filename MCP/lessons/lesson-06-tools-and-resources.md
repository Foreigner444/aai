# Lesson 6: Tools and Resources

## Deep Dive: Resources vs Tools

Understanding when to use resources vs tools is crucial for good MCP server design.

### Resources

**When to use**:
- Data that exists independently
- Read-only information
- Content that can be browsed
- Static or slowly-changing data

**Examples**:
- Documents in a database
- API responses
- File contents
- Database records

**Think of resources as**: Nouns (things that exist)

### Tools

**When to use**:
- Actions that do something
- Operations that modify state
- Computations
- Queries with parameters

**Examples**:
- Search functions
- CRUD operations
- Calculations
- API calls

**Think of tools as**: Verbs (actions to perform)

### Example: Library System

**Resources**:
- `book://id/123` - A specific book
- `book://id/124` - Another book
- `catalog://recent` - Recent additions list

**Tools**:
- `search_books(query, filters)` - Search the catalog
- `checkout_book(book_id, user_id)` - Check out a book
- `reserve_book(book_id)` - Reserve a book

## Advanced Resource Patterns

### 1. Hierarchical Resources

Organize resources in a hierarchy:

```python
from mcp.server import Server
from mcp.types import Resource

server = Server("library-server")

@server.list_resources()
async def list_resources() -> list[Resource]:
    return [
        Resource(
            uri="library://books",
            name="All Books",
            description="Root collection of all books"
        ),
        Resource(
            uri="library://books/fiction",
            name="Fiction Books",
            description="Fiction category"
        ),
        Resource(
            uri="library://books/fiction/123",
            name="The Great Gatsby",
            description="A specific fiction book"
        ),
        Resource(
            uri="library://books/non-fiction",
            name="Non-Fiction Books",
            description="Non-fiction category"
        ),
    ]
```

### 2. Dynamic Resources

Generate resources based on data:

```python
class BookLibrary:
    def __init__(self):
        self.books = [
            {"id": 1, "title": "1984", "author": "Orwell", "category": "fiction"},
            {"id": 2, "title": "Sapiens", "author": "Harari", "category": "non-fiction"},
        ]
    
    def get_all_books(self):
        return self.books

library = BookLibrary()

@server.list_resources()
async def list_resources() -> list[Resource]:
    resources = []
    
    for book in library.get_all_books():
        resource = Resource(
            uri=f"book:///{book['id']}",
            name=book['title'],
            description=f"By {book['author']} - {book['category']}",
            mimeType="application/json"
        )
        resources.append(resource)
    
    return resources

@server.read_resource()
async def read_resource(uri: str) -> str:
    import json
    
    if not uri.startswith("book:///"):
        raise ValueError(f"Invalid URI: {uri}")
    
    book_id = int(uri.replace("book:///", ""))
    
    for book in library.get_all_books():
        if book['id'] == book_id:
            return json.dumps(book, indent=2)
    
    raise ValueError(f"Book {book_id} not found")
```

### 3. Resource Templates

Create resources from templates:

```python
from mcp.types import ResourceTemplate

@server.list_resource_templates()
async def list_templates() -> list[ResourceTemplate]:
    return [
        ResourceTemplate(
            uriTemplate="book:///{id}",
            name="Book by ID",
            description="Access any book by its ID",
            mimeType="application/json"
        ),
        ResourceTemplate(
            uriTemplate="author:///{name}/books",
            name="Books by Author",
            description="All books by a specific author"
        )
    ]
```

### 4. Resource Subscriptions

Allow clients to subscribe to resource changes:

```python
from mcp.types import ResourceUpdated
import asyncio

@server.list_resources()
async def list_resources() -> list[Resource]:
    return [
        Resource(
            uri="data:///live-feed",
            name="Live Data Feed",
            description="Real-time data updates"
        )
    ]

async def monitor_changes():
    while True:
        await asyncio.sleep(5)
        
        await server.request_context.session.send_resource_updated(
            ResourceUpdated(uri="data:///live-feed")
        )
```

## Advanced Tool Patterns

### 1. Complex Input Schemas

Use JSON Schema features for detailed validation:

```python
Tool(
    name="create_user",
    description="Create a new user account",
    inputSchema={
        "type": "object",
        "properties": {
            "username": {
                "type": "string",
                "minLength": 3,
                "maxLength": 20,
                "pattern": "^[a-zA-Z0-9_]+$",
                "description": "Username (alphanumeric and underscore only)"
            },
            "email": {
                "type": "string",
                "format": "email",
                "description": "Valid email address"
            },
            "age": {
                "type": "integer",
                "minimum": 13,
                "maximum": 120,
                "description": "User age"
            },
            "role": {
                "type": "string",
                "enum": ["user", "admin", "moderator"],
                "default": "user",
                "description": "User role"
            },
            "preferences": {
                "type": "object",
                "properties": {
                    "theme": {
                        "type": "string",
                        "enum": ["light", "dark", "auto"]
                    },
                    "notifications": {
                        "type": "boolean"
                    }
                },
                "additionalProperties": False
            }
        },
        "required": ["username", "email"],
        "additionalProperties": False
    }
)
```

### 2. Multi-Step Tools

Tools that coordinate multiple operations:

```python
from mcp.types import TextContent
import asyncio

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "deploy_website":
        site_name = arguments["site_name"]
        results = []
        
        results.append(f"Starting deployment of {site_name}...")
        
        await asyncio.sleep(1)
        results.append("✓ Building assets...")
        
        await asyncio.sleep(1)
        results.append("✓ Uploading files...")
        
        await asyncio.sleep(1)
        results.append("✓ Configuring server...")
        
        await asyncio.sleep(0.5)
        results.append(f"✓ Deployment complete! Site live at https://{site_name}.example.com")
        
        return [TextContent(type="text", text="\n".join(results))]
```

### 3. Tools with Progress Updates

For long-running operations:

```python
from mcp.types import LoggingLevel

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "process_large_file":
        file_path = arguments["file_path"]
        
        await server.request_context.session.send_log_message(
            level=LoggingLevel.INFO,
            data=f"Starting to process {file_path}"
        )
        
        for i in range(1, 11):
            await asyncio.sleep(0.5)
            await server.request_context.session.send_log_message(
                level=LoggingLevel.INFO,
                data=f"Progress: {i*10}%"
            )
        
        return [TextContent(
            type="text",
            text=f"Processed {file_path} successfully"
        )]
```

### 4. Tools with Rich Output

Return multiple content types:

```python
from mcp.types import TextContent, ImageContent, EmbeddedResource

@server.call_tool()
async def call_tool(name: str, arguments: dict):
    if name == "generate_report":
        report_data = {"sales": 1000, "growth": 15}
        
        return [
            TextContent(
                type="text",
                text=f"Sales Report:\nTotal: ${report_data['sales']}\nGrowth: {report_data['growth']}%"
            ),
            EmbeddedResource(
                type="resource",
                resource=Resource(
                    uri="report:///2024-q4",
                    name="Q4 2024 Report",
                    mimeType="application/json"
                )
            )
        ]
```

## Tool Design Best Practices

### 1. Clear Descriptions

The AI uses descriptions to decide when to call your tool. Be specific:

❌ **Bad**:
```python
"description": "Get data"
```

✅ **Good**:
```python
"description": "Retrieve customer order history for a given customer ID. Returns list of orders with dates, amounts, and status."
```

### 2. Sensible Defaults

Provide defaults for optional parameters:

```python
{
    "type": "object",
    "properties": {
        "limit": {
            "type": "integer",
            "default": 10,
            "description": "Maximum results to return"
        },
        "sort_by": {
            "type": "string",
            "default": "date",
            "enum": ["date", "amount", "status"]
        }
    }
}
```

### 3. Input Validation

Always validate inputs:

```python
@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "transfer_money":
        amount = arguments.get("amount")
        recipient = arguments.get("recipient")
        
        if not isinstance(amount, (int, float)) or amount <= 0:
            return [TextContent(type="text", text="Error: Amount must be positive")]
        
        if amount > 10000:
            return [TextContent(type="text", text="Error: Amount exceeds limit")]
        
        if not isinstance(recipient, str) or not recipient.strip():
            return [TextContent(type="text", text="Error: Invalid recipient")]
        
        result = f"Transferred ${amount:.2f} to {recipient}"
        return [TextContent(type="text", text=result)]
```

### 4. Meaningful Error Messages

Help users understand what went wrong:

```python
if len(username) < 3:
    return [TextContent(
        type="text",
        text="Error: Username must be at least 3 characters long. You provided: '{username}' ({len(username)} characters)"
    )]
```

### 5. Idempotency

Make tools safe to call multiple times:

```python
@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "create_user":
        username = arguments["username"]
        
        existing_user = database.get_user(username)
        if existing_user:
            return [TextContent(
                type="text",
                text=f"User '{username}' already exists (created {existing_user.created_at})"
            )]
        
        user = database.create_user(username)
        return [TextContent(type="text", text=f"Created user '{username}'")]
```

## Resource Design Best Practices

### 1. Meaningful URIs

Use descriptive, hierarchical URIs:

❌ **Bad**:
```python
"uri": "res1"
```

✅ **Good**:
```python
"uri": "library://books/fiction/classic/1984"
```

### 2. Appropriate MIME Types

Set correct MIME types:

```python
Resource(
    uri="doc:///report.pdf",
    mimeType="application/pdf"
)

Resource(
    uri="doc:///data.json",
    mimeType="application/json"
)

Resource(
    uri="doc:///page.html",
    mimeType="text/html"
)
```

### 3. Helpful Descriptions

Add context to resources:

```python
Resource(
    uri="sales:///2024/q4",
    name="Q4 2024 Sales",
    description="Sales data for October-December 2024. Includes revenue, units sold, and regional breakdown.",
    mimeType="application/json"
)
```

### 4. Efficient Listing

Don't return too many resources at once:

```python
@server.list_resources()
async def list_resources() -> list[Resource]:
    all_resources = database.get_all_resources()
    
    recent_resources = all_resources[:100]
    
    return [
        Resource(
            uri=f"doc:///{r.id}",
            name=r.name,
            description=r.description
        )
        for r in recent_resources
    ]
```

## Combining Resources and Tools

Create powerful combinations:

```python
from dataclasses import dataclass, field
from typing import List
import json

@dataclass
class DocumentStore:
    documents: List[dict] = field(default_factory=list)
    
    def add_document(self, title: str, content: str):
        doc = {"id": len(self.documents) + 1, "title": title, "content": content}
        self.documents.append(doc)
        return doc
    
    def search(self, query: str):
        return [d for d in self.documents if query.lower() in d["title"].lower()]

store = DocumentStore()
server = Server("doc-server")

@server.list_resources()
async def list_resources() -> list[Resource]:
    return [
        Resource(
            uri=f"doc:///{doc['id']}",
            name=doc['title'],
            mimeType="text/plain"
        )
        for doc in store.documents
    ]

@server.read_resource()
async def read_resource(uri: str) -> str:
    doc_id = int(uri.replace("doc:///", ""))
    doc = next((d for d in store.documents if d["id"] == doc_id), None)
    if not doc:
        raise ValueError("Document not found")
    return doc["content"]

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="add_document",
            description="Add a new document to the store",
            inputSchema={
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["title", "content"]
            }
        ),
        Tool(
            name="search_documents",
            description="Search documents by title",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "add_document":
        doc = store.add_document(arguments["title"], arguments["content"])
        return [TextContent(type="text", text=f"Created document {doc['id']}: {doc['title']}")]
    
    elif name == "search_documents":
        results = store.search(arguments["query"])
        if not results:
            return [TextContent(type="text", text="No documents found")]
        
        result_text = f"Found {len(results)} documents:\n"
        for doc in results:
            result_text += f"- doc:///{doc['id']} - {doc['title']}\n"
        
        return [TextContent(type="text", text=result_text)]
```

**Flow**:
1. Use `add_document` tool to create documents
2. Documents appear as resources automatically
3. Use `search_documents` tool to find documents
4. Read specific documents as resources

## Practice Exercise

Build a **Task Management MCP Server** with:

### Resources
1. Individual tasks: `task:///1`, `task:///2`, etc.
2. Task lists by status: `task:///status/todo`, `task:///status/done`
3. Task statistics: `task:///stats`

### Tools
1. `create_task(title, description, priority)` - Create new task
2. `update_task(id, status, priority)` - Update task
3. `assign_task(id, assignee)` - Assign to someone
4. `search_tasks(query, status, priority)` - Search with filters
5. `get_stats()` - Get task statistics

### Requirements
- Validate all inputs
- Return helpful error messages
- Use proper JSON schemas
- Implement all CRUD operations
- Support filtering and search

Save as `task_manager_server.py` and test thoroughly!

## Summary

- Resources are for data (nouns), tools are for actions (verbs)
- Use hierarchical URIs for organization
- JSON Schema provides powerful validation
- Clear descriptions help AI make good decisions
- Combine resources and tools for powerful servers
- Always validate inputs and handle errors
- Follow best practices for maintainable code

---

**Next**: [Lesson 7: Pydantic AI Agents](lesson-07-pydantic-ai-agents.md)
