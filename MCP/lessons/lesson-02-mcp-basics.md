# Lesson 2: MCP Basics

## Understanding the Protocol

MCP is built on **JSON-RPC 2.0**, a simple protocol for remote procedure calls. Don't worry if you haven't used JSON-RPC before - we'll explain everything you need to know.

### JSON-RPC Primer

JSON-RPC messages are just JSON objects with specific fields:

```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "search",
    "arguments": {"query": "python"}
  },
  "id": 1
}
```

**Key fields**:
- `jsonrpc`: Always "2.0"
- `method`: What you want to do
- `params`: Arguments for the method
- `id`: Request identifier (for matching responses)

## MCP Core Concepts Deep Dive

### 1. Resources

Resources are **read-only data** that the server exposes. Think of them as endpoints that return information.

#### Resource Structure

```python
{
  "uri": "file:///documents/report.pdf",
  "name": "Q4 Report",
  "mimeType": "application/pdf",
  "description": "Quarterly financial report"
}
```

#### Resource URIs

URIs uniquely identify resources and follow standard URI syntax:

```
file:///path/to/file.txt
http://api.example.com/data
database://localhost/users/123
custom://my-resource/item
```

#### Example: Listing Resources

**Client Request**:
```json
{
  "jsonrpc": "2.0",
  "method": "resources/list",
  "id": 1
}
```

**Server Response**:
```json
{
  "jsonrpc": "2.0",
  "result": {
    "resources": [
      {
        "uri": "weather://current/london",
        "name": "London Weather",
        "mimeType": "application/json",
        "description": "Current weather in London"
      }
    ]
  },
  "id": 1
}
```

#### Example: Reading a Resource

**Client Request**:
```json
{
  "jsonrpc": "2.0",
  "method": "resources/read",
  "params": {
    "uri": "weather://current/london"
  },
  "id": 2
}
```

**Server Response**:
```json
{
  "jsonrpc": "2.0",
  "result": {
    "contents": [{
      "uri": "weather://current/london",
      "mimeType": "application/json",
      "text": "{\"temp\": 15, \"condition\": \"cloudy\"}"
    }]
  },
  "id": 2
}
```

### 2. Tools

Tools are **actions** the AI can perform. Unlike resources (which are read-only), tools can modify data, trigger operations, or perform computations.

#### Tool Structure

```python
{
  "name": "search_weather",
  "description": "Search weather by location",
  "inputSchema": {
    "type": "object",
    "properties": {
      "location": {
        "type": "string",
        "description": "City name"
      },
      "units": {
        "type": "string",
        "enum": ["celsius", "fahrenheit"],
        "default": "celsius"
      }
    },
    "required": ["location"]
  }
}
```

**Key fields**:
- `name`: Tool identifier
- `description`: What the tool does (AI uses this to decide when to call it)
- `inputSchema`: JSON Schema defining parameters

#### Example: Listing Tools

**Client Request**:
```json
{
  "jsonrpc": "2.0",
  "method": "tools/list",
  "id": 3
}
```

**Server Response**:
```json
{
  "jsonrpc": "2.0",
  "result": {
    "tools": [
      {
        "name": "get_forecast",
        "description": "Get weather forecast for a city",
        "inputSchema": {
          "type": "object",
          "properties": {
            "city": {"type": "string"},
            "days": {"type": "number", "default": 7}
          },
          "required": ["city"]
        }
      }
    ]
  },
  "id": 3
}
```

#### Example: Calling a Tool

**Client Request**:
```json
{
  "jsonrpc": "2.0",
  "method": "tools/call",
  "params": {
    "name": "get_forecast",
    "arguments": {
      "city": "Tokyo",
      "days": 5
    }
  },
  "id": 4
}
```

**Server Response**:
```json
{
  "jsonrpc": "2.0",
  "result": {
    "content": [
      {
        "type": "text",
        "text": "5-day forecast for Tokyo: Mon 22°C, Tue 24°C, Wed 23°C..."
      }
    ]
  },
  "id": 4
}
```

### 3. Prompts

Prompts are **pre-defined templates** that help structure AI interactions. They're less commonly used than resources and tools, but useful for standardized queries.

#### Prompt Structure

```python
{
  "name": "weather_report",
  "description": "Generate a weather report",
  "arguments": [
    {
      "name": "location",
      "description": "Location for report",
      "required": True
    }
  ]
}
```

## Message Flow

Let's trace a complete interaction:

### Scenario: AI wants to get weather data

```
┌─────────┐                                    ┌─────────┐
│  Client │                                    │  Server │
│ (AI App)│                                    │         │
└────┬────┘                                    └────┬────┘
     │                                              │
     │ 1. List available tools                     │
     │────────────────────────────────────────────→│
     │   {"method": "tools/list"}                  │
     │                                              │
     │ 2. Return tools                             │
     │←────────────────────────────────────────────│
     │   [{"name": "get_weather",...}]             │
     │                                              │
     │ 3. AI decides to call get_weather           │
     │    (based on user request and tool desc)    │
     │                                              │
     │ 4. Call tool with arguments                 │
     │────────────────────────────────────────────→│
     │   {"method": "tools/call",                  │
     │    "params": {"name": "get_weather",        │
     │               "arguments": {...}}}          │
     │                                              │
     │ 5. Server executes tool                     │
     │    (fetches from API, DB, etc.)             │
     │                                              │
     │ 6. Return result                            │
     │←────────────────────────────────────────────│
     │   {"result": {"content": [...]}}            │
     │                                              │
     │ 7. AI processes result and responds to user │
     │                                              │
```

## Transport Mechanisms

MCP supports two main transport types:

### 1. Standard Input/Output (stdio)
Server runs as a subprocess, communicates via stdin/stdout.

**Use case**: Local servers, CLI tools

**Advantages**:
- Simple to implement
- No network configuration
- Built-in process management

**Example** (Python conceptual):
```python
import subprocess
import json

server_process = subprocess.Popen(
    ['python', 'mcp_server.py'],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE
)

request = {"jsonrpc": "2.0", "method": "tools/list", "id": 1}
server_process.stdin.write(json.dumps(request).encode() + b'\n')
server_process.stdin.flush()

response = json.loads(server_process.stdout.readline())
```

### 2. HTTP with Server-Sent Events (SSE)
Server exposes HTTP endpoint, uses SSE for server-to-client messages.

**Use case**: Remote servers, web services

**Advantages**:
- Works over network
- Standard HTTP infrastructure
- Can be deployed anywhere

## Error Handling

MCP uses standard JSON-RPC error responses:

```json
{
  "jsonrpc": "2.0",
  "error": {
    "code": -32601,
    "message": "Method not found",
    "data": {"method": "invalid/method"}
  },
  "id": 5
}
```

**Common error codes**:
- `-32700`: Parse error
- `-32600`: Invalid request
- `-32601`: Method not found
- `-32602`: Invalid params
- `-32603`: Internal error

## Capabilities

Servers declare their capabilities during initialization:

```json
{
  "capabilities": {
    "resources": {
      "subscribe": true,
      "listChanged": true
    },
    "tools": {
      "listChanged": false
    },
    "prompts": {
      "listChanged": false
    }
  }
}
```

**What this means**:
- `subscribe`: Client can subscribe to resource changes
- `listChanged`: Server will notify when list of items changes

## Security Considerations

### 1. Authentication
MCP itself doesn't specify authentication, but you should implement it:

```python
{
  "method": "tools/call",
  "params": {
    "name": "delete_file",
    "arguments": {"path": "/important.txt"}
  },
  "meta": {
    "auth_token": "user-token-here"
  }
}
```

### 2. Input Validation
Always validate tool arguments:

```python
def validate_city_name(city: str) -> bool:
    if len(city) > 100:
        return False
    if not city.replace(' ', '').replace('-', '').isalpha():
        return False
    return True
```

### 3. Rate Limiting
Prevent abuse by limiting requests:

```python
from collections import defaultdict
from time import time

class RateLimiter:
    def __init__(self, max_requests=100, window=60):
        self.requests = defaultdict(list)
        self.max_requests = max_requests
        self.window = window
    
    def allow_request(self, client_id: str) -> bool:
        now = time()
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if now - req_time < self.window
        ]
        
        if len(self.requests[client_id]) < self.max_requests:
            self.requests[client_id].append(now)
            return True
        return False
```

## Best Practices

### 1. Tool Descriptions
Write clear, specific descriptions so the AI knows when to use each tool:

❌ **Bad**:
```python
"description": "Search stuff"
```

✅ **Good**:
```python
"description": "Search for academic papers by keywords. Returns title, authors, abstract, and publication date for up to 10 papers."
```

### 2. Resource URIs
Use meaningful, hierarchical URIs:

❌ **Bad**:
```python
"uri": "resource1"
```

✅ **Good**:
```python
"uri": "papers://arxiv/cs.AI/2024/12345"
```

### 3. Error Messages
Return helpful error messages:

❌ **Bad**:
```python
{"error": {"message": "Error"}}
```

✅ **Good**:
```python
{"error": {
  "message": "City not found",
  "data": {
    "city": "Atlantis",
    "suggestion": "Check spelling or try nearby cities"
  }
}}
```

## Practice Exercise

Design an MCP server for a **todo list application**:

1. **Resources**: What data should be exposed?
   - Hint: Think about different views of todos (all, completed, by category)

2. **Tools**: What actions should be possible?
   - Hint: CRUD operations (Create, Read, Update, Delete)

3. **URIs**: How would you structure resource URIs?
   - Hint: `todo://` prefix, hierarchical structure

4. **Tool Schemas**: Write JSON schemas for at least 2 tools

Write your design in a text file. In upcoming lessons, you'll implement this!

## Summary

- MCP uses JSON-RPC 2.0 for communication
- **Resources** provide read-only data with URIs
- **Tools** enable actions with typed parameters
- **Prompts** offer reusable templates
- Communication happens over stdio or HTTP+SSE
- Security and validation are crucial

---

**Next**: [Lesson 3: Pydantic AI Introduction](lesson-03-pydantic-ai-intro.md)
