# Lesson 1: Introduction to MCP

## What is MCP?

**Model Context Protocol (MCP)** is an open standard created by Anthropic that defines how AI applications should connect to external data sources and tools. Think of it as a universal adapter that lets AI models interact with databases, APIs, file systems, and other services in a standardized way.

### The Problem MCP Solves

Before MCP, every AI application needed custom code for each integration:

```
AI App 1 → Custom connector → Database
AI App 1 → Custom connector → API
AI App 1 → Custom connector → File System

AI App 2 → Different custom connector → Database
AI App 2 → Different custom connector → API
...and so on
```

This meant:
- Duplicate code across projects
- Inconsistent implementations
- Hard to maintain
- Difficult to share integrations

### The MCP Solution

With MCP, you build **one server** per data source, and **any MCP client** can use it:

```
AI App 1 (MCP Client) ┐
AI App 2 (MCP Client) ├→ MCP Server → Database
AI App 3 (MCP Client) ┘

AI App 1 (MCP Client) ┐
AI App 2 (MCP Client) ├→ MCP Server → API
AI App 3 (MCP Client) ┘
```

Benefits:
- ✅ Write once, use everywhere
- ✅ Consistent interface
- ✅ Easy to share and reuse
- ✅ Better security and control

## MCP Architecture

MCP uses a **client-server** architecture:

### MCP Client
- The AI application (your Pydantic AI agent)
- Requests resources and invokes tools
- Receives structured responses

### MCP Server
- Exposes data sources and capabilities
- Implements tools and resources
- Handles requests from clients

### Communication Flow

```
┌─────────────────┐          ┌─────────────────┐
│   MCP Client    │          │   MCP Server    │
│                 │          │                 │
│ (Pydantic AI    │          │  (Your custom   │
│  with Gemini)   │          │   server)       │
└────────┬────────┘          └────────┬────────┘
         │                            │
         │  1. Request resource       │
         ├───────────────────────────→│
         │                            │
         │  2. Return data            │
         │←───────────────────────────┤
         │                            │
         │  3. Invoke tool            │
         ├───────────────────────────→│
         │                            │
         │  4. Return result          │
         │←───────────────────────────┤
         │                            │
```

## Key Concepts

### 1. Resources
**What they are**: Data sources that the server makes available (like documents, database records, API responses)

**Example**: A weather MCP server might expose:
- Current weather data for a city
- Historical weather records
- Weather forecasts

### 2. Tools
**What they are**: Actions the AI can perform through the server (like search, update, calculate)

**Example**: A weather MCP server might provide tools to:
- Search weather by location
- Get forecast for N days
- Convert temperature units

### 3. Prompts
**What they are**: Pre-defined prompt templates the server can provide

**Example**: A weather server might offer:
- "Weather report prompt" template
- "Travel advisory prompt" template

## Real-World Use Cases

### 1. Enterprise Data Access
**Scenario**: AI assistant needs to access company databases, CRM, and internal APIs

**MCP Solution**:
```
AI Assistant (Client)
  ├→ MCP Server (Database)
  ├→ MCP Server (Salesforce)
  └→ MCP Server (Internal API)
```

### 2. Research Assistant
**Scenario**: AI needs to search papers, fetch citations, and analyze documents

**MCP Solution**:
```
Research AI (Client)
  ├→ MCP Server (ArXiv)
  ├→ MCP Server (Google Scholar)
  └→ MCP Server (PDF Parser)
```

### 3. Development Tools
**Scenario**: AI coding assistant needs file access, git operations, and testing

**MCP Solution**:
```
Coding AI (Client)
  ├→ MCP Server (Filesystem)
  ├→ MCP Server (Git)
  └→ MCP Server (Test Runner)
```

## Why Pydantic AI + Gemini?

### Pydantic AI
- **Type-safe**: Built on Pydantic for data validation
- **Clean API**: Simple, Pythonic interface
- **Flexible**: Works with multiple AI models
- **Structured outputs**: Get validated responses

### Google Gemini
- **Powerful**: State-of-the-art multimodal AI
- **Fast**: Low latency responses
- **Affordable**: Competitive pricing
- **Accessible**: Easy to get started with API

### Together
Pydantic AI provides the framework for building agents, Gemini provides the intelligence, and MCP provides the connections to data.

## What You'll Build

Throughout this guide, you'll build a **Smart Research Assistant** that:

1. **Fetches data** from multiple sources via MCP servers
2. **Processes queries** using Google Gemini
3. **Returns structured responses** validated by Pydantic
4. **Performs actions** through MCP tools

Example interaction:
```
User: "What's the latest research on Python async programming?"

Assistant (via Gemini):
→ Calls MCP Server (ArXiv) tool: search_papers("Python async")
→ Gets structured results
→ Analyzes with Gemini
→ Returns: "I found 15 recent papers. The most relevant is..."
```

## Prerequisites

Before starting this guide, you should have:

- Basic Python knowledge (functions, classes, async/await)
- Python 3.10+ installed
- Pip package manager
- A text editor or IDE
- Internet connection

Don't worry if you're not an expert! Each lesson builds gradually.

## Next Steps

In the next lesson, you'll dive deeper into:
- How MCP protocol works under the hood
- Message types and communication patterns
- Building a mental model for MCP development

## Practice Exercise

Before moving on, think about a use case where MCP would be helpful:

1. What data sources would your AI need to access?
2. What actions should it be able to perform?
3. How would MCP make this easier than custom code?

Write down your answers and refer back to them as you progress through the course.

---

**Next**: [Lesson 2: MCP Basics](lesson-02-mcp-basics.md)
