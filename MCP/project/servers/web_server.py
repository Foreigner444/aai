import asyncio
import json
from mcp.server import Server
from mcp.types import Tool, TextContent
import mcp.server.stdio

server = Server("web-server")

mock_web_results = {
    "transformer": [
        {
            "title": "Transformer Architecture Explained",
            "url": "https://example.com/transformer-guide",
            "snippet": "A comprehensive guide to understanding transformer architecture in deep learning..."
        },
        {
            "title": "Attention Mechanism in Transformers",
            "url": "https://example.com/attention-guide",
            "snippet": "Deep dive into self-attention and multi-head attention mechanisms..."
        }
    ],
    "machine learning": [
        {
            "title": "Introduction to Machine Learning",
            "url": "https://example.com/ml-intro",
            "snippet": "Learn the fundamentals of machine learning, from basic concepts to advanced techniques..."
        },
        {
            "title": "Machine Learning Best Practices",
            "url": "https://example.com/ml-practices",
            "snippet": "Best practices for building production ML systems..."
        }
    ]
}

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="search",
            description="Search the web for information. Returns titles, URLs, and snippets of relevant web pages.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "max_results": {
                        "type": "integer",
                        "default": 3,
                        "minimum": 1,
                        "maximum": 10,
                        "description": "Maximum number of results"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get_page_content",
            description="Fetch and extract text content from a web page URL",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL of the page to fetch"
                    }
                },
                "required": ["url"]
            }
        )
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "search":
        query = arguments["query"].lower()
        max_results = arguments.get("max_results", 3)
        
        results = []
        for keyword, pages in mock_web_results.items():
            if keyword in query:
                results.extend(pages)
        
        if not results:
            results = [
                {
                    "title": f"Result for: {query}",
                    "url": f"https://example.com/search?q={query.replace(' ', '+')}",
                    "snippet": f"Information about {query} from various sources..."
                }
            ]
        
        results = results[:max_results]
        
        return [TextContent(type="text", text=json.dumps(results, indent=2))]
    
    elif name == "get_page_content":
        url = arguments["url"]
        
        mock_content = f"""
        # Article Content
        
        This is mock content from {url}.
        
        ## Introduction
        
        This article discusses various topics related to the URL you requested.
        In a real implementation, this would fetch and parse the actual page content.
        
        ## Main Content
        
        Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod
        tempor incididunt ut labore et dolore magna aliqua.
        
        ## Conclusion
        
        This demonstrates how web content would be fetched and returned.
        """
        
        return [TextContent(type="text", text=mock_content)]
    
    return [TextContent(type="text", text=json.dumps({"error": "Unknown tool"}))]

async def main():
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())
