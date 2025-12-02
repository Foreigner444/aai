"""
Tests for MCP servers
"""

import pytest
import asyncio
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

@pytest.mark.asyncio
async def test_papers_server():
    """Test papers server functionality"""
    server_params = StdioServerParameters(
        command="python",
        args=["servers/papers_server.py"],
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            tools = await session.list_tools()
            assert len(tools.tools) > 0
            assert any(t.name == "search_papers" for t in tools.tools)
            
            result = await session.call_tool(
                "search_papers",
                {"query": "transformer", "max_results": 3}
            )
            
            assert result.content
            papers = json.loads(result.content[0].text)
            assert isinstance(papers, list)
            assert len(papers) > 0
            assert "title" in papers[0]
            assert "authors" in papers[0]

@pytest.mark.asyncio
async def test_web_server():
    """Test web server functionality"""
    server_params = StdioServerParameters(
        command="python",
        args=["servers/web_server.py"],
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            tools = await session.list_tools()
            assert any(t.name == "search" for t in tools.tools)
            
            result = await session.call_tool(
                "search",
                {"query": "machine learning", "max_results": 2}
            )
            
            assert result.content
            results = json.loads(result.content[0].text)
            assert isinstance(results, list)
            assert len(results) > 0

@pytest.mark.asyncio
async def test_notes_server():
    """Test notes server functionality"""
    server_params = StdioServerParameters(
        command="python",
        args=["servers/notes_server.py"],
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            tools = await session.list_tools()
            assert any(t.name == "add_note" for t in tools.tools)
            assert any(t.name == "list_all_notes" for t in tools.tools)
            
            result = await session.call_tool(
                "add_note",
                {
                    "title": "Test Note",
                    "content": "This is a test note",
                    "tags": ["test"]
                }
            )
            
            assert result.content
            assert "Created note" in result.content[0].text
            
            result = await session.call_tool("list_all_notes", {})
            assert result.content
            assert "Test Note" in result.content[0].text

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
