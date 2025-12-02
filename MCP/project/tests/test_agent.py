"""
Tests for research agent
"""

import pytest
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import MCPConnectionManager
from agent.research_agent import run_research
from agent.models import ResearchResult

@pytest.mark.asyncio
async def test_research_agent_basic():
    """Test basic research agent functionality"""
    manager = MCPConnectionManager()
    
    try:
        sessions = await manager.connect_to_servers()
        
        if not sessions:
            pytest.skip("No MCP servers available")
        
        result = await run_research(
            query="test query",
            mcp_sessions=sessions,
            max_papers=2
        )
        
        assert isinstance(result, ResearchResult)
        assert result.query
        assert isinstance(result.papers, list)
        assert isinstance(result.summary, str)
        assert isinstance(result.key_findings, list)
        assert isinstance(result.recommended_reading_order, list)
    
    finally:
        await manager.disconnect_all()

@pytest.mark.asyncio
async def test_research_agent_papers():
    """Test that research agent finds papers"""
    manager = MCPConnectionManager()
    
    try:
        sessions = await manager.connect_to_servers()
        
        if not sessions:
            pytest.skip("No MCP servers available")
        
        result = await run_research(
            query="transformer architecture",
            mcp_sessions=sessions,
            max_papers=3
        )
        
        assert len(result.papers) > 0
        
        for paper in result.papers:
            assert paper.title
            assert paper.authors
            assert paper.url
    
    finally:
        await manager.disconnect_all()

@pytest.mark.asyncio
async def test_research_agent_recommendations():
    """Test that research agent provides reading recommendations"""
    manager = MCPConnectionManager()
    
    try:
        sessions = await manager.connect_to_servers()
        
        if not sessions:
            pytest.skip("No MCP servers available")
        
        result = await run_research(
            query="nlp models",
            mcp_sessions=sessions,
            max_papers=3
        )
        
        assert len(result.recommended_reading_order) > 0
        
        for idx in result.recommended_reading_order:
            assert idx < len(result.papers)
    
    finally:
        await manager.disconnect_all()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
