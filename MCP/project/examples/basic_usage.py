"""
Basic usage example for the Smart Research Assistant.

This example shows how to:
1. Connect to MCP servers
2. Run a simple research query
3. Display results
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import MCPConnectionManager
from agent.research_agent import run_research

async def main():
    print("Smart Research Assistant - Basic Usage Example")
    print("=" * 60)
    print()
    
    manager = MCPConnectionManager()
    
    sessions = await manager.connect_to_servers()
    
    if not sessions:
        print("Failed to connect to servers")
        return
    
    print(f"Connected to {len(sessions)} servers\n")
    
    query = "attention mechanisms in neural networks"
    
    print(f"Query: {query}\n")
    
    result = await run_research(
        query=query,
        mcp_sessions=sessions,
        max_papers=3,
        include_web_search=False
    )
    
    print("Results:")
    print("-" * 60)
    print(f"Summary: {result.summary}\n")
    
    print(f"Papers ({len(result.papers)}):")
    for i, paper in enumerate(result.papers, 1):
        print(f"  {i}. {paper.title}")
        print(f"     {', '.join(paper.authors)} ({paper.year})")
    
    print(f"\nKey Findings:")
    for finding in result.key_findings:
        print(f"  - {finding}")
    
    await manager.disconnect_all()
    print("\nDone!")

if __name__ == "__main__":
    asyncio.run(main())
