"""
Advanced usage example for the Smart Research Assistant.

This example shows how to:
1. Perform multiple research queries
2. Save notes about findings
3. Search through saved notes
4. Get detailed paper information
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import MCPConnectionManager
from agent.research_agent import run_research

async def main():
    print("Smart Research Assistant - Advanced Usage Example")
    print("=" * 60)
    print()
    
    manager = MCPConnectionManager()
    sessions = await manager.connect_to_servers()
    
    if not sessions:
        print("Failed to connect to servers")
        return
    
    queries = [
        "transformer attention mechanisms",
        "BERT pretraining techniques",
        "GPT architecture evolution"
    ]
    
    all_results = []
    
    for i, query in enumerate(queries, 1):
        print(f"\n[Query {i}/{len(queries)}] {query}")
        print("-" * 60)
        
        result = await run_research(
            query=query,
            mcp_sessions=sessions,
            max_papers=3,
            include_web_search=True
        )
        
        all_results.append(result)
        
        print(f"Found {len(result.papers)} papers")
        print(f"Summary: {result.summary[:150]}...")
        
        note_title = f"Research: {query}"
        note_content = f"""
Query: {query}

Summary:
{result.summary}

Key Findings:
{chr(10).join(f'- {f}' for f in result.key_findings)}

Papers:
{chr(10).join(f'{i+1}. {p.title} - {p.url}' for i, p in enumerate(result.papers))}
"""
        
        notes_session = sessions.get('notes')
        if notes_session:
            try:
                await notes_session.call_tool(
                    "add_note",
                    {
                        "title": note_title,
                        "content": note_content,
                        "tags": ["research", "nlp", query.split()[0]]
                    }
                )
                print(f"✓ Saved note: {note_title}")
            except Exception as e:
                print(f"✗ Failed to save note: {e}")
    
    print("\n" + "=" * 60)
    print("SUMMARY OF ALL QUERIES")
    print("=" * 60)
    
    total_papers = sum(len(r.papers) for r in all_results)
    print(f"\nTotal papers found: {total_papers}")
    print(f"Queries performed: {len(queries)}")
    
    if sessions.get('notes'):
        print("\n" + "-" * 60)
        print("Saved Notes:")
        print("-" * 60)
        try:
            result = await sessions['notes'].call_tool("list_all_notes", {})
            print(result.content[0].text)
        except Exception as e:
            print(f"Failed to list notes: {e}")
    
    await manager.disconnect_all()
    print("\nDone!")

if __name__ == "__main__":
    asyncio.run(main())
