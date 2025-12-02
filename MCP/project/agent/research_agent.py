import json
import os
from pydantic_ai import Agent, RunContext
from typing import List
from .models import (
    ResearchResult,
    ResearchContext,
    Paper,
    WebResult
)

research_agent = Agent(
    model=os.getenv('GEMINI_MODEL', 'gemini-1.5-flash'),
    result_type=ResearchResult,
    deps_type=ResearchContext,
    system_prompt="""
    You are an expert research assistant specializing in academic paper analysis.
    
    Your workflow:
    1. Search for relevant academic papers using the search_papers tool
    2. Optionally search the web for supplementary information
    3. Analyze paper abstracts for relevance and quality
    4. Identify key findings and insights across papers
    5. Recommend a reading order based on:
       - Prerequisites (foundational papers first)
       - Citation count (influential papers)
       - Publication date (newer papers often build on older ones)
       - Relevance to the query
    
    Be thorough but concise. Focus on actionable insights.
    Explain your reasoning for the recommended reading order.
    """
)

@research_agent.tool
async def search_papers(
    ctx: RunContext[ResearchContext],
    query: str,
    max_results: int = 5
) -> str:
    """
    Search for academic papers using the papers MCP server.
    Returns JSON string with paper details.
    """
    papers_session = ctx.deps.mcp_sessions.get('papers')
    if not papers_session:
        return json.dumps({"error": "Papers server not available"})
    
    try:
        result = await papers_session.call_tool(
            "search_papers",
            {"query": query, "max_results": min(max_results, ctx.deps.max_papers)}
        )
        
        return result.content[0].text if result.content else json.dumps([])
    except Exception as e:
        return json.dumps({"error": f"Failed to search papers: {str(e)}"})

@research_agent.tool
async def get_paper_details(
    ctx: RunContext[ResearchContext],
    paper_id: int
) -> str:
    """
    Get detailed information about a specific paper by ID.
    """
    papers_session = ctx.deps.mcp_sessions.get('papers')
    if not papers_session:
        return json.dumps({"error": "Papers server not available"})
    
    try:
        result = await papers_session.call_tool(
            "get_paper_details",
            {"paper_id": paper_id}
        )
        
        return result.content[0].text if result.content else json.dumps({})
    except Exception as e:
        return json.dumps({"error": f"Failed to get paper details: {str(e)}"})

@research_agent.tool
async def search_web(
    ctx: RunContext[ResearchContext],
    query: str,
    max_results: int = 3
) -> str:
    """
    Search the web for supplementary information.
    Returns JSON string with web results.
    """
    if not ctx.deps.include_web_search:
        return json.dumps([])
    
    web_session = ctx.deps.mcp_sessions.get('web')
    if not web_session:
        return json.dumps({"error": "Web server not available"})
    
    try:
        result = await web_session.call_tool(
            "search",
            {"query": query, "max_results": max_results}
        )
        
        return result.content[0].text if result.content else json.dumps([])
    except Exception as e:
        return json.dumps({"error": f"Failed to search web: {str(e)}"})

@research_agent.tool
async def save_research_note(
    ctx: RunContext[ResearchContext],
    title: str,
    content: str,
    tags: List[str] = None
) -> str:
    """
    Save a research note with findings.
    """
    notes_session = ctx.deps.mcp_sessions.get('notes')
    if not notes_session:
        return "Notes server not available"
    
    try:
        result = await notes_session.call_tool(
            "add_note",
            {
                "title": title,
                "content": content,
                "tags": tags or [ctx.deps.research_area]
            }
        )
        
        return result.content[0].text if result.content else "Failed to save note"
    except Exception as e:
        return f"Failed to save note: {str(e)}"

@research_agent.tool
async def get_citation_info(
    ctx: RunContext[ResearchContext],
    paper_id: int
) -> str:
    """
    Get citation information for a paper to assess its impact.
    """
    papers_session = ctx.deps.mcp_sessions.get('papers')
    if not papers_session:
        return json.dumps({"error": "Papers server not available"})
    
    try:
        result = await papers_session.call_tool(
            "get_citations",
            {"paper_id": paper_id}
        )
        
        return result.content[0].text if result.content else json.dumps({})
    except Exception as e:
        return json.dumps({"error": f"Failed to get citations: {str(e)}"})

async def run_research(
    query: str,
    mcp_sessions: dict,
    user_id: str = "default_user",
    max_papers: int = 10,
    include_web_search: bool = True
) -> ResearchResult:
    """
    Main entry point for running a research query.
    
    Args:
        query: Research query/topic
        mcp_sessions: Dictionary of connected MCP client sessions
        user_id: User identifier
        max_papers: Maximum number of papers to retrieve
        include_web_search: Whether to include web search results
    
    Returns:
        ResearchResult with papers, analysis, and recommendations
    """
    context = ResearchContext(
        user_id=user_id,
        research_area=query,
        mcp_sessions=mcp_sessions,
        max_papers=max_papers,
        include_web_search=include_web_search
    )
    
    result = await research_agent.run(
        f"Research this topic thoroughly: {query}",
        deps=context
    )
    
    return result.data
