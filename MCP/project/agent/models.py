from pydantic import BaseModel, Field
from typing import List, Literal
from dataclasses import dataclass

class Paper(BaseModel):
    """Represents an academic paper"""
    title: str
    authors: List[str]
    abstract: str = Field(max_length=1000, description="Paper abstract")
    url: str
    year: int | None = None
    citations: int | None = Field(default=None, description="Number of citations")

class WebResult(BaseModel):
    """Represents a web search result"""
    title: str
    url: str
    snippet: str = Field(description="Brief excerpt from the page")

class ResearchResult(BaseModel):
    """Complete research result with papers, analysis, and recommendations"""
    query: str = Field(description="Original research query")
    papers: List[Paper] = Field(description="List of relevant papers found")
    summary: str = Field(description="High-level summary of findings")
    key_findings: List[str] = Field(description="Key insights from the research")
    recommended_reading_order: List[int] = Field(
        description="Indices of papers in recommended reading order (0-based)"
    )
    web_resources: List[WebResult] = Field(
        default_factory=list,
        description="Additional web resources found"
    )

@dataclass
class ResearchContext:
    """Context passed to the research agent"""
    user_id: str
    research_area: str
    mcp_sessions: dict
    max_papers: int = 10
    include_web_search: bool = True

class NoteCreationResult(BaseModel):
    """Result of creating a research note"""
    note_id: int
    title: str
    success: bool
    message: str

class SearchStats(BaseModel):
    """Statistics about a search operation"""
    query: str
    papers_found: int
    web_results: int
    total_citations: int
    year_range: tuple[int, int] | None = None
