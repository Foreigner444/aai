"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                 LESSON 9: CITATION MODELS WITH PYDANTIC                      ║
║                                                                              ║
║              Creating Proper Source Attribution for RAG Responses            ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

📚 CONCEPT OVERVIEW
==================

What & Why:
-----------
Citations are ESSENTIAL for trustworthy RAG systems. When your AI gives an
answer, users need to know WHERE that information came from. Citations:

- Build trust by showing sources
- Allow users to verify information
- Help identify outdated or incorrect sources
- Required for many professional/legal contexts

🎯 Real-World Analogy:
----------------------
Think of citations like receipts for information:

Without Citations:
Customer: "Why was I charged $50?"
Support: "According to our records, you were charged for premium service."
Customer: "Which records? When? Show me proof!"

With Citations:
Customer: "Why was I charged $50?"
Support: "According to your invoice #12345 dated March 15, 2024 (section 3.2),
          you were charged $50 for the Premium Plan upgrade."
Customer: "Ah, I can check that. Thanks!"

Citations transform vague claims into verifiable statements!

🔒 Type Safety Benefit:
-----------------------
With Pydantic citation models:
- Citations always have required fields (source, title)
- Page numbers and dates are properly typed
- Confidence scores are validated ranges
- Missing citation data is caught immediately


💻 CODE IMPLEMENTATION
=====================
"""

from typing import Optional, Literal
from datetime import datetime, date
from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum


class SourceType(str, Enum):
    """Types of sources that can be cited."""
    DOCUMENT = "document"
    WEBPAGE = "webpage"
    DATABASE = "database"
    API = "api"
    USER_INPUT = "user_input"
    KNOWLEDGE_BASE = "knowledge_base"


class Citation(BaseModel):
    """
    A single citation pointing to a source.
    This is the core model for source attribution.
    """
    
    source_id: str = Field(
        description="Unique identifier for the source"
    )
    
    source_type: SourceType = Field(
        default=SourceType.DOCUMENT,
        description="Type of source"
    )
    
    title: str = Field(
        min_length=1,
        description="Title of the source document"
    )
    
    source_path: str = Field(
        description="Path or URL to the source"
    )
    
    chunk_index: Optional[int] = Field(
        default=None,
        ge=0,
        description="Which chunk of the document"
    )
    
    page_number: Optional[int] = Field(
        default=None,
        ge=1,
        description="Page number if applicable"
    )
    
    section: Optional[str] = Field(
        default=None,
        description="Section or heading"
    )
    
    author: Optional[str] = Field(
        default=None,
        description="Author of the source"
    )
    
    publication_date: Optional[date] = Field(
        default=None,
        description="When the source was published"
    )
    
    accessed_at: datetime = Field(
        default_factory=datetime.now,
        description="When this source was accessed"
    )
    
    relevance_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="How relevant this source is to the answer"
    )
    
    excerpt: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Brief excerpt from the source"
    )
    
    def format_citation(self, style: str = "simple") -> str:
        """Format citation in different styles."""
        if style == "simple":
            parts = [self.title]
            if self.author:
                parts.insert(0, self.author)
            if self.page_number:
                parts.append(f"p. {self.page_number}")
            return ", ".join(parts)
        
        elif style == "academic":
            parts = []
            if self.author:
                parts.append(self.author)
            if self.publication_date:
                parts.append(f"({self.publication_date.year})")
            parts.append(f'"{self.title}"')
            if self.page_number:
                parts.append(f"p. {self.page_number}")
            return " ".join(parts)
        
        elif style == "inline":
            if self.page_number:
                return f"[{self.title}, p. {self.page_number}]"
            return f"[{self.title}]"
        
        return self.title


class CitationGroup(BaseModel):
    """
    A group of citations supporting a single claim or statement.
    Multiple sources can support the same claim.
    """
    
    claim: str = Field(
        description="The statement or claim being cited"
    )
    
    citations: list[Citation] = Field(
        min_length=1,
        description="Citations supporting this claim"
    )
    
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Confidence in this claim based on citations"
    )
    
    @property
    def citation_count(self) -> int:
        """Number of citations in this group."""
        return len(self.citations)
    
    @property
    def average_relevance(self) -> float:
        """Average relevance score of citations."""
        if not self.citations:
            return 0.0
        return sum(c.relevance_score for c in self.citations) / len(self.citations)
    
    def get_unique_sources(self) -> list[str]:
        """Get unique source titles."""
        return list(set(c.title for c in self.citations))


class CitedAnswer(BaseModel):
    """
    An answer with inline citations.
    The main output format for RAG systems.
    """
    
    answer: str = Field(
        min_length=1,
        description="The generated answer"
    )
    
    citations: list[Citation] = Field(
        default_factory=list,
        description="All citations used"
    )
    
    citation_groups: list[CitationGroup] = Field(
        default_factory=list,
        description="Citations grouped by claim"
    )
    
    overall_confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Overall confidence in the answer"
    )
    
    sources_consulted: int = Field(
        default=0,
        ge=0,
        description="Number of sources consulted"
    )
    
    generated_at: datetime = Field(
        default_factory=datetime.now,
        description="When this answer was generated"
    )
    
    model_used: str = Field(
        default="gemini-1.5-flash",
        description="Model that generated the answer"
    )
    
    @property
    def has_citations(self) -> bool:
        """Check if answer has any citations."""
        return len(self.citations) > 0
    
    @property
    def citation_count(self) -> int:
        """Total number of citations."""
        return len(self.citations)
    
    def get_citations_by_relevance(self) -> list[Citation]:
        """Get citations sorted by relevance."""
        return sorted(self.citations, key=lambda c: c.relevance_score, reverse=True)
    
    def format_with_footnotes(self) -> str:
        """Format answer with footnote-style citations."""
        if not self.citations:
            return self.answer
        
        result = self.answer
        footnotes = []
        
        for i, citation in enumerate(self.citations, 1):
            footnotes.append(f"[{i}] {citation.format_citation('simple')}")
        
        return f"{result}\n\n---\nSources:\n" + "\n".join(footnotes)
    
    def format_with_inline_citations(self) -> str:
        """Format answer with inline citations."""
        if not self.citation_groups:
            return self.answer
        
        result = self.answer
        for group in self.citation_groups:
            citation_refs = ", ".join(c.format_citation('inline') for c in group.citations[:2])
            result += f" {citation_refs}"
        
        return result


class CitationBuilder:
    """
    Helper class for building citations from retrieved chunks.
    """
    
    def __init__(self):
        self._citations: list[Citation] = []
        self._groups: list[CitationGroup] = []
    
    def add_from_chunk(
        self,
        chunk_id: str,
        content: str,
        source: str,
        title: str,
        similarity_score: float,
        chunk_index: int = 0,
        **extra_metadata
    ) -> Citation:
        """Create a citation from a retrieved chunk."""
        citation = Citation(
            source_id=chunk_id,
            source_type=SourceType.DOCUMENT,
            title=title,
            source_path=source,
            chunk_index=chunk_index,
            relevance_score=similarity_score,
            excerpt=content[:200] if content else None,
            **{k: v for k, v in extra_metadata.items() if v is not None}
        )
        self._citations.append(citation)
        return citation
    
    def create_group(
        self,
        claim: str,
        citation_ids: list[str],
        confidence: float = 0.8
    ) -> CitationGroup:
        """Create a citation group for a specific claim."""
        matching_citations = [
            c for c in self._citations
            if c.source_id in citation_ids
        ]
        
        if not matching_citations:
            raise ValueError(f"No citations found for IDs: {citation_ids}")
        
        group = CitationGroup(
            claim=claim,
            citations=matching_citations,
            confidence=confidence
        )
        self._groups.append(group)
        return group
    
    def build_cited_answer(
        self,
        answer: str,
        model: str = "gemini-1.5-flash",
        overall_confidence: Optional[float] = None
    ) -> CitedAnswer:
        """Build the final cited answer."""
        if overall_confidence is None:
            if self._citations:
                overall_confidence = sum(c.relevance_score for c in self._citations) / len(self._citations)
            else:
                overall_confidence = 0.5
        
        return CitedAnswer(
            answer=answer,
            citations=self._citations.copy(),
            citation_groups=self._groups.copy(),
            overall_confidence=overall_confidence,
            sources_consulted=len(set(c.source_path for c in self._citations)),
            model_used=model
        )
    
    def clear(self):
        """Clear all citations."""
        self._citations = []
        self._groups = []


class DocumentMetadata(BaseModel):
    """Metadata for a retrieved document."""
    source: str
    title: Optional[str] = None
    chunk_index: int = 0


class RetrievedChunk(BaseModel):
    """A chunk retrieved from vector store."""
    chunk_id: str
    content: str
    metadata: DocumentMetadata
    similarity_score: float = Field(ge=0, le=1)


def create_citations_from_chunks(
    chunks: list[RetrievedChunk]
) -> list[Citation]:
    """
    Utility function to create citations from retrieved chunks.
    """
    citations = []
    for chunk in chunks:
        citation = Citation(
            source_id=chunk.chunk_id,
            source_type=SourceType.DOCUMENT,
            title=chunk.metadata.title or chunk.metadata.source,
            source_path=chunk.metadata.source,
            chunk_index=chunk.metadata.chunk_index,
            relevance_score=chunk.similarity_score,
            excerpt=chunk.content[:200]
        )
        citations.append(citation)
    return citations


def demonstrate_citation_models():
    """
    Demonstrate citation models and formatting.
    """
    
    print("=" * 70)
    print("CITATION MODELS WITH PYDANTIC DEMONSTRATION")
    print("=" * 70)
    
    print("\n📝 CREATING INDIVIDUAL CITATIONS")
    print("-" * 50)
    
    citation1 = Citation(
        source_id="doc_001_chunk_0",
        source_type=SourceType.DOCUMENT,
        title="Employee Handbook 2024",
        source_path="hr/employee_handbook.pdf",
        page_number=42,
        section="3.2 Vacation Policy",
        author="HR Department",
        publication_date=date(2024, 1, 15),
        relevance_score=0.95,
        excerpt="All full-time employees are entitled to 20 days of paid vacation per year..."
    )
    
    citation2 = Citation(
        source_id="doc_002_chunk_0",
        source_type=SourceType.DOCUMENT,
        title="PTO Guidelines",
        source_path="hr/pto_guidelines.md",
        section="Carryover Rules",
        relevance_score=0.88,
        excerpt="Unused vacation days can be carried over to the next year, up to 5 days maximum."
    )
    
    citation3 = Citation(
        source_id="kb_003",
        source_type=SourceType.KNOWLEDGE_BASE,
        title="FAQ: Time Off",
        source_path="support/faq/time-off",
        relevance_score=0.72,
        excerpt="Q: How do I request time off? A: Submit through the HR portal at least 2 weeks in advance."
    )
    
    print(f"\nCitation 1:")
    print(f"   Title: {citation1.title}")
    print(f"   Type: {citation1.source_type.value}")
    print(f"   Page: {citation1.page_number}")
    print(f"   Section: {citation1.section}")
    print(f"   Relevance: {citation1.relevance_score:.0%}")
    
    print("\n📚 CITATION FORMATTING STYLES")
    print("-" * 50)
    
    print(f"\nSimple: {citation1.format_citation('simple')}")
    print(f"Academic: {citation1.format_citation('academic')}")
    print(f"Inline: {citation1.format_citation('inline')}")
    
    print("\n" + "=" * 70)
    print("CITATION GROUPS")
    print("=" * 70)
    
    group1 = CitationGroup(
        claim="Employees receive 20 days of paid vacation per year",
        citations=[citation1, citation2],
        confidence=0.95
    )
    
    group2 = CitationGroup(
        claim="Unused days can be carried over up to 5 days",
        citations=[citation2],
        confidence=0.88
    )
    
    print(f"\nGroup 1: '{group1.claim}'")
    print(f"   Citation count: {group1.citation_count}")
    print(f"   Confidence: {group1.confidence:.0%}")
    print(f"   Average relevance: {group1.average_relevance:.0%}")
    print(f"   Unique sources: {group1.get_unique_sources()}")
    
    print("\n" + "=" * 70)
    print("BUILDING CITED ANSWERS")
    print("=" * 70)
    
    builder = CitationBuilder()
    
    chunks = [
        RetrievedChunk(
            chunk_id="doc_001_chunk_0",
            content="All full-time employees are entitled to 20 days of paid vacation per year. "
                   "Part-time employees receive prorated vacation based on hours worked.",
            metadata=DocumentMetadata(
                source="hr/employee_handbook.pdf",
                title="Employee Handbook 2024",
                chunk_index=0
            ),
            similarity_score=0.95
        ),
        RetrievedChunk(
            chunk_id="doc_002_chunk_0",
            content="Unused vacation days can be carried over to the next year, with a maximum "
                   "of 5 days. Days exceeding this limit will be forfeited on January 1st.",
            metadata=DocumentMetadata(
                source="hr/pto_guidelines.md",
                title="PTO Guidelines",
                chunk_index=0
            ),
            similarity_score=0.88
        ),
    ]
    
    print("\n📥 Adding citations from retrieved chunks...")
    
    for chunk in chunks:
        builder.add_from_chunk(
            chunk_id=chunk.chunk_id,
            content=chunk.content,
            source=chunk.metadata.source,
            title=chunk.metadata.title or "Unknown",
            similarity_score=chunk.similarity_score,
            chunk_index=chunk.metadata.chunk_index
        )
    
    builder.create_group(
        claim="20 days vacation for full-time employees",
        citation_ids=["doc_001_chunk_0"],
        confidence=0.95
    )
    
    builder.create_group(
        claim="Up to 5 days carryover allowed",
        citation_ids=["doc_002_chunk_0"],
        confidence=0.88
    )
    
    answer = (
        "Based on the Employee Handbook, full-time employees are entitled to 20 days "
        "of paid vacation per year. According to the PTO Guidelines, you can carry over "
        "up to 5 unused days to the next year. Any days beyond this limit will be "
        "forfeited on January 1st."
    )
    
    cited_answer = builder.build_cited_answer(
        answer=answer,
        model="gemini-1.5-flash"
    )
    
    print(f"\n📋 CITED ANSWER:")
    print("-" * 50)
    print(f"\nAnswer: {cited_answer.answer}")
    print(f"\nHas citations: {cited_answer.has_citations}")
    print(f"Citation count: {cited_answer.citation_count}")
    print(f"Sources consulted: {cited_answer.sources_consulted}")
    print(f"Overall confidence: {cited_answer.overall_confidence:.0%}")
    print(f"Generated at: {cited_answer.generated_at}")
    print(f"Model: {cited_answer.model_used}")
    
    print("\n" + "=" * 70)
    print("FORMATTED OUTPUT STYLES")
    print("=" * 70)
    
    print("\n📝 With Footnotes:")
    print("-" * 50)
    print(cited_answer.format_with_footnotes())
    
    print("\n📝 Citations by Relevance:")
    print("-" * 50)
    for i, cit in enumerate(cited_answer.get_citations_by_relevance(), 1):
        print(f"{i}. [{cit.relevance_score:.0%}] {cit.title}")
    
    print("\n" + "=" * 70)
    print("VALIDATION EXAMPLES")
    print("=" * 70)
    
    print("\n Testing citation validation...")
    
    try:
        bad_citation = Citation(
            source_id="test",
            title="",
            source_path="test.md"
        )
    except Exception as e:
        print(f"\n✅ Caught empty title!")
        print(f"   Error: {e}")
    
    try:
        bad_citation = Citation(
            source_id="test",
            title="Test",
            source_path="test.md",
            relevance_score=1.5
        )
    except Exception as e:
        print(f"\n✅ Caught invalid relevance score!")
        print(f"   Error: {e}")
    
    try:
        bad_citation = Citation(
            source_id="test",
            title="Test",
            source_path="test.md",
            page_number=0
        )
    except Exception as e:
        print(f"\n✅ Caught invalid page number (must be >= 1)!")
        print(f"   Error: {e}")
    
    try:
        bad_group = CitationGroup(
            claim="Test claim",
            citations=[]
        )
    except Exception as e:
        print(f"\n✅ Caught empty citation group!")
        print(f"   Error: {e}")


if __name__ == "__main__":
    demonstrate_citation_models()


"""
🧪 TEST & APPLY
===============

How to Test It:
---------------
    python lesson_09_citation_models_with_pydantic.py

Expected Output:
----------------
1. Individual citation creation and details
2. Multiple formatting styles (simple, academic, inline)
3. Citation groups with claims
4. Complete cited answer building
5. Formatted output with footnotes
6. Validation examples


⚠️ COMMON STUMBLING BLOCKS
==========================

1. "Empty title not allowed"
   
   Error: ValidationError: String should have at least 1 character
   
   Fix: Always provide a title for citations:
   citation = Citation(title=source_name or "Unknown Source", ...)

2. "Page number must be positive"
   
   Error: ValidationError: Input should be greater than or equal to 1
   
   Fix: Page numbers start at 1, not 0:
   citation = Citation(page_number=1, ...)  # First page

3. "No citations found for IDs"
   
   Error: ValueError: No citations found for IDs: [...]
   
   Fix: Add citations before creating groups:
   builder.add_from_chunk(...)  # Add first
   builder.create_group(...)     # Then create group

4. "CitationGroup needs at least one citation"
   
   Error: ValidationError: List should have at least 1 item
   
   Fix: Never create empty citation groups. If no citations,
   don't create a group for that claim.


🎯 KEY TAKEAWAYS
================

1. Citations Build Trust
   - Users can verify information
   - Shows sources are real
   - Professional and credible

2. Multiple Format Styles
   - Simple for casual use
   - Academic for formal
   - Inline for readability

3. Citation Groups Link Claims
   - Each claim can have multiple sources
   - Stronger claims = more citations
   - Confidence based on evidence

4. Pydantic Validates Everything
   - Required fields enforced
   - Scores in valid ranges
   - Page numbers positive


📚 NEXT LESSON
==============
In Lesson 10, we'll learn Source Tracking and Attribution - managing
the flow of information through your RAG pipeline!
"""
