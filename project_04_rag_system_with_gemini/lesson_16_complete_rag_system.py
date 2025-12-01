"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    LESSON 16: COMPLETE RAG SYSTEM                            ║
║                                                                              ║
║        Putting Everything Together into a Production-Ready Application       ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

📚 CONCEPT OVERVIEW
==================

What & Why:
-----------
This is the culmination of everything we've learned! We're building a complete,
production-ready RAG system that combines:

- Document processing and chunking
- Gemini embeddings and vector storage
- Similarity search with context management
- Dependency injection for clean architecture
- Citations and source tracking
- Confidence scoring and validation
- Streaming responses

This is a REAL system you can deploy and use!

🎯 What We're Building:
-----------------------
A complete RAG pipeline that:
1. Ingests and processes documents
2. Creates and stores embeddings
3. Retrieves relevant context for queries
4. Generates validated, cited answers
5. Streams responses with confidence scores

🔒 Type Safety Throughout:
--------------------------
Every component uses Pydantic models, ensuring:
- All data is validated at every step
- Errors are caught early and clearly
- The system is self-documenting
- Refactoring is safe and easy


💻 CODE IMPLEMENTATION
=====================
"""

import os
import asyncio
import hashlib
from dataclasses import dataclass
from typing import Optional, AsyncIterator
from datetime import datetime
from pydantic import BaseModel, Field, computed_field
from enum import Enum


class DocumentMetadata(BaseModel):
    """Metadata for documents in the RAG system."""
    source: str = Field(description="Source path or identifier")
    title: Optional[str] = Field(default=None, description="Document title")
    author: Optional[str] = Field(default=None, description="Document author")
    created_at: datetime = Field(default_factory=datetime.now)
    chunk_index: int = Field(default=0, ge=0)
    total_chunks: int = Field(default=1, ge=1)


class DocumentChunk(BaseModel):
    """A chunk of a document ready for embedding."""
    chunk_id: str = Field(description="Unique chunk identifier")
    document_id: str = Field(description="Parent document ID")
    content: str = Field(min_length=1, description="Chunk content")
    metadata: DocumentMetadata = Field(description="Chunk metadata")
    token_estimate: int = Field(default=0, ge=0)


class RetrievedChunk(BaseModel):
    """A chunk retrieved from the vector store."""
    chunk_id: str
    content: str
    metadata: DocumentMetadata
    similarity_score: float = Field(ge=0, le=1)
    

class Citation(BaseModel):
    """A citation to a source document."""
    source_id: str
    source: str
    title: Optional[str] = None
    relevance: float = Field(ge=0, le=1)
    excerpt: Optional[str] = None


class ConfidenceLevel(str, Enum):
    """Confidence levels for answers."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RAGResponse(BaseModel):
    """Complete RAG response with all metadata."""
    
    query: str = Field(description="Original query")
    answer: str = Field(description="Generated answer")
    
    citations: list[Citation] = Field(
        default_factory=list,
        description="Sources cited"
    )
    
    confidence_score: float = Field(
        ge=0, le=1,
        description="Confidence in the answer"
    )
    
    confidence_level: ConfidenceLevel = Field(
        description="Human-readable confidence"
    )
    
    chunks_used: int = Field(ge=0, description="Number of chunks used")
    
    retrieval_time_ms: float = Field(ge=0)
    generation_time_ms: float = Field(ge=0)
    total_time_ms: float = Field(ge=0)
    
    model_used: str = Field(description="Model that generated answer")
    generated_at: datetime = Field(default_factory=datetime.now)
    
    validation_passed: bool = Field(default=True)
    validation_warnings: list[str] = Field(default_factory=list)
    
    @computed_field
    @property
    def is_reliable(self) -> bool:
        """Check if response is reliable enough to show."""
        return self.confidence_score >= 0.5 and self.validation_passed
    
    def format_with_citations(self) -> str:
        """Format answer with citation footnotes."""
        if not self.citations:
            return self.answer
        
        footnotes = []
        for i, citation in enumerate(self.citations, 1):
            source_name = citation.title or citation.source
            footnotes.append(f"[{i}] {source_name}")
        
        return f"{self.answer}\n\n---\nSources:\n" + "\n".join(footnotes)


class RAGConfig(BaseModel):
    """Configuration for the RAG system."""
    
    collection_name: str = Field(default="rag_documents")
    
    chunk_size: int = Field(default=500, ge=100, le=2000)
    chunk_overlap: int = Field(default=50, ge=0)
    
    top_k: int = Field(default=5, ge=1, le=20)
    min_similarity: float = Field(default=0.5, ge=0, le=1)
    
    max_context_tokens: int = Field(default=4000, ge=100)
    
    model_name: str = Field(default="gemini-1.5-flash")
    temperature: float = Field(default=0.7, ge=0, le=2)
    
    enable_validation: bool = Field(default=True)
    enable_streaming: bool = Field(default=True)
    
    confidence_threshold: float = Field(default=0.5, ge=0, le=1)


class MockVectorStore:
    """Mock vector store for demonstration."""
    
    def __init__(self):
        self._documents: list[dict] = []
    
    def add(self, chunk_id: str, content: str, embedding: list[float], 
            metadata: dict) -> None:
        self._documents.append({
            "id": chunk_id,
            "content": content,
            "embedding": embedding,
            "metadata": metadata
        })
    
    def search(self, query_embedding: list[float], top_k: int) -> list[dict]:
        import math
        
        def cosine_sim(v1, v2):
            dot = sum(a * b for a, b in zip(v1, v2))
            m1 = math.sqrt(sum(a * a for a in v1))
            m2 = math.sqrt(sum(b * b for b in v2))
            if m1 == 0 or m2 == 0:
                return 0
            return (dot / (m1 * m2) + 1) / 2
        
        results = []
        for doc in self._documents:
            sim = cosine_sim(query_embedding, doc["embedding"])
            results.append({**doc, "similarity": sim})
        
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]
    
    def count(self) -> int:
        return len(self._documents)


class MockEmbedder:
    """Mock embedder for demonstration."""
    
    def __init__(self, dimensions: int = 768):
        self.dimensions = dimensions
    
    def embed(self, text: str) -> list[float]:
        import random
        random.seed(hash(text.lower()) % 2**32)
        return [random.gauss(0, 0.3) for _ in range(self.dimensions)]
    
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]


@dataclass
class RAGDependencies:
    """Dependencies for the RAG system."""
    vector_store: MockVectorStore
    embedder: MockEmbedder
    config: RAGConfig


class DocumentProcessor:
    """Processes documents into chunks."""
    
    def __init__(self, config: RAGConfig):
        self.config = config
    
    def _generate_id(self, content: str) -> str:
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def chunk_document(
        self,
        content: str,
        source: str,
        title: Optional[str] = None
    ) -> list[DocumentChunk]:
        """Split document into chunks."""
        chunks = []
        doc_id = self._generate_id(content + source)
        
        words = content.split()
        chunk_words = self.config.chunk_size // 5
        overlap_words = self.config.chunk_overlap // 5
        
        start = 0
        chunk_idx = 0
        
        while start < len(words):
            end = min(start + chunk_words, len(words))
            chunk_content = " ".join(words[start:end])
            
            chunk = DocumentChunk(
                chunk_id=f"{doc_id}_chunk_{chunk_idx}",
                document_id=doc_id,
                content=chunk_content,
                metadata=DocumentMetadata(
                    source=source,
                    title=title,
                    chunk_index=chunk_idx,
                    total_chunks=0
                ),
                token_estimate=len(chunk_content) // 4
            )
            chunks.append(chunk)
            
            start = end - overlap_words if end < len(words) else end
            chunk_idx += 1
        
        for chunk in chunks:
            chunk.metadata.total_chunks = len(chunks)
        
        return chunks


class RAGPipeline:
    """
    Complete RAG pipeline with all components.
    """
    
    def __init__(self, deps: RAGDependencies):
        """Initialize the pipeline with dependencies."""
        self.deps = deps
        self.processor = DocumentProcessor(deps.config)
    
    def ingest_document(
        self,
        content: str,
        source: str,
        title: Optional[str] = None
    ) -> int:
        """
        Ingest a document into the RAG system.
        Returns number of chunks created.
        """
        chunks = self.processor.chunk_document(content, source, title)
        
        for chunk in chunks:
            embedding = self.deps.embedder.embed(chunk.content)
            
            self.deps.vector_store.add(
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                embedding=embedding,
                metadata=chunk.metadata.model_dump()
            )
        
        return len(chunks)
    
    def retrieve(self, query: str) -> list[RetrievedChunk]:
        """Retrieve relevant chunks for a query."""
        query_embedding = self.deps.embedder.embed(query)
        
        results = self.deps.vector_store.search(
            query_embedding,
            self.deps.config.top_k
        )
        
        chunks = []
        for result in results:
            if result["similarity"] >= self.deps.config.min_similarity:
                chunk = RetrievedChunk(
                    chunk_id=result["id"],
                    content=result["content"],
                    metadata=DocumentMetadata(**result["metadata"]),
                    similarity_score=result["similarity"]
                )
                chunks.append(chunk)
        
        return chunks
    
    def _build_context(self, chunks: list[RetrievedChunk]) -> str:
        """Build context string from chunks."""
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk.metadata.title or chunk.metadata.source
            context_parts.append(f"[Source {i}: {source}]\n{chunk.content}")
        
        return "\n\n".join(context_parts)
    
    def _generate_answer(
        self,
        query: str,
        context: str,
        chunks: list[RetrievedChunk]
    ) -> str:
        """Generate answer from context (simulated)."""
        if not chunks:
            return "I couldn't find relevant information to answer your question."
        
        main_content = chunks[0].content
        
        if "vacation" in query.lower() or "pto" in query.lower():
            return (
                "Based on the documentation, here's what I found about vacation policy:\n\n"
                f"{main_content}\n\n"
                "This information comes from the official company policy documents."
            )
        elif "password" in query.lower():
            return (
                "To reset your password, follow these steps:\n\n"
                f"{main_content}\n\n"
                "If you continue to have issues, contact IT support."
            )
        else:
            return f"Based on the available documentation:\n\n{main_content}"
    
    def _calculate_confidence(
        self,
        chunks: list[RetrievedChunk],
        answer: str
    ) -> tuple[float, ConfidenceLevel]:
        """Calculate confidence score and level."""
        if not chunks:
            return 0.0, ConfidenceLevel.LOW
        
        avg_similarity = sum(c.similarity_score for c in chunks) / len(chunks)
        
        coverage = min(len(chunks) / 3, 1.0)
        
        length_factor = min(len(answer) / 100, 1.0)
        
        confidence = (avg_similarity * 0.5 + coverage * 0.3 + length_factor * 0.2)
        
        if confidence >= 0.7:
            level = ConfidenceLevel.HIGH
        elif confidence >= 0.5:
            level = ConfidenceLevel.MEDIUM
        else:
            level = ConfidenceLevel.LOW
        
        return confidence, level
    
    def _create_citations(
        self,
        chunks: list[RetrievedChunk]
    ) -> list[Citation]:
        """Create citations from retrieved chunks."""
        citations = []
        seen_sources = set()
        
        for chunk in chunks:
            source_key = chunk.metadata.source
            if source_key not in seen_sources:
                citations.append(Citation(
                    source_id=chunk.chunk_id,
                    source=chunk.metadata.source,
                    title=chunk.metadata.title,
                    relevance=chunk.similarity_score,
                    excerpt=chunk.content[:100] + "..."
                ))
                seen_sources.add(source_key)
        
        return citations
    
    def _validate_answer(
        self,
        answer: str,
        chunks: list[RetrievedChunk]
    ) -> tuple[bool, list[str]]:
        """Validate the generated answer."""
        warnings = []
        
        if len(answer) < 20:
            warnings.append("Answer is very short")
        
        if not chunks:
            warnings.append("No source documents found")
            return False, warnings
        
        answer_lower = answer.lower()
        source_content = " ".join(c.content.lower() for c in chunks)
        
        answer_words = set(answer_lower.split())
        source_words = set(source_content.split())
        overlap = len(answer_words & source_words) / len(answer_words) if answer_words else 0
        
        if overlap < 0.2:
            warnings.append("Answer may contain information not in sources")
        
        passed = len(warnings) == 0 or all("may" in w for w in warnings)
        
        return passed, warnings
    
    def query(self, query: str) -> RAGResponse:
        """
        Execute a complete RAG query.
        
        This is the main entry point for the RAG system.
        """
        import time
        
        start_time = time.time()
        
        retrieval_start = time.time()
        chunks = self.retrieve(query)
        retrieval_time = (time.time() - retrieval_start) * 1000
        
        generation_start = time.time()
        context = self._build_context(chunks)
        answer = self._generate_answer(query, context, chunks)
        generation_time = (time.time() - generation_start) * 1000
        
        confidence, level = self._calculate_confidence(chunks, answer)
        
        citations = self._create_citations(chunks)
        
        validation_passed, warnings = (True, [])
        if self.deps.config.enable_validation:
            validation_passed, warnings = self._validate_answer(answer, chunks)
        
        total_time = (time.time() - start_time) * 1000
        
        return RAGResponse(
            query=query,
            answer=answer,
            citations=citations,
            confidence_score=confidence,
            confidence_level=level,
            chunks_used=len(chunks),
            retrieval_time_ms=retrieval_time,
            generation_time_ms=generation_time,
            total_time_ms=total_time,
            model_used=self.deps.config.model_name,
            validation_passed=validation_passed,
            validation_warnings=warnings
        )
    
    async def query_streaming(
        self,
        query: str
    ) -> AsyncIterator[str]:
        """
        Execute a streaming RAG query.
        Yields answer chunks progressively.
        """
        chunks = self.retrieve(query)
        context = self._build_context(chunks)
        answer = self._generate_answer(query, context, chunks)
        
        words = answer.split()
        
        for i in range(0, len(words), 3):
            chunk = " ".join(words[i:i+3]) + " "
            yield chunk
            await asyncio.sleep(0.05)
    
    def get_stats(self) -> dict:
        """Get system statistics."""
        return {
            "total_chunks": self.deps.vector_store.count(),
            "config": self.deps.config.model_dump()
        }


def create_rag_system(config: Optional[RAGConfig] = None) -> RAGPipeline:
    """Factory function to create a RAG system."""
    config = config or RAGConfig()
    
    deps = RAGDependencies(
        vector_store=MockVectorStore(),
        embedder=MockEmbedder(),
        config=config
    )
    
    return RAGPipeline(deps)


def demonstrate_complete_system():
    """
    Demonstrate the complete RAG system.
    """
    
    print("=" * 70)
    print("COMPLETE RAG SYSTEM DEMONSTRATION")
    print("=" * 70)
    
    print("\n📦 CREATING RAG SYSTEM")
    print("-" * 50)
    
    config = RAGConfig(
        collection_name="company_knowledge",
        chunk_size=400,
        chunk_overlap=50,
        top_k=5,
        min_similarity=0.4,
        model_name="gemini-1.5-flash",
        enable_validation=True
    )
    
    rag = create_rag_system(config)
    
    print(f"✅ RAG System created")
    print(f"   Model: {config.model_name}")
    print(f"   Chunk size: {config.chunk_size}")
    print(f"   Top K: {config.top_k}")
    
    print("\n" + "=" * 70)
    print("INGESTING DOCUMENTS")
    print("=" * 70)
    
    documents = [
        {
            "content": """
            Vacation Policy for Full-Time Employees
            
            All full-time employees are entitled to 20 days of paid vacation per year.
            Vacation time begins accruing from your start date and can be used after
            completing 90 days of employment.
            
            Requesting Time Off:
            - Submit vacation requests through the HR portal
            - Requests must be submitted at least 2 weeks in advance
            - Manager approval is required within 48 hours
            - Emergency requests will be considered on a case-by-case basis
            
            Carryover Policy:
            - Up to 5 unused days can be carried over to the next year
            - Carried-over days must be used by March 31st
            - Days exceeding the limit will be forfeited
            
            Part-time employees receive prorated vacation based on scheduled hours.
            """,
            "source": "hr/vacation_policy.md",
            "title": "Vacation Policy"
        },
        {
            "content": """
            Password and Security Guidelines
            
            Creating Strong Passwords:
            - Minimum 12 characters
            - Include uppercase and lowercase letters
            - Include numbers and special characters
            - Avoid personal information
            
            Resetting Your Password:
            1. Go to the login page
            2. Click "Forgot Password"
            3. Enter your email address
            4. Check your email for the reset link
            5. Create a new password following the guidelines above
            
            Two-Factor Authentication:
            - Enable 2FA in Settings > Security
            - Use an authenticator app (recommended) or SMS
            - Backup codes are provided for emergencies
            
            If you're locked out, contact IT Support at support@company.com.
            """,
            "source": "it/security_guide.md",
            "title": "Security Guidelines"
        },
        {
            "content": """
            Remote Work Policy
            
            Eligibility:
            - Available to employees after 6 months
            - Requires manager approval
            - Some roles may require on-site presence
            
            Equipment:
            - Company provides laptop and necessary software
            - $500 annual stipend for home office setup
            - IT support available remotely
            
            Expectations:
            - Maintain regular working hours (with flexibility)
            - Attend virtual meetings as scheduled
            - Respond to communications within 2 hours
            - Ensure reliable internet connection
            
            Hybrid arrangements are available for those preferring a mix of
            remote and on-site work. Discuss options with your manager.
            """,
            "source": "hr/remote_work.md",
            "title": "Remote Work Policy"
        }
    ]
    
    total_chunks = 0
    for doc in documents:
        chunks = rag.ingest_document(
            content=doc["content"],
            source=doc["source"],
            title=doc["title"]
        )
        total_chunks += chunks
        print(f"   📄 {doc['title']}: {chunks} chunks")
    
    print(f"\n   Total chunks ingested: {total_chunks}")
    
    print("\n" + "=" * 70)
    print("QUERYING THE SYSTEM")
    print("=" * 70)
    
    queries = [
        "How many vacation days do I get as a full-time employee?",
        "How do I reset my password?",
        "What is the remote work policy?",
        "What is the meaning of life?",
    ]
    
    for query in queries:
        print(f"\n🔍 Query: '{query}'")
        print("-" * 50)
        
        response = rag.query(query)
        
        print(f"\n📝 Answer:")
        print(f"   {response.answer[:200]}...")
        
        print(f"\n📊 Metrics:")
        print(f"   Confidence: {response.confidence_score:.0%} ({response.confidence_level.value})")
        print(f"   Chunks Used: {response.chunks_used}")
        print(f"   Retrieval: {response.retrieval_time_ms:.1f}ms")
        print(f"   Generation: {response.generation_time_ms:.1f}ms")
        print(f"   Total: {response.total_time_ms:.1f}ms")
        print(f"   Reliable: {response.is_reliable}")
        
        if response.citations:
            print(f"\n📚 Citations:")
            for citation in response.citations[:2]:
                print(f"   • {citation.title or citation.source} ({citation.relevance:.0%})")
        
        if response.validation_warnings:
            print(f"\n⚠️  Warnings:")
            for warning in response.validation_warnings:
                print(f"   • {warning}")
    
    print("\n" + "=" * 70)
    print("FORMATTED OUTPUT WITH CITATIONS")
    print("=" * 70)
    
    response = rag.query("What is the vacation policy?")
    print(f"\n{response.format_with_citations()}")
    
    print("\n" + "=" * 70)
    print("STREAMING DEMONSTRATION")
    print("=" * 70)
    
    async def demo_streaming():
        query = "How do I request time off?"
        print(f"\n🔍 Streaming Query: '{query}'")
        print("-" * 50)
        print()
        
        async for chunk in rag.query_streaming(query):
            print(chunk, end="", flush=True)
        
        print("\n")
    
    asyncio.run(demo_streaming())
    
    print("\n" + "=" * 70)
    print("SYSTEM STATISTICS")
    print("=" * 70)
    
    stats = rag.get_stats()
    print(f"\n📈 Statistics:")
    print(f"   Total Chunks: {stats['total_chunks']}")
    print(f"   Model: {stats['config']['model_name']}")
    print(f"   Chunk Size: {stats['config']['chunk_size']}")
    
    print("\n" + "=" * 70)
    print("🎉 COMPLETE RAG SYSTEM DEMONSTRATION FINISHED!")
    print("=" * 70)
    print("""
You've now seen the complete RAG system in action! This system includes:

✅ Document processing and chunking
✅ Vector storage with similarity search  
✅ Context-aware answer generation
✅ Confidence scoring and validation
✅ Citation generation
✅ Streaming responses

To use this in production, replace the mock components with:
- Real Gemini embeddings (text-embedding-004)
- Real vector store (ChromaDB, Pinecone, etc.)
- Real Gemini generation (gemini-1.5-flash/pro)

Congratulations on completing the RAG System project! 🚀
""")


if __name__ == "__main__":
    demonstrate_complete_system()


"""
🧪 TEST & APPLY
===============

How to Test It:
---------------
    python lesson_16_complete_rag_system.py

Expected Output:
----------------
1. System creation and configuration
2. Document ingestion with chunk counts
3. Multiple query examples with full metrics
4. Formatted output with citations
5. Streaming demonstration
6. System statistics


🎓 PROJECT COMPLETE!
====================

Congratulations! You've completed the RAG System with Gemini project!

What You've Learned:
-------------------
1. RAG Architecture - How retrieval augmented generation works
2. Document Processing - Extracting and preparing content
3. Text Chunking - Splitting documents effectively
4. Embeddings - Converting text to vectors with Gemini
5. Vector Storage - Storing and searching embeddings
6. Similarity Search - Finding relevant content
7. Context Management - Optimizing context for Gemini
8. Dependencies - Clean, testable architecture
9. Citations - Proper source attribution
10. Source Tracking - Full provenance tracking
11. Multi-Document - Searching across collections
12. Confidence Scoring - Evaluating answer reliability
13. Answer Validation - Ensuring accuracy
14. Fact-Checking - Verifying specific claims
15. Streaming - Real-time response delivery
16. Complete System - Production-ready RAG

Skills Acquired:
---------------
✅ Building type-safe AI systems with Pydantic AI
✅ Using Google Gemini for embeddings and generation
✅ Implementing vector similarity search
✅ Creating validated, cited responses
✅ Streaming responses for better UX
✅ Testing with dependency injection


🚀 NEXT STEPS
=============

To make this production-ready:

1. Replace Mock Components:
   - Use google.generativeai for real embeddings
   - Use chromadb or similar for vector storage
   - Use Pydantic AI Agent for generation

2. Add Error Handling:
   - Retry logic for API calls
   - Graceful degradation
   - Comprehensive logging

3. Optimize Performance:
   - Batch embedding creation
   - Caching for repeated queries
   - Async operations throughout

4. Add Features:
   - User authentication
   - Query history
   - Feedback collection
   - Analytics dashboard

5. Deploy:
   - FastAPI REST API (Project 8!)
   - Docker containerization
   - Cloud deployment


You're now ready to build production RAG systems! 🎉
"""
