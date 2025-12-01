"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                LESSON 6: SIMILARITY SEARCH IMPLEMENTATION                    ║
║                                                                              ║
║              Finding the Most Relevant Documents for Any Query               ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

📚 CONCEPT OVERVIEW
==================

What & Why:
-----------
Similarity search is the heart of RAG - it finds documents whose meaning is
closest to a user's question. Unlike keyword search, similarity search
understands that "reset password" and "change login credentials" are related,
even without shared words.

Why this is CRUCIAL:
- Users don't know what words are in your documents
- Same concept can be expressed many ways
- Semantic search finds relevant content regardless of wording
- This is what makes RAG feel "intelligent"

🎯 Real-World Analogy:
----------------------
Imagine you're at a huge library and you ask:
"I need books about getting better at public speaking"

Keyword Search Librarian:
- Looks for books with "public" and "speaking" in the title
- Misses "The Art of Persuasion" and "Confident Communication"

Similarity Search Librarian:
- Understands you want self-improvement in presentation skills
- Finds related books even with different titles
- Returns "The Art of Persuasion", "Speak with Confidence", etc.

That's the power of similarity search!

🔒 Type Safety Benefit:
-----------------------
With Pydantic, our search results are:
- Properly typed with scores in valid ranges
- Include validated metadata for citations
- Guarantee non-empty results when matches exist
- Structured for easy processing downstream


💻 CODE IMPLEMENTATION
=====================
"""

import os
import math
from typing import Optional, Literal
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, model_validator


class SearchConfig(BaseModel):
    """
    Configuration for similarity search.
    """
    
    top_k: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Number of results to return"
    )
    
    min_similarity: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum similarity threshold"
    )
    
    rerank: bool = Field(
        default=False,
        description="Whether to rerank results"
    )
    
    include_metadata: bool = Field(
        default=True,
        description="Include metadata in results"
    )
    
    diversity_factor: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Factor to promote diverse results (MMR)"
    )


class DocumentMetadata(BaseModel):
    """Metadata for search results."""
    source: str = Field(description="Document source")
    title: Optional[str] = Field(default=None)
    chunk_index: int = Field(default=0, ge=0)
    total_chunks: int = Field(default=1, ge=1)
    created_at: Optional[str] = Field(default=None)


class SearchResult(BaseModel):
    """
    A single search result with similarity score.
    """
    
    document_id: str = Field(description="Unique document identifier")
    content: str = Field(description="Document content")
    similarity: float = Field(
        ge=0.0,
        le=1.0,
        description="Similarity score (0-1)"
    )
    metadata: DocumentMetadata = Field(description="Document metadata")
    rank: int = Field(ge=1, description="Position in results")
    
    @property
    def is_highly_relevant(self) -> bool:
        """Check if this result is highly relevant (>0.8 similarity)."""
        return self.similarity >= 0.8


class SearchResults(BaseModel):
    """
    Complete search results with statistics.
    """
    
    query: str = Field(description="Original query")
    results: list[SearchResult] = Field(description="Search results")
    total_found: int = Field(ge=0, description="Total matching documents")
    search_time_ms: float = Field(ge=0, description="Search duration")
    config_used: SearchConfig = Field(description="Search configuration")
    
    @property
    def has_results(self) -> bool:
        """Check if any results were found."""
        return len(self.results) > 0
    
    @property
    def best_result(self) -> Optional[SearchResult]:
        """Get the highest-scoring result."""
        return self.results[0] if self.results else None
    
    @property
    def average_similarity(self) -> float:
        """Calculate average similarity of results."""
        if not self.results:
            return 0.0
        return sum(r.similarity for r in self.results) / len(self.results)
    
    def filter_by_source(self, source_pattern: str) -> list[SearchResult]:
        """Filter results by source pattern."""
        return [r for r in self.results if source_pattern in r.metadata.source]
    
    def get_unique_sources(self) -> list[str]:
        """Get list of unique sources in results."""
        return list(set(r.metadata.source for r in self.results))


class SimilaritySearcher:
    """
    Main class for performing similarity search on embedded documents.
    """
    
    def __init__(self, config: Optional[SearchConfig] = None):
        """Initialize the searcher."""
        self.config = config or SearchConfig()
        self._documents: list[dict] = []
    
    @staticmethod
    def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
        """
        Calculate cosine similarity between two vectors.
        Returns a value between 0 and 1 (1 = identical).
        """
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        similarity = dot_product / (magnitude1 * magnitude2)
        return max(0.0, min(1.0, (similarity + 1) / 2))
    
    @staticmethod
    def euclidean_distance(vec1: list[float], vec2: list[float]) -> float:
        """Calculate Euclidean distance between two vectors."""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(vec1, vec2)))
    
    @staticmethod
    def euclidean_to_similarity(distance: float) -> float:
        """Convert Euclidean distance to similarity score."""
        return 1 / (1 + distance)
    
    def add_document(
        self,
        document_id: str,
        content: str,
        embedding: list[float],
        metadata: DocumentMetadata
    ) -> None:
        """Add a document to the search index."""
        self._documents.append({
            "id": document_id,
            "content": content,
            "embedding": embedding,
            "metadata": metadata
        })
    
    def add_documents(
        self,
        documents: list[dict]
    ) -> int:
        """
        Add multiple documents to the search index.
        
        Each document should have: id, content, embedding, metadata
        """
        for doc in documents:
            self._documents.append({
                "id": doc["id"],
                "content": doc["content"],
                "embedding": doc["embedding"],
                "metadata": doc["metadata"]
            })
        return len(documents)
    
    def search(
        self,
        query_embedding: list[float],
        query_text: str = "",
        config: Optional[SearchConfig] = None
    ) -> SearchResults:
        """
        Search for similar documents.
        
        Args:
            query_embedding: The query vector
            query_text: Original query text
            config: Override default search config
            
        Returns:
            SearchResults with ranked documents
        """
        import time
        
        start_time = time.time()
        search_config = config or self.config
        
        scored_docs = []
        for doc in self._documents:
            similarity = self.cosine_similarity(query_embedding, doc["embedding"])
            
            if similarity >= search_config.min_similarity:
                scored_docs.append({
                    **doc,
                    "similarity": similarity
                })
        
        scored_docs.sort(key=lambda x: x["similarity"], reverse=True)
        
        if search_config.diversity_factor > 0:
            scored_docs = self._apply_mmr(
                scored_docs,
                query_embedding,
                search_config.diversity_factor,
                search_config.top_k
            )
        
        top_docs = scored_docs[:search_config.top_k]
        
        results = []
        for rank, doc in enumerate(top_docs, 1):
            result = SearchResult(
                document_id=doc["id"],
                content=doc["content"],
                similarity=doc["similarity"],
                metadata=doc["metadata"],
                rank=rank
            )
            results.append(result)
        
        search_time = (time.time() - start_time) * 1000
        
        return SearchResults(
            query=query_text,
            results=results,
            total_found=len(scored_docs),
            search_time_ms=search_time,
            config_used=search_config
        )
    
    def _apply_mmr(
        self,
        scored_docs: list[dict],
        query_embedding: list[float],
        diversity_factor: float,
        top_k: int
    ) -> list[dict]:
        """
        Apply Maximal Marginal Relevance for diverse results.
        
        MMR balances relevance with diversity by penalizing documents
        that are too similar to already-selected documents.
        """
        if not scored_docs:
            return []
        
        selected = [scored_docs[0]]
        candidates = scored_docs[1:]
        
        while len(selected) < top_k and candidates:
            best_score = -1
            best_idx = 0
            
            for i, candidate in enumerate(candidates):
                relevance = candidate["similarity"]
                
                max_sim_to_selected = max(
                    self.cosine_similarity(
                        candidate["embedding"],
                        s["embedding"]
                    )
                    for s in selected
                )
                
                mmr_score = (
                    (1 - diversity_factor) * relevance -
                    diversity_factor * max_sim_to_selected
                )
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i
            
            selected.append(candidates[best_idx])
            candidates.pop(best_idx)
        
        return selected
    
    def search_with_filter(
        self,
        query_embedding: list[float],
        query_text: str,
        source_filter: Optional[str] = None,
        metadata_filter: Optional[dict] = None
    ) -> SearchResults:
        """
        Search with metadata filtering.
        
        Args:
            query_embedding: Query vector
            query_text: Original query
            source_filter: Filter by source substring
            metadata_filter: Dict of metadata field -> value filters
        """
        filtered_docs = self._documents.copy()
        
        if source_filter:
            filtered_docs = [
                d for d in filtered_docs
                if source_filter in d["metadata"].source
            ]
        
        if metadata_filter:
            for key, value in metadata_filter.items():
                filtered_docs = [
                    d for d in filtered_docs
                    if getattr(d["metadata"], key, None) == value
                ]
        
        original_docs = self._documents
        self._documents = filtered_docs
        
        results = self.search(query_embedding, query_text)
        
        self._documents = original_docs
        
        return results
    
    def get_document_count(self) -> int:
        """Get number of documents in the index."""
        return len(self._documents)
    
    def clear(self) -> None:
        """Clear all documents from the index."""
        self._documents = []


def demonstrate_similarity_search():
    """
    Demonstrate similarity search with sample documents.
    """
    import random
    
    print("=" * 70)
    print("SIMILARITY SEARCH IMPLEMENTATION DEMONSTRATION")
    print("=" * 70)
    
    def create_pseudo_embedding(text: str, dim: int = 768) -> list[float]:
        """Create deterministic pseudo-embedding based on text content."""
        random.seed(hash(text.lower()) % 2**32)
        base = [random.gauss(0, 0.3) for _ in range(dim)]
        
        keywords = {
            "password": ([0, 1, 2, 3, 4], 1.0),
            "reset": ([5, 6, 7, 8, 9], 0.9),
            "change": ([5, 6, 10, 11], 0.8),
            "login": ([0, 1, 12, 13, 14], 0.85),
            "credentials": ([0, 1, 2, 15, 16], 0.9),
            "forgot": ([17, 18, 19], 0.7),
            "vacation": ([100, 101, 102, 103], 1.0),
            "time off": ([100, 104, 105], 0.95),
            "policy": ([106, 107, 108], 0.8),
            "days": ([109, 110], 0.6),
            "python": ([200, 201, 202, 203], 1.0),
            "code": ([204, 205, 206], 0.85),
            "programming": ([200, 207, 208], 0.9),
            "tutorial": ([209, 210, 211], 0.7),
        }
        
        text_lower = text.lower()
        for keyword, (indices, strength) in keywords.items():
            if keyword in text_lower:
                for idx in indices:
                    base[idx] += strength
        
        mag = math.sqrt(sum(x**2 for x in base))
        return [x / mag for x in base] if mag > 0 else base
    
    documents = [
        {
            "id": "doc_001",
            "content": "To reset your password, navigate to Settings > Security > Change Password. "
                      "Enter your current password followed by your new password.",
            "metadata": DocumentMetadata(
                source="help/password_guide.md",
                title="Password Reset Guide",
                chunk_index=0,
                total_chunks=1
            )
        },
        {
            "id": "doc_002",
            "content": "If you forgot your login credentials, click the 'Forgot Password' link "
                      "on the sign-in page. A reset link will be sent to your email.",
            "metadata": DocumentMetadata(
                source="help/login_help.md",
                title="Login Troubleshooting",
                chunk_index=0,
                total_chunks=2
            )
        },
        {
            "id": "doc_003",
            "content": "For security reasons, passwords must be at least 12 characters long "
                      "and include uppercase, lowercase, numbers, and special characters.",
            "metadata": DocumentMetadata(
                source="help/login_help.md",
                title="Login Troubleshooting",
                chunk_index=1,
                total_chunks=2
            )
        },
        {
            "id": "doc_004",
            "content": "Our vacation policy grants 20 days of paid time off per year. "
                      "Unused days can be carried over up to a maximum of 5 days.",
            "metadata": DocumentMetadata(
                source="hr/vacation_policy.md",
                title="Vacation Policy",
                chunk_index=0,
                total_chunks=2
            )
        },
        {
            "id": "doc_005",
            "content": "To request time off, submit a vacation request at least two weeks "
                      "in advance through the HR portal. Manager approval is required.",
            "metadata": DocumentMetadata(
                source="hr/vacation_policy.md",
                title="Vacation Policy",
                chunk_index=1,
                total_chunks=2
            )
        },
        {
            "id": "doc_006",
            "content": "Python is a versatile programming language perfect for beginners. "
                      "Start with 'Hello World': print('Hello, World!')",
            "metadata": DocumentMetadata(
                source="tutorials/python_basics.md",
                title="Python Basics",
                chunk_index=0,
                total_chunks=3
            )
        },
        {
            "id": "doc_007",
            "content": "Python variables don't need type declarations. Just assign values: "
                      "name = 'Alice' and age = 30. Python figures out the types.",
            "metadata": DocumentMetadata(
                source="tutorials/python_basics.md",
                title="Python Basics",
                chunk_index=1,
                total_chunks=3
            )
        },
        {
            "id": "doc_008",
            "content": "Functions in Python are defined with 'def'. Example: "
                      "def greet(name): return f'Hello, {name}!'",
            "metadata": DocumentMetadata(
                source="tutorials/python_basics.md",
                title="Python Basics",
                chunk_index=2,
                total_chunks=3
            )
        },
    ]
    
    searcher = SimilaritySearcher(SearchConfig(top_k=5))
    
    print("\n📚 Loading documents into search index...")
    for doc in documents:
        doc["embedding"] = create_pseudo_embedding(doc["content"])
        searcher.add_document(
            document_id=doc["id"],
            content=doc["content"],
            embedding=doc["embedding"],
            metadata=doc["metadata"]
        )
    print(f"   Loaded {searcher.get_document_count()} documents")
    
    print("\n" + "=" * 70)
    print("BASIC SIMILARITY SEARCH")
    print("=" * 70)
    
    queries = [
        "How do I change my password?",
        "What is the vacation policy?",
        "How to learn Python programming?",
    ]
    
    for query in queries:
        print(f"\n🔍 Query: '{query}'")
        print("-" * 50)
        
        query_embedding = create_pseudo_embedding(query)
        results = searcher.search(query_embedding, query)
        
        print(f"   Found {results.total_found} matches in {results.search_time_ms:.2f}ms")
        print(f"   Average similarity: {results.average_similarity:.3f}")
        print()
        
        for result in results.results[:3]:
            relevance = "🟢" if result.is_highly_relevant else "🟡"
            print(f"   {relevance} Rank {result.rank}: [{result.similarity:.3f}] {result.metadata.title}")
            print(f"      Source: {result.metadata.source}")
            print(f"      Content: {result.content[:60]}...")
            print()
    
    print("\n" + "=" * 70)
    print("SEARCH WITH MINIMUM THRESHOLD")
    print("=" * 70)
    
    query = "employee benefits"
    query_embedding = create_pseudo_embedding(query)
    
    strict_config = SearchConfig(top_k=5, min_similarity=0.5)
    results = searcher.search(query_embedding, query, strict_config)
    
    print(f"\n🔍 Query: '{query}'")
    print(f"   Min similarity threshold: {strict_config.min_similarity}")
    print(f"   Results above threshold: {results.total_found}")
    
    for result in results.results:
        print(f"\n   [{result.similarity:.3f}] {result.metadata.title}")
    
    print("\n" + "=" * 70)
    print("SEARCH WITH DIVERSITY (MMR)")
    print("=" * 70)
    
    query = "Python coding tutorial"
    query_embedding = create_pseudo_embedding(query)
    
    print(f"\n🔍 Query: '{query}'")
    
    print("\n   Without diversity (may get similar chunks):")
    normal_config = SearchConfig(top_k=3, diversity_factor=0.0)
    results = searcher.search(query_embedding, query, normal_config)
    
    for result in results.results:
        print(f"   [{result.similarity:.3f}] {result.metadata.title} (chunk {result.metadata.chunk_index})")
    
    print("\n   With diversity factor 0.5 (more varied results):")
    diverse_config = SearchConfig(top_k=3, diversity_factor=0.5)
    results = searcher.search(query_embedding, query, diverse_config)
    
    for result in results.results:
        print(f"   [{result.similarity:.3f}] {result.metadata.title} (chunk {result.metadata.chunk_index})")
    
    print("\n" + "=" * 70)
    print("FILTERED SEARCH")
    print("=" * 70)
    
    query = "how to do something"
    query_embedding = create_pseudo_embedding(query)
    
    print(f"\n🔍 Query: '{query}'")
    print("   Filter: source contains 'help'")
    
    results = searcher.search_with_filter(
        query_embedding=query_embedding,
        query_text=query,
        source_filter="help"
    )
    
    print(f"   Found {results.total_found} results in help docs:\n")
    for result in results.results:
        print(f"   [{result.similarity:.3f}] {result.metadata.source}")
    
    print("\n" + "=" * 70)
    print("SEARCH RESULTS ANALYSIS")
    print("=" * 70)
    
    query = "company policies"
    query_embedding = create_pseudo_embedding(query)
    results = searcher.search(query_embedding, query, SearchConfig(top_k=5))
    
    print(f"\n📊 Results Analysis for: '{query}'")
    print(f"   Has results: {results.has_results}")
    print(f"   Total found: {results.total_found}")
    print(f"   Average similarity: {results.average_similarity:.3f}")
    
    if results.best_result:
        print(f"   Best result: {results.best_result.metadata.title}")
        print(f"   Best similarity: {results.best_result.similarity:.3f}")
    
    print(f"   Unique sources: {results.get_unique_sources()}")
    
    hr_results = results.filter_by_source("hr/")
    print(f"   HR-specific results: {len(hr_results)}")


if __name__ == "__main__":
    demonstrate_similarity_search()


"""
🧪 TEST & APPLY
===============

How to Test It:
---------------
    python lesson_06_similarity_search_implementation.py

Expected Output:
----------------
1. Documents loaded into search index
2. Basic similarity search results for multiple queries
3. Search with minimum threshold filtering
4. Diversity search (MMR) comparison
5. Filtered search by source
6. Search results analysis


⚠️ COMMON STUMBLING BLOCKS
==========================

1. "All similarities are very low"
   
   Possible causes:
   - Embeddings are random/not meaningful
   - Query and documents use very different vocabulary
   
   Fix: With real Gemini embeddings, this improves dramatically.
   The demo uses pseudo-embeddings for illustration.

2. "Too many similar results from same document"
   
   This happens when multiple chunks from one document all match.
   
   Fix: Use diversity_factor in SearchConfig:
   config = SearchConfig(diversity_factor=0.3)

3. "min_similarity filters everything out"
   
   Error: No results returned
   
   Fix: Start with low threshold and increase gradually:
   config = SearchConfig(min_similarity=0.3)

4. "Search is slow with many documents"
   
   The simple implementation is O(n) per search.
   
   Fix: Use a proper vector database (ChromaDB, Pinecone)
   which uses approximate nearest neighbor algorithms.


🎯 KEY TAKEAWAYS
================

1. Cosine Similarity is Standard
   - Measures angle between vectors
   - Range: -1 to 1 (normalized to 0-1)
   - Magnitude-independent

2. MMR Balances Relevance & Diversity
   - Prevents duplicate-ish results
   - diversity_factor controls trade-off
   - 0.0 = pure relevance, 1.0 = pure diversity

3. Filtering Narrows Search Space
   - Metadata filters before similarity
   - Useful for multi-tenant, multi-category data
   - Combine with similarity for precise results

4. Type Safety Enables Rich Results
   - Pydantic models with computed properties
   - Easy filtering and analysis methods
   - Validated scores and metadata


📚 NEXT LESSON
==============
In Lesson 7, we'll learn Context Window Management - how to optimize
what context we send to Gemini for the best answers!
"""
