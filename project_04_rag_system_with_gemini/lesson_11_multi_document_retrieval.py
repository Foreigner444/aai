"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                   LESSON 11: MULTI-DOCUMENT RETRIEVAL                        ║
║                                                                              ║
║          Searching Across Multiple Collections and Combining Results         ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

📚 CONCEPT OVERVIEW
==================

What & Why:
-----------
Real-world RAG systems rarely have just one document collection. You might
have separate collections for different departments, document types, or 
access levels. Multi-document retrieval lets you:

- Search across multiple collections simultaneously
- Combine results intelligently
- Apply different weights to different sources
- Maintain separation of concerns

🎯 Real-World Analogy:
----------------------
Think of multi-document retrieval like a librarian searching multiple sections:

Question: "What's the best programming language for beginners?"

Single Section Search:
- Only searches "Programming" section
- Misses relevant books in "Education" and "Career" sections

Multi-Section Search:
- Searches "Programming" (technical perspective)
- Searches "Education" (learning perspective)
- Searches "Career" (practical perspective)
- Combines findings for comprehensive answer

Multi-document retrieval gives you the full picture!

🔒 Type Safety Benefit:
-----------------------
With Pydantic, multi-document retrieval ensures:
- All collections return consistent result types
- Scores are normalized and comparable
- Source attribution is maintained
- Combined results are properly structured


💻 CODE IMPLEMENTATION
=====================
"""

from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field, computed_field
from enum import Enum
import math


class CollectionType(str, Enum):
    """Types of document collections."""
    TECHNICAL = "technical"
    HR_POLICIES = "hr_policies"
    PRODUCT = "product"
    SUPPORT = "support"
    GENERAL = "general"


class DocumentMetadata(BaseModel):
    """Metadata for a document."""
    source: str
    title: Optional[str] = None
    collection: str = "default"
    chunk_index: int = 0
    created_at: Optional[datetime] = None


class RetrievedChunk(BaseModel):
    """A chunk retrieved from a collection."""
    
    chunk_id: str = Field(description="Unique chunk identifier")
    content: str = Field(description="Text content")
    metadata: DocumentMetadata = Field(description="Document metadata")
    similarity_score: float = Field(ge=0, le=1, description="Raw similarity")
    collection_name: str = Field(description="Source collection")
    
    normalized_score: float = Field(
        default=0.0,
        ge=0,
        le=1,
        description="Normalized score for cross-collection comparison"
    )
    
    weighted_score: float = Field(
        default=0.0,
        ge=0,
        description="Score after collection weighting"
    )


class CollectionConfig(BaseModel):
    """Configuration for a single collection."""
    
    name: str = Field(description="Collection name")
    collection_type: CollectionType = Field(description="Type of collection")
    weight: float = Field(
        default=1.0,
        ge=0.0,
        le=5.0,
        description="Weight for this collection's results"
    )
    
    min_similarity: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum similarity threshold"
    )
    
    max_results: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum results from this collection"
    )
    
    enabled: bool = Field(
        default=True,
        description="Whether to search this collection"
    )
    
    boost_recent: bool = Field(
        default=False,
        description="Boost recent documents"
    )
    
    recency_days: int = Field(
        default=30,
        ge=1,
        description="Days to consider 'recent'"
    )


class MultiRetrievalConfig(BaseModel):
    """Configuration for multi-document retrieval."""
    
    collections: list[CollectionConfig] = Field(
        description="Collections to search"
    )
    
    total_results: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Total results to return"
    )
    
    deduplication: bool = Field(
        default=True,
        description="Remove duplicate content"
    )
    
    similarity_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Threshold for considering duplicates"
    )
    
    fusion_method: str = Field(
        default="weighted",
        description="How to combine results: weighted, rrf, max"
    )
    
    normalize_scores: bool = Field(
        default=True,
        description="Normalize scores across collections"
    )
    
    def get_enabled_collections(self) -> list[CollectionConfig]:
        """Get only enabled collections."""
        return [c for c in self.collections if c.enabled]


class CollectionResults(BaseModel):
    """Results from a single collection."""
    
    collection_name: str = Field(description="Collection name")
    chunks: list[RetrievedChunk] = Field(description="Retrieved chunks")
    search_time_ms: float = Field(ge=0, description="Search time")
    total_in_collection: int = Field(ge=0, description="Total docs in collection")
    
    @computed_field
    @property
    def result_count(self) -> int:
        return len(self.chunks)
    
    @computed_field
    @property
    def avg_similarity(self) -> float:
        if not self.chunks:
            return 0.0
        return sum(c.similarity_score for c in self.chunks) / len(self.chunks)


class MultiRetrievalResults(BaseModel):
    """Combined results from multiple collections."""
    
    query: str = Field(description="Original query")
    
    collection_results: list[CollectionResults] = Field(
        description="Results per collection"
    )
    
    combined_chunks: list[RetrievedChunk] = Field(
        description="Final combined results"
    )
    
    total_search_time_ms: float = Field(
        ge=0,
        description="Total search time"
    )
    
    collections_searched: int = Field(
        ge=0,
        description="Number of collections searched"
    )
    
    duplicates_removed: int = Field(
        default=0,
        ge=0,
        description="Duplicates removed"
    )
    
    config_used: MultiRetrievalConfig = Field(
        description="Configuration used"
    )
    
    @computed_field
    @property
    def total_results(self) -> int:
        return len(self.combined_chunks)
    
    @computed_field
    @property
    def results_per_collection(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for chunk in self.combined_chunks:
            name = chunk.collection_name
            counts[name] = counts.get(name, 0) + 1
        return counts
    
    def get_top_results(self, n: int = 5) -> list[RetrievedChunk]:
        """Get top N results."""
        return self.combined_chunks[:n]
    
    def filter_by_collection(self, collection_name: str) -> list[RetrievedChunk]:
        """Filter results by collection."""
        return [c for c in self.combined_chunks if c.collection_name == collection_name]


class MultiDocumentRetriever:
    """
    Retrieves and combines results from multiple document collections.
    """
    
    def __init__(self, config: MultiRetrievalConfig):
        """Initialize with configuration."""
        self.config = config
        self._collections: dict[str, list[dict]] = {}
    
    def add_mock_collection(
        self,
        name: str,
        documents: list[dict]
    ) -> None:
        """Add a mock collection for demonstration."""
        self._collections[name] = documents
    
    def _create_mock_embedding(self, text: str) -> list[float]:
        """Create deterministic pseudo-embedding."""
        import random
        random.seed(hash(text.lower()) % 2**32)
        return [random.gauss(0, 0.3) for _ in range(768)]
    
    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity."""
        dot = sum(a * b for a, b in zip(vec1, vec2))
        mag1 = math.sqrt(sum(a * a for a in vec1))
        mag2 = math.sqrt(sum(b * b for b in vec2))
        if mag1 == 0 or mag2 == 0:
            return 0.0
        return max(0.0, min(1.0, (dot / (mag1 * mag2) + 1) / 2))
    
    def _search_collection(
        self,
        collection_name: str,
        query_embedding: list[float],
        collection_config: CollectionConfig
    ) -> CollectionResults:
        """Search a single collection."""
        import time
        
        start = time.time()
        docs = self._collections.get(collection_name, [])
        
        results = []
        for doc in docs:
            doc_embedding = self._create_mock_embedding(doc["content"])
            similarity = self._cosine_similarity(query_embedding, doc_embedding)
            
            if similarity >= collection_config.min_similarity:
                chunk = RetrievedChunk(
                    chunk_id=doc["id"],
                    content=doc["content"],
                    metadata=DocumentMetadata(
                        source=doc.get("source", "unknown"),
                        title=doc.get("title"),
                        collection=collection_name,
                        chunk_index=doc.get("chunk_index", 0)
                    ),
                    similarity_score=similarity,
                    collection_name=collection_name
                )
                results.append(chunk)
        
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        results = results[:collection_config.max_results]
        
        search_time = (time.time() - start) * 1000
        
        return CollectionResults(
            collection_name=collection_name,
            chunks=results,
            search_time_ms=search_time,
            total_in_collection=len(docs)
        )
    
    def _normalize_scores(
        self,
        all_results: list[CollectionResults]
    ) -> None:
        """Normalize scores across collections."""
        all_scores = []
        for result in all_results:
            for chunk in result.chunks:
                all_scores.append(chunk.similarity_score)
        
        if not all_scores:
            return
        
        min_score = min(all_scores)
        max_score = max(all_scores)
        score_range = max_score - min_score if max_score > min_score else 1.0
        
        for result in all_results:
            for chunk in result.chunks:
                chunk.normalized_score = (chunk.similarity_score - min_score) / score_range
    
    def _apply_weights(
        self,
        all_results: list[CollectionResults]
    ) -> None:
        """Apply collection weights to scores."""
        config_map = {c.name: c for c in self.config.collections}
        
        for result in all_results:
            collection_config = config_map.get(result.collection_name)
            weight = collection_config.weight if collection_config else 1.0
            
            for chunk in result.chunks:
                base_score = chunk.normalized_score if self.config.normalize_scores else chunk.similarity_score
                chunk.weighted_score = base_score * weight
    
    def _deduplicate(
        self,
        chunks: list[RetrievedChunk]
    ) -> tuple[list[RetrievedChunk], int]:
        """Remove duplicate or near-duplicate content."""
        if not self.config.deduplication:
            return chunks, 0
        
        unique: list[RetrievedChunk] = []
        removed = 0
        
        for chunk in chunks:
            is_duplicate = False
            
            for existing in unique:
                chunk_emb = self._create_mock_embedding(chunk.content)
                existing_emb = self._create_mock_embedding(existing.content)
                similarity = self._cosine_similarity(chunk_emb, existing_emb)
                
                if similarity >= self.config.similarity_threshold:
                    is_duplicate = True
                    if chunk.weighted_score > existing.weighted_score:
                        unique.remove(existing)
                        unique.append(chunk)
                    break
            
            if not is_duplicate:
                unique.append(chunk)
            else:
                removed += 1
        
        return unique, removed
    
    def _combine_results(
        self,
        all_results: list[CollectionResults]
    ) -> tuple[list[RetrievedChunk], int]:
        """Combine results from all collections."""
        all_chunks = []
        for result in all_results:
            all_chunks.extend(result.chunks)
        
        if self.config.fusion_method == "weighted":
            all_chunks.sort(key=lambda x: x.weighted_score, reverse=True)
        elif self.config.fusion_method == "max":
            all_chunks.sort(key=lambda x: x.similarity_score, reverse=True)
        elif self.config.fusion_method == "rrf":
            self._apply_rrf(all_results, all_chunks)
            all_chunks.sort(key=lambda x: x.weighted_score, reverse=True)
        
        unique_chunks, removed = self._deduplicate(all_chunks)
        
        final_chunks = unique_chunks[:self.config.total_results]
        
        return final_chunks, removed
    
    def _apply_rrf(
        self,
        all_results: list[CollectionResults],
        all_chunks: list[RetrievedChunk]
    ) -> None:
        """Apply Reciprocal Rank Fusion."""
        k = 60
        chunk_scores: dict[str, float] = {}
        
        for result in all_results:
            for rank, chunk in enumerate(result.chunks, 1):
                rrf_score = 1 / (k + rank)
                current = chunk_scores.get(chunk.chunk_id, 0)
                chunk_scores[chunk.chunk_id] = current + rrf_score
        
        for chunk in all_chunks:
            chunk.weighted_score = chunk_scores.get(chunk.chunk_id, 0)
    
    def retrieve(self, query: str) -> MultiRetrievalResults:
        """
        Retrieve from all configured collections.
        """
        import time
        
        start = time.time()
        query_embedding = self._create_mock_embedding(query)
        
        collection_results = []
        for collection_config in self.config.get_enabled_collections():
            result = self._search_collection(
                collection_name=collection_config.name,
                query_embedding=query_embedding,
                collection_config=collection_config
            )
            collection_results.append(result)
        
        self._normalize_scores(collection_results)
        self._apply_weights(collection_results)
        
        combined_chunks, duplicates_removed = self._combine_results(collection_results)
        
        total_time = (time.time() - start) * 1000
        
        return MultiRetrievalResults(
            query=query,
            collection_results=collection_results,
            combined_chunks=combined_chunks,
            total_search_time_ms=total_time,
            collections_searched=len(collection_results),
            duplicates_removed=duplicates_removed,
            config_used=self.config
        )


def demonstrate_multi_document_retrieval():
    """
    Demonstrate multi-document retrieval.
    """
    
    print("=" * 70)
    print("MULTI-DOCUMENT RETRIEVAL DEMONSTRATION")
    print("=" * 70)
    
    config = MultiRetrievalConfig(
        collections=[
            CollectionConfig(
                name="hr_policies",
                collection_type=CollectionType.HR_POLICIES,
                weight=1.5,
                min_similarity=0.3,
                max_results=3
            ),
            CollectionConfig(
                name="technical_docs",
                collection_type=CollectionType.TECHNICAL,
                weight=1.0,
                min_similarity=0.3,
                max_results=3
            ),
            CollectionConfig(
                name="support_kb",
                collection_type=CollectionType.SUPPORT,
                weight=1.2,
                min_similarity=0.3,
                max_results=3
            ),
        ],
        total_results=5,
        deduplication=True,
        fusion_method="weighted",
        normalize_scores=True
    )
    
    print("\n📋 RETRIEVAL CONFIGURATION")
    print("-" * 50)
    print(f"Total results: {config.total_results}")
    print(f"Deduplication: {config.deduplication}")
    print(f"Fusion method: {config.fusion_method}")
    print(f"\nCollections:")
    for coll in config.collections:
        print(f"   • {coll.name} (weight: {coll.weight}, max: {coll.max_results})")
    
    retriever = MultiDocumentRetriever(config)
    
    hr_docs = [
        {
            "id": "hr_001",
            "content": "Employees receive 20 days of paid vacation per year. "
                      "Vacation requests must be submitted 2 weeks in advance.",
            "source": "hr/vacation_policy.md",
            "title": "Vacation Policy"
        },
        {
            "id": "hr_002",
            "content": "Unused vacation days can be carried over, up to 5 days maximum. "
                      "Excess days are forfeited on January 1st.",
            "source": "hr/vacation_policy.md",
            "title": "Vacation Policy"
        },
        {
            "id": "hr_003",
            "content": "Sick leave provides 10 days per year. No advance notice required "
                      "for sick days, but manager should be notified same day.",
            "source": "hr/sick_leave.md",
            "title": "Sick Leave Policy"
        },
    ]
    
    tech_docs = [
        {
            "id": "tech_001",
            "content": "To reset your password, go to Settings > Security > Change Password. "
                      "Passwords must be at least 12 characters.",
            "source": "tech/password_guide.md",
            "title": "Password Guide"
        },
        {
            "id": "tech_002",
            "content": "VPN access requires installing the company VPN client. "
                      "Download from internal.company.com/vpn",
            "source": "tech/vpn_setup.md",
            "title": "VPN Setup"
        },
    ]
    
    support_docs = [
        {
            "id": "support_001",
            "content": "For time off questions, contact HR at hr@company.com or "
                      "submit a ticket through the employee portal.",
            "source": "support/faq.md",
            "title": "Employee FAQ"
        },
        {
            "id": "support_002",
            "content": "Vacation requests are reviewed within 48 hours. "
                      "For urgent requests, contact your manager directly.",
            "source": "support/faq.md",
            "title": "Employee FAQ"
        },
    ]
    
    retriever.add_mock_collection("hr_policies", hr_docs)
    retriever.add_mock_collection("technical_docs", tech_docs)
    retriever.add_mock_collection("support_kb", support_docs)
    
    print("\n" + "=" * 70)
    print("PERFORMING MULTI-COLLECTION SEARCH")
    print("=" * 70)
    
    query = "How many vacation days do I get and how do I request time off?"
    print(f"\n🔍 Query: '{query}'")
    
    results = retriever.retrieve(query)
    
    print(f"\n📊 SEARCH STATISTICS")
    print("-" * 50)
    print(f"Collections searched: {results.collections_searched}")
    print(f"Total search time: {results.total_search_time_ms:.2f}ms")
    print(f"Duplicates removed: {results.duplicates_removed}")
    print(f"Final results: {results.total_results}")
    
    print(f"\n📚 RESULTS PER COLLECTION")
    print("-" * 50)
    for coll_result in results.collection_results:
        print(f"\n{coll_result.collection_name}:")
        print(f"   Found: {coll_result.result_count}")
        print(f"   Avg similarity: {coll_result.avg_similarity:.3f}")
        print(f"   Search time: {coll_result.search_time_ms:.2f}ms")
    
    print(f"\n📈 DISTRIBUTION IN FINAL RESULTS")
    print("-" * 50)
    for name, count in results.results_per_collection.items():
        bar = "█" * count
        print(f"   {name}: {bar} ({count})")
    
    print("\n" + "=" * 70)
    print("COMBINED RESULTS (Ranked)")
    print("=" * 70)
    
    for i, chunk in enumerate(results.combined_chunks, 1):
        print(f"\n{i}. [{chunk.weighted_score:.3f}] {chunk.metadata.title}")
        print(f"   Collection: {chunk.collection_name}")
        print(f"   Similarity: {chunk.similarity_score:.3f}")
        print(f"   Normalized: {chunk.normalized_score:.3f}")
        print(f"   Content: {chunk.content[:80]}...")
    
    print("\n" + "=" * 70)
    print("FILTERING RESULTS")
    print("=" * 70)
    
    hr_only = results.filter_by_collection("hr_policies")
    print(f"\n📁 HR Policies only: {len(hr_only)} results")
    for chunk in hr_only:
        print(f"   • {chunk.metadata.title}: {chunk.content[:50]}...")
    
    top_3 = results.get_top_results(3)
    print(f"\n🔝 Top 3 results:")
    for chunk in top_3:
        print(f"   • [{chunk.weighted_score:.3f}] {chunk.metadata.title}")
    
    print("\n" + "=" * 70)
    print("DIFFERENT FUSION METHODS")
    print("=" * 70)
    
    for method in ["weighted", "max", "rrf"]:
        config_copy = config.model_copy()
        config_copy.fusion_method = method
        
        retriever_temp = MultiDocumentRetriever(config_copy)
        retriever_temp._collections = retriever._collections
        
        results_temp = retriever_temp.retrieve(query)
        
        print(f"\n{method.upper()} fusion - Top 3:")
        for chunk in results_temp.get_top_results(3):
            print(f"   [{chunk.weighted_score:.3f}] {chunk.metadata.title} ({chunk.collection_name})")


if __name__ == "__main__":
    demonstrate_multi_document_retrieval()


"""
🧪 TEST & APPLY
===============

How to Test It:
---------------
    python lesson_11_multi_document_retrieval.py

Expected Output:
----------------
1. Retrieval configuration display
2. Multi-collection search statistics
3. Results per collection
4. Combined ranked results
5. Filtered results examples
6. Different fusion method comparisons


⚠️ COMMON STUMBLING BLOCKS
==========================

1. "Collection not found"
   
   Error: Empty results for collection
   
   Fix: Register collection before searching:
   retriever.add_mock_collection("my_collection", docs)

2. "All results from one collection"
   
   Might mean weights are imbalanced.
   
   Fix: Adjust collection weights:
   CollectionConfig(weight=1.0, ...)  # Equal weight

3. "Too few results after deduplication"
   
   Similarity threshold might be too low.
   
   Fix: Increase threshold or disable dedup:
   config = MultiRetrievalConfig(
       deduplication=False,  # Or
       similarity_threshold=0.95  # Stricter
   )

4. "Scores not comparable across collections"
   
   Fix: Enable score normalization:
   config = MultiRetrievalConfig(normalize_scores=True)


🎯 KEY TAKEAWAYS
================

1. Multiple Collections = Better Coverage
   - Different perspectives
   - Comprehensive answers
   - Organized document management

2. Weights Control Importance
   - Boost authoritative sources
   - Demote less reliable ones
   - Tune for your use case

3. Fusion Methods
   - Weighted: Uses configured weights
   - Max: Takes highest raw score
   - RRF: Rank-based, robust fusion

4. Deduplication is Essential
   - Same content in multiple collections
   - Near-duplicates waste context
   - Keep highest-scoring version


📚 NEXT LESSON
==============
In Lesson 12, we'll learn Confidence Scoring - evaluating how
confident we should be in our RAG answers!
"""
