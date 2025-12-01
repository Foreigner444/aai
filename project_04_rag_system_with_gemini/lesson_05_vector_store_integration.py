"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    LESSON 5: VECTOR STORE INTEGRATION                        ║
║                                                                              ║
║             Storing Embeddings in ChromaDB for Fast Retrieval                ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

📚 CONCEPT OVERVIEW
==================

What & Why:
-----------
A vector store (or vector database) is a specialized database optimized for
storing and searching embedding vectors. Unlike traditional databases that
search by exact matches or keywords, vector stores find items by SIMILARITY.

Why we need a vector store:
- Regular databases can't efficiently search 768-dimensional vectors
- Vector stores use special algorithms (like HNSW) for fast similarity search
- They scale to millions of documents while keeping search fast
- They store both vectors AND metadata for rich retrieval

🎯 Real-World Analogy:
----------------------
Think of a vector store like a smart filing cabinet for a library:

Traditional Filing Cabinet:
- Books organized by title (A-Z)
- To find related books, you need to know exact titles
- Finding "similar" books requires reading each one

Vector Store Filing Cabinet:
- Books organized by MEANING in a multi-dimensional space
- Similar books are automatically placed near each other
- Ask "books about space exploration" → instantly find related books
  even if they don't contain those exact words!

🔒 Type Safety Benefit:
-----------------------
With Pydantic models, we ensure:
- All stored documents have required fields
- Metadata is properly structured
- Query results are validated
- Collection configurations are correct


💻 CODE IMPLEMENTATION
=====================
"""

import os
import uuid
from typing import Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field, field_validator


class VectorStoreConfig(BaseModel):
    """
    Configuration for the vector store.
    """
    
    collection_name: str = Field(
        default="rag_documents",
        min_length=1,
        max_length=64,
        description="Name of the collection"
    )
    
    persist_directory: Optional[str] = Field(
        default="./chroma_db",
        description="Directory to persist the database"
    )
    
    embedding_dimension: int = Field(
        default=768,
        ge=1,
        description="Dimension of embedding vectors"
    )
    
    distance_metric: str = Field(
        default="cosine",
        description="Distance metric for similarity"
    )
    
    @field_validator('distance_metric')
    @classmethod
    def validate_metric(cls, v: str) -> str:
        valid = {"cosine", "l2", "ip"}
        if v not in valid:
            raise ValueError(f"Invalid metric: {v}. Must be one of {valid}")
        return v


class DocumentMetadata(BaseModel):
    """Metadata stored with each document in the vector store."""
    
    source: str = Field(description="Original source of the document")
    title: Optional[str] = Field(default=None)
    author: Optional[str] = Field(default=None)
    chunk_index: int = Field(ge=0, default=0)
    total_chunks: int = Field(ge=1, default=1)
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    document_id: Optional[str] = Field(default=None)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for ChromaDB storage."""
        return {k: v for k, v in self.model_dump().items() if v is not None}


class StoredDocument(BaseModel):
    """
    A document stored in the vector store.
    Includes the content, embedding, and metadata.
    """
    
    id: str = Field(description="Unique identifier in the store")
    content: str = Field(description="The text content")
    embedding: list[float] = Field(description="The embedding vector")
    metadata: DocumentMetadata = Field(description="Document metadata")


class QueryResult(BaseModel):
    """
    A single result from a vector store query.
    """
    
    id: str = Field(description="Document ID")
    content: str = Field(description="Document content")
    metadata: DocumentMetadata = Field(description="Document metadata")
    similarity_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Similarity to query (0-1, higher is better)"
    )
    distance: float = Field(
        ge=0.0,
        description="Distance from query (lower is better)"
    )


class QueryResults(BaseModel):
    """
    Results from a vector store query.
    """
    
    query: str = Field(description="The original query")
    results: list[QueryResult] = Field(description="Matching documents")
    total_results: int = Field(ge=0, description="Number of results")
    query_time_ms: float = Field(ge=0, description="Query execution time")


class CollectionStats(BaseModel):
    """Statistics about a vector store collection."""
    
    name: str = Field(description="Collection name")
    document_count: int = Field(ge=0, description="Number of documents")
    embedding_dimension: int = Field(ge=1, description="Vector dimensions")
    metadata_fields: list[str] = Field(description="Available metadata fields")


class VectorStore:
    """
    Vector store implementation using ChromaDB.
    Provides methods for storing and querying document embeddings.
    """
    
    def __init__(self, config: Optional[VectorStoreConfig] = None):
        """Initialize the vector store."""
        self.config = config or VectorStoreConfig()
        self._client = None
        self._collection = None
    
    def _get_client(self):
        """Get or create ChromaDB client."""
        if self._client is None:
            try:
                import chromadb
                from chromadb.config import Settings
                
                if self.config.persist_directory:
                    self._client = chromadb.PersistentClient(
                        path=self.config.persist_directory
                    )
                else:
                    self._client = chromadb.Client()
            except ImportError:
                raise ImportError(
                    "ChromaDB not installed. Install with: pip install chromadb"
                )
        return self._client
    
    def _get_collection(self):
        """Get or create the collection."""
        if self._collection is None:
            client = self._get_client()
            
            metadata = {"hnsw:space": self.config.distance_metric}
            
            self._collection = client.get_or_create_collection(
                name=self.config.collection_name,
                metadata=metadata
            )
        return self._collection
    
    def add_documents(
        self,
        contents: list[str],
        embeddings: list[list[float]],
        metadatas: list[DocumentMetadata],
        ids: Optional[list[str]] = None
    ) -> list[str]:
        """
        Add documents to the vector store.
        
        Args:
            contents: List of text contents
            embeddings: List of embedding vectors
            metadatas: List of metadata objects
            ids: Optional list of IDs (generated if not provided)
            
        Returns:
            List of document IDs
        """
        if not (len(contents) == len(embeddings) == len(metadatas)):
            raise ValueError(
                "Contents, embeddings, and metadatas must have same length"
            )
        
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in contents]
        
        collection = self._get_collection()
        
        metadata_dicts = [m.to_dict() for m in metadatas]
        
        collection.add(
            documents=contents,
            embeddings=embeddings,
            metadatas=metadata_dicts,
            ids=ids
        )
        
        return ids
    
    def add_document(
        self,
        content: str,
        embedding: list[float],
        metadata: DocumentMetadata,
        doc_id: Optional[str] = None
    ) -> str:
        """Add a single document to the vector store."""
        doc_id = doc_id or str(uuid.uuid4())
        
        self.add_documents(
            contents=[content],
            embeddings=[embedding],
            metadatas=[metadata],
            ids=[doc_id]
        )
        
        return doc_id
    
    def query(
        self,
        query_embedding: list[float],
        query_text: str = "",
        n_results: int = 5,
        where: Optional[dict] = None,
        where_document: Optional[dict] = None
    ) -> QueryResults:
        """
        Query the vector store for similar documents.
        
        Args:
            query_embedding: The query vector
            query_text: Original query text (for logging)
            n_results: Number of results to return
            where: Metadata filter
            where_document: Document content filter
            
        Returns:
            QueryResults with matching documents
        """
        import time
        
        start_time = time.time()
        collection = self._get_collection()
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            where_document=where_document,
            include=["documents", "metadatas", "distances"]
        )
        
        query_time = (time.time() - start_time) * 1000
        
        query_results = []
        
        if results['ids'] and results['ids'][0]:
            for i, doc_id in enumerate(results['ids'][0]):
                distance = results['distances'][0][i] if results['distances'] else 0
                
                if self.config.distance_metric == "cosine":
                    similarity = 1 - distance
                else:
                    similarity = 1 / (1 + distance)
                
                similarity = max(0.0, min(1.0, similarity))
                
                meta_dict = results['metadatas'][0][i] if results['metadatas'] else {}
                
                metadata = DocumentMetadata(
                    source=meta_dict.get('source', 'unknown'),
                    title=meta_dict.get('title'),
                    author=meta_dict.get('author'),
                    chunk_index=meta_dict.get('chunk_index', 0),
                    total_chunks=meta_dict.get('total_chunks', 1),
                    created_at=meta_dict.get('created_at', datetime.now().isoformat()),
                    document_id=meta_dict.get('document_id')
                )
                
                query_result = QueryResult(
                    id=doc_id,
                    content=results['documents'][0][i] if results['documents'] else "",
                    metadata=metadata,
                    similarity_score=similarity,
                    distance=distance
                )
                query_results.append(query_result)
        
        return QueryResults(
            query=query_text,
            results=query_results,
            total_results=len(query_results),
            query_time_ms=query_time
        )
    
    def delete_documents(self, ids: list[str]) -> None:
        """Delete documents by ID."""
        collection = self._get_collection()
        collection.delete(ids=ids)
    
    def get_stats(self) -> CollectionStats:
        """Get statistics about the collection."""
        collection = self._get_collection()
        
        count = collection.count()
        
        sample = collection.peek(1)
        metadata_fields = []
        if sample['metadatas'] and sample['metadatas'][0]:
            metadata_fields = list(sample['metadatas'][0].keys())
        
        return CollectionStats(
            name=self.config.collection_name,
            document_count=count,
            embedding_dimension=self.config.embedding_dimension,
            metadata_fields=metadata_fields
        )
    
    def clear_collection(self) -> None:
        """Delete all documents in the collection."""
        client = self._get_client()
        try:
            client.delete_collection(self.config.collection_name)
        except Exception:
            pass
        self._collection = None


def demonstrate_vector_store():
    """
    Demonstrate vector store operations with simulated embeddings.
    """
    import random
    import math
    import shutil
    
    print("=" * 70)
    print("VECTOR STORE INTEGRATION DEMONSTRATION")
    print("=" * 70)
    
    test_dir = "./test_chroma_db"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    
    config = VectorStoreConfig(
        collection_name="demo_collection",
        persist_directory=test_dir,
        distance_metric="cosine"
    )
    
    print(f"\n📦 Vector Store Configuration:")
    print(f"   Collection: {config.collection_name}")
    print(f"   Directory: {config.persist_directory}")
    print(f"   Metric: {config.distance_metric}")
    print(f"   Dimensions: {config.embedding_dimension}")
    
    store = VectorStore(config)
    
    print("\n" + "=" * 70)
    print("ADDING DOCUMENTS")
    print("=" * 70)
    
    def create_pseudo_embedding(text: str, dim: int = 768) -> list[float]:
        """Create deterministic pseudo-embedding."""
        random.seed(hash(text.lower()) % 2**32)
        base = [random.gauss(0, 0.3) for _ in range(dim)]
        keywords = {
            "password": ([0, 1, 2], 0.8),
            "reset": ([3, 4, 5], 0.7),
            "login": ([0, 1, 6], 0.6),
            "vacation": ([50, 51, 52], 0.8),
            "policy": ([53, 54, 55], 0.7),
            "python": ([100, 101, 102], 0.8),
            "code": ([103, 104, 105], 0.7),
        }
        for keyword, (indices, strength) in keywords.items():
            if keyword in text.lower():
                for idx in indices:
                    base[idx] += strength
        mag = math.sqrt(sum(x**2 for x in base))
        return [x / mag for x in base]
    
    documents = [
        {
            "content": "To reset your password, go to Settings > Security > Change Password. "
                      "Enter your current password and then your new password twice.",
            "metadata": DocumentMetadata(
                source="help_docs/password.md",
                title="Password Reset Guide",
                chunk_index=0,
                total_chunks=1
            )
        },
        {
            "content": "If you forgot your login credentials, click 'Forgot Password' on the "
                      "login page. We'll send a reset link to your registered email.",
            "metadata": DocumentMetadata(
                source="help_docs/login.md",
                title="Login Troubleshooting",
                chunk_index=0,
                total_chunks=1
            )
        },
        {
            "content": "Our vacation policy allows 20 days of paid time off per year. "
                      "Unused days can be carried over up to 5 days maximum.",
            "metadata": DocumentMetadata(
                source="hr_docs/vacation.md",
                title="Vacation Policy",
                chunk_index=0,
                total_chunks=1
            )
        },
        {
            "content": "To request time off, submit a vacation request through the HR portal "
                      "at least two weeks before your planned absence.",
            "metadata": DocumentMetadata(
                source="hr_docs/vacation.md",
                title="Vacation Policy",
                chunk_index=1,
                total_chunks=2
            )
        },
        {
            "content": "Python is a versatile programming language. To write your first program, "
                      "create a file called hello.py and add: print('Hello, World!')",
            "metadata": DocumentMetadata(
                source="tutorials/python.md",
                title="Python Getting Started",
                chunk_index=0,
                total_chunks=1
            )
        },
    ]
    
    print("\nAdding documents to vector store...\n")
    
    for i, doc in enumerate(documents):
        embedding = create_pseudo_embedding(doc["content"])
        doc_id = store.add_document(
            content=doc["content"],
            embedding=embedding,
            metadata=doc["metadata"]
        )
        print(f"✅ Added: {doc['metadata'].title} (chunk {doc['metadata'].chunk_index})")
        print(f"   ID: {doc_id[:8]}...")
    
    print("\n" + "=" * 70)
    print("QUERYING THE VECTOR STORE")
    print("=" * 70)
    
    queries = [
        "How do I change my password?",
        "How many vacation days do I get?",
        "How to write Python code?",
    ]
    
    for query in queries:
        print(f"\n🔍 Query: '{query}'")
        print("-" * 50)
        
        query_embedding = create_pseudo_embedding(query)
        results = store.query(
            query_embedding=query_embedding,
            query_text=query,
            n_results=3
        )
        
        print(f"   Found {results.total_results} results in {results.query_time_ms:.2f}ms\n")
        
        for i, result in enumerate(results.results, 1):
            print(f"   {i}. [{result.similarity_score:.3f}] {result.metadata.title}")
            print(f"      Source: {result.metadata.source}")
            print(f"      Content: {result.content[:60]}...")
            print()
    
    print("\n" + "=" * 70)
    print("FILTERED QUERIES")
    print("=" * 70)
    
    print("\n🔍 Query with metadata filter (source contains 'hr_docs'):")
    
    query = "company policies"
    query_embedding = create_pseudo_embedding(query)
    
    results = store.query(
        query_embedding=query_embedding,
        query_text=query,
        n_results=5,
        where={"source": {"$contains": "hr_docs"}}
    )
    
    print(f"   Query: '{query}'")
    print(f"   Filter: source contains 'hr_docs'")
    print(f"   Results: {results.total_results}\n")
    
    for result in results.results:
        print(f"   • {result.metadata.title}")
        print(f"     {result.content[:50]}...")
    
    print("\n" + "=" * 70)
    print("COLLECTION STATISTICS")
    print("=" * 70)
    
    stats = store.get_stats()
    
    print(f"\n📊 Collection: {stats.name}")
    print(f"   Documents: {stats.document_count}")
    print(f"   Dimensions: {stats.embedding_dimension}")
    print(f"   Metadata fields: {stats.metadata_fields}")
    
    print("\n" + "=" * 70)
    print("VALIDATION EXAMPLES")
    print("=" * 70)
    
    print("\n Testing configuration validation...")
    
    try:
        bad_config = VectorStoreConfig(
            distance_metric="invalid"
        )
    except Exception as e:
        print(f"\n✅ Caught invalid distance metric!")
        print(f"   Error: {e}")
    
    try:
        bad_config = VectorStoreConfig(
            collection_name=""
        )
    except Exception as e:
        print(f"\n✅ Caught empty collection name!")
        print(f"   Error: {e}")
    
    print("\n🧹 Cleaning up test database...")
    store.clear_collection()
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    print("   Done!")


if __name__ == "__main__":
    demonstrate_vector_store()


"""
🧪 TEST & APPLY
===============

How to Test It:
---------------
First install ChromaDB:
    pip install chromadb

Then run:
    python lesson_05_vector_store_integration.py

Expected Output:
----------------
1. Vector store configuration details
2. Documents added with their IDs
3. Query results with similarity scores
4. Filtered query results
5. Collection statistics
6. Validation examples


⚠️ COMMON STUMBLING BLOCKS
==========================

1. "ChromaDB not installed"
   
   Error: ImportError: ChromaDB not installed
   
   Fix: pip install chromadb

2. "Collection already exists with different settings"
   
   Error: ValueError: Collection exists with different metadata
   
   This happens when you change distance_metric on existing collection.
   Fix: Delete the collection first or use a new name:
   
   store.clear_collection()  # Deletes existing collection

3. "Dimension mismatch"
   
   Error: chromadb.errors.InvalidDimensionException
   
   All embeddings in a collection must have the same dimensions.
   Fix: Ensure all embeddings are created with the same model/settings.

4. "Query returns no results"
   
   Possible causes:
   - Collection is empty
   - Filter (where clause) is too restrictive
   - n_results is 0
   
   Fix: Check stats and remove filters to debug:
   
   stats = store.get_stats()
   print(f"Documents: {stats.document_count}")


🎯 KEY TAKEAWAYS
================

1. Vector Stores are Specialized
   - Optimized for similarity search, not exact matching
   - Use HNSW or similar algorithms for speed
   - Scale to millions of vectors

2. ChromaDB is Simple but Powerful
   - Local, embedded database (no server needed)
   - Persistent or in-memory modes
   - Supports metadata filtering

3. Metadata Enables Rich Queries
   - Filter by source, date, author, etc.
   - Combine similarity with structured filters
   - Essential for multi-document RAG

4. Type Safety with Pydantic
   - Validated configurations
   - Structured query results
   - Consistent metadata


📚 NEXT LESSON
==============
In Lesson 6, we'll implement Similarity Search - the core retrieval
logic that finds the most relevant documents for any query!
"""
