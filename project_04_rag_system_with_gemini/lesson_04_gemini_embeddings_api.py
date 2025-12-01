"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                     LESSON 4: GEMINI EMBEDDINGS API                          ║
║                                                                              ║
║            Converting Text to Vectors for Semantic Similarity                ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

📚 CONCEPT OVERVIEW
==================

What & Why:
-----------
Embeddings are the magic that makes semantic search possible. An embedding
converts text into a vector (a list of numbers) that captures its meaning.
Similar texts have similar vectors, allowing us to find relevant documents
even when they don't share exact words.

Why embeddings are CRUCIAL:
- "reset password" and "change my login credentials" mean similar things
- Without embeddings, search only finds exact keyword matches
- With embeddings, search finds semantically similar content
- This is how RAG finds relevant documents for any question!

🎯 Real-World Analogy:
----------------------
Imagine a GPS coordinate system for MEANING instead of location:

- "I love pizza" → [40.7128, -74.0060, ...] (thousands of dimensions)
- "Pizza is delicious" → [40.7131, -74.0058, ...] (very close!)
- "The weather is nice" → [51.5074, -0.1278, ...] (far away)

Just like cities close together on a map have similar coordinates,
texts with similar meanings have similar embedding vectors.

Gemini's embedding model uses 768 dimensions (not just 2!) to capture
the nuances of meaning in text.

🔒 Type Safety Benefit:
-----------------------
With Pydantic, we ensure:
- Embeddings are always lists of floats
- Dimension counts are correct (768 for text-embedding-004)
- Token counts don't exceed limits
- All metadata is properly typed


💻 CODE IMPLEMENTATION
=====================
"""

import os
from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
import google.generativeai as genai


class EmbeddingConfig(BaseModel):
    """
    Configuration for the Gemini embedding model.
    """
    
    model_name: str = Field(
        default="models/text-embedding-004",
        description="Gemini embedding model to use"
    )
    
    task_type: str = Field(
        default="retrieval_document",
        description="Task type affects embedding optimization"
    )
    
    output_dimensionality: Optional[int] = Field(
        default=None,
        ge=1,
        le=768,
        description="Output dimensions (None = full 768)"
    )
    
    @field_validator('task_type')
    @classmethod
    def validate_task_type(cls, v: str) -> str:
        """Validate task type is supported."""
        valid_types = {
            "retrieval_document",
            "retrieval_query",
            "semantic_similarity",
            "classification",
            "clustering"
        }
        if v not in valid_types:
            raise ValueError(
                f"Invalid task type: {v}. "
                f"Must be one of: {valid_types}"
            )
        return v


class EmbeddingVector(BaseModel):
    """
    A single embedding vector with metadata.
    This is what gets stored in the vector database.
    """
    
    embedding_id: str = Field(
        description="Unique identifier for this embedding"
    )
    
    source_text: str = Field(
        description="Original text that was embedded"
    )
    
    vector: list[float] = Field(
        description="The embedding vector (768 dimensions)"
    )
    
    model_used: str = Field(
        description="Model that created this embedding"
    )
    
    dimensions: int = Field(
        ge=1,
        description="Number of dimensions in the vector"
    )
    
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="When this embedding was created"
    )
    
    task_type: str = Field(
        description="Task type used for embedding"
    )
    
    @field_validator('vector')
    @classmethod
    def validate_vector(cls, v: list[float]) -> list[float]:
        """Ensure vector contains valid floats."""
        if not v:
            raise ValueError("Embedding vector cannot be empty")
        for i, val in enumerate(v[:10]):
            if not isinstance(val, (int, float)):
                raise ValueError(f"Vector element {i} is not a number")
        return v


class BatchEmbeddingResult(BaseModel):
    """
    Result of embedding multiple texts at once.
    Batch embedding is more efficient than one-by-one.
    """
    
    embeddings: list[EmbeddingVector] = Field(
        description="List of embedding results"
    )
    
    total_texts: int = Field(
        ge=0,
        description="Number of texts embedded"
    )
    
    model_used: str = Field(
        description="Model used for embeddings"
    )
    
    processing_time_ms: float = Field(
        ge=0,
        description="Time taken in milliseconds"
    )
    
    total_characters: int = Field(
        ge=0,
        description="Total characters processed"
    )


class DocumentMetadata(BaseModel):
    """Metadata for document chunks."""
    source: str
    title: Optional[str] = None
    chunk_index: Optional[int] = None


class DocumentChunk(BaseModel):
    """A chunk of document to be embedded."""
    chunk_id: str
    document_id: str
    content: str
    metadata: DocumentMetadata


class EmbeddedChunk(BaseModel):
    """
    A document chunk with its embedding.
    This is the final form before storage in vector DB.
    """
    
    chunk: DocumentChunk = Field(
        description="The original chunk"
    )
    
    embedding: list[float] = Field(
        description="The embedding vector"
    )
    
    embedding_model: str = Field(
        description="Model used for embedding"
    )


class GeminiEmbedder:
    """
    Main class for creating embeddings using Google's Gemini API.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[EmbeddingConfig] = None
    ):
        """
        Initialize the embedder.
        
        Args:
            api_key: Google AI API key (or set GOOGLE_API_KEY env var)
            config: Embedding configuration
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.config = config or EmbeddingConfig()
        
        if self.api_key:
            genai.configure(api_key=self.api_key)
        
        self.model_name = self.config.model_name
    
    def embed_text(
        self,
        text: str,
        embedding_id: str,
        task_type: Optional[str] = None
    ) -> EmbeddingVector:
        """
        Create an embedding for a single text.
        
        Args:
            text: The text to embed
            embedding_id: Unique ID for this embedding
            task_type: Override default task type
        """
        task = task_type or self.config.task_type
        
        result = genai.embed_content(
            model=self.model_name,
            content=text,
            task_type=task,
            output_dimensionality=self.config.output_dimensionality
        )
        
        vector = result['embedding']
        
        return EmbeddingVector(
            embedding_id=embedding_id,
            source_text=text,
            vector=vector,
            model_used=self.model_name,
            dimensions=len(vector),
            task_type=task
        )
    
    def embed_batch(
        self,
        texts: list[str],
        id_prefix: str = "emb"
    ) -> BatchEmbeddingResult:
        """
        Create embeddings for multiple texts efficiently.
        
        Args:
            texts: List of texts to embed
            id_prefix: Prefix for generated embedding IDs
        """
        import time
        
        start_time = time.time()
        embeddings = []
        
        for i, text in enumerate(texts):
            emb_id = f"{id_prefix}_{i}"
            embedding = self.embed_text(text, emb_id)
            embeddings.append(embedding)
        
        processing_time = (time.time() - start_time) * 1000
        total_chars = sum(len(t) for t in texts)
        
        return BatchEmbeddingResult(
            embeddings=embeddings,
            total_texts=len(texts),
            model_used=self.model_name,
            processing_time_ms=processing_time,
            total_characters=total_chars
        )
    
    def embed_chunks(
        self,
        chunks: list[DocumentChunk]
    ) -> list[EmbeddedChunk]:
        """
        Embed a list of document chunks.
        This is the main method for RAG ingestion.
        """
        embedded_chunks = []
        
        for chunk in chunks:
            result = genai.embed_content(
                model=self.model_name,
                content=chunk.content,
                task_type="retrieval_document",
                output_dimensionality=self.config.output_dimensionality
            )
            
            embedded_chunk = EmbeddedChunk(
                chunk=chunk,
                embedding=result['embedding'],
                embedding_model=self.model_name
            )
            embedded_chunks.append(embedded_chunk)
        
        return embedded_chunks
    
    def embed_query(self, query: str) -> list[float]:
        """
        Embed a search query.
        Uses "retrieval_query" task type for optimal search.
        """
        result = genai.embed_content(
            model=self.model_name,
            content=query,
            task_type="retrieval_query",
            output_dimensionality=self.config.output_dimensionality
        )
        
        return result['embedding']


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """
    Calculate cosine similarity between two vectors.
    Returns a value between -1 and 1 (1 = identical, 0 = unrelated).
    """
    import math
    
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))
    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    
    return dot_product / (magnitude1 * magnitude2)


def demonstrate_embeddings_simulated():
    """
    Demonstrate embedding concepts with simulated data.
    (No API key required for this demonstration)
    """
    import random
    import math
    
    print("=" * 70)
    print("GEMINI EMBEDDINGS API DEMONSTRATION")
    print("(Simulated - No API Key Required)")
    print("=" * 70)
    
    print("\n📊 UNDERSTANDING EMBEDDINGS")
    print("-" * 50)
    
    print("""
Embeddings convert text to vectors (lists of numbers) that capture meaning:

Text: "How do I reset my password?"
   ↓
Embedding: [0.023, -0.156, 0.891, 0.042, ...] (768 numbers)

Similar meanings → Similar vectors → Found via similarity search!
""")
    
    print("\n" + "=" * 70)
    print("TASK TYPES EXPLAINED")
    print("=" * 70)
    
    task_types = {
        "retrieval_document": "Optimized for documents being searched",
        "retrieval_query": "Optimized for search queries",
        "semantic_similarity": "For comparing text similarity",
        "classification": "For categorization tasks",
        "clustering": "For grouping similar items"
    }
    
    print("\nGemini supports different task types that optimize embeddings:\n")
    for task, description in task_types.items():
        print(f"  • {task}")
        print(f"    {description}\n")
    
    print("\n" + "=" * 70)
    print("SIMULATED SIMILARITY DEMONSTRATION")
    print("=" * 70)
    
    def create_pseudo_embedding(text: str) -> list[float]:
        """Create deterministic pseudo-embedding based on text features."""
        random.seed(hash(text.lower()) % 2**32)
        base = [random.gauss(0, 0.3) for _ in range(768)]
        
        keywords = {
            "password": ([0, 1, 2], 0.8),
            "reset": ([3, 4, 5], 0.7),
            "login": ([0, 1, 6], 0.6),
            "account": ([1, 7, 8], 0.5),
            "weather": ([100, 101, 102], 0.9),
            "rain": ([103, 104, 105], 0.7),
            "sun": ([106, 107, 108], 0.7),
            "python": ([200, 201, 202], 0.8),
            "code": ([203, 204, 205], 0.7),
            "programming": ([200, 206, 207], 0.6),
        }
        
        for keyword, (indices, strength) in keywords.items():
            if keyword in text.lower():
                for idx in indices:
                    base[idx] += strength
        
        magnitude = math.sqrt(sum(x**2 for x in base))
        return [x / magnitude for x in base]
    
    texts = [
        "How do I reset my password?",
        "I need to change my login credentials",
        "What's the weather like today?",
        "Will it rain tomorrow?",
        "How to write Python code",
        "Programming tutorial for beginners"
    ]
    
    print("\nSample texts and their similarity relationships:\n")
    
    embeddings = {text: create_pseudo_embedding(text) for text in texts}
    
    for i, text1 in enumerate(texts):
        print(f"\n📝 '{text1}'")
        print("   Similarities:")
        
        similarities = []
        for j, text2 in enumerate(texts):
            if i != j:
                sim = cosine_similarity(embeddings[text1], embeddings[text2])
                similarities.append((text2, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        for text2, sim in similarities[:2]:
            print(f"   → {sim:.3f} with '{text2}'")
    
    print("\n" + "=" * 70)
    print("EMBEDDING VECTOR STRUCTURE")
    print("=" * 70)
    
    sample_embedding = EmbeddingVector(
        embedding_id="emb_001",
        source_text="How do I reset my password?",
        vector=embeddings["How do I reset my password?"],
        model_used="models/text-embedding-004",
        dimensions=768,
        task_type="retrieval_document"
    )
    
    print(f"""
EmbeddingVector Structure:
  embedding_id: {sample_embedding.embedding_id}
  source_text: "{sample_embedding.source_text}"
  model_used: {sample_embedding.model_used}
  dimensions: {sample_embedding.dimensions}
  task_type: {sample_embedding.task_type}
  vector: [{sample_embedding.vector[0]:.4f}, {sample_embedding.vector[1]:.4f}, ... {sample_embedding.vector[-1]:.4f}]
""")
    
    print("\n" + "=" * 70)
    print("SEARCH SIMULATION")
    print("=" * 70)
    
    documents = [
        "To reset your password, click the 'Forgot Password' link on the login page.",
        "Account security best practices include using strong passwords.",
        "The weather forecast shows sunny skies for the weekend.",
        "Python is a popular programming language for data science.",
        "Login issues can often be resolved by clearing browser cookies.",
    ]
    
    doc_embeddings = {doc: create_pseudo_embedding(doc) for doc in documents}
    
    query = "I can't remember my password"
    query_embedding = create_pseudo_embedding(query)
    
    print(f"\n🔍 Query: '{query}'")
    print("\nSearching documents...\n")
    
    results = []
    for doc, emb in doc_embeddings.items():
        sim = cosine_similarity(query_embedding, emb)
        results.append((doc, sim))
    
    results.sort(key=lambda x: x[1], reverse=True)
    
    print("Results (ranked by similarity):\n")
    for i, (doc, sim) in enumerate(results, 1):
        bar = "█" * int(sim * 20)
        print(f"  {i}. [{sim:.3f}] {bar}")
        print(f"     {doc[:70]}...")
        print()
    
    print("\n" + "=" * 70)
    print("CONFIGURATION VALIDATION")
    print("=" * 70)
    
    print("\n Testing configuration validation...")
    
    valid_config = EmbeddingConfig(
        model_name="models/text-embedding-004",
        task_type="retrieval_document",
        output_dimensionality=256
    )
    print(f"\n✅ Valid config created:")
    print(f"   Model: {valid_config.model_name}")
    print(f"   Task: {valid_config.task_type}")
    print(f"   Dimensions: {valid_config.output_dimensionality}")
    
    try:
        bad_config = EmbeddingConfig(
            task_type="invalid_type"
        )
    except Exception as e:
        print(f"\n✅ Caught invalid task type!")
        print(f"   Error: {e}")
    
    try:
        bad_config = EmbeddingConfig(
            output_dimensionality=10000
        )
    except Exception as e:
        print(f"\n✅ Caught invalid dimensions!")
        print(f"   Error: {e}")


def demonstrate_embeddings_live():
    """
    Demonstrate embeddings with actual Gemini API.
    Requires GOOGLE_API_KEY environment variable.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        print("\n⚠️  No GOOGLE_API_KEY found. Running simulated demo instead.")
        print("   To use real embeddings, set: export GOOGLE_API_KEY='your-key'")
        demonstrate_embeddings_simulated()
        return
    
    print("=" * 70)
    print("GEMINI EMBEDDINGS API - LIVE DEMONSTRATION")
    print("=" * 70)
    
    embedder = GeminiEmbedder(api_key=api_key)
    
    print("\n📊 Creating embeddings with Gemini...")
    
    texts = [
        "How do I reset my password?",
        "I need to change my login credentials",
        "What's the weather like today?",
    ]
    
    embeddings = []
    for i, text in enumerate(texts):
        emb = embedder.embed_text(text, f"demo_{i}")
        embeddings.append(emb)
        print(f"\n✅ Embedded: '{text}'")
        print(f"   Dimensions: {emb.dimensions}")
        print(f"   First 5 values: {emb.vector[:5]}")
    
    print("\n📊 Similarity Analysis:")
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sim = cosine_similarity(embeddings[i].vector, embeddings[j].vector)
            print(f"\n   '{texts[i][:30]}...'")
            print(f"   vs '{texts[j][:30]}...'")
            print(f"   Similarity: {sim:.4f}")
    
    print("\n🔍 Testing query embedding...")
    query = "password help"
    query_emb = embedder.embed_query(query)
    print(f"   Query: '{query}'")
    print(f"   Dimensions: {len(query_emb)}")
    
    print("\n   Finding most similar document...")
    best_sim = 0
    best_text = ""
    for text, emb in zip(texts, embeddings):
        sim = cosine_similarity(query_emb, emb.vector)
        if sim > best_sim:
            best_sim = sim
            best_text = text
    
    print(f"   Best match: '{best_text}'")
    print(f"   Similarity: {best_sim:.4f}")


if __name__ == "__main__":
    demonstrate_embeddings_simulated()
    print("\n" + "=" * 70)
    print("To run with real Gemini API, set GOOGLE_API_KEY and call:")
    print("demonstrate_embeddings_live()")
    print("=" * 70)


"""
🧪 TEST & APPLY
===============

How to Test It:
---------------
Without API key (simulated demo):
    python lesson_04_gemini_embeddings_api.py

With API key (real embeddings):
    export GOOGLE_API_KEY='your-api-key'
    python lesson_04_gemini_embeddings_api.py

Expected Output:
----------------
1. Explanation of embeddings and task types
2. Simulated similarity demonstration
3. Search simulation with ranked results
4. Configuration validation examples


⚠️ COMMON STUMBLING BLOCKS
==========================

1. "API key not found" error
   
   Error: google.api_core.exceptions.DefaultCredentialsError
   
   Fix: Set your API key:
   export GOOGLE_API_KEY='your-key-here'
   
   Get a key at: https://makersuite.google.com/app/apikey

2. "Invalid task_type" validation error
   
   Error: ValidationError: Invalid task type: my_task
   
   Fix: Use one of the valid task types:
   - retrieval_document (for documents)
   - retrieval_query (for queries)
   - semantic_similarity
   - classification
   - clustering

3. "Why are my similarity scores low?"
   
   Possible causes:
   - Texts are genuinely unrelated
   - Task type mismatch (use retrieval_document for docs, retrieval_query for queries)
   - Very short texts have less semantic signal
   
   Fix: Ensure task types match your use case.

4. "Rate limit exceeded"
   
   Error: google.api_core.exceptions.ResourceExhausted
   
   Gemini has rate limits. Fix:
   - Add delays between requests
   - Use batch embedding when possible
   - Upgrade to paid tier for higher limits


🎯 KEY TAKEAWAYS
================

1. Embeddings Capture Meaning
   - Not just keywords, but semantic content
   - Similar meanings → similar vectors
   - Enables semantic search

2. Task Types Matter
   - Use retrieval_document for indexing documents
   - Use retrieval_query for search queries
   - This asymmetry improves search quality!

3. Gemini's text-embedding-004
   - 768 dimensions by default
   - Can reduce dimensions for faster search
   - Very capable for most RAG tasks

4. Cosine Similarity
   - Standard way to compare embeddings
   - Range: -1 to 1 (1 = identical)
   - Fast to compute


📚 NEXT LESSON
==============
In Lesson 5, we'll learn Vector Store Integration - storing our
embeddings in ChromaDB for lightning-fast similarity search!
"""
