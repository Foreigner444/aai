"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    LESSON 1: UNDERSTANDING RAG ARCHITECTURE                  ║
║                                                                              ║
║            Building AI That Answers Questions From Your Documents            ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

📚 CONCEPT OVERVIEW
==================

What & Why:
-----------
RAG (Retrieval Augmented Generation) is a technique that makes AI smarter by 
giving it access to YOUR specific documents and data. Instead of relying only 
on what the AI was trained on, RAG retrieves relevant information from your 
documents and uses it to generate accurate, grounded answers.

This is CRUCIAL because:
- AI models have knowledge cutoffs (they don't know recent events)
- AI models don't know YOUR specific data (company docs, research papers, etc.)
- AI models can "hallucinate" - make up information that sounds right but isn't
- RAG grounds the AI's responses in REAL documents you provide

🎯 Real-World Analogy:
----------------------
Imagine you're an expert being asked questions on a TV quiz show. Without RAG,
you can only answer from memory - sometimes you're wrong, sometimes you can't 
remember, and you definitely don't know things that happened after your last
"update."

With RAG, it's like having a research assistant who:
1. Hears the question
2. Runs to a library of relevant books
3. Finds the most relevant pages
4. Hands them to you before you answer

Now you can give accurate, well-sourced answers because you're reading from
actual documents, not just guessing from memory!

🔒 Type Safety Benefit:
-----------------------
Pydantic AI ensures that every step of the RAG pipeline produces validated,
typed outputs. When your AI retrieves documents and generates answers, you
KNOW the structure is correct, citations are properly formatted, and confidence
scores are valid numbers. No surprises in production!


📊 THE RAG ARCHITECTURE
======================

Here's the complete RAG pipeline we'll build:

┌─────────────────────────────────────────────────────────────────────────────┐
│                          RAG SYSTEM ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   INGESTION PHASE (One-time, when you add documents):                       │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────┐         │
│   │ Documents │───▶│ Chunking │───▶│Embedding │───▶│ Vector Store │         │
│   │  (PDFs,   │    │  (Split  │    │ (Convert │    │   (Store     │         │
│   │   Text)   │    │ to small │    │ to math  │    │  searchable  │         │
│   └──────────┘    │  pieces) │    │ vectors) │    │   vectors)   │         │
│                    └──────────┘    └──────────┘    └──────────────┘         │
│                                                                             │
│   QUERY PHASE (Every time user asks a question):                            │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────┐         │
│   │  User    │───▶│ Embed    │───▶│ Search   │───▶│   Retrieve   │         │
│   │ Question │    │ Question │    │  Vector  │    │   Relevant   │         │
│   └──────────┘    └──────────┘    │  Store   │    │   Chunks     │         │
│                                    └──────────┘    └──────┬───────┘         │
│                                                           │                 │
│                                                           ▼                 │
│   ┌──────────┐    ┌──────────────────────────────────────────────┐         │
│   │  Answer  │◀───│  Gemini + Retrieved Context + System Prompt  │         │
│   │   with   │    │                                              │         │
│   │Citations │    │  "Using these documents, answer the question" │         │
│   └──────────┘    └──────────────────────────────────────────────┘         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘


🧩 KEY COMPONENTS
=================

Let's understand each component we'll build:

1. DOCUMENT PROCESSOR
   - Takes raw documents (PDF, TXT, Markdown)
   - Extracts clean text content
   - Preserves metadata (source, page numbers, dates)

2. TEXT CHUNKER
   - Splits large documents into smaller pieces
   - Why? Embeddings work best with focused content
   - Maintains context with overlapping chunks

3. EMBEDDING MODEL (Gemini's text-embedding-004)
   - Converts text to mathematical vectors (lists of numbers)
   - Similar text → similar vectors (close in "vector space")
   - This is how we find relevant content!

4. VECTOR STORE (ChromaDB - simple and local)
   - Database optimized for vector similarity search
   - Stores chunks + their embeddings + metadata
   - Lightning-fast similarity search

5. RETRIEVER
   - Takes a question
   - Converts it to a vector
   - Finds most similar document chunks
   - Returns them with metadata

6. RAG AGENT (Pydantic AI + Gemini)
   - Receives question + retrieved context
   - Generates answer using Gemini
   - Returns structured response with citations
   - All validated by Pydantic!


💻 CODE IMPLEMENTATION
=====================
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class RAGArchitectureOverview(BaseModel):
    """
    This model represents the conceptual architecture of our RAG system.
    We use Pydantic to define clear, typed structures from the very beginning.
    """
    
    component_name: str = Field(
        description="Name of the RAG component"
    )
    
    purpose: str = Field(
        description="What this component does in the pipeline"
    )
    
    input_type: str = Field(
        description="What kind of data goes into this component"
    )
    
    output_type: str = Field(
        description="What kind of data comes out of this component"
    )
    
    gemini_model_used: Optional[str] = Field(
        default=None,
        description="Which Gemini model powers this component, if any"
    )


class DocumentMetadata(BaseModel):
    """
    Every document in our system will have metadata.
    This helps us track sources and provide proper citations.
    """
    
    source: str = Field(
        description="Original source of the document (filename, URL, etc.)"
    )
    
    title: Optional[str] = Field(
        default=None,
        description="Document title if available"
    )
    
    author: Optional[str] = Field(
        default=None,
        description="Document author if known"
    )
    
    created_date: Optional[datetime] = Field(
        default=None,
        description="When the document was created"
    )
    
    page_number: Optional[int] = Field(
        default=None,
        description="Page number for PDFs"
    )
    
    chunk_index: Optional[int] = Field(
        default=None,
        description="Index of this chunk within the document"
    )


class DocumentChunk(BaseModel):
    """
    A chunk is a piece of a document that's small enough to embed
    and retrieve effectively. This is the core unit of our RAG system.
    """
    
    chunk_id: str = Field(
        description="Unique identifier for this chunk"
    )
    
    content: str = Field(
        description="The actual text content of this chunk"
    )
    
    metadata: DocumentMetadata = Field(
        description="Metadata about where this chunk came from"
    )
    
    token_count: Optional[int] = Field(
        default=None,
        description="Approximate number of tokens in this chunk"
    )


class RetrievedContext(BaseModel):
    """
    When we search for relevant documents, we get back retrieved context.
    This includes the chunks and how similar they are to the query.
    """
    
    chunks: list[DocumentChunk] = Field(
        description="List of relevant document chunks"
    )
    
    similarity_scores: list[float] = Field(
        description="How similar each chunk is to the query (0-1)"
    )
    
    query: str = Field(
        description="The original query that was searched"
    )


class RAGResponse(BaseModel):
    """
    The final output of our RAG system - a validated, structured response
    with proper citations and confidence scoring.
    """
    
    answer: str = Field(
        description="The generated answer to the user's question"
    )
    
    citations: list[DocumentMetadata] = Field(
        description="Sources used to generate this answer"
    )
    
    confidence_score: float = Field(
        ge=0.0,
        le=1.0,
        description="How confident the system is in this answer (0-1)"
    )
    
    retrieved_chunks_count: int = Field(
        description="Number of document chunks used for context"
    )
    
    model_used: str = Field(
        description="Which Gemini model generated this response"
    )


def demonstrate_architecture():
    """
    Let's create some example instances to see how our types work together.
    """
    
    components = [
        RAGArchitectureOverview(
            component_name="Document Processor",
            purpose="Extract and clean text from various document formats",
            input_type="Raw files (PDF, TXT, MD)",
            output_type="Clean text with metadata",
            gemini_model_used=None
        ),
        RAGArchitectureOverview(
            component_name="Text Chunker",
            purpose="Split documents into smaller, focused pieces",
            input_type="Full document text",
            output_type="List of DocumentChunk objects",
            gemini_model_used=None
        ),
        RAGArchitectureOverview(
            component_name="Embedding Generator",
            purpose="Convert text to mathematical vectors for similarity search",
            input_type="Text strings",
            output_type="Vector embeddings (768 dimensions)",
            gemini_model_used="text-embedding-004"
        ),
        RAGArchitectureOverview(
            component_name="Vector Store",
            purpose="Store and search document embeddings efficiently",
            input_type="Embeddings + metadata",
            output_type="Similar documents for a query",
            gemini_model_used=None
        ),
        RAGArchitectureOverview(
            component_name="RAG Agent",
            purpose="Generate answers using retrieved context",
            input_type="Question + relevant document chunks",
            output_type="RAGResponse with citations",
            gemini_model_used="gemini-1.5-flash"
        ),
    ]
    
    print("=" * 70)
    print("RAG SYSTEM COMPONENTS")
    print("=" * 70)
    
    for i, component in enumerate(components, 1):
        print(f"\n{i}. {component.component_name}")
        print(f"   Purpose: {component.purpose}")
        print(f"   Input:   {component.input_type}")
        print(f"   Output:  {component.output_type}")
        if component.gemini_model_used:
            print(f"   Model:   {component.gemini_model_used}")
    
    print("\n" + "=" * 70)
    print("EXAMPLE: Document Processing Flow")
    print("=" * 70)
    
    metadata = DocumentMetadata(
        source="company_handbook.pdf",
        title="Employee Handbook 2024",
        author="HR Department",
        created_date=datetime(2024, 1, 15),
        page_number=12,
        chunk_index=3
    )
    
    chunk = DocumentChunk(
        chunk_id="handbook_p12_c3",
        content="Employees are entitled to 20 days of paid vacation per year. "
                "Unused vacation days can be carried over to the next year, "
                "up to a maximum of 5 days. Vacation requests should be "
                "submitted at least two weeks in advance.",
        metadata=metadata,
        token_count=52
    )
    
    print(f"\nDocument: {chunk.metadata.title}")
    print(f"Source: {chunk.metadata.source}, Page {chunk.metadata.page_number}")
    print(f"Chunk ID: {chunk.chunk_id}")
    print(f"Content Preview: {chunk.content[:100]}...")
    print(f"Token Count: {chunk.token_count}")
    
    print("\n" + "=" * 70)
    print("EXAMPLE: RAG Response")
    print("=" * 70)
    
    response = RAGResponse(
        answer="According to the Employee Handbook, you are entitled to 20 days "
               "of paid vacation per year. You can carry over up to 5 unused days "
               "to the next year. Make sure to submit your vacation request at "
               "least two weeks in advance.",
        citations=[metadata],
        confidence_score=0.92,
        retrieved_chunks_count=3,
        model_used="gemini-1.5-flash"
    )
    
    print(f"\nQuestion: How many vacation days do I get?")
    print(f"\nAnswer: {response.answer}")
    print(f"\nCitations:")
    for citation in response.citations:
        print(f"  - {citation.title}, Page {citation.page_number}")
    print(f"\nConfidence: {response.confidence_score:.0%}")
    print(f"Chunks Used: {response.retrieved_chunks_count}")
    print(f"Model: {response.model_used}")


if __name__ == "__main__":
    demonstrate_architecture()


"""
🧪 TEST & APPLY
===============

How to Test It:
---------------
Run this file to see the architecture demonstration:

    python lesson_01_understanding_rag_architecture.py

Expected Output:
----------------
You'll see:
1. A list of all RAG components with their purposes
2. An example of a processed document chunk
3. An example of a complete RAG response with citations

This shows you the structure we'll build - all with proper types!


⚠️ COMMON STUMBLING BLOCKS
==========================

1. "Why not just give Gemini all my documents?"
   
   Problem: Gemini has a context window limit (even though it's huge at 1M+ tokens).
   More importantly, throwing everything at the model:
   - Wastes tokens (costs money)
   - Dilutes relevant information with noise
   - Makes it harder for the model to find what matters
   
   Solution: RAG retrieves ONLY the relevant parts, making responses faster,
   cheaper, and more accurate.

2. "What's the difference between embeddings and the actual text?"
   
   Embeddings are mathematical representations (lists of ~768 numbers) that
   capture the MEANING of text. Similar meanings = similar numbers.
   
   Text: "How do I reset my password?"
   Embedding: [0.023, -0.156, 0.891, ...] (768 numbers)
   
   We can compare embeddings mathematically to find similar content,
   even if the exact words are different!

3. "Why do we need Pydantic models for everything?"
   
   Without Pydantic:
   - No guarantee the AI returns proper citations
   - Confidence scores could be invalid (like 1.5 or "high")
   - Metadata might be missing or malformed
   - Errors happen at runtime, in production
   
   With Pydantic:
   - Every response is validated automatically
   - Types are checked before your code uses them
   - Errors are caught immediately with clear messages
   - Your IDE provides autocomplete for everything


🎯 KEY TAKEAWAYS
================

1. RAG = Retrieval + Generation
   - Retrieve relevant documents first
   - Then generate answers using that context

2. The Pipeline:
   Documents → Chunks → Embeddings → Vector Store → Retrieval → Generation

3. Type Safety Throughout:
   - Every component has typed inputs and outputs
   - Pydantic validates at every step
   - No surprises in production

4. Gemini's Role:
   - text-embedding-004: Creates embeddings for similarity search
   - gemini-1.5-flash/pro: Generates answers from retrieved context


📚 NEXT LESSON
==============
In Lesson 2, we'll build the Document Processing Pipeline - taking raw
documents and preparing them for our RAG system!
"""
