"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    LESSON 3: TEXT CHUNKING STRATEGIES                        ║
║                                                                              ║
║           Splitting Documents into Optimal Pieces for Retrieval              ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

📚 CONCEPT OVERVIEW
==================

What & Why:
-----------
Text chunking is the process of splitting large documents into smaller pieces
(chunks) that are optimal for embedding and retrieval. This is a CRITICAL step
because chunk size and strategy directly impact your RAG system's quality.

Why chunking matters:
- Embeddings work best on focused, coherent text (not 50-page documents)
- Smaller chunks = more precise retrieval
- Too small = lose context; Too large = dilute relevance
- Good chunking preserves meaning at chunk boundaries

🎯 Real-World Analogy:
----------------------
Imagine you're organizing a massive recipe book for quick lookup. You could:

1. Keep it as one huge book (bad) - Can't find anything quickly
2. Tear out random pieces (bad) - Ingredients on one page, instructions elsewhere
3. Cut each recipe into its own card, with slight overlap at the edges so 
   continued instructions make sense (good!) - Each card is self-contained
   but references connect them.

That's exactly what we do with text chunking! We create focused pieces that
make sense on their own while maintaining connections through overlap.

🔒 Type Safety Benefit:
-----------------------
Our Pydantic models ensure:
- Chunk sizes are within specified limits
- Overlap percentages are valid (0-100%)
- Each chunk has required metadata
- Token counts are positive integers


💻 CODE IMPLEMENTATION
=====================
"""

import re
import uuid
from typing import Optional, Literal
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, model_validator


class ChunkingConfig(BaseModel):
    """
    Configuration for how documents should be chunked.
    Different use cases need different chunking strategies.
    """
    
    chunk_size: int = Field(
        default=512,
        ge=100,
        le=8000,
        description="Target size of each chunk in characters"
    )
    
    chunk_overlap: int = Field(
        default=50,
        ge=0,
        description="Number of characters to overlap between chunks"
    )
    
    min_chunk_size: int = Field(
        default=100,
        ge=50,
        description="Minimum chunk size - smaller chunks are merged"
    )
    
    strategy: Literal["fixed", "sentence", "paragraph", "semantic"] = Field(
        default="sentence",
        description="Chunking strategy to use"
    )
    
    preserve_sentences: bool = Field(
        default=True,
        description="Avoid cutting in the middle of sentences"
    )
    
    @model_validator(mode='after')
    def validate_overlap(self) -> 'ChunkingConfig':
        """Ensure overlap is less than chunk size."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"Overlap ({self.chunk_overlap}) must be less than "
                f"chunk size ({self.chunk_size})"
            )
        return self


class DocumentMetadata(BaseModel):
    """Metadata from the original document."""
    
    source: str = Field(description="Original source of the document")
    title: Optional[str] = Field(default=None, description="Document title")
    author: Optional[str] = Field(default=None, description="Document author")
    created_date: Optional[datetime] = Field(default=None)


class DocumentChunk(BaseModel):
    """
    A single chunk of a document, ready for embedding.
    This is the fundamental unit that gets stored in our vector database.
    """
    
    chunk_id: str = Field(
        description="Unique identifier for this chunk"
    )
    
    document_id: str = Field(
        description="ID of the parent document"
    )
    
    content: str = Field(
        min_length=1,
        description="The text content of this chunk"
    )
    
    metadata: DocumentMetadata = Field(
        description="Metadata from the parent document"
    )
    
    chunk_index: int = Field(
        ge=0,
        description="Position of this chunk within the document (0-indexed)"
    )
    
    total_chunks: int = Field(
        ge=1,
        description="Total number of chunks in the parent document"
    )
    
    start_char: int = Field(
        ge=0,
        description="Starting character position in original document"
    )
    
    end_char: int = Field(
        ge=0,
        description="Ending character position in original document"
    )
    
    token_count_estimate: int = Field(
        ge=1,
        description="Estimated token count (chars / 4)"
    )
    
    @field_validator('content')
    @classmethod
    def strip_content(cls, v: str) -> str:
        """Clean up chunk content."""
        return v.strip()
    
    @property
    def char_count(self) -> int:
        """Get character count of content."""
        return len(self.content)


class ChunkingResult(BaseModel):
    """
    Result of chunking a document.
    Contains all chunks plus statistics about the operation.
    """
    
    document_id: str = Field(
        description="ID of the document that was chunked"
    )
    
    chunks: list[DocumentChunk] = Field(
        description="List of chunks created from the document"
    )
    
    config_used: ChunkingConfig = Field(
        description="Configuration used for chunking"
    )
    
    original_length: int = Field(
        ge=0,
        description="Length of original document in characters"
    )
    
    total_chunks: int = Field(
        ge=0,
        description="Number of chunks created"
    )
    
    avg_chunk_size: float = Field(
        ge=0,
        description="Average chunk size in characters"
    )
    
    processing_time_ms: float = Field(
        ge=0,
        description="Time taken to chunk the document in milliseconds"
    )


class TextChunker:
    """
    Main class for chunking documents using various strategies.
    """
    
    def __init__(self, config: Optional[ChunkingConfig] = None):
        """Initialize chunker with configuration."""
        self.config = config or ChunkingConfig()
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (roughly 4 characters per token)."""
        return max(1, len(text) // 4)
    
    def _find_sentence_boundary(self, text: str, position: int, search_range: int = 100) -> int:
        """
        Find the nearest sentence boundary to the given position.
        Returns the position after the sentence-ending punctuation.
        """
        search_start = max(0, position - search_range)
        search_end = min(len(text), position + search_range)
        search_text = text[search_start:search_end]
        
        sentence_endings = [
            (m.end() + search_start) for m in re.finditer(r'[.!?]+[\s"]', search_text)
        ]
        
        if not sentence_endings:
            return position
        
        closest = min(sentence_endings, key=lambda x: abs(x - position))
        return closest
    
    def _find_paragraph_boundary(self, text: str, position: int, search_range: int = 200) -> int:
        """Find the nearest paragraph boundary to the given position."""
        search_start = max(0, position - search_range)
        search_end = min(len(text), position + search_range)
        
        before_text = text[search_start:position]
        after_text = text[position:search_end]
        
        para_before = before_text.rfind('\n\n')
        para_after = after_text.find('\n\n')
        
        if para_before != -1 and para_after != -1:
            dist_before = position - (search_start + para_before)
            dist_after = para_after
            if dist_before < dist_after:
                return search_start + para_before + 2
            else:
                return position + para_after + 2
        elif para_before != -1:
            return search_start + para_before + 2
        elif para_after != -1:
            return position + para_after + 2
        
        return position
    
    def chunk_fixed(
        self,
        text: str,
        document_id: str,
        metadata: DocumentMetadata
    ) -> list[DocumentChunk]:
        """
        Simple fixed-size chunking.
        Splits text into equal-sized chunks with overlap.
        """
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = min(start + self.config.chunk_size, len(text))
            
            if self.config.preserve_sentences and end < len(text):
                end = self._find_sentence_boundary(text, end)
            
            chunk_content = text[start:end]
            
            if len(chunk_content.strip()) >= self.config.min_chunk_size:
                chunk = DocumentChunk(
                    chunk_id=f"{document_id}_chunk_{chunk_index}",
                    document_id=document_id,
                    content=chunk_content,
                    metadata=metadata,
                    chunk_index=chunk_index,
                    total_chunks=0,
                    start_char=start,
                    end_char=end,
                    token_count_estimate=self._estimate_tokens(chunk_content)
                )
                chunks.append(chunk)
                chunk_index += 1
            
            start = end - self.config.chunk_overlap
            if start <= chunks[-1].start_char if chunks else 0:
                start = end
        
        for chunk in chunks:
            object.__setattr__(chunk, 'total_chunks', len(chunks))
        
        return chunks
    
    def chunk_by_sentence(
        self,
        text: str,
        document_id: str,
        metadata: DocumentMetadata
    ) -> list[DocumentChunk]:
        """
        Chunk by sentence boundaries.
        Groups sentences until reaching target chunk size.
        """
        sentence_pattern = r'(?<=[.!?])\s+'
        sentences = re.split(sentence_pattern, text)
        
        chunks = []
        current_chunk = ""
        chunk_index = 0
        start_char = 0
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > self.config.chunk_size and current_chunk:
                chunk = DocumentChunk(
                    chunk_id=f"{document_id}_chunk_{chunk_index}",
                    document_id=document_id,
                    content=current_chunk.strip(),
                    metadata=metadata,
                    chunk_index=chunk_index,
                    total_chunks=0,
                    start_char=start_char,
                    end_char=start_char + len(current_chunk),
                    token_count_estimate=self._estimate_tokens(current_chunk)
                )
                chunks.append(chunk)
                chunk_index += 1
                
                if self.config.chunk_overlap > 0:
                    overlap_text = current_chunk[-self.config.chunk_overlap:]
                    start_char = start_char + len(current_chunk) - len(overlap_text)
                    current_chunk = overlap_text + " " + sentence
                else:
                    start_char = start_char + len(current_chunk)
                    current_chunk = sentence
            else:
                current_chunk += (" " if current_chunk else "") + sentence
        
        if current_chunk.strip() and len(current_chunk.strip()) >= self.config.min_chunk_size:
            chunk = DocumentChunk(
                chunk_id=f"{document_id}_chunk_{chunk_index}",
                document_id=document_id,
                content=current_chunk.strip(),
                metadata=metadata,
                chunk_index=chunk_index,
                total_chunks=0,
                start_char=start_char,
                end_char=start_char + len(current_chunk),
                token_count_estimate=self._estimate_tokens(current_chunk)
            )
            chunks.append(chunk)
        elif current_chunk.strip() and chunks:
            last_chunk = chunks[-1]
            merged_content = last_chunk.content + " " + current_chunk.strip()
            chunks[-1] = DocumentChunk(
                chunk_id=last_chunk.chunk_id,
                document_id=last_chunk.document_id,
                content=merged_content,
                metadata=last_chunk.metadata,
                chunk_index=last_chunk.chunk_index,
                total_chunks=0,
                start_char=last_chunk.start_char,
                end_char=start_char + len(current_chunk),
                token_count_estimate=self._estimate_tokens(merged_content)
            )
        
        for chunk in chunks:
            object.__setattr__(chunk, 'total_chunks', len(chunks))
        
        return chunks
    
    def chunk_by_paragraph(
        self,
        text: str,
        document_id: str,
        metadata: DocumentMetadata
    ) -> list[DocumentChunk]:
        """
        Chunk by paragraph boundaries.
        Each chunk contains one or more complete paragraphs.
        """
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        chunks = []
        current_chunk = ""
        chunk_index = 0
        start_char = 0
        char_position = 0
        
        for para in paragraphs:
            para_start = text.find(para, char_position)
            if para_start != -1:
                char_position = para_start
            
            if len(current_chunk) + len(para) > self.config.chunk_size and current_chunk:
                chunk = DocumentChunk(
                    chunk_id=f"{document_id}_chunk_{chunk_index}",
                    document_id=document_id,
                    content=current_chunk.strip(),
                    metadata=metadata,
                    chunk_index=chunk_index,
                    total_chunks=0,
                    start_char=start_char,
                    end_char=start_char + len(current_chunk),
                    token_count_estimate=self._estimate_tokens(current_chunk)
                )
                chunks.append(chunk)
                chunk_index += 1
                start_char = char_position
                current_chunk = para
            else:
                current_chunk += ("\n\n" if current_chunk else "") + para
        
        if current_chunk.strip():
            if len(current_chunk.strip()) >= self.config.min_chunk_size:
                chunk = DocumentChunk(
                    chunk_id=f"{document_id}_chunk_{chunk_index}",
                    document_id=document_id,
                    content=current_chunk.strip(),
                    metadata=metadata,
                    chunk_index=chunk_index,
                    total_chunks=0,
                    start_char=start_char,
                    end_char=len(text),
                    token_count_estimate=self._estimate_tokens(current_chunk)
                )
                chunks.append(chunk)
            elif chunks:
                last_chunk = chunks[-1]
                merged_content = last_chunk.content + "\n\n" + current_chunk.strip()
                chunks[-1] = DocumentChunk(
                    chunk_id=last_chunk.chunk_id,
                    document_id=last_chunk.document_id,
                    content=merged_content,
                    metadata=last_chunk.metadata,
                    chunk_index=last_chunk.chunk_index,
                    total_chunks=0,
                    start_char=last_chunk.start_char,
                    end_char=len(text),
                    token_count_estimate=self._estimate_tokens(merged_content)
                )
        
        for chunk in chunks:
            object.__setattr__(chunk, 'total_chunks', len(chunks))
        
        return chunks
    
    def chunk_document(
        self,
        text: str,
        document_id: str,
        metadata: DocumentMetadata
    ) -> ChunkingResult:
        """
        Chunk a document using the configured strategy.
        Returns a ChunkingResult with all chunks and statistics.
        """
        import time
        
        start_time = time.time()
        
        if self.config.strategy == "fixed":
            chunks = self.chunk_fixed(text, document_id, metadata)
        elif self.config.strategy == "sentence":
            chunks = self.chunk_by_sentence(text, document_id, metadata)
        elif self.config.strategy == "paragraph":
            chunks = self.chunk_by_paragraph(text, document_id, metadata)
        else:
            chunks = self.chunk_by_sentence(text, document_id, metadata)
        
        processing_time = (time.time() - start_time) * 1000
        
        avg_size = sum(c.char_count for c in chunks) / len(chunks) if chunks else 0
        
        return ChunkingResult(
            document_id=document_id,
            chunks=chunks,
            config_used=self.config,
            original_length=len(text),
            total_chunks=len(chunks),
            avg_chunk_size=avg_size,
            processing_time_ms=processing_time
        )


def demonstrate_chunking_strategies():
    """
    Demonstrate different chunking strategies with sample text.
    """
    
    sample_document = """
# Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing algorithms that can access data and use it to learn for themselves.

## Types of Machine Learning

There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning.

### Supervised Learning

Supervised learning is the most common type. In this approach, the algorithm learns from labeled training data. Each training example includes an input and the desired output. The algorithm learns to map inputs to outputs.

Common applications include spam detection, image classification, and price prediction. The model learns patterns from historical data and applies them to new, unseen data.

### Unsupervised Learning

Unsupervised learning works with unlabeled data. The algorithm tries to find patterns and structure in the data without guidance. Common techniques include clustering and dimensionality reduction.

Customer segmentation is a popular use case. The algorithm groups customers based on behavior without being told what groups to create.

### Reinforcement Learning

Reinforcement learning involves an agent learning through trial and error. The agent takes actions in an environment and receives rewards or penalties. Over time, it learns to maximize cumulative rewards.

Game playing AI and robotics commonly use reinforcement learning. The famous AlphaGo system used reinforcement learning to master the game of Go.

## Key Concepts

### Features and Labels

Features are the input variables used to make predictions. Labels are the output values we want to predict. Good feature engineering is crucial for model performance.

### Training and Testing

Data is typically split into training and testing sets. The model learns from training data and is evaluated on testing data it hasn't seen before. This helps assess how well the model generalizes.

### Overfitting and Underfitting

Overfitting occurs when a model learns the training data too well, including noise. It performs well on training data but poorly on new data. Underfitting happens when the model is too simple to capture underlying patterns.

## Conclusion

Machine learning has transformed many industries and continues to evolve rapidly. Understanding these fundamentals is essential for anyone working in data science or AI development.
"""

    metadata = DocumentMetadata(
        source="ml_introduction.md",
        title="Introduction to Machine Learning",
        author="AI Academy"
    )
    
    document_id = "ml_intro_001"
    
    print("=" * 70)
    print("TEXT CHUNKING STRATEGIES DEMONSTRATION")
    print("=" * 70)
    print(f"\nOriginal Document Length: {len(sample_document):,} characters")
    print(f"Estimated Tokens: ~{len(sample_document) // 4:,}")
    
    print("\n" + "=" * 70)
    print("STRATEGY 1: FIXED-SIZE CHUNKING")
    print("=" * 70)
    
    fixed_config = ChunkingConfig(
        chunk_size=500,
        chunk_overlap=50,
        strategy="fixed",
        preserve_sentences=True
    )
    fixed_chunker = TextChunker(fixed_config)
    fixed_result = fixed_chunker.chunk_document(sample_document, document_id, metadata)
    
    print(f"\nConfiguration:")
    print(f"  - Chunk Size: {fixed_config.chunk_size} chars")
    print(f"  - Overlap: {fixed_config.chunk_overlap} chars")
    print(f"  - Preserve Sentences: {fixed_config.preserve_sentences}")
    
    print(f"\nResults:")
    print(f"  - Total Chunks: {fixed_result.total_chunks}")
    print(f"  - Average Size: {fixed_result.avg_chunk_size:.0f} chars")
    print(f"  - Processing Time: {fixed_result.processing_time_ms:.2f}ms")
    
    print(f"\nChunk Sizes:")
    for chunk in fixed_result.chunks:
        print(f"  Chunk {chunk.chunk_index}: {chunk.char_count} chars, "
              f"~{chunk.token_count_estimate} tokens")
    
    print("\n" + "=" * 70)
    print("STRATEGY 2: SENTENCE-BASED CHUNKING")
    print("=" * 70)
    
    sentence_config = ChunkingConfig(
        chunk_size=500,
        chunk_overlap=50,
        strategy="sentence"
    )
    sentence_chunker = TextChunker(sentence_config)
    sentence_result = sentence_chunker.chunk_document(sample_document, document_id, metadata)
    
    print(f"\nResults:")
    print(f"  - Total Chunks: {sentence_result.total_chunks}")
    print(f"  - Average Size: {sentence_result.avg_chunk_size:.0f} chars")
    
    print(f"\nChunk Sizes:")
    for chunk in sentence_result.chunks:
        print(f"  Chunk {chunk.chunk_index}: {chunk.char_count} chars, "
              f"~{chunk.token_count_estimate} tokens")
    
    print("\n" + "=" * 70)
    print("STRATEGY 3: PARAGRAPH-BASED CHUNKING")
    print("=" * 70)
    
    para_config = ChunkingConfig(
        chunk_size=800,
        chunk_overlap=0,
        strategy="paragraph",
        min_chunk_size=100
    )
    para_chunker = TextChunker(para_config)
    para_result = para_chunker.chunk_document(sample_document, document_id, metadata)
    
    print(f"\nResults:")
    print(f"  - Total Chunks: {para_result.total_chunks}")
    print(f"  - Average Size: {para_result.avg_chunk_size:.0f} chars")
    
    print(f"\nChunk Sizes:")
    for chunk in para_result.chunks:
        print(f"  Chunk {chunk.chunk_index}: {chunk.char_count} chars, "
              f"~{chunk.token_count_estimate} tokens")
    
    print("\n" + "=" * 70)
    print("SAMPLE CHUNK CONTENT")
    print("=" * 70)
    
    print("\n📄 Sentence-Based Chunk 0:")
    print("-" * 50)
    print(sentence_result.chunks[0].content[:400] + "...")
    
    print(f"\n📝 Chunk Metadata:")
    print(f"  - Chunk ID: {sentence_result.chunks[0].chunk_id}")
    print(f"  - Document ID: {sentence_result.chunks[0].document_id}")
    print(f"  - Position: {sentence_result.chunks[0].chunk_index + 1} of {sentence_result.chunks[0].total_chunks}")
    print(f"  - Source: {sentence_result.chunks[0].metadata.source}")
    
    print("\n" + "=" * 70)
    print("VALIDATION EXAMPLES")
    print("=" * 70)
    
    print("\n Testing invalid configurations...")
    
    try:
        bad_config = ChunkingConfig(
            chunk_size=100,
            chunk_overlap=150
        )
    except Exception as e:
        print(f"\n✅ Caught overlap > chunk_size!")
        print(f"   Error: {e}")
    
    try:
        bad_config = ChunkingConfig(
            chunk_size=50
        )
    except Exception as e:
        print(f"\n✅ Caught chunk_size < minimum!")
        print(f"   Error: {e}")


if __name__ == "__main__":
    demonstrate_chunking_strategies()


"""
🧪 TEST & APPLY
===============

How to Test It:
---------------
    python lesson_03_text_chunking_strategies.py

Expected Output:
----------------
1. Fixed-size chunking results
2. Sentence-based chunking results  
3. Paragraph-based chunking results
4. Sample chunk content preview
5. Validation error demonstrations


⚠️ COMMON STUMBLING BLOCKS
==========================

1. "My chunks are too small/large"
   
   Fix: Adjust chunk_size based on your use case:
   - Q&A systems: 300-500 chars (focused retrieval)
   - Summarization: 800-1500 chars (more context)
   - Code: 500-1000 chars (preserve function boundaries)

2. "Chunks cut off mid-sentence"
   
   This happens when:
   - preserve_sentences=False
   - Sentences are very long
   
   Fix: Enable preserve_sentences and increase search range:
   boundary = self._find_sentence_boundary(text, end, search_range=200)

3. "Getting empty chunks"
   
   Error: ValidationError: String should have at least 1 character
   
   This is the validation working! Usually caused by:
   - Documents with lots of whitespace
   - Very small min_chunk_size
   
   Fix: Increase min_chunk_size or add whitespace normalization.

4. "Overlap doesn't seem to work"
   
   Ensure overlap < chunk_size (validated by Pydantic).
   The overlap creates continuity between chunks, which helps
   when information spans chunk boundaries.


🎯 KEY TAKEAWAYS
================

1. Strategy Matters
   - Fixed: Simple, predictable sizes
   - Sentence: Preserves linguistic boundaries  
   - Paragraph: Preserves semantic units

2. Size Guidelines
   - Too small: Lose context, more searches needed
   - Too large: Dilute relevance, slower processing
   - Sweet spot: 300-800 chars for most RAG systems

3. Overlap Creates Continuity
   - 10-20% overlap is common
   - Helps with information at boundaries
   - Trade-off: More chunks, more storage

4. Validation Prevents Issues
   - Pydantic catches invalid configs early
   - min_chunk_size prevents useless tiny chunks
   - Type safety throughout


📚 NEXT LESSON
==============
In Lesson 4, we'll learn about the Gemini Embeddings API - converting
our chunks into mathematical vectors for similarity search!
"""
