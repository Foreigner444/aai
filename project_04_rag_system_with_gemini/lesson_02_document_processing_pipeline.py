"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                  LESSON 2: DOCUMENT PROCESSING PIPELINE                      ║
║                                                                              ║
║              Extracting Clean Text from Various Document Formats             ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

📚 CONCEPT OVERVIEW
==================

What & Why:
-----------
Before we can search through documents, we need to process them into a clean,
consistent format. Documents come in many formats (PDF, Word, plain text, 
Markdown), and each needs special handling to extract the actual content.

This is CRUCIAL because:
- Raw PDFs contain formatting data, images, and hidden characters
- Different formats have different structures
- We need consistent text that can be chunked and embedded
- Metadata (like page numbers) must be preserved for citations

🎯 Real-World Analogy:
----------------------
Think of document processing like preparing ingredients for cooking:
- You start with whole vegetables (raw documents)
- You wash them (remove formatting artifacts)
- You peel them (extract just the text)
- You keep track of what's what (metadata)
- Now they're ready for chopping (chunking) and cooking (embedding)

Just like a chef can't cook with dirty, unpeeled vegetables, our RAG system
can't work effectively with unprocessed documents!

🔒 Type Safety Benefit:
-----------------------
Our Pydantic models ensure every processed document has:
- Valid content (non-empty string)
- Required metadata (source is always present)
- Proper types (dates are dates, numbers are numbers)

If a document fails processing, we know immediately - not when a user
asks a question and gets garbage!


💻 CODE IMPLEMENTATION
=====================
"""

import os
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator, model_validator


class DocumentMetadata(BaseModel):
    """
    Metadata about a document that we need for citations and tracking.
    This is the same model from Lesson 1 - consistency is key!
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
    
    file_type: Optional[str] = Field(
        default=None,
        description="Type of file (pdf, txt, md, etc.)"
    )
    
    file_size_bytes: Optional[int] = Field(
        default=None,
        ge=0,
        description="Size of the original file in bytes"
    )
    
    page_count: Optional[int] = Field(
        default=None,
        ge=1,
        description="Number of pages (for PDFs)"
    )


class ProcessedDocument(BaseModel):
    """
    A fully processed document ready for chunking.
    This is the output of our document processing pipeline.
    """
    
    document_id: str = Field(
        description="Unique identifier for this document"
    )
    
    content: str = Field(
        min_length=1,
        description="The extracted text content"
    )
    
    metadata: DocumentMetadata = Field(
        description="Metadata about the document"
    )
    
    processing_timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When this document was processed"
    )
    
    character_count: int = Field(
        ge=0,
        description="Number of characters in the content"
    )
    
    word_count: int = Field(
        ge=0,
        description="Approximate number of words"
    )
    
    @field_validator('content')
    @classmethod
    def clean_content(cls, v: str) -> str:
        """Remove excessive whitespace and normalize line breaks."""
        import re
        v = re.sub(r'\n{3,}', '\n\n', v)
        v = re.sub(r' {2,}', ' ', v)
        v = v.strip()
        return v
    
    @model_validator(mode='after')
    def calculate_counts(self) -> 'ProcessedDocument':
        """Automatically calculate character and word counts."""
        object.__setattr__(self, 'character_count', len(self.content))
        object.__setattr__(self, 'word_count', len(self.content.split()))
        return self


class ProcessingError(BaseModel):
    """
    When document processing fails, we capture the error in a structured way.
    This helps with debugging and monitoring.
    """
    
    source: str = Field(
        description="The document that failed to process"
    )
    
    error_type: str = Field(
        description="Type of error that occurred"
    )
    
    error_message: str = Field(
        description="Detailed error message"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When the error occurred"
    )
    
    recoverable: bool = Field(
        default=False,
        description="Whether this error might be fixed by retrying"
    )


class ProcessingResult(BaseModel):
    """
    The result of processing one or more documents.
    Contains both successes and failures.
    """
    
    successful: list[ProcessedDocument] = Field(
        default_factory=list,
        description="Successfully processed documents"
    )
    
    failed: list[ProcessingError] = Field(
        default_factory=list,
        description="Documents that failed processing"
    )
    
    total_processed: int = Field(
        default=0,
        description="Total number of documents attempted"
    )
    
    processing_time_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Time taken to process all documents"
    )
    
    @property
    def success_rate(self) -> float:
        """Calculate the percentage of successful processing."""
        if self.total_processed == 0:
            return 0.0
        return len(self.successful) / self.total_processed


class DocumentProcessor:
    """
    Main class for processing documents into a format ready for RAG.
    Handles multiple file types and extracts clean text with metadata.
    """
    
    SUPPORTED_EXTENSIONS = {'.txt', '.md', '.py', '.json', '.csv', '.html'}
    
    def __init__(self):
        self.processed_count = 0
    
    def generate_document_id(self, source: str, content: str) -> str:
        """
        Generate a unique ID for a document based on its source and content.
        This ensures the same document always gets the same ID.
        """
        hash_input = f"{source}:{content[:1000]}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]
    
    def process_text_file(self, file_path: Path) -> ProcessedDocument:
        """
        Process a plain text file.
        """
        content = file_path.read_text(encoding='utf-8')
        
        stat = file_path.stat()
        
        metadata = DocumentMetadata(
            source=str(file_path),
            title=file_path.stem,
            file_type=file_path.suffix.lstrip('.'),
            file_size_bytes=stat.st_size,
            created_date=datetime.fromtimestamp(stat.st_mtime)
        )
        
        doc_id = self.generate_document_id(str(file_path), content)
        
        return ProcessedDocument(
            document_id=doc_id,
            content=content,
            metadata=metadata,
            character_count=len(content),
            word_count=len(content.split())
        )
    
    def process_markdown_file(self, file_path: Path) -> ProcessedDocument:
        """
        Process a Markdown file, preserving structure but removing formatting.
        """
        import re
        
        content = file_path.read_text(encoding='utf-8')
        
        title = None
        title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        if title_match:
            title = title_match.group(1).strip()
        
        clean_content = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', content)
        clean_content = re.sub(r'!\[([^\]]*)\]\([^\)]+\)', r'[Image: \1]', clean_content)
        clean_content = re.sub(r'```[\s\S]*?```', '[Code Block]', clean_content)
        clean_content = re.sub(r'`([^`]+)`', r'\1', clean_content)
        clean_content = re.sub(r'\*\*([^*]+)\*\*', r'\1', clean_content)
        clean_content = re.sub(r'\*([^*]+)\*', r'\1', clean_content)
        
        stat = file_path.stat()
        
        metadata = DocumentMetadata(
            source=str(file_path),
            title=title or file_path.stem,
            file_type='markdown',
            file_size_bytes=stat.st_size,
            created_date=datetime.fromtimestamp(stat.st_mtime)
        )
        
        doc_id = self.generate_document_id(str(file_path), clean_content)
        
        return ProcessedDocument(
            document_id=doc_id,
            content=clean_content,
            metadata=metadata,
            character_count=len(clean_content),
            word_count=len(clean_content.split())
        )
    
    def process_string(
        self,
        content: str,
        source: str = "direct_input",
        title: Optional[str] = None
    ) -> ProcessedDocument:
        """
        Process a raw string directly (useful for testing and APIs).
        """
        metadata = DocumentMetadata(
            source=source,
            title=title,
            file_type='text',
            created_date=datetime.now()
        )
        
        doc_id = self.generate_document_id(source, content)
        
        return ProcessedDocument(
            document_id=doc_id,
            content=content,
            metadata=metadata,
            character_count=len(content),
            word_count=len(content.split())
        )
    
    def process_file(self, file_path: str | Path) -> ProcessedDocument:
        """
        Process any supported file type.
        Automatically detects the file type and uses the appropriate processor.
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        extension = path.suffix.lower()
        
        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {extension}. "
                f"Supported types: {self.SUPPORTED_EXTENSIONS}"
            )
        
        if extension == '.md':
            return self.process_markdown_file(path)
        else:
            return self.process_text_file(path)
    
    def process_directory(
        self,
        directory_path: str | Path,
        recursive: bool = True
    ) -> ProcessingResult:
        """
        Process all supported documents in a directory.
        Returns a result with both successful and failed processing.
        """
        import time
        
        start_time = time.time()
        path = Path(directory_path)
        
        if not path.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory_path}")
        
        successful = []
        failed = []
        
        if recursive:
            files = path.rglob('*')
        else:
            files = path.glob('*')
        
        for file_path in files:
            if file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                continue
            
            if not file_path.is_file():
                continue
            
            try:
                doc = self.process_file(file_path)
                successful.append(doc)
                self.processed_count += 1
            except Exception as e:
                error = ProcessingError(
                    source=str(file_path),
                    error_type=type(e).__name__,
                    error_message=str(e),
                    recoverable=isinstance(e, (IOError, OSError))
                )
                failed.append(error)
        
        processing_time = time.time() - start_time
        
        return ProcessingResult(
            successful=successful,
            failed=failed,
            total_processed=len(successful) + len(failed),
            processing_time_seconds=processing_time
        )


def demonstrate_document_processing():
    """
    Demonstrate the document processing pipeline with sample documents.
    """
    
    processor = DocumentProcessor()
    
    print("=" * 70)
    print("DOCUMENT PROCESSING PIPELINE DEMONSTRATION")
    print("=" * 70)
    
    sample_content = """
    # Company Vacation Policy
    
    ## Overview
    All full-time employees are entitled to paid time off (PTO) as outlined
    in this policy. Our goal is to ensure work-life balance while maintaining
    operational efficiency.
    
    ## Annual Allowance
    - **Entry Level (0-2 years)**: 15 days per year
    - **Mid Level (2-5 years)**: 20 days per year
    - **Senior Level (5+ years)**: 25 days per year
    
    ## Requesting Time Off
    1. Submit requests through the HR portal
    2. Provide at least 2 weeks notice for requests over 3 days
    3. Manager approval required within 48 hours
    
    ## Carryover Policy
    Unused PTO can be carried over to the next year, with a maximum
    accumulation of 5 days. Any excess will be forfeited on January 1st.
    
    For questions, contact hr@company.com or visit the [HR Portal](https://hr.company.com).
    """
    
    print("\n📄 Processing raw string content...")
    print("-" * 50)
    
    doc = processor.process_string(
        content=sample_content,
        source="hr_policies/vacation_policy.md",
        title="Company Vacation Policy"
    )
    
    print(f"Document ID: {doc.document_id}")
    print(f"Title: {doc.metadata.title}")
    print(f"Source: {doc.metadata.source}")
    print(f"Character Count: {doc.character_count:,}")
    print(f"Word Count: {doc.word_count:,}")
    print(f"Processed At: {doc.processing_timestamp}")
    
    print("\n📝 Content Preview (first 300 chars):")
    print("-" * 50)
    print(doc.content[:300] + "...")
    
    print("\n" + "=" * 70)
    print("CREATING SAMPLE FILES FOR TESTING")
    print("=" * 70)
    
    sample_dir = Path("data/documents")
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    readme_content = """# Project Documentation

## Getting Started

Welcome to our project! This guide will help you get up and running quickly.

### Prerequisites
- Python 3.9 or higher
- pip package manager
- Virtual environment (recommended)

### Installation

```bash
pip install -r requirements.txt
```

### Quick Start
1. Clone the repository
2. Install dependencies
3. Run the application

For more details, see the full documentation.
"""
    
    faq_content = """Frequently Asked Questions

Q: What is RAG?
A: RAG stands for Retrieval Augmented Generation. It's a technique that combines 
information retrieval with text generation to produce more accurate and grounded
responses from AI models.

Q: Why use Pydantic AI?
A: Pydantic AI provides type-safe structured outputs from language models. This
means you can define exactly what structure you expect, and the framework will
validate that the model's response matches your specification.

Q: Which Gemini model should I use?
A: For most RAG applications, gemini-1.5-flash offers the best balance of speed
and capability. For complex reasoning tasks, consider gemini-1.5-pro.

Q: How many documents can the system handle?
A: The system can handle thousands of documents efficiently. The vector store
performs similarity search in milliseconds regardless of the collection size.
"""
    
    (sample_dir / "README.md").write_text(readme_content)
    (sample_dir / "faq.txt").write_text(faq_content)
    
    print(f"\n📁 Created sample files in {sample_dir}")
    
    print("\n" + "=" * 70)
    print("PROCESSING DIRECTORY")
    print("=" * 70)
    
    result = processor.process_directory(sample_dir)
    
    print(f"\n📊 Processing Results:")
    print(f"   Total Attempted: {result.total_processed}")
    print(f"   Successful: {len(result.successful)}")
    print(f"   Failed: {len(result.failed)}")
    print(f"   Success Rate: {result.success_rate:.1%}")
    print(f"   Processing Time: {result.processing_time_seconds:.3f}s")
    
    print("\n📄 Processed Documents:")
    for doc in result.successful:
        print(f"\n   {doc.metadata.title or doc.metadata.source}")
        print(f"   - ID: {doc.document_id}")
        print(f"   - Type: {doc.metadata.file_type}")
        print(f"   - Words: {doc.word_count:,}")
        print(f"   - Characters: {doc.character_count:,}")
    
    if result.failed:
        print("\n❌ Failed Documents:")
        for error in result.failed:
            print(f"\n   {error.source}")
            print(f"   - Error: {error.error_type}")
            print(f"   - Message: {error.error_message}")
    
    print("\n" + "=" * 70)
    print("VALIDATION DEMONSTRATION")
    print("=" * 70)
    
    print("\n Testing Pydantic validation...")
    
    try:
        bad_doc = ProcessedDocument(
            document_id="test",
            content="",
            metadata=DocumentMetadata(source="test.txt"),
            character_count=0,
            word_count=0
        )
    except Exception as e:
        print(f"\n✅ Validation caught empty content!")
        print(f"   Error: {e}")
    
    try:
        bad_metadata = DocumentMetadata(
            source="test.txt",
            file_size_bytes=-100
        )
    except Exception as e:
        print(f"\n✅ Validation caught negative file size!")
        print(f"   Error: {e}")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    demonstrate_document_processing()


"""
🧪 TEST & APPLY
===============

How to Test It:
---------------
Run this file from the project directory:

    cd project_04_rag_system_with_gemini
    python lesson_02_document_processing_pipeline.py

Expected Output:
----------------
1. A processed document from raw string content
2. Sample files created in the data/documents directory
3. Directory processing results
4. Validation demonstrations


⚠️ COMMON STUMBLING BLOCKS
==========================

1. "FileNotFoundError when processing files"
   
   Error: FileNotFoundError: File not found: documents/policy.pdf
   
   This happens when:
   - The file path is wrong
   - The current working directory is different from expected
   
   Fix: Use absolute paths or ensure you're running from the project directory.
   
   # Good: Use Path for cross-platform compatibility
   from pathlib import Path
   file_path = Path(__file__).parent / "data" / "documents" / "file.txt"

2. "UnicodeDecodeError when reading files"
   
   Error: UnicodeDecodeError: 'utf-8' codec can't decode byte...
   
   This happens when:
   - The file uses a different encoding (like Latin-1 or Windows-1252)
   - The file is actually binary (like a PDF)
   
   Fix: Either specify the encoding or add error handling:
   
   content = file_path.read_text(encoding='utf-8', errors='replace')

3. "ValidationError for empty content"
   
   Error: ValidationError: String should have at least 1 character
   
   This is GOOD! Our Pydantic model is protecting us from empty documents
   that would cause problems later in the pipeline.
   
   Fix: Either skip empty files or handle them explicitly:
   
   if not content.strip():
       raise ValueError("Document is empty")

4. "Why isn't my PDF being processed?"
   
   Our simple processor handles text files only. For PDFs, you need
   additional libraries like PyPDF2 or pdfplumber. We'll cover this
   in a later lesson, but for now, convert PDFs to text externally.


🎯 KEY TAKEAWAYS
================

1. Clean Input = Clean Output
   - Document processing is the foundation of RAG quality
   - Garbage in → garbage out (no matter how good Gemini is!)

2. Preserve Metadata
   - Source information is crucial for citations
   - Timestamps help with relevance
   - File types affect processing logic

3. Structured Error Handling
   - Don't just crash on bad files
   - Collect errors with context
   - Continue processing good files

4. Pydantic Validation
   - Catches invalid documents immediately
   - Ensures consistent data structure
   - Makes debugging much easier


📚 NEXT LESSON
==============
In Lesson 3, we'll learn Text Chunking Strategies - how to split our
processed documents into optimal pieces for embedding and retrieval!
"""
