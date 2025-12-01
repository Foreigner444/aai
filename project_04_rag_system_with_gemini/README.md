# 📚 Project 4: RAG System with Gemini

A complete, production-ready Retrieval Augmented Generation (RAG) system using Pydantic AI and Google Gemini models.

## 🎯 What You'll Build

A RAG system that:
- ✅ Ingests and processes documents into searchable chunks
- ✅ Creates embeddings using Gemini's text-embedding-004
- ✅ Stores and retrieves vectors using similarity search
- ✅ Generates validated, cited answers with confidence scores
- ✅ Streams responses in real-time
- ✅ Tracks sources and validates factual accuracy

## 📋 Lessons

| # | Lesson | Description |
|---|--------|-------------|
| 01 | Understanding RAG Architecture | Core concepts and pipeline design |
| 02 | Document Processing Pipeline | Extracting text from various formats |
| 03 | Text Chunking Strategies | Splitting documents effectively |
| 04 | Gemini Embeddings API | Converting text to vectors |
| 05 | Vector Store Integration | Storing embeddings in ChromaDB |
| 06 | Similarity Search Implementation | Finding relevant documents |
| 07 | Context Window Management | Optimizing context for Gemini |
| 08 | Building RAG Dependencies | Dependency injection patterns |
| 09 | Citation Models with Pydantic | Source attribution |
| 10 | Source Tracking and Attribution | Full provenance tracking |
| 11 | Multi-Document Retrieval | Searching across collections |
| 12 | Confidence Scoring | Evaluating answer reliability |
| 13 | Answer Validation | Ensuring accuracy |
| 14 | Fact-Checking Patterns | Verifying specific claims |
| 15 | Streaming RAG Responses | Real-time response delivery |
| 16 | Complete RAG System | Production-ready implementation |

## 🚀 Getting Started

### Prerequisites

```bash
# Python 3.9+
python --version

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Set your API key (for lessons using real Gemini API)
export GOOGLE_API_KEY='your-api-key-here'
```

### Running Lessons

Each lesson is a standalone Python file that can be run directly:

```bash
# Run any lesson
python lesson_01_understanding_rag_architecture.py
python lesson_02_document_processing_pipeline.py
# ... and so on

# Run the complete system
python lesson_16_complete_rag_system.py
```

## 📁 Project Structure

```
project_04_rag_system_with_gemini/
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
├── lesson_01_understanding_rag_architecture.py
├── lesson_02_document_processing_pipeline.py
├── lesson_03_text_chunking_strategies.py
├── lesson_04_gemini_embeddings_api.py
├── lesson_05_vector_store_integration.py
├── lesson_06_similarity_search_implementation.py
├── lesson_07_context_window_management.py
├── lesson_08_building_rag_dependencies.py
├── lesson_09_citation_models_with_pydantic.py
├── lesson_10_source_tracking_and_attribution.py
├── lesson_11_multi_document_retrieval.py
├── lesson_12_confidence_scoring.py
├── lesson_13_answer_validation.py
├── lesson_14_fact_checking_patterns.py
├── lesson_15_streaming_rag_responses.py
├── lesson_16_complete_rag_system.py
├── src/                                   # Source code modules
│   ├── models/                            # Pydantic models
│   ├── agents/                            # Pydantic AI agents
│   ├── tools/                             # Custom tools
│   ├── dependencies/                      # Dependency providers
│   └── utils/                             # Utility functions
├── tests/                                 # Test files
└── data/
    └── documents/                         # Sample documents
```

## 🔧 Key Technologies

- **Pydantic AI**: Type-safe AI framework
- **Pydantic v2**: Data validation
- **Google Gemini**: AI models
  - `text-embedding-004`: Embeddings
  - `gemini-1.5-flash`: Fast generation
  - `gemini-1.5-pro`: Complex reasoning
- **ChromaDB**: Vector database
- **Python 3.9+**: Modern Python features

## 💡 Key Concepts

### RAG Pipeline

```
Documents → Chunks → Embeddings → Vector Store
                                      ↓
Query → Embed → Search → Retrieve → Generate → Validate → Response
```

### Type Safety with Pydantic

Every component uses validated Pydantic models:
- `DocumentChunk`: Validated document pieces
- `RetrievedChunk`: Search results with scores
- `RAGResponse`: Complete responses with citations
- `ConfidenceScore`: Reliability metrics

### Dependency Injection

Clean, testable architecture:
```python
@dataclass
class RAGDependencies:
    vector_store: VectorStoreProtocol
    embedder: EmbedderProtocol
    config: RAGConfig
```

## 📊 Features

### Confidence Scoring
```python
confidence = ConfidenceScore(
    overall_score=0.85,
    level=ConfidenceLevel.HIGH,
    factors=ConfidenceFactors(...)
)
```

### Citation Tracking
```python
citation = Citation(
    source="hr/policy.md",
    title="HR Policy",
    relevance=0.95,
    excerpt="..."
)
```

### Streaming Responses
```python
async for chunk in rag.query_streaming(query):
    print(chunk, end="", flush=True)
```

## 🎓 Learning Outcomes

After completing this project, you will be able to:

1. **Design RAG Systems**: Understand the complete RAG architecture
2. **Process Documents**: Extract and chunk text effectively
3. **Use Embeddings**: Create and compare vector representations
4. **Implement Search**: Build similarity-based retrieval
5. **Generate Answers**: Create grounded, validated responses
6. **Track Sources**: Maintain full provenance and citations
7. **Validate Quality**: Score confidence and check facts
8. **Stream Responses**: Deliver real-time user experiences

## 🔗 Next Steps

After completing this project:
- **Project 5**: Conversational AI with memory
- **Project 6**: Multi-Agent Orchestration
- **Project 8**: FastAPI Integration

## 📚 Resources

- [Pydantic AI Documentation](https://ai.pydantic.dev/)
- [Google Gemini API](https://ai.google.dev/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Pydantic v2 Documentation](https://docs.pydantic.dev/)

---

Happy learning! 🚀 You're building production-ready AI systems!
