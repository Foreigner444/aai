"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                    LESSON 8: BUILDING RAG DEPENDENCIES                       ║
║                                                                              ║
║           Dependency Injection for Clean, Testable RAG Systems               ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

📚 CONCEPT OVERVIEW
==================

What & Why:
-----------
Dependencies in Pydantic AI are a way to pass context, configuration, and
services to your agents in a clean, organized way. Instead of using global
variables or passing tons of parameters, you bundle everything your agent
needs into a dependency object.

Why dependencies are CRUCIAL for RAG:
- RAG needs access to vector stores, embedders, and config
- Testing requires swapping real services with mocks
- Different users might have different contexts
- Clean separation of concerns = maintainable code

🎯 Real-World Analogy:
----------------------
Think of dependencies like a chef's mise en place (everything in its place):

Without Dependencies (Chaos):
- Chef runs around kitchen grabbing ingredients
- Sometimes grabs wrong items
- Hard to prep for different dishes
- Can't easily substitute ingredients

With Dependencies (Organized):
- Everything chef needs is on their station
- Pre-measured, pre-prepped
- Easy to swap ingredients for dietary needs
- Same recipe works with different setups

Dependencies are your RAG agent's mise en place!

🔒 Type Safety Benefit:
-----------------------
With Pydantic dependency models:
- All required services are guaranteed to exist
- Configuration is validated before use
- IDE autocomplete works everywhere
- Tests can safely mock dependencies


💻 CODE IMPLEMENTATION
=====================
"""

from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable
from datetime import datetime
from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Metadata for documents."""
    source: str
    title: Optional[str] = None
    chunk_index: int = 0


class RetrievedChunk(BaseModel):
    """A chunk retrieved from the vector store."""
    chunk_id: str
    content: str
    metadata: DocumentMetadata
    similarity_score: float = Field(ge=0, le=1)


class UserContext(BaseModel):
    """
    Information about the current user.
    Useful for personalization and access control.
    """
    
    user_id: str = Field(description="Unique user identifier")
    username: str = Field(description="Display name")
    role: str = Field(default="user", description="User role")
    allowed_sources: list[str] = Field(
        default_factory=list,
        description="Sources this user can access"
    )
    preferences: dict = Field(
        default_factory=dict,
        description="User preferences"
    )
    
    def can_access(self, source: str) -> bool:
        """Check if user can access a source."""
        if not self.allowed_sources:
            return True
        return any(allowed in source for allowed in self.allowed_sources)


class RAGConfig(BaseModel):
    """
    Configuration for the RAG system.
    All settings in one validated place.
    """
    
    collection_name: str = Field(
        default="documents",
        description="Vector store collection"
    )
    
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of chunks to retrieve"
    )
    
    min_similarity: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum similarity threshold"
    )
    
    max_context_tokens: int = Field(
        default=4000,
        ge=100,
        description="Maximum context tokens"
    )
    
    model_name: str = Field(
        default="gemini-1.5-flash",
        description="Gemini model to use"
    )
    
    include_citations: bool = Field(
        default=True,
        description="Include source citations"
    )
    
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Model temperature"
    )


@runtime_checkable
class VectorStoreProtocol(Protocol):
    """
    Protocol defining what a vector store must provide.
    This allows easy swapping of implementations.
    """
    
    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        filter_sources: Optional[list[str]] = None
    ) -> list[RetrievedChunk]:
        """Search for similar documents."""
        ...
    
    def add_documents(
        self,
        contents: list[str],
        embeddings: list[list[float]],
        metadatas: list[DocumentMetadata]
    ) -> list[str]:
        """Add documents to the store."""
        ...


@runtime_checkable
class EmbedderProtocol(Protocol):
    """
    Protocol defining what an embedder must provide.
    """
    
    def embed_text(self, text: str) -> list[float]:
        """Embed a single text."""
        ...
    
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts."""
        ...


class MockVectorStore:
    """
    Mock vector store for testing and demonstration.
    """
    
    def __init__(self):
        self._documents: list[dict] = []
    
    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        filter_sources: Optional[list[str]] = None
    ) -> list[RetrievedChunk]:
        """Return mock search results."""
        results = []
        for i, doc in enumerate(self._documents[:top_k]):
            if filter_sources:
                if not any(s in doc["metadata"].source for s in filter_sources):
                    continue
            
            results.append(RetrievedChunk(
                chunk_id=doc["id"],
                content=doc["content"],
                metadata=doc["metadata"],
                similarity_score=0.9 - (i * 0.1)
            ))
        return results
    
    def add_documents(
        self,
        contents: list[str],
        embeddings: list[list[float]],
        metadatas: list[DocumentMetadata]
    ) -> list[str]:
        """Add mock documents."""
        ids = []
        for i, (content, metadata) in enumerate(zip(contents, metadatas)):
            doc_id = f"doc_{len(self._documents) + i}"
            self._documents.append({
                "id": doc_id,
                "content": content,
                "metadata": metadata
            })
            ids.append(doc_id)
        return ids


class MockEmbedder:
    """
    Mock embedder for testing and demonstration.
    """
    
    def __init__(self, dimension: int = 768):
        self.dimension = dimension
    
    def embed_text(self, text: str) -> list[float]:
        """Return mock embedding."""
        import hashlib
        hash_bytes = hashlib.md5(text.encode()).digest()
        base = [b / 255.0 for b in hash_bytes]
        return (base * (self.dimension // 16 + 1))[:self.dimension]
    
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Return mock embeddings for batch."""
        return [self.embed_text(text) for text in texts]


@dataclass
class RAGDependencies:
    """
    All dependencies needed by the RAG agent.
    This is what gets passed to every agent call.
    
    Using @dataclass for Pydantic AI compatibility.
    """
    
    vector_store: VectorStoreProtocol
    embedder: EmbedderProtocol
    config: RAGConfig
    user_context: Optional[UserContext] = None
    
    def retrieve(self, query: str) -> list[RetrievedChunk]:
        """
        Convenience method to retrieve documents for a query.
        Handles embedding and filtering automatically.
        """
        query_embedding = self.embedder.embed_text(query)
        
        filter_sources = None
        if self.user_context and self.user_context.allowed_sources:
            filter_sources = self.user_context.allowed_sources
        
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=self.config.top_k,
            filter_sources=filter_sources
        )
        
        filtered = [r for r in results if r.similarity_score >= self.config.min_similarity]
        
        return filtered
    
    def get_model_name(self) -> str:
        """Get the configured model name."""
        return self.config.model_name


class RAGDependencyFactory:
    """
    Factory for creating RAG dependencies.
    Makes it easy to create dependencies for different scenarios.
    """
    
    @staticmethod
    def create_mock_dependencies(
        config: Optional[RAGConfig] = None,
        user: Optional[UserContext] = None
    ) -> RAGDependencies:
        """Create dependencies with mock services (for testing)."""
        return RAGDependencies(
            vector_store=MockVectorStore(),
            embedder=MockEmbedder(),
            config=config or RAGConfig(),
            user_context=user
        )
    
    @staticmethod
    def create_production_dependencies(
        vector_store: VectorStoreProtocol,
        embedder: EmbedderProtocol,
        config: Optional[RAGConfig] = None,
        user: Optional[UserContext] = None
    ) -> RAGDependencies:
        """Create dependencies with real services."""
        return RAGDependencies(
            vector_store=vector_store,
            embedder=embedder,
            config=config or RAGConfig(),
            user_context=user
        )


class ConversationHistory(BaseModel):
    """
    Tracks conversation history for multi-turn RAG.
    """
    
    messages: list[dict] = Field(default_factory=list)
    max_messages: int = Field(default=10)
    
    def add_turn(self, query: str, response: str) -> None:
        """Add a conversation turn."""
        self.messages.append({
            "query": query,
            "response": response,
            "timestamp": datetime.now().isoformat()
        })
        
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
    
    def get_context_summary(self) -> str:
        """Get summary of recent conversation for context."""
        if not self.messages:
            return ""
        
        recent = self.messages[-3:]
        summary_parts = []
        for msg in recent:
            summary_parts.append(f"User: {msg['query']}")
            summary_parts.append(f"Assistant: {msg['response'][:100]}...")
        
        return "\n".join(summary_parts)


@dataclass
class ConversationalRAGDependencies(RAGDependencies):
    """
    Extended dependencies for conversational RAG.
    Includes conversation history.
    """
    
    conversation: ConversationHistory = None
    
    def __post_init__(self):
        if self.conversation is None:
            self.conversation = ConversationHistory()
    
    def retrieve_with_history(self, query: str) -> tuple[list[RetrievedChunk], str]:
        """
        Retrieve documents considering conversation history.
        Returns chunks and conversation context.
        """
        chunks = self.retrieve(query)
        history_context = self.conversation.get_context_summary()
        return chunks, history_context


def demonstrate_rag_dependencies():
    """
    Demonstrate RAG dependency patterns.
    """
    
    print("=" * 70)
    print("RAG DEPENDENCIES DEMONSTRATION")
    print("=" * 70)
    
    print("\n📦 CREATING RAG CONFIGURATION")
    print("-" * 50)
    
    config = RAGConfig(
        collection_name="company_docs",
        top_k=5,
        min_similarity=0.6,
        max_context_tokens=4000,
        model_name="gemini-1.5-flash",
        include_citations=True,
        temperature=0.7
    )
    
    print(f"\nConfiguration created:")
    print(f"   Collection: {config.collection_name}")
    print(f"   Top K: {config.top_k}")
    print(f"   Min Similarity: {config.min_similarity}")
    print(f"   Model: {config.model_name}")
    print(f"   Temperature: {config.temperature}")
    
    print("\n" + "=" * 70)
    print("CREATING USER CONTEXT")
    print("=" * 70)
    
    admin_user = UserContext(
        user_id="admin_001",
        username="Alice Admin",
        role="admin",
        allowed_sources=[],
        preferences={"language": "en", "detail_level": "high"}
    )
    
    regular_user = UserContext(
        user_id="user_042",
        username="Bob User",
        role="user",
        allowed_sources=["public_docs", "help"],
        preferences={"language": "en", "detail_level": "medium"}
    )
    
    print(f"\nAdmin User:")
    print(f"   ID: {admin_user.user_id}")
    print(f"   Role: {admin_user.role}")
    print(f"   Can access 'internal_docs': {admin_user.can_access('internal_docs')}")
    
    print(f"\nRegular User:")
    print(f"   ID: {regular_user.user_id}")
    print(f"   Role: {regular_user.role}")
    print(f"   Allowed sources: {regular_user.allowed_sources}")
    print(f"   Can access 'public_docs': {regular_user.can_access('public_docs')}")
    print(f"   Can access 'internal_docs': {regular_user.can_access('internal_docs')}")
    
    print("\n" + "=" * 70)
    print("CREATING DEPENDENCIES WITH FACTORY")
    print("=" * 70)
    
    deps = RAGDependencyFactory.create_mock_dependencies(
        config=config,
        user=regular_user
    )
    
    print(f"\nDependencies created:")
    print(f"   Vector Store: {type(deps.vector_store).__name__}")
    print(f"   Embedder: {type(deps.embedder).__name__}")
    print(f"   Config: {deps.config.collection_name}")
    print(f"   User: {deps.user_context.username if deps.user_context else 'None'}")
    
    print("\n" + "=" * 70)
    print("USING DEPENDENCIES")
    print("=" * 70)
    
    docs = [
        ("How to reset your password: Go to Settings > Security.", 
         DocumentMetadata(source="help/password.md", title="Password Help")),
        ("Vacation policy: 20 days PTO per year for full-time employees.",
         DocumentMetadata(source="hr/vacation.md", title="Vacation Policy")),
        ("Internal metrics dashboard is at metrics.internal.company.com",
         DocumentMetadata(source="internal/tools.md", title="Internal Tools")),
    ]
    
    contents = [d[0] for d in docs]
    metadatas = [d[1] for d in docs]
    embeddings = deps.embedder.embed_batch(contents)
    
    doc_ids = deps.vector_store.add_documents(contents, embeddings, metadatas)
    print(f"\nAdded {len(doc_ids)} documents to vector store")
    
    query = "How do I change my password?"
    print(f"\n🔍 Query: '{query}'")
    
    results = deps.retrieve(query)
    
    print(f"\nRetrieved {len(results)} chunks:")
    for result in results:
        print(f"\n   [{result.similarity_score:.2f}] {result.metadata.title}")
        print(f"   Source: {result.metadata.source}")
        print(f"   Content: {result.content[:50]}...")
    
    print("\n" + "=" * 70)
    print("ACCESS CONTROL IN ACTION")
    print("=" * 70)
    
    admin_deps = RAGDependencyFactory.create_mock_dependencies(
        config=config,
        user=admin_user
    )
    admin_deps.vector_store = deps.vector_store
    admin_deps.embedder = deps.embedder
    
    print(f"\n👤 Admin user searching:")
    admin_results = admin_deps.retrieve(query)
    print(f"   Found {len(admin_results)} results (no restrictions)")
    
    print(f"\n👤 Regular user searching:")
    user_results = deps.retrieve(query)
    print(f"   Found {len(user_results)} results (filtered by allowed_sources)")
    
    print("\n" + "=" * 70)
    print("CONVERSATIONAL DEPENDENCIES")
    print("=" * 70)
    
    conv_deps = ConversationalRAGDependencies(
        vector_store=deps.vector_store,
        embedder=deps.embedder,
        config=config,
        user_context=regular_user,
        conversation=ConversationHistory()
    )
    
    conv_deps.conversation.add_turn(
        query="What's the vacation policy?",
        response="You get 20 days of PTO per year as a full-time employee."
    )
    
    conv_deps.conversation.add_turn(
        query="Can I carry over unused days?",
        response="Yes, up to 5 days can be carried over to the next year."
    )
    
    print(f"\nConversation History:")
    print("-" * 50)
    print(conv_deps.conversation.get_context_summary())
    
    new_query = "What about sick days?"
    chunks, history = conv_deps.retrieve_with_history(new_query)
    
    print(f"\n🔍 New Query: '{new_query}'")
    print(f"   Retrieved chunks: {len(chunks)}")
    print(f"   History context included: {len(history)} chars")
    
    print("\n" + "=" * 70)
    print("VALIDATION EXAMPLES")
    print("=" * 70)
    
    print("\n Testing configuration validation...")
    
    try:
        bad_config = RAGConfig(top_k=100)
    except Exception as e:
        print(f"\n✅ Caught top_k too high!")
        print(f"   Error: {e}")
    
    try:
        bad_config = RAGConfig(min_similarity=1.5)
    except Exception as e:
        print(f"\n✅ Caught invalid similarity!")
        print(f"   Error: {e}")
    
    try:
        bad_config = RAGConfig(temperature=-0.5)
    except Exception as e:
        print(f"\n✅ Caught invalid temperature!")
        print(f"   Error: {e}")
    
    print("\n" + "=" * 70)
    print("DEPENDENCY PATTERNS FOR PYDANTIC AI")
    print("=" * 70)
    
    print("""
In Pydantic AI, you use dependencies like this:

    from pydantic_ai import Agent
    
    # Define your agent with dependency type
    agent = Agent(
        'gemini-1.5-flash',
        deps_type=RAGDependencies,
        system_prompt='You are a helpful assistant...'
    )
    
    # Define tools that use dependencies
    @agent.tool
    async def search_documents(ctx: RunContext[RAGDependencies], query: str) -> str:
        chunks = ctx.deps.retrieve(query)
        return format_chunks(chunks)
    
    # Run agent with dependencies
    deps = RAGDependencyFactory.create_production_dependencies(
        vector_store=my_vector_store,
        embedder=my_embedder,
        config=my_config,
        user=current_user
    )
    
    result = await agent.run('How do I reset my password?', deps=deps)
""")


if __name__ == "__main__":
    demonstrate_rag_dependencies()


"""
🧪 TEST & APPLY
===============

How to Test It:
---------------
    python lesson_08_building_rag_dependencies.py

Expected Output:
----------------
1. RAG configuration details
2. User context with access control
3. Dependencies created with factory
4. Document retrieval demonstration
5. Access control comparison (admin vs user)
6. Conversational dependencies with history
7. Validation examples


⚠️ COMMON STUMBLING BLOCKS
==========================

1. "Dependencies not passed to tools"
   
   Error: AttributeError: 'NoneType' has no attribute 'retrieve'
   
   Fix: Ensure you pass deps when running the agent:
   result = await agent.run(query, deps=deps)

2. "Protocol not implemented"
   
   Error: TypeError: Class doesn't implement protocol
   
   Fix: Ensure your class has all required methods:
   class MyStore:
       def search(...) -> list[RetrievedChunk]: ...
       def add_documents(...) -> list[str]: ...

3. "User context is None"
   
   Fix: Check user_context before accessing:
   if deps.user_context:
       allowed = deps.user_context.allowed_sources

4. "Config validation fails on creation"
   
   Fix: Check field constraints in RAGConfig:
   - top_k: 1-20
   - min_similarity: 0.0-1.0
   - temperature: 0.0-2.0


🎯 KEY TAKEAWAYS
================

1. Dependencies Bundle Everything
   - All services in one place
   - Easy to pass around
   - Clear what agent needs

2. Protocols Enable Swapping
   - Define interface, not implementation
   - Mock for testing
   - Swap for different backends

3. Factory Pattern Helps
   - create_mock_dependencies for tests
   - create_production_dependencies for real
   - Easy to add new scenarios

4. User Context Enables Features
   - Access control
   - Personalization
   - Audit logging


📚 NEXT LESSON
==============
In Lesson 9, we'll learn Citation Models with Pydantic - creating
proper source attribution for RAG responses!
"""
