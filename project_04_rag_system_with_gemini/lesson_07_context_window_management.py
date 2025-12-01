"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                   LESSON 7: CONTEXT WINDOW MANAGEMENT                        ║
║                                                                              ║
║            Optimizing What Context We Send to Gemini for Best Results        ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

📚 CONCEPT OVERVIEW
==================

What & Why:
-----------
The context window is the "working memory" of the AI model - it's how much
text Gemini can "see" when generating a response. While Gemini has MASSIVE
context windows (up to 2M tokens!), we still need to manage context wisely:

- More context = slower responses and higher costs
- Irrelevant context = confused answers
- Well-organized context = better, more accurate responses

Context window management is about sending the RIGHT information, not ALL
information.

🎯 Real-World Analogy:
----------------------
Imagine you're taking an open-book exam:

Bad Strategy:
- Bring 50 textbooks to your desk
- Spend all your time searching through them
- Get confused by contradictory information

Good Strategy:
- Bring only the most relevant chapters
- Organize them with tabs and highlights
- Focus on quality over quantity

That's context window management - curating the best context for the task!

🔒 Type Safety Benefit:
-----------------------
With Pydantic, we ensure:
- Token counts are within model limits
- Context chunks are properly prioritized
- Metadata for context is complete
- Output is structured and predictable


💻 CODE IMPLEMENTATION
=====================
"""

from typing import Optional, Literal
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, model_validator


class ModelLimits(BaseModel):
    """
    Token limits for different Gemini models.
    """
    
    model_name: str = Field(description="Model identifier")
    max_input_tokens: int = Field(ge=1, description="Maximum input tokens")
    max_output_tokens: int = Field(ge=1, description="Maximum output tokens")
    context_window: int = Field(ge=1, description="Total context window")
    
    @classmethod
    def gemini_flash(cls) -> "ModelLimits":
        """Gemini 1.5 Flash limits."""
        return cls(
            model_name="gemini-1.5-flash",
            max_input_tokens=1_000_000,
            max_output_tokens=8_192,
            context_window=1_000_000
        )
    
    @classmethod
    def gemini_pro(cls) -> "ModelLimits":
        """Gemini 1.5 Pro limits."""
        return cls(
            model_name="gemini-1.5-pro",
            max_input_tokens=2_000_000,
            max_output_tokens=8_192,
            context_window=2_000_000
        )


class ContextConfig(BaseModel):
    """
    Configuration for context window management.
    """
    
    max_context_tokens: int = Field(
        default=4000,
        ge=100,
        description="Maximum tokens for context"
    )
    
    max_chunks: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum number of chunks to include"
    )
    
    reserved_for_response: int = Field(
        default=1000,
        ge=100,
        description="Tokens reserved for model response"
    )
    
    include_chunk_metadata: bool = Field(
        default=True,
        description="Include source info with chunks"
    )
    
    prioritization: Literal["similarity", "recency", "hybrid"] = Field(
        default="similarity",
        description="How to prioritize chunks"
    )
    
    recency_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight for recency in hybrid mode"
    )


class ContextChunk(BaseModel):
    """
    A chunk of context with metadata and priority.
    """
    
    chunk_id: str = Field(description="Unique identifier")
    content: str = Field(description="Text content")
    source: str = Field(description="Source document")
    similarity_score: float = Field(ge=0, le=1, description="Relevance score")
    token_count: int = Field(ge=1, description="Token count")
    
    title: Optional[str] = Field(default=None)
    chunk_index: Optional[int] = Field(default=None, ge=0)
    created_at: Optional[datetime] = Field(default=None)
    priority_score: float = Field(default=0.0, description="Computed priority")
    
    def format_for_context(self, include_metadata: bool = True) -> str:
        """Format this chunk for inclusion in context."""
        if include_metadata:
            header = f"[Source: {self.source}"
            if self.title:
                header += f" - {self.title}"
            header += "]"
            return f"{header}\n{self.content}"
        return self.content


class ContextWindow(BaseModel):
    """
    A managed context window ready to send to Gemini.
    """
    
    chunks: list[ContextChunk] = Field(description="Selected chunks")
    total_tokens: int = Field(ge=0, description="Total token count")
    formatted_context: str = Field(description="Ready-to-use context")
    
    config_used: ContextConfig = Field(description="Configuration used")
    chunks_dropped: int = Field(ge=0, description="Chunks not included")
    tokens_available: int = Field(ge=0, description="Remaining token budget")
    
    @property
    def chunk_count(self) -> int:
        """Number of chunks in context."""
        return len(self.chunks)
    
    @property
    def utilization(self) -> float:
        """Context window utilization percentage."""
        if self.config_used.max_context_tokens == 0:
            return 0.0
        return self.total_tokens / self.config_used.max_context_tokens
    
    def get_sources(self) -> list[str]:
        """Get unique sources used in context."""
        return list(set(c.source for c in self.chunks))


class PromptTemplate(BaseModel):
    """
    Template for constructing RAG prompts.
    """
    
    system_instruction: str = Field(
        description="System instruction for the model"
    )
    
    context_prefix: str = Field(
        default="Use the following context to answer the question:",
        description="Text before context"
    )
    
    context_suffix: str = Field(
        default="",
        description="Text after context"
    )
    
    question_prefix: str = Field(
        default="Question:",
        description="Text before question"
    )
    
    instruction_suffix: str = Field(
        default="Answer based only on the context provided. If the context doesn't contain enough information, say so.",
        description="Final instruction"
    )
    
    def build_prompt(
        self,
        context: str,
        question: str
    ) -> str:
        """Build the complete prompt."""
        parts = [
            self.context_prefix,
            "",
            context,
            "",
        ]
        
        if self.context_suffix:
            parts.append(self.context_suffix)
            parts.append("")
        
        parts.extend([
            self.question_prefix,
            question,
            "",
            self.instruction_suffix
        ])
        
        return "\n".join(parts)
    
    def estimate_overhead_tokens(self) -> int:
        """Estimate tokens used by template (not context/question)."""
        overhead = (
            self.context_prefix +
            self.context_suffix +
            self.question_prefix +
            self.instruction_suffix
        )
        return len(overhead) // 4


class ContextWindowManager:
    """
    Manages context window construction for RAG queries.
    """
    
    def __init__(
        self,
        config: Optional[ContextConfig] = None,
        model_limits: Optional[ModelLimits] = None
    ):
        """Initialize the context window manager."""
        self.config = config or ContextConfig()
        self.model_limits = model_limits or ModelLimits.gemini_flash()
    
    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Estimate token count for text (roughly 4 chars per token)."""
        return max(1, len(text) // 4)
    
    def _calculate_priority(
        self,
        chunk: ContextChunk,
        max_age_hours: float = 168
    ) -> float:
        """
        Calculate priority score for a chunk.
        Combines similarity with recency based on config.
        """
        similarity_score = chunk.similarity_score
        
        if self.config.prioritization == "similarity":
            return similarity_score
        
        recency_score = 1.0
        if chunk.created_at:
            age_hours = (datetime.now() - chunk.created_at).total_seconds() / 3600
            recency_score = max(0.0, 1.0 - (age_hours / max_age_hours))
        
        if self.config.prioritization == "recency":
            return recency_score
        
        weight = self.config.recency_weight
        return (1 - weight) * similarity_score + weight * recency_score
    
    def build_context(
        self,
        chunks: list[ContextChunk],
        query: str = ""
    ) -> ContextWindow:
        """
        Build an optimized context window from retrieved chunks.
        
        Args:
            chunks: Retrieved chunks with similarity scores
            query: The user's query (for token budgeting)
            
        Returns:
            ContextWindow ready for use
        """
        for chunk in chunks:
            chunk.priority_score = self._calculate_priority(chunk)
        
        sorted_chunks = sorted(
            chunks,
            key=lambda c: c.priority_score,
            reverse=True
        )
        
        query_tokens = self.estimate_tokens(query)
        available_tokens = (
            self.config.max_context_tokens -
            self.config.reserved_for_response -
            query_tokens -
            100
        )
        
        selected_chunks = []
        total_tokens = 0
        chunks_dropped = 0
        
        for chunk in sorted_chunks:
            if len(selected_chunks) >= self.config.max_chunks:
                chunks_dropped += 1
                continue
            
            chunk_tokens = chunk.token_count
            if self.config.include_chunk_metadata:
                chunk_tokens += 20
            
            if total_tokens + chunk_tokens <= available_tokens:
                selected_chunks.append(chunk)
                total_tokens += chunk_tokens
            else:
                chunks_dropped += 1
        
        formatted_parts = []
        for i, chunk in enumerate(selected_chunks, 1):
            formatted = chunk.format_for_context(self.config.include_chunk_metadata)
            formatted_parts.append(f"--- Context {i} ---\n{formatted}")
        
        formatted_context = "\n\n".join(formatted_parts)
        
        return ContextWindow(
            chunks=selected_chunks,
            total_tokens=total_tokens,
            formatted_context=formatted_context,
            config_used=self.config,
            chunks_dropped=chunks_dropped,
            tokens_available=available_tokens - total_tokens
        )
    
    def optimize_for_model(
        self,
        context_window: ContextWindow
    ) -> ContextWindow:
        """
        Further optimize context for specific model limits.
        """
        max_safe_tokens = self.model_limits.max_input_tokens * 0.8
        
        if context_window.total_tokens <= max_safe_tokens:
            return context_window
        
        chunks_to_keep = []
        running_total = 0
        
        for chunk in context_window.chunks:
            if running_total + chunk.token_count <= max_safe_tokens:
                chunks_to_keep.append(chunk)
                running_total += chunk.token_count
        
        formatted_parts = []
        for i, chunk in enumerate(chunks_to_keep, 1):
            formatted = chunk.format_for_context(self.config.include_chunk_metadata)
            formatted_parts.append(f"--- Context {i} ---\n{formatted}")
        
        return ContextWindow(
            chunks=chunks_to_keep,
            total_tokens=running_total,
            formatted_context="\n\n".join(formatted_parts),
            config_used=self.config,
            chunks_dropped=len(context_window.chunks) - len(chunks_to_keep),
            tokens_available=int(max_safe_tokens - running_total)
        )


class RAGPromptBuilder:
    """
    Builds complete RAG prompts with managed context.
    """
    
    def __init__(
        self,
        context_manager: Optional[ContextWindowManager] = None,
        template: Optional[PromptTemplate] = None
    ):
        """Initialize the prompt builder."""
        self.context_manager = context_manager or ContextWindowManager()
        self.template = template or PromptTemplate(
            system_instruction="You are a helpful assistant that answers questions based on provided context."
        )
    
    def build_rag_prompt(
        self,
        chunks: list[ContextChunk],
        question: str
    ) -> tuple[str, ContextWindow]:
        """
        Build a complete RAG prompt.
        
        Returns:
            Tuple of (formatted_prompt, context_window)
        """
        context_window = self.context_manager.build_context(chunks, question)
        
        prompt = self.template.build_prompt(
            context=context_window.formatted_context,
            question=question
        )
        
        return prompt, context_window


def demonstrate_context_management():
    """
    Demonstrate context window management techniques.
    """
    
    print("=" * 70)
    print("CONTEXT WINDOW MANAGEMENT DEMONSTRATION")
    print("=" * 70)
    
    print("\n📊 GEMINI MODEL LIMITS")
    print("-" * 50)
    
    flash_limits = ModelLimits.gemini_flash()
    pro_limits = ModelLimits.gemini_pro()
    
    print(f"\n{flash_limits.model_name}:")
    print(f"   Context Window: {flash_limits.context_window:,} tokens")
    print(f"   Max Output: {flash_limits.max_output_tokens:,} tokens")
    
    print(f"\n{pro_limits.model_name}:")
    print(f"   Context Window: {pro_limits.context_window:,} tokens")
    print(f"   Max Output: {pro_limits.max_output_tokens:,} tokens")
    
    print("\n" + "=" * 70)
    print("CONTEXT WINDOW CONFIGURATION")
    print("=" * 70)
    
    config = ContextConfig(
        max_context_tokens=2000,
        max_chunks=5,
        reserved_for_response=500,
        prioritization="similarity"
    )
    
    print(f"\nConfiguration:")
    print(f"   Max context tokens: {config.max_context_tokens:,}")
    print(f"   Max chunks: {config.max_chunks}")
    print(f"   Reserved for response: {config.reserved_for_response}")
    print(f"   Prioritization: {config.prioritization}")
    
    print("\n" + "=" * 70)
    print("BUILDING CONTEXT FROM CHUNKS")
    print("=" * 70)
    
    sample_chunks = [
        ContextChunk(
            chunk_id="chunk_001",
            content="To reset your password, go to Settings > Security > Change Password. "
                   "You'll need to enter your current password and then type your new "
                   "password twice to confirm.",
            source="help_docs/password.md",
            title="Password Reset Guide",
            similarity_score=0.95,
            token_count=45,
            chunk_index=0
        ),
        ContextChunk(
            chunk_id="chunk_002",
            content="If you've forgotten your password, click 'Forgot Password' on the "
                   "login page. We'll send a reset link to your registered email address.",
            source="help_docs/login.md",
            title="Login Help",
            similarity_score=0.88,
            token_count=35,
            chunk_index=0
        ),
        ContextChunk(
            chunk_id="chunk_003",
            content="For security, passwords must be at least 12 characters and include "
                   "uppercase, lowercase, numbers, and special characters.",
            source="help_docs/security.md",
            title="Security Guidelines",
            similarity_score=0.72,
            token_count=30,
            chunk_index=0
        ),
        ContextChunk(
            chunk_id="chunk_004",
            content="Two-factor authentication adds an extra layer of security. Enable it "
                   "in Settings > Security > Two-Factor Authentication.",
            source="help_docs/security.md",
            title="Security Guidelines",
            similarity_score=0.45,
            token_count=30,
            chunk_index=1
        ),
        ContextChunk(
            chunk_id="chunk_005",
            content="Contact support at support@company.com if you're still having "
                   "trouble accessing your account after trying these steps.",
            source="help_docs/contact.md",
            title="Contact Support",
            similarity_score=0.35,
            token_count=25,
            chunk_index=0
        ),
    ]
    
    manager = ContextWindowManager(config)
    
    question = "How do I reset my password?"
    context_window = manager.build_context(sample_chunks, question)
    
    print(f"\nInput: {len(sample_chunks)} chunks")
    print(f"Question: '{question}'")
    
    print(f"\n📦 Context Window Built:")
    print(f"   Chunks included: {context_window.chunk_count}")
    print(f"   Chunks dropped: {context_window.chunks_dropped}")
    print(f"   Total tokens: {context_window.total_tokens}")
    print(f"   Tokens available: {context_window.tokens_available}")
    print(f"   Utilization: {context_window.utilization:.1%}")
    print(f"   Sources: {context_window.get_sources()}")
    
    print("\n📝 Selected Chunks (by priority):")
    for chunk in context_window.chunks:
        print(f"\n   [{chunk.similarity_score:.2f}] {chunk.title}")
        print(f"   Source: {chunk.source}")
        print(f"   Tokens: {chunk.token_count}")
    
    print("\n" + "=" * 70)
    print("FORMATTED CONTEXT")
    print("=" * 70)
    
    print("\n" + context_window.formatted_context[:500] + "...")
    
    print("\n" + "=" * 70)
    print("COMPLETE RAG PROMPT")
    print("=" * 70)
    
    template = PromptTemplate(
        system_instruction="You are a helpful IT support assistant.",
        context_prefix="Here is relevant information from our documentation:",
        instruction_suffix="Based on this context, provide a clear, helpful answer. "
                          "If the answer isn't in the context, say so."
    )
    
    builder = RAGPromptBuilder(manager, template)
    prompt, ctx = builder.build_rag_prompt(sample_chunks, question)
    
    print(f"\nFull Prompt ({ContextWindowManager.estimate_tokens(prompt)} tokens):")
    print("-" * 50)
    print(prompt[:800] + "\n...")
    
    print("\n" + "=" * 70)
    print("DIFFERENT PRIORITIZATION STRATEGIES")
    print("=" * 70)
    
    for strategy in ["similarity", "recency", "hybrid"]:
        config = ContextConfig(
            max_context_tokens=1000,
            max_chunks=3,
            prioritization=strategy,
            recency_weight=0.4
        )
        manager = ContextWindowManager(config)
        context = manager.build_context(sample_chunks, question)
        
        print(f"\n{strategy.upper()} prioritization:")
        for chunk in context.chunks:
            print(f"   [{chunk.priority_score:.2f}] {chunk.title}")
    
    print("\n" + "=" * 70)
    print("VALIDATION EXAMPLES")
    print("=" * 70)
    
    print("\n Testing configuration validation...")
    
    try:
        bad_config = ContextConfig(max_context_tokens=50)
    except Exception as e:
        print(f"\n✅ Caught too-small context!")
        print(f"   Error: {e}")
    
    try:
        bad_config = ContextConfig(max_chunks=100)
    except Exception as e:
        print(f"\n✅ Caught too-many chunks!")
        print(f"   Error: {e}")


if __name__ == "__main__":
    demonstrate_context_management()


"""
🧪 TEST & APPLY
===============

How to Test It:
---------------
    python lesson_07_context_window_management.py

Expected Output:
----------------
1. Gemini model limits comparison
2. Context configuration details
3. Built context window statistics
4. Formatted context preview
5. Complete RAG prompt example
6. Different prioritization strategies
7. Validation examples


⚠️ COMMON STUMBLING BLOCKS
==========================

1. "Context too large for model"
   
   Error: Token count exceeds model limit
   
   Fix: Use optimize_for_model() or reduce max_context_tokens:
   context = manager.optimize_for_model(context_window)

2. "Important chunks being dropped"
   
   Chunks with lower similarity might be important.
   
   Fix: Increase max_chunks or max_context_tokens:
   config = ContextConfig(max_chunks=10, max_context_tokens=4000)

3. "Response cut off"
   
   Not enough tokens reserved for response.
   
   Fix: Increase reserved_for_response:
   config = ContextConfig(reserved_for_response=2000)

4. "Same source repeated many times"
   
   Multiple chunks from one document dominating.
   
   Fix: Add source diversity logic or use MMR from Lesson 6.


🎯 KEY TAKEAWAYS
================

1. Quality Over Quantity
   - More context isn't always better
   - Relevant context = better answers
   - Irrelevant context = confused model

2. Token Budgeting
   - Reserve tokens for response
   - Account for prompt overhead
   - Leave buffer for safety

3. Prioritization Matters
   - Similarity: Best for factual Q&A
   - Recency: Best for time-sensitive info
   - Hybrid: Balances both factors

4. Gemini's Massive Context
   - Flash: 1M tokens (huge!)
   - Pro: 2M tokens (enormous!)
   - Still budget wisely for cost/speed


📚 NEXT LESSON
==============
In Lesson 8, we'll build RAG Dependencies - the dependency injection
pattern for clean, testable RAG systems with Pydantic AI!
"""
