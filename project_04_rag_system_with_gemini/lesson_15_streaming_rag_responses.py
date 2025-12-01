"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                   LESSON 15: STREAMING RAG RESPONSES                         ║
║                                                                              ║
║            Delivering Real-Time Answers with Progressive Context             ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

📚 CONCEPT OVERVIEW
==================

What & Why:
-----------
Streaming delivers RAG responses progressively, word by word or chunk by chunk,
instead of waiting for the complete answer. This dramatically improves user
experience because:

- Users see immediate feedback (system is working)
- First content appears in milliseconds, not seconds
- Perceived latency is much lower
- Users can start reading while generation continues

🎯 Real-World Analogy:
----------------------
Think of streaming like a waiter bringing your meal:

Without Streaming:
- Order placed
- Wait 20 minutes
- Entire meal arrives at once
(Long wait, then everything at once)

With Streaming:
- Order placed
- Bread arrives immediately
- Appetizer comes next
- Main course follows
- Dessert last
(Constant engagement, shorter perceived wait)

Streaming keeps users engaged throughout the response!

🔒 Type Safety Benefit:
-----------------------
With Pydantic streaming models:
- Each chunk is validated as it arrives
- Progress tracking is structured
- Final assembly is type-safe
- Errors are caught progressively


💻 CODE IMPLEMENTATION
=====================
"""

import asyncio
from typing import Optional, AsyncIterator
from datetime import datetime
from pydantic import BaseModel, Field, computed_field
from enum import Enum


class StreamEventType(str, Enum):
    """Types of events in a streaming response."""
    STARTED = "started"
    RETRIEVAL_COMPLETE = "retrieval_complete"
    GENERATION_STARTED = "generation_started"
    CONTENT_CHUNK = "content_chunk"
    CITATION_ADDED = "citation_added"
    COMPLETED = "completed"
    ERROR = "error"


class DocumentMetadata(BaseModel):
    """Metadata for a document."""
    source: str
    title: Optional[str] = None


class RetrievedChunk(BaseModel):
    """A retrieved chunk."""
    chunk_id: str
    content: str
    metadata: DocumentMetadata
    similarity_score: float = Field(ge=0, le=1)


class Citation(BaseModel):
    """A citation reference."""
    source: str
    title: Optional[str] = None
    relevance: float = Field(ge=0, le=1)


class StreamEvent(BaseModel):
    """
    A single event in a streaming response.
    """
    
    event_type: StreamEventType = Field(
        description="Type of this event"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When this event occurred"
    )
    
    content: Optional[str] = Field(
        default=None,
        description="Text content if applicable"
    )
    
    citation: Optional[Citation] = Field(
        default=None,
        description="Citation if this is a citation event"
    )
    
    metadata: dict = Field(
        default_factory=dict,
        description="Additional metadata"
    )
    
    progress: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Progress percentage (0-1)"
    )
    
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if error event"
    )


class StreamProgress(BaseModel):
    """
    Tracks progress of a streaming response.
    """
    
    total_chunks_retrieved: int = Field(default=0, ge=0)
    chunks_processed: int = Field(default=0, ge=0)
    tokens_generated: int = Field(default=0, ge=0)
    estimated_total_tokens: int = Field(default=100, ge=1)
    citations_added: int = Field(default=0, ge=0)
    
    retrieval_time_ms: float = Field(default=0.0, ge=0)
    generation_time_ms: float = Field(default=0.0, ge=0)
    
    current_phase: str = Field(default="initializing")
    
    @computed_field
    @property
    def generation_progress(self) -> float:
        """Progress of generation phase."""
        if self.estimated_total_tokens == 0:
            return 0.0
        return min(self.tokens_generated / self.estimated_total_tokens, 1.0)
    
    @computed_field
    @property
    def overall_progress(self) -> float:
        """Overall progress estimate."""
        if self.current_phase == "initializing":
            return 0.0
        elif self.current_phase == "retrieving":
            return 0.1
        elif self.current_phase == "generating":
            return 0.2 + (self.generation_progress * 0.7)
        elif self.current_phase == "completed":
            return 1.0
        return 0.5


class StreamingResponse(BaseModel):
    """
    Complete streaming response with all events.
    """
    
    query: str = Field(description="Original query")
    
    events: list[StreamEvent] = Field(
        default_factory=list,
        description="All stream events"
    )
    
    final_content: str = Field(
        default="",
        description="Assembled final content"
    )
    
    citations: list[Citation] = Field(
        default_factory=list,
        description="All citations"
    )
    
    progress: StreamProgress = Field(
        default_factory=StreamProgress,
        description="Progress tracking"
    )
    
    started_at: datetime = Field(
        default_factory=datetime.now,
        description="When streaming started"
    )
    
    completed_at: Optional[datetime] = Field(
        default=None,
        description="When streaming completed"
    )
    
    @computed_field
    @property
    def is_complete(self) -> bool:
        """Check if streaming is complete."""
        return self.completed_at is not None
    
    @computed_field
    @property
    def duration_ms(self) -> float:
        """Total duration in milliseconds."""
        end = self.completed_at or datetime.now()
        return (end - self.started_at).total_seconds() * 1000
    
    def add_event(self, event: StreamEvent) -> None:
        """Add an event to the stream."""
        self.events.append(event)
        
        if event.event_type == StreamEventType.CONTENT_CHUNK and event.content:
            self.final_content += event.content
            self.progress.tokens_generated += len(event.content.split())
        
        elif event.event_type == StreamEventType.CITATION_ADDED and event.citation:
            self.citations.append(event.citation)
            self.progress.citations_added += 1
        
        elif event.event_type == StreamEventType.COMPLETED:
            self.completed_at = datetime.now()
            self.progress.current_phase = "completed"


class StreamConfig(BaseModel):
    """Configuration for streaming."""
    
    chunk_delay_ms: int = Field(
        default=50,
        ge=0,
        description="Delay between chunks (for simulation)"
    )
    
    include_progress_events: bool = Field(
        default=True,
        description="Include progress update events"
    )
    
    stream_citations: bool = Field(
        default=True,
        description="Stream citations as they're added"
    )
    
    words_per_chunk: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Words per streaming chunk"
    )


class StreamingRAGEngine:
    """
    RAG engine with streaming support.
    """
    
    def __init__(self, config: Optional[StreamConfig] = None):
        """Initialize the streaming engine."""
        self.config = config or StreamConfig()
    
    async def _simulate_retrieval(
        self,
        query: str,
        chunks: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        """Simulate retrieval phase."""
        await asyncio.sleep(0.1)
        return chunks
    
    async def _generate_streaming_response(
        self,
        query: str,
        context: str,
        progress: StreamProgress
    ) -> AsyncIterator[str]:
        """
        Generate response with streaming.
        Simulates word-by-word generation.
        """
        
        response_template = (
            f"Based on the available documentation, here is the answer to your question: "
            f"Full-time employees receive 20 days of paid vacation per year. "
            f"Vacation requests must be submitted at least two weeks in advance "
            f"through the HR portal. Unused days can be carried over to the next year, "
            f"with a maximum of 5 days. Any days exceeding this limit will be forfeited "
            f"on January 1st. Part-time employees receive prorated vacation based on "
            f"their scheduled hours."
        )
        
        words = response_template.split()
        
        for i in range(0, len(words), self.config.words_per_chunk):
            chunk_words = words[i:i + self.config.words_per_chunk]
            chunk_text = " ".join(chunk_words) + " "
            
            await asyncio.sleep(self.config.chunk_delay_ms / 1000)
            
            yield chunk_text
    
    async def stream_response(
        self,
        query: str,
        chunks: list[RetrievedChunk]
    ) -> AsyncIterator[StreamEvent]:
        """
        Stream a complete RAG response.
        Yields events as they occur.
        """
        
        progress = StreamProgress()
        
        yield StreamEvent(
            event_type=StreamEventType.STARTED,
            metadata={"query": query},
            progress=0.0
        )
        
        progress.current_phase = "retrieving"
        
        import time
        retrieval_start = time.time()
        retrieved = await self._simulate_retrieval(query, chunks)
        progress.retrieval_time_ms = (time.time() - retrieval_start) * 1000
        progress.total_chunks_retrieved = len(retrieved)
        
        yield StreamEvent(
            event_type=StreamEventType.RETRIEVAL_COMPLETE,
            metadata={
                "chunks_found": len(retrieved),
                "retrieval_time_ms": progress.retrieval_time_ms
            },
            progress=0.2
        )
        
        progress.current_phase = "generating"
        
        yield StreamEvent(
            event_type=StreamEventType.GENERATION_STARTED,
            metadata={"model": "gemini-1.5-flash"},
            progress=0.25
        )
        
        context = "\n\n".join(c.content for c in retrieved)
        
        citations_to_add = [
            Citation(
                source=c.metadata.source,
                title=c.metadata.title,
                relevance=c.similarity_score
            )
            for c in retrieved[:3]
        ]
        
        citation_indices = [20, 40, 60]
        citation_idx = 0
        word_count = 0
        
        generation_start = time.time()
        
        async for chunk_text in self._generate_streaming_response(query, context, progress):
            word_count += len(chunk_text.split())
            
            yield StreamEvent(
                event_type=StreamEventType.CONTENT_CHUNK,
                content=chunk_text,
                progress=0.25 + (word_count / 100) * 0.65
            )
            
            if (citation_idx < len(citation_indices) and 
                word_count >= citation_indices[citation_idx] and
                citation_idx < len(citations_to_add)):
                
                if self.config.stream_citations:
                    yield StreamEvent(
                        event_type=StreamEventType.CITATION_ADDED,
                        citation=citations_to_add[citation_idx],
                        progress=0.25 + (word_count / 100) * 0.65
                    )
                citation_idx += 1
        
        progress.generation_time_ms = (time.time() - generation_start) * 1000
        progress.current_phase = "completed"
        
        yield StreamEvent(
            event_type=StreamEventType.COMPLETED,
            metadata={
                "total_tokens": word_count,
                "generation_time_ms": progress.generation_time_ms,
                "citations_count": len(citations_to_add)
            },
            progress=1.0
        )
    
    async def collect_response(
        self,
        query: str,
        chunks: list[RetrievedChunk]
    ) -> StreamingResponse:
        """
        Collect all streaming events into a complete response.
        """
        response = StreamingResponse(query=query)
        
        async for event in self.stream_response(query, chunks):
            response.add_event(event)
        
        return response


class StreamingDisplay:
    """
    Utility for displaying streaming responses.
    """
    
    def __init__(self, show_events: bool = True):
        self.show_events = show_events
    
    async def display_stream(
        self,
        engine: StreamingRAGEngine,
        query: str,
        chunks: list[RetrievedChunk]
    ) -> StreamingResponse:
        """Display streaming response in real-time."""
        
        response = StreamingResponse(query=query)
        content_buffer = ""
        
        print(f"\n🔍 Query: {query}")
        print("-" * 50)
        
        async for event in engine.stream_response(query, chunks):
            response.add_event(event)
            
            if event.event_type == StreamEventType.STARTED:
                print("⏳ Starting...")
            
            elif event.event_type == StreamEventType.RETRIEVAL_COMPLETE:
                print(f"📚 Retrieved {event.metadata.get('chunks_found', 0)} chunks")
            
            elif event.event_type == StreamEventType.GENERATION_STARTED:
                print("✨ Generating response...")
                print()
            
            elif event.event_type == StreamEventType.CONTENT_CHUNK:
                if event.content:
                    print(event.content, end="", flush=True)
                    content_buffer += event.content
            
            elif event.event_type == StreamEventType.CITATION_ADDED:
                if event.citation and self.show_events:
                    print(f" [{event.citation.title or event.citation.source}]", end="")
            
            elif event.event_type == StreamEventType.COMPLETED:
                print("\n")
                print("-" * 50)
                print(f"✅ Complete in {response.duration_ms:.0f}ms")
        
        return response


async def demonstrate_streaming():
    """
    Demonstrate streaming RAG responses.
    """
    
    print("=" * 70)
    print("STREAMING RAG RESPONSES DEMONSTRATION")
    print("=" * 70)
    
    chunks = [
        RetrievedChunk(
            chunk_id="chunk_001",
            content="Full-time employees receive 20 days of paid vacation per year. "
                   "Vacation requests must be submitted at least two weeks in advance.",
            metadata=DocumentMetadata(
                source="hr/vacation_policy.md",
                title="Vacation Policy"
            ),
            similarity_score=0.95
        ),
        RetrievedChunk(
            chunk_id="chunk_002",
            content="Unused vacation days can be carried over, up to 5 days maximum. "
                   "Days exceeding this limit are forfeited on January 1st.",
            metadata=DocumentMetadata(
                source="hr/vacation_policy.md",
                title="Vacation Policy"
            ),
            similarity_score=0.88
        ),
        RetrievedChunk(
            chunk_id="chunk_003",
            content="Part-time employees receive prorated vacation based on hours.",
            metadata=DocumentMetadata(
                source="hr/vacation_policy.md",
                title="Vacation Policy"
            ),
            similarity_score=0.72
        ),
    ]
    
    print("\n" + "=" * 70)
    print("STREAMING WITH REAL-TIME DISPLAY")
    print("=" * 70)
    
    config = StreamConfig(
        chunk_delay_ms=30,
        words_per_chunk=3,
        stream_citations=True
    )
    
    engine = StreamingRAGEngine(config)
    display = StreamingDisplay(show_events=True)
    
    query = "What is the vacation policy for employees?"
    response = await display.display_stream(engine, query, chunks)
    
    print("\n" + "=" * 70)
    print("RESPONSE STATISTICS")
    print("=" * 70)
    
    print(f"\n📊 Statistics:")
    print(f"   Total Events: {len(response.events)}")
    print(f"   Total Duration: {response.duration_ms:.0f}ms")
    print(f"   Citations: {len(response.citations)}")
    print(f"   Final Content Length: {len(response.final_content)} chars")
    print(f"   Is Complete: {response.is_complete}")
    
    print(f"\n📚 Citations Added:")
    for citation in response.citations:
        print(f"   • {citation.title or citation.source} (relevance: {citation.relevance:.0%})")
    
    print("\n" + "=" * 70)
    print("EVENT TIMELINE")
    print("=" * 70)
    
    print("\n📅 Events (first 10):")
    for event in response.events[:10]:
        time_offset = (event.timestamp - response.started_at).total_seconds() * 1000
        content_preview = ""
        if event.content:
            content_preview = f": '{event.content[:20]}...'"
        elif event.citation:
            content_preview = f": {event.citation.source}"
        
        print(f"   [{time_offset:6.0f}ms] {event.event_type.value}{content_preview}")
    
    if len(response.events) > 10:
        print(f"   ... and {len(response.events) - 10} more events")
    
    print("\n" + "=" * 70)
    print("COLLECTED RESPONSE (NON-STREAMING)")
    print("=" * 70)
    
    fast_config = StreamConfig(chunk_delay_ms=10, words_per_chunk=10)
    fast_engine = StreamingRAGEngine(fast_config)
    
    collected = await fast_engine.collect_response(query, chunks)
    
    print(f"\n📝 Final Content (collected):")
    print(f"   {collected.final_content[:200]}...")
    print(f"\n   Duration: {collected.duration_ms:.0f}ms")
    print(f"   Events: {len(collected.events)}")
    
    print("\n" + "=" * 70)
    print("CONFIGURATION OPTIONS")
    print("=" * 70)
    
    configs = [
        ("Fast (10ms, 10 words)", StreamConfig(chunk_delay_ms=10, words_per_chunk=10)),
        ("Smooth (50ms, 3 words)", StreamConfig(chunk_delay_ms=50, words_per_chunk=3)),
        ("Typewriter (100ms, 1 word)", StreamConfig(chunk_delay_ms=100, words_per_chunk=1)),
    ]
    
    print("\n📊 Config Comparison:")
    for name, cfg in configs:
        test_engine = StreamingRAGEngine(cfg)
        test_response = await test_engine.collect_response(query, chunks)
        print(f"\n   {name}:")
        print(f"      Duration: {test_response.duration_ms:.0f}ms")
        print(f"      Events: {len(test_response.events)}")


def demonstrate_streaming_sync():
    """Synchronous wrapper for demonstration."""
    asyncio.run(demonstrate_streaming())


if __name__ == "__main__":
    demonstrate_streaming_sync()


"""
🧪 TEST & APPLY
===============

How to Test It:
---------------
    python lesson_15_streaming_rag_responses.py

Expected Output:
----------------
1. Real-time streaming display with progressive text
2. Response statistics
3. Event timeline
4. Collected response comparison
5. Configuration options comparison


⚠️ COMMON STUMBLING BLOCKS
==========================

1. "Async function not awaited"
   
   Error: RuntimeWarning: coroutine was never awaited
   
   Fix: Use asyncio.run() or await:
   asyncio.run(demonstrate_streaming())
   # or
   await engine.stream_response(...)

2. "Events arriving too fast/slow"
   
   Fix: Adjust chunk_delay_ms:
   config = StreamConfig(chunk_delay_ms=100)  # Slower
   config = StreamConfig(chunk_delay_ms=10)   # Faster

3. "Missing content in final response"
   
   Make sure to collect all events:
   async for event in engine.stream_response(...):
       response.add_event(event)

4. "Citations not appearing"
   
   Check config:
   config = StreamConfig(stream_citations=True)


🎯 KEY TAKEAWAYS
================

1. Streaming Improves UX
   - Immediate feedback
   - Lower perceived latency
   - Progressive content delivery

2. Event-Driven Architecture
   - Each event is typed and validated
   - Progress tracking built-in
   - Easy to extend with new event types

3. Configuration Flexibility
   - Adjust speed for your use case
   - Control chunk size
   - Enable/disable features

4. Collect or Stream
   - Stream for real-time display
   - Collect for batch processing


📚 NEXT LESSON
==============
In Lesson 16, we'll build the Complete RAG System - putting
everything together into a production-ready application!
"""
