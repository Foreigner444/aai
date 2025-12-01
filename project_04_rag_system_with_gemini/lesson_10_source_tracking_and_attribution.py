"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                LESSON 10: SOURCE TRACKING AND ATTRIBUTION                    ║
║                                                                              ║
║            Managing Information Flow Through Your RAG Pipeline               ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

📚 CONCEPT OVERVIEW
==================

What & Why:
-----------
Source tracking goes beyond citations - it's about maintaining a complete
audit trail of WHERE information came from, HOW it was processed, and
WHEN it was accessed. This is essential for:

- Compliance and auditing requirements
- Debugging incorrect responses
- Updating stale information
- Understanding answer provenance

🎯 Real-World Analogy:
----------------------
Think of source tracking like a food supply chain:

Simple Citation: "This beef comes from Texas"

Full Source Tracking:
- Ranch: "Sunny Valley Ranch, Amarillo, TX"
- Processing: "Packed at Central Processing on March 10"
- Transport: "Shipped via refrigerated truck #4521"
- Store: "Received at Store #123 on March 12, stored in cooler B"
- Checkout: "Purchased on March 14 at 3:42 PM"

If there's a problem, you can trace it back through every step!

🔒 Type Safety Benefit:
-----------------------
With Pydantic source tracking:
- Complete chain of custody is guaranteed
- Timestamps are always valid
- Processing steps are logged consistently
- Attribution is traceable end-to-end


💻 CODE IMPLEMENTATION
=====================
"""

import uuid
from typing import Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field, computed_field
from enum import Enum


class ProcessingStage(str, Enum):
    """Stages in the RAG processing pipeline."""
    INGESTION = "ingestion"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    STORAGE = "storage"
    RETRIEVAL = "retrieval"
    RANKING = "ranking"
    GENERATION = "generation"
    VALIDATION = "validation"


class SourceOrigin(BaseModel):
    """
    The original source of a document.
    Captures where the content originally came from.
    """
    
    origin_type: str = Field(
        description="Type: file, url, api, database, manual"
    )
    
    origin_path: str = Field(
        description="Path, URL, or identifier"
    )
    
    original_filename: Optional[str] = Field(
        default=None,
        description="Original filename if uploaded"
    )
    
    mime_type: Optional[str] = Field(
        default=None,
        description="MIME type of original file"
    )
    
    file_hash: Optional[str] = Field(
        default=None,
        description="Hash of original file for integrity"
    )
    
    created_by: Optional[str] = Field(
        default=None,
        description="Who created/uploaded this source"
    )
    
    created_at: datetime = Field(
        default_factory=datetime.now,
        description="When source was added to system"
    )
    
    last_modified: Optional[datetime] = Field(
        default=None,
        description="Last modification time of source"
    )


class ProcessingRecord(BaseModel):
    """
    A record of a processing step in the pipeline.
    """
    
    stage: ProcessingStage = Field(
        description="Which stage of processing"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When this processing occurred"
    )
    
    processor_name: str = Field(
        description="Name of the processor/model used"
    )
    
    processor_version: Optional[str] = Field(
        default=None,
        description="Version of the processor"
    )
    
    input_hash: Optional[str] = Field(
        default=None,
        description="Hash of input data"
    )
    
    output_hash: Optional[str] = Field(
        default=None,
        description="Hash of output data"
    )
    
    parameters: dict = Field(
        default_factory=dict,
        description="Parameters used in processing"
    )
    
    metrics: dict = Field(
        default_factory=dict,
        description="Metrics from processing (time, count, etc.)"
    )
    
    status: str = Field(
        default="success",
        description="Processing status"
    )
    
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if failed"
    )


class SourceTracker(BaseModel):
    """
    Complete tracking information for a piece of content.
    Maintains full provenance from origin through generation.
    """
    
    tracking_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique tracking identifier"
    )
    
    origin: SourceOrigin = Field(
        description="Original source information"
    )
    
    processing_history: list[ProcessingRecord] = Field(
        default_factory=list,
        description="All processing steps"
    )
    
    current_stage: ProcessingStage = Field(
        default=ProcessingStage.INGESTION,
        description="Current processing stage"
    )
    
    chunk_id: Optional[str] = Field(
        default=None,
        description="ID after chunking"
    )
    
    embedding_id: Optional[str] = Field(
        default=None,
        description="ID in vector store"
    )
    
    access_count: int = Field(
        default=0,
        ge=0,
        description="Times this source was accessed"
    )
    
    last_accessed: Optional[datetime] = Field(
        default=None,
        description="Last access time"
    )
    
    is_stale: bool = Field(
        default=False,
        description="Whether source needs refresh"
    )
    
    staleness_reason: Optional[str] = Field(
        default=None,
        description="Why source is stale"
    )
    
    def add_processing_record(
        self,
        stage: ProcessingStage,
        processor_name: str,
        **kwargs
    ) -> ProcessingRecord:
        """Add a new processing record."""
        record = ProcessingRecord(
            stage=stage,
            processor_name=processor_name,
            **kwargs
        )
        self.processing_history.append(record)
        self.current_stage = stage
        return record
    
    def record_access(self) -> None:
        """Record that this source was accessed."""
        self.access_count += 1
        self.last_accessed = datetime.now()
    
    def mark_stale(self, reason: str) -> None:
        """Mark this source as stale."""
        self.is_stale = True
        self.staleness_reason = reason
    
    @computed_field
    @property
    def age_hours(self) -> float:
        """How old this source is in hours."""
        age = datetime.now() - self.origin.created_at
        return age.total_seconds() / 3600
    
    @computed_field
    @property
    def processing_count(self) -> int:
        """Number of processing steps."""
        return len(self.processing_history)
    
    def get_processing_timeline(self) -> list[dict]:
        """Get timeline of all processing."""
        return [
            {
                "stage": record.stage.value,
                "timestamp": record.timestamp.isoformat(),
                "processor": record.processor_name,
                "status": record.status
            }
            for record in self.processing_history
        ]


class AttributionChain(BaseModel):
    """
    Chain of attribution from answer back to sources.
    Shows how each piece of the answer relates to sources.
    """
    
    answer_segment: str = Field(
        description="Part of the answer"
    )
    
    source_trackers: list[SourceTracker] = Field(
        description="Sources that contributed to this segment"
    )
    
    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Confidence in this attribution"
    )
    
    attribution_type: str = Field(
        default="direct",
        description="direct, inferred, synthesized"
    )
    
    @property
    def source_count(self) -> int:
        """Number of sources for this segment."""
        return len(self.source_trackers)
    
    @property
    def primary_source(self) -> Optional[SourceTracker]:
        """Get the primary (first) source."""
        return self.source_trackers[0] if self.source_trackers else None


class TrackedResponse(BaseModel):
    """
    A complete RAG response with full source tracking.
    """
    
    response_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique response identifier"
    )
    
    query: str = Field(
        description="Original query"
    )
    
    answer: str = Field(
        description="Generated answer"
    )
    
    attribution_chains: list[AttributionChain] = Field(
        default_factory=list,
        description="Attribution for answer segments"
    )
    
    all_sources: list[SourceTracker] = Field(
        default_factory=list,
        description="All sources consulted"
    )
    
    model_used: str = Field(
        description="Model that generated response"
    )
    
    generated_at: datetime = Field(
        default_factory=datetime.now,
        description="Generation timestamp"
    )
    
    generation_time_ms: float = Field(
        default=0.0,
        ge=0.0,
        description="Time to generate response"
    )
    
    @property
    def total_sources(self) -> int:
        """Total unique sources."""
        return len(self.all_sources)
    
    @property
    def has_attribution(self) -> bool:
        """Check if response has attribution."""
        return len(self.attribution_chains) > 0
    
    def get_stale_sources(self) -> list[SourceTracker]:
        """Get any stale sources used."""
        return [s for s in self.all_sources if s.is_stale]
    
    def generate_audit_report(self) -> dict:
        """Generate an audit report for this response."""
        return {
            "response_id": self.response_id,
            "query": self.query,
            "generated_at": self.generated_at.isoformat(),
            "model": self.model_used,
            "generation_time_ms": self.generation_time_ms,
            "total_sources": self.total_sources,
            "stale_sources": len(self.get_stale_sources()),
            "attribution_chains": len(self.attribution_chains),
            "sources": [
                {
                    "tracking_id": s.tracking_id,
                    "origin": s.origin.origin_path,
                    "stage": s.current_stage.value,
                    "access_count": s.access_count,
                    "age_hours": round(s.age_hours, 2)
                }
                for s in self.all_sources
            ]
        }


class SourceRegistry:
    """
    Registry for managing all tracked sources.
    Provides lookup, staleness checking, and statistics.
    """
    
    def __init__(self):
        self._sources: dict[str, SourceTracker] = {}
        self._by_origin: dict[str, str] = {}
    
    def register_source(self, tracker: SourceTracker) -> str:
        """Register a new source tracker."""
        self._sources[tracker.tracking_id] = tracker
        self._by_origin[tracker.origin.origin_path] = tracker.tracking_id
        return tracker.tracking_id
    
    def get_by_id(self, tracking_id: str) -> Optional[SourceTracker]:
        """Get tracker by ID."""
        return self._sources.get(tracking_id)
    
    def get_by_origin(self, origin_path: str) -> Optional[SourceTracker]:
        """Get tracker by origin path."""
        tracking_id = self._by_origin.get(origin_path)
        if tracking_id:
            return self._sources.get(tracking_id)
        return None
    
    def check_staleness(self, max_age_hours: float = 168) -> list[SourceTracker]:
        """Check for stale sources (default: 7 days)."""
        stale = []
        for tracker in self._sources.values():
            if tracker.age_hours > max_age_hours:
                tracker.mark_stale(f"Age exceeds {max_age_hours} hours")
                stale.append(tracker)
        return stale
    
    def get_statistics(self) -> dict:
        """Get registry statistics."""
        total = len(self._sources)
        if total == 0:
            return {"total_sources": 0}
        
        stale_count = sum(1 for s in self._sources.values() if s.is_stale)
        total_access = sum(s.access_count for s in self._sources.values())
        avg_age = sum(s.age_hours for s in self._sources.values()) / total
        
        return {
            "total_sources": total,
            "stale_sources": stale_count,
            "total_accesses": total_access,
            "average_age_hours": round(avg_age, 2),
            "sources_by_stage": self._count_by_stage()
        }
    
    def _count_by_stage(self) -> dict[str, int]:
        """Count sources by current stage."""
        counts: dict[str, int] = {}
        for tracker in self._sources.values():
            stage = tracker.current_stage.value
            counts[stage] = counts.get(stage, 0) + 1
        return counts


def demonstrate_source_tracking():
    """
    Demonstrate source tracking and attribution.
    """
    
    print("=" * 70)
    print("SOURCE TRACKING AND ATTRIBUTION DEMONSTRATION")
    print("=" * 70)
    
    registry = SourceRegistry()
    
    print("\n📦 CREATING SOURCE TRACKERS")
    print("-" * 50)
    
    source1 = SourceTracker(
        origin=SourceOrigin(
            origin_type="file",
            origin_path="hr/employee_handbook.pdf",
            original_filename="Employee_Handbook_2024.pdf",
            mime_type="application/pdf",
            file_hash="abc123def456",
            created_by="hr_admin",
            last_modified=datetime(2024, 1, 15)
        )
    )
    
    source2 = SourceTracker(
        origin=SourceOrigin(
            origin_type="url",
            origin_path="https://company.com/policies/vacation",
            created_by="system_crawler"
        )
    )
    
    registry.register_source(source1)
    registry.register_source(source2)
    
    print(f"\nSource 1: {source1.origin.origin_path}")
    print(f"   Tracking ID: {source1.tracking_id[:8]}...")
    print(f"   Type: {source1.origin.origin_type}")
    print(f"   Created by: {source1.origin.created_by}")
    
    print(f"\nSource 2: {source2.origin.origin_path}")
    print(f"   Tracking ID: {source2.tracking_id[:8]}...")
    print(f"   Type: {source2.origin.origin_type}")
    
    print("\n" + "=" * 70)
    print("RECORDING PROCESSING STEPS")
    print("=" * 70)
    
    source1.add_processing_record(
        stage=ProcessingStage.INGESTION,
        processor_name="PDFProcessor",
        processor_version="1.2.0",
        parameters={"extract_images": False},
        metrics={"pages_processed": 50, "time_ms": 1200}
    )
    
    source1.add_processing_record(
        stage=ProcessingStage.CHUNKING,
        processor_name="SentenceChunker",
        parameters={"chunk_size": 500, "overlap": 50},
        metrics={"chunks_created": 45, "avg_chunk_size": 480}
    )
    source1.chunk_id = "chunk_001"
    
    source1.add_processing_record(
        stage=ProcessingStage.EMBEDDING,
        processor_name="text-embedding-004",
        processor_version="004",
        parameters={"dimensions": 768},
        metrics={"embedding_time_ms": 85}
    )
    source1.embedding_id = "emb_001"
    
    source1.add_processing_record(
        stage=ProcessingStage.STORAGE,
        processor_name="ChromaDB",
        parameters={"collection": "company_docs"},
        metrics={"storage_time_ms": 12}
    )
    
    print(f"\n📜 Processing Timeline for Source 1:")
    for record in source1.get_processing_timeline():
        print(f"   {record['stage']:12} | {record['processor']:20} | {record['status']}")
    
    print(f"\n   Current Stage: {source1.current_stage.value}")
    print(f"   Processing Steps: {source1.processing_count}")
    print(f"   Age: {source1.age_hours:.2f} hours")
    
    print("\n" + "=" * 70)
    print("RECORDING ACCESS")
    print("=" * 70)
    
    print("\n🔍 Simulating document access...")
    
    source1.record_access()
    source1.record_access()
    source1.record_access()
    source2.record_access()
    
    print(f"\nSource 1 access count: {source1.access_count}")
    print(f"Source 1 last accessed: {source1.last_accessed}")
    print(f"Source 2 access count: {source2.access_count}")
    
    print("\n" + "=" * 70)
    print("CREATING TRACKED RESPONSE")
    print("=" * 70)
    
    source1.add_processing_record(
        stage=ProcessingStage.RETRIEVAL,
        processor_name="SimilaritySearch",
        parameters={"top_k": 5, "threshold": 0.7},
        metrics={"similarity_score": 0.92}
    )
    
    attribution1 = AttributionChain(
        answer_segment="Full-time employees receive 20 days of paid vacation",
        source_trackers=[source1],
        confidence=0.95,
        attribution_type="direct"
    )
    
    attribution2 = AttributionChain(
        answer_segment="Vacation requests require 2 weeks advance notice",
        source_trackers=[source1, source2],
        confidence=0.88,
        attribution_type="synthesized"
    )
    
    response = TrackedResponse(
        query="What is the vacation policy?",
        answer="Full-time employees receive 20 days of paid vacation per year. "
               "Vacation requests require 2 weeks advance notice through the HR portal.",
        attribution_chains=[attribution1, attribution2],
        all_sources=[source1, source2],
        model_used="gemini-1.5-flash",
        generation_time_ms=245.5
    )
    
    print(f"\n📋 Tracked Response:")
    print(f"   Response ID: {response.response_id[:8]}...")
    print(f"   Query: '{response.query}'")
    print(f"   Model: {response.model_used}")
    print(f"   Generation Time: {response.generation_time_ms:.1f}ms")
    print(f"   Total Sources: {response.total_sources}")
    print(f"   Attribution Chains: {len(response.attribution_chains)}")
    
    print("\n📚 Attribution Details:")
    for i, attr in enumerate(response.attribution_chains, 1):
        print(f"\n   Chain {i}: '{attr.answer_segment[:50]}...'")
        print(f"   Type: {attr.attribution_type}")
        print(f"   Confidence: {attr.confidence:.0%}")
        print(f"   Sources: {attr.source_count}")
        if attr.primary_source:
            print(f"   Primary: {attr.primary_source.origin.origin_path}")
    
    print("\n" + "=" * 70)
    print("AUDIT REPORT")
    print("=" * 70)
    
    audit = response.generate_audit_report()
    
    print(f"\n📊 Audit Report for Response {audit['response_id'][:8]}...")
    print(f"   Generated: {audit['generated_at']}")
    print(f"   Model: {audit['model']}")
    print(f"   Generation Time: {audit['generation_time_ms']}ms")
    print(f"   Total Sources: {audit['total_sources']}")
    print(f"   Stale Sources: {audit['stale_sources']}")
    
    print("\n   Source Details:")
    for src in audit['sources']:
        print(f"   - {src['origin']}")
        print(f"     Stage: {src['stage']}, Accesses: {src['access_count']}, Age: {src['age_hours']}h")
    
    print("\n" + "=" * 70)
    print("REGISTRY STATISTICS")
    print("=" * 70)
    
    stats = registry.get_statistics()
    
    print(f"\n📈 Registry Statistics:")
    print(f"   Total Sources: {stats['total_sources']}")
    print(f"   Stale Sources: {stats['stale_sources']}")
    print(f"   Total Accesses: {stats['total_accesses']}")
    print(f"   Average Age: {stats['average_age_hours']} hours")
    print(f"   By Stage: {stats['sources_by_stage']}")
    
    print("\n" + "=" * 70)
    print("STALENESS CHECKING")
    print("=" * 70)
    
    source2.mark_stale("Content updated on website")
    
    stale_sources = registry.check_staleness(max_age_hours=0.01)
    
    print(f"\n⚠️  Stale Sources Found: {len(stale_sources) + 1}")
    
    for source in [source2]:
        if source.is_stale:
            print(f"   - {source.origin.origin_path}")
            print(f"     Reason: {source.staleness_reason}")
    
    stale_in_response = response.get_stale_sources()
    print(f"\n   Stale sources in response: {len(stale_in_response)}")


if __name__ == "__main__":
    demonstrate_source_tracking()


"""
🧪 TEST & APPLY
===============

How to Test It:
---------------
    python lesson_10_source_tracking_and_attribution.py

Expected Output:
----------------
1. Source tracker creation with origins
2. Processing history timeline
3. Access recording demonstration
4. Tracked response with attribution chains
5. Audit report generation
6. Registry statistics
7. Staleness checking


⚠️ COMMON STUMBLING BLOCKS
==========================

1. "Tracking ID not found"
   
   Error: KeyError or None returned
   
   Fix: Always register sources before lookup:
   registry.register_source(tracker)
   later = registry.get_by_id(tracker.tracking_id)

2. "Processing history out of order"
   
   Processing records use current timestamp by default.
   
   Fix: Records are appended in call order. For testing,
   you can override timestamp:
   record = ProcessingRecord(timestamp=specific_time, ...)

3. "Attribution chain has no primary source"
   
   This happens when source_trackers is empty.
   
   Fix: Check before accessing:
   if attr.primary_source:
       print(attr.primary_source.origin.origin_path)

4. "Staleness not detected"
   
   Sources are only marked stale when explicitly checked.
   
   Fix: Call check_staleness periodically:
   stale = registry.check_staleness(max_age_hours=24)


🎯 KEY TAKEAWAYS
================

1. Full Provenance Tracking
   - Know exactly where data came from
   - Track every processing step
   - Enable complete auditing

2. Processing Timeline
   - Each stage is recorded
   - Parameters and metrics captured
   - Status and errors logged

3. Attribution Chains
   - Link answer segments to sources
   - Support multiple sources per claim
   - Track confidence levels

4. Staleness Management
   - Detect outdated sources
   - Warn users of stale data
   - Trigger refresh workflows


📚 NEXT LESSON
==============
In Lesson 11, we'll learn Multi-Document Retrieval - searching across
multiple collections and combining results effectively!
"""
