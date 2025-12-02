# Lesson 15: Building Extraction Pipeline

## A. Concept Overview

### What & Why
**An extraction pipeline orchestrates the entire flow from raw text to validated structured data, handling errors, retries, filtering, and quality control.** Individual extractions are useful, but production systems need robust pipelines that process text at scale, handle failures gracefully, track metrics, and ensure data quality end-to-end.

### Analogy
Think of an extraction pipeline like a manufacturing assembly line:
- **Raw material**: Unstructured text input
- **Quality check**: Input validation
- **Processing stations**: Extraction agents
- **Quality control**: Confidence filtering
- **Packaging**: Output formatting
- **Shipping**: Storing/delivering results

Each stage has clear inputs, outputs, error handling, and quality checks.

### Type Safety Benefit
Pipeline design with Pydantic provides **end-to-end type safety**:
- Type-safe input validation
- Type-safe transformation at each stage
- Type-safe error handling
- Type-safe metrics collection
- Validated outputs guaranteed
- Clear contracts between pipeline stages

---

## B. Pipeline Architecture

**Pipeline Stages:**

```
Input → Validation → Extraction → Filtering → Enrichment → Output
  ↓         ↓           ↓           ↓            ↓          ↓
Error   Rejection   Retry      Low Quality  Enhancement  Success
```

**Core Components:**

1. **Input Validator**: Validates raw text
2. **Extractor**: Gemini-powered extraction
3. **Quality Filter**: Confidence-based filtering
4. **Enricher**: Add computed fields, relationships
5. **Output Formatter**: Format for downstream systems
6. **Error Handler**: Handle failures gracefully
7. **Metrics Collector**: Track performance

---

## C. Implementation Pattern

```python
from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum
from datetime import datetime


class PipelineStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILED = "failed"
    REJECTED = "rejected"


class PipelineInput(BaseModel):
    """Pipeline input validation."""
    text: str = Field(..., min_length=10, max_length=100000)
    source_id: str
    metadata: dict = Field(default_factory=dict)
    
    @property
    def word_count(self) -> int:
        return len(self.text.split())


class ExtractionStage(BaseModel):
    """Result from extraction stage."""
    entities: List[Entity]
    raw_confidence: float
    extraction_time_ms: int


class FilteringStage(BaseModel):
    """Result from filtering stage."""
    accepted_entities: List[Entity]
    rejected_entities: List[Entity]
    acceptance_rate: float


class EnrichmentStage(BaseModel):
    """Result from enrichment stage."""
    enriched_entities: List[Entity]
    relationships: List[Relationship]
    computed_fields: dict


class PipelineOutput(BaseModel):
    """Final pipeline output."""
    status: PipelineStatus
    input_id: str
    
    # Successful extraction data
    entities: List[Entity] = Field(default_factory=list)
    relationships: List[Relationship] = Field(default_factory=list)
    
    # Quality metrics
    confidence: float = 0.0
    entity_count: int = 0
    
    # Pipeline metadata
    processing_time_ms: int
    stages_completed: List[str] = Field(default_factory=list)
    
    # Error info (if failed)
    error_message: Optional[str] = None
    error_stage: Optional[str] = None


class Pipeline:
    """Extraction pipeline orchestrator."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.metrics = PipelineMetrics()
    
    async def process(self, text: str, source_id: str) -> PipelineOutput:
        """Process text through full pipeline."""
        start_time = datetime.now()
        
        try:
            # Stage 1: Validate Input
            pipeline_input = self._validate_input(text, source_id)
            
            # Stage 2: Extract Entities
            extraction = await self._extract(pipeline_input)
            
            # Stage 3: Filter by Quality
            filtered = self._filter(extraction)
            
            # Stage 4: Enrich Data
            enriched = self._enrich(filtered)
            
            # Stage 5: Format Output
            output = self._format_output(enriched, source_id, start_time)
            
            self.metrics.record_success()
            return output
            
        except ValidationError as e:
            return self._handle_validation_error(e, source_id, start_time)
        except ExtractionError as e:
            return self._handle_extraction_error(e, source_id, start_time)
        except Exception as e:
            return self._handle_unexpected_error(e, source_id, start_time)
    
    def _validate_input(self, text: str, source_id: str) -> PipelineInput:
        """Stage 1: Validate input."""
        return PipelineInput(text=text, source_id=source_id)
    
    async def _extract(self, pipeline_input: PipelineInput) -> ExtractionStage:
        """Stage 2: Extract entities with Gemini."""
        # Call Pydantic AI agent
        result = await self.extractor.run(pipeline_input.text)
        
        return ExtractionStage(
            entities=result.data.entities,
            raw_confidence=result.data.confidence,
            extraction_time_ms=result.processing_time
        )
    
    def _filter(self, extraction: ExtractionStage) -> FilteringStage:
        """Stage 3: Filter by confidence threshold."""
        accepted = [
            e for e in extraction.entities 
            if e.confidence >= self.config.min_confidence
        ]
        rejected = [
            e for e in extraction.entities 
            if e.confidence < self.config.min_confidence
        ]
        
        return FilteringStage(
            accepted_entities=accepted,
            rejected_entities=rejected,
            acceptance_rate=len(accepted) / len(extraction.entities) if extraction.entities else 0
        )
    
    def _enrich(self, filtered: FilteringStage) -> EnrichmentStage:
        """Stage 4: Enrich with computed fields and relationships."""
        # Add relationships between entities
        relationships = self._extract_relationships(filtered.accepted_entities)
        
        # Add computed fields
        enriched = self._add_computed_fields(filtered.accepted_entities)
        
        return EnrichmentStage(
            enriched_entities=enriched,
            relationships=relationships,
            computed_fields={}
        )
    
    def _format_output(
        self, 
        enriched: EnrichmentStage, 
        source_id: str,
        start_time: datetime
    ) -> PipelineOutput:
        """Stage 5: Format final output."""
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return PipelineOutput(
            status=PipelineStatus.SUCCESS,
            input_id=source_id,
            entities=enriched.enriched_entities,
            relationships=enriched.relationships,
            confidence=self._calculate_overall_confidence(enriched.enriched_entities),
            entity_count=len(enriched.enriched_entities),
            processing_time_ms=int(processing_time),
            stages_completed=["validation", "extraction", "filtering", "enrichment", "formatting"]
        )
```

---

## D. Pipeline Patterns

**1. Retry Logic**
```python
async def extract_with_retry(text: str, max_retries: int = 3) -> ExtractionResult:
    """Extract with exponential backoff retry."""
    for attempt in range(max_retries):
        try:
            return await extractor.run(text)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            wait_time = 2 ** attempt  # Exponential backoff
            await asyncio.sleep(wait_time)
```

**2. Batch Processing**
```python
async def process_batch(texts: List[str]) -> List[PipelineOutput]:
    """Process multiple texts in parallel."""
    tasks = [pipeline.process(text, f"batch-{i}") for i, text in enumerate(texts)]
    return await asyncio.gather(*tasks, return_exceptions=True)
```

**3. Progressive Enhancement**
```python
def enhance_progressively(entities: List[Entity]) -> List[Entity]:
    """Add enhancements based on confidence."""
    enhanced = []
    for entity in entities:
        if entity.confidence >= 0.9:
            # High confidence: add full enrichment
            entity = add_full_enrichment(entity)
        elif entity.confidence >= 0.7:
            # Medium confidence: basic enrichment
            entity = add_basic_enrichment(entity)
        # Low confidence: no enrichment
        enhanced.append(entity)
    return enhanced
```

**4. Metrics Collection**
```python
class PipelineMetrics(BaseModel):
    total_processed: int = 0
    successful: int = 0
    failed: int = 0
    average_confidence: float = 0.0
    average_processing_time_ms: float = 0.0
    
    def record_success(self, confidence: float, time_ms: int):
        self.total_processed += 1
        self.successful += 1
        self._update_averages(confidence, time_ms)
    
    @property
    def success_rate(self) -> float:
        return self.successful / self.total_processed if self.total_processed > 0 else 0.0
```

---

## E. Error Handling

```python
class PipelineError(Exception):
    """Base pipeline error."""
    pass

class ValidationError(PipelineError):
    """Input validation failed."""
    pass

class ExtractionError(PipelineError):
    """Extraction stage failed."""
    pass

class FilteringError(PipelineError):
    """Filtering stage failed."""
    pass


def handle_error(error: Exception, stage: str) -> PipelineOutput:
    """Handle pipeline errors gracefully."""
    return PipelineOutput(
        status=PipelineStatus.FAILED,
        input_id="error",
        error_message=str(error),
        error_stage=stage,
        processing_time_ms=0
    )
```

---

## F. Best Practices

**1. Fail Fast on Invalid Input**
```python
# Validate input before expensive operations
if len(text) < 10:
    raise ValidationError("Text too short")
if len(text) > 100000:
    raise ValidationError("Text too long")
```

**2. Track Progress**
```python
@dataclass
class PipelineProgress:
    stage: str
    progress: float  # 0.0 to 1.0
    message: str

def report_progress(stage: str, progress: float, message: str):
    # Log or emit progress events
    pass
```

**3. Circuit Breaker**
```python
class CircuitBreaker:
    """Prevent cascading failures."""
    
    def __init__(self, failure_threshold: int = 5):
        self.failures = 0
        self.threshold = failure_threshold
        self.is_open = False
    
    def record_failure(self):
        self.failures += 1
        if self.failures >= self.threshold:
            self.is_open = True
    
    def should_allow_request(self) -> bool:
        return not self.is_open
```

---

## 🎯 Next Steps

Excellent work! You now understand:
- ✅ How to design extraction pipelines
- ✅ How to handle errors at each stage
- ✅ How to implement retry logic
- ✅ How to collect metrics
- ✅ How to process data in batches

In the final lesson, **Complete Data Extraction System**, we'll build a full end-to-end system integrating everything you've learned!

**Ready for Lesson 16 (Final Lesson)?** 🚀
