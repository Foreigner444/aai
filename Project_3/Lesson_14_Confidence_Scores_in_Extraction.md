# Lesson 14: Confidence Scores in Extraction

## A. Concept Overview

### What & Why
**Confidence scores quantify how certain the AI is about extracted data, enabling smart filtering and quality control.** Not all extractions are equal—some are obvious ("Tim Cook is CEO of Apple"), others ambiguous ("he said the company would..."). Confidence scores let you automatically filter, validate, and prioritize extracted data based on quality.

### Analogy
Think of confidence scores like credit scores for data:
- **High confidence (0.9+)**: Gold-rated, use immediately
- **Medium confidence (0.7-0.9)**: Silver-rated, use with caution
- **Low confidence (<0.7)**: Needs review before use

Just as you treat financial decisions differently based on credit scores, you handle extracted data differently based on confidence scores.

### Type Safety Benefit
Confidence-scored extraction provides **quantified reliability**:
- Numeric confidence for every extraction
- Type-safe thresholds (0.0 to 1.0)
- Automatic filtering by quality
- Graduated processing pipelines
- Audit trails showing why data was accepted/rejected
- Clear quality metrics for monitoring

---

## B. Key Patterns

**Pattern 1: Per-Field Confidence**
```python
class ConfidentField(BaseModel):
    value: str
    confidence: float = Field(..., ge=0.0, le=1.0)

class Person(BaseModel):
    name: ConfidentField
    age: Optional[ConfidentField] = None
    email: Optional[ConfidentField] = None
```

**Pattern 2: Entity-Level Confidence**
```python
class Person(BaseModel):
    name: str
    age: Optional[int] = None
    email: Optional[str] = None
    
    # Single confidence for entire entity
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall extraction confidence")
```

**Pattern 3: Tiered Confidence**
```python
class ConfidenceLevel(str, Enum):
    VERY_HIGH = "very_high"  # 0.95+
    HIGH = "high"  # 0.85-0.95
    MEDIUM = "medium"  # 0.70-0.85
    LOW = "low"  # 0.50-0.70
    VERY_LOW = "very_low"  # <0.50

class Entity(BaseModel):
    text: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    
    @property
    def confidence_level(self) -> ConfidenceLevel:
        if self.confidence >= 0.95:
            return ConfidenceLevel.VERY_HIGH
        elif self.confidence >= 0.85:
            return ConfidenceLevel.HIGH
        elif self.confidence >= 0.70:
            return ConfidenceLevel.MEDIUM
        elif self.confidence >= 0.50:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
```

**Pattern 4: Confidence with Evidence**
```python
class EvidencedExtraction(BaseModel):
    entity: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    evidence: List[str] = Field(..., description="Text snippets supporting extraction")
    source_sentences: List[str] = Field(..., description="Sentences entity appeared in")
```

---

## C. Confidence-Based Processing

**1. Automatic Filtering**
```python
class FilteredExtraction(BaseModel):
    all_entities: List[Entity]
    min_confidence: float = 0.7
    
    @property
    def high_confidence_entities(self) -> List[Entity]:
        return [e for e in self.all_entities if e.confidence >= self.min_confidence]
    
    @property
    def needs_review(self) -> List[Entity]:
        return [e for e in self.all_entities if e.confidence < self.min_confidence]
```

**2. Graduated Pipelines**
```python
def process_extraction(entity: Entity):
    """Process based on confidence level."""
    if entity.confidence >= 0.9:
        # Auto-approve and use immediately
        approve_automatically(entity)
    elif entity.confidence >= 0.7:
        # Use but mark for spot-check
        use_with_review_flag(entity)
    elif entity.confidence >= 0.5:
        # Queue for manual review
        queue_for_review(entity)
    else:
        # Reject automatically
        reject_low_quality(entity)
```

**3. Weighted Aggregation**
```python
class WeightedExtraction(BaseModel):
    entities: List[Entity]
    
    def get_average_confidence(self) -> float:
        if not self.entities:
            return 0.0
        return sum(e.confidence for e in self.entities) / len(self.entities)
    
    def get_weighted_entities(self) -> List[Entity]:
        """Return entities weighted by confidence."""
        return sorted(self.entities, key=lambda e: e.confidence, reverse=True)
```

**4. Confidence Calibration**
```python
class CalibratedEntity(BaseModel):
    text: str
    raw_confidence: float
    
    @property
    def calibrated_confidence(self) -> float:
        """Apply calibration based on entity type, length, etc."""
        confidence = self.raw_confidence
        
        # Penalize very short entities
        if len(self.text) < 3:
            confidence *= 0.8
        
        # Boost entities with proper capitalization
        if self.text[0].isupper():
            confidence *= 1.1
        
        return min(1.0, confidence)
```

---

## D. System Prompt for Confidence

```python
system_prompt = """
You are an expert at extracting entities with confidence scores.

For each extraction, provide a confidence score (0.0 to 1.0):

HIGH CONFIDENCE (0.9-1.0):
- Entity explicitly named with clear context
- No ambiguity in interpretation
- Multiple mentions support extraction
- Example: "Apple Inc. announced" → confidence: 0.95

MEDIUM CONFIDENCE (0.7-0.9):
- Entity clearly mentioned but with some ambiguity
- Context provides reasonable interpretation
- Single clear mention
- Example: "The company" (after mentioning Apple) → confidence: 0.8

LOW CONFIDENCE (0.5-0.7):
- Entity implied or inferred from context
- Some ambiguity in interpretation
- Unclear references
- Example: "They" (unclear antecedent) → confidence: 0.6

VERY LOW CONFIDENCE (<0.5):
- Highly ambiguous
- Multiple possible interpretations
- Insufficient context
- Example: "It" (unclear reference) → confidence: 0.4

Be honest about uncertainty. It's better to give low confidence than to
be overconfident about ambiguous extractions.
"""
```

---

## E. Best Practices

**1. Confidence Thresholds**
```python
class QualityThresholds(BaseModel):
    """Standard quality thresholds."""
    auto_accept: float = 0.90  # Automatically use
    needs_review: float = 0.70  # Use with review
    manual_review: float = 0.50  # Human review required
    auto_reject: float = 0.50  # Automatically discard
```

**2. Confidence Tracking**
```python
class ExtractionMetrics(BaseModel):
    total_entities: int
    high_confidence: int
    medium_confidence: int
    low_confidence: int
    average_confidence: float
    
    @property
    def quality_rate(self) -> float:
        """Percentage of high-confidence extractions."""
        return self.high_confidence / self.total_entities if self.total_entities > 0 else 0.0
```

**3. Confidence Propagation**
```python
class Relationship(BaseModel):
    subject: str
    predicate: str
    object: str
    subject_confidence: float
    object_confidence: float
    
    @property
    def relationship_confidence(self) -> float:
        """Relationship confidence is minimum of entity confidences."""
        return min(self.subject_confidence, self.object_confidence)
```

**4. Confidence Visualization**
```python
def format_with_confidence(entity: str, confidence: float) -> str:
    """Format entity with confidence indicator."""
    if confidence >= 0.9:
        return f"{entity} ✅"  # High confidence
    elif confidence >= 0.7:
        return f"{entity} ⚠️"  # Medium confidence
    else:
        return f"{entity} ❓"  # Low confidence
```

---

## F. Common Stumbling Blocks

### Proactive Debugging

**Mistake 1: Treating all confidences equally**
```python
# ❌ Using low-confidence data without checking
for entity in extraction.entities:
    process(entity)  # Some might be unreliable!

# ✅ Filter by confidence
for entity in extraction.entities:
    if entity.confidence >= 0.7:
        process(entity)
    else:
        queue_for_review(entity)
```

**Mistake 2: No confidence calibration**
```python
# ❌ Raw confidence might not be accurate
confidence = 0.95  # Model says 95%, but is it calibrated?

# ✅ Calibrate based on validation data
calibrated_confidence = calibrate(0.95, entity_type="person", text_length=len(text))
```

**Mistake 3: Binary accept/reject**
```python
# ❌ Only two states: use or reject
if confidence >= 0.7:
    use(entity)
else:
    reject(entity)

# ✅ Graduated processing
if confidence >= 0.9:
    auto_approve(entity)
elif confidence >= 0.7:
    use_with_flag(entity)
elif confidence >= 0.5:
    queue_for_review(entity)
else:
    reject(entity)
```

### Type Safety Gotchas

1. **Confidence range**: Always constrain to [0.0, 1.0] with `ge=0.0, le=1.0`
2. **Percentage vs decimal**: 0.9 = 90%, not 9%
3. **Aggregation**: Combining confidences requires careful math (min, average, product?)
4. **Calibration**: Raw scores may not reflect true confidence
5. **Threshold tuning**: Optimal thresholds depend on use case

---

## 🎯 Next Steps

Excellent! You now understand:
- ✅ How to work with confidence scores
- ✅ How to filter extractions by quality
- ✅ How to build graduated processing pipelines
- ✅ How to track extraction quality metrics
- ✅ How to calibrate and use confidence scores effectively

In the next lesson, **Building Extraction Pipeline**, we'll combine all these patterns into a production-ready pipeline.

**Ready for Lesson 15?** 🚀
