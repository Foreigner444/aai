# Lesson 8: Field Constraints and Limits

## A. Concept Overview

### What & Why
**Field constraints provide declarative validation rules directly in your field definitions, making common validation patterns simple and readable.** Instead of writing custom validators for every range check, length limit, or pattern match, Pydantic's `Field()` function provides built-in constraints for numbers, strings, collections, and more. This makes your models self-documenting and ensures data extracted by Gemini meets precise specifications.

### Analogy
Think of field constraints like building codes for construction:
- **Without constraints**: "Build a room" (no specifications—could be any size, shape, or quality)
- **With constraints**: "Build a room that's 10-20 feet wide, 12-25 feet long, with 8-10 foot ceilings" (precise, enforceable specifications)

When Gemini extracts data, field constraints are those building codes—they specify exact requirements that are automatically checked and enforced.

### Type Safety Benefit
Field constraints provide **declarative validation**:
- Self-documenting—the field definition shows all constraints at a glance
- Type-checked—constraints are validated at model creation time
- IDE-visible—constraints appear in IDE tooltips and documentation
- Automatic errors—violations produce clear, specific error messages
- No custom code needed—common validations are built-in
- Performance—constraints are optimized and faster than custom validators

---

## B. Code Implementation

### File Structure
```
data_extraction_pipeline/
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── constraint_models.py  # New: Field constraint patterns
│   └── examples/
│       ├── __init__.py
│       └── constraint_demo.py     # New: This lesson
```

### Complete Code Implementation

**File: `src/models/constraint_models.py`**

```python
"""Models demonstrating field constraints and limits."""

from pydantic import BaseModel, Field
from typing import List, Optional, Set
from datetime import datetime
from decimal import Decimal


# ============================================================================
# NUMERIC CONSTRAINTS
# ============================================================================

class NumericConstraints(BaseModel):
    """Numeric field constraints."""
    # Integer constraints
    age: int = Field(..., ge=0, le=150, description="Age in years")
    score: int = Field(..., gt=0, lt=100, description="Score (exclusive bounds)")
    count: int = Field(..., ge=1, description="Count (minimum 1)")
    
    # Float constraints
    price: float = Field(..., gt=0.0, description="Price (must be positive)")
    discount_rate: float = Field(..., ge=0.0, le=100.0, description="Discount percentage")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    
    # Decimal for precise money
    precise_price: Decimal = Field(..., ge=Decimal("0.01"), max_digits=10, decimal_places=2)
    
    # Multiple constraints
    temperature_f: float = Field(
        ...,
        ge=-459.67,  # Absolute zero in Fahrenheit
        le=1000.0,
        description="Temperature in Fahrenheit"
    )


class RangeValidation(BaseModel):
    """Common range validation patterns."""
    # Percentage (0-100)
    percentage: float = Field(..., ge=0.0, le=100.0)
    
    # Probability (0-1)
    probability: float = Field(..., ge=0.0, le=1.0)
    
    # Rating (1-5)
    rating: int = Field(..., ge=1, le=5)
    
    # Year (reasonable range)
    year: int = Field(..., ge=1900, le=2100)
    
    # Page number (positive)
    page: int = Field(..., ge=1)
    
    # Index (non-negative)
    index: int = Field(..., ge=0)


# ============================================================================
# STRING CONSTRAINTS
# ============================================================================

class StringConstraints(BaseModel):
    """String field constraints."""
    # Length constraints
    username: str = Field(..., min_length=3, max_length=20)
    password: str = Field(..., min_length=8, max_length=128)
    title: str = Field(..., min_length=1, max_length=200)
    description: str = Field(..., max_length=1000)
    
    # Pattern matching (regex)
    email: str = Field(..., pattern=r'^[\w\.-]+@[\w\.-]+\.\w+$')
    phone: str = Field(..., pattern=r'^\+?[\d\s\-\(\)]+$')
    zip_code: str = Field(..., pattern=r'^\d{5}(-\d{4})?$')
    
    # Exact length
    state_code: str = Field(..., min_length=2, max_length=2, pattern=r'^[A-Z]{2}$')
    
    # URL pattern
    website: str = Field(
        ...,
        pattern=r'^https?://[^\s/$.?#].[^\s]*$',
        description="Valid HTTP/HTTPS URL"
    )


class TextContent(BaseModel):
    """Text content with length limits."""
    # Short text
    tag: str = Field(..., min_length=1, max_length=50)
    
    # Medium text
    excerpt: str = Field(..., min_length=10, max_length=500)
    
    # Long text
    body: str = Field(..., min_length=100, max_length=50000)
    
    # Optional with max length
    notes: Optional[str] = Field(None, max_length=2000)


# ============================================================================
# COLLECTION CONSTRAINTS
# ============================================================================

class CollectionConstraints(BaseModel):
    """Collection field constraints."""
    # List length constraints
    tags: List[str] = Field(..., min_length=1, max_length=10)
    authors: List[str] = Field(..., min_length=1, max_length=5)
    keywords: List[str] = Field(..., min_length=3, max_length=20)
    
    # Empty collections allowed
    comments: List[str] = Field(default_factory=list, max_length=100)
    
    # Set with size constraints
    unique_ids: Set[str] = Field(..., min_length=1)
    
    # List of numbers with constraints
    scores: List[float] = Field(
        ...,
        min_length=1,
        max_length=10,
        description="List of scores between 0 and 1"
    )


class EntityExtraction(BaseModel):
    """Entity extraction with constrained collections."""
    # Must extract at least one entity
    people: List[str] = Field(..., min_length=1, max_length=50)
    
    # Organizations optional but limited
    organizations: List[str] = Field(default_factory=list, max_length=20)
    
    # Locations optional but limited
    locations: List[str] = Field(default_factory=list, max_length=30)
    
    # Confidence scores must match entity count (validated separately)
    confidence_scores: List[float] = Field(..., min_length=1, max_length=50)


# ============================================================================
# COMBINED CONSTRAINTS
# ============================================================================

class ProductListing(BaseModel):
    """Product listing with multiple constraint types."""
    # String constraints
    name: str = Field(..., min_length=1, max_length=100)
    description: str = Field(..., min_length=10, max_length=2000)
    sku: str = Field(..., pattern=r'^[A-Z0-9\-]+$', min_length=5, max_length=20)
    
    # Numeric constraints
    price: Decimal = Field(..., ge=Decimal("0.01"), le=Decimal("1000000.00"), decimal_places=2)
    quantity: int = Field(..., ge=0, le=10000)
    weight_kg: float = Field(..., gt=0.0, le=1000.0)
    
    # Collection constraints
    categories: List[str] = Field(..., min_length=1, max_length=5)
    images: List[str] = Field(..., min_length=1, max_length=10)
    tags: List[str] = Field(default_factory=list, max_length=20)
    
    # Optional with constraints
    discount_percentage: Optional[float] = Field(None, ge=0.0, le=100.0)
    rating: Optional[float] = Field(None, ge=0.0, le=5.0)


class UserProfile(BaseModel):
    """User profile with comprehensive constraints."""
    # Identity
    user_id: str = Field(..., pattern=r'^[a-z0-9\-]+$', min_length=5, max_length=50)
    username: str = Field(..., min_length=3, max_length=20, pattern=r'^[a-zA-Z0-9_]+$')
    email: str = Field(..., pattern=r'^[\w\.-]+@[\w\.-]+\.\w+$')
    
    # Personal info with constraints
    age: int = Field(..., ge=13, le=120)
    bio: Optional[str] = Field(None, max_length=500)
    
    # Skills with limits
    skills: List[str] = Field(..., min_length=1, max_length=50)
    
    # Social links (optional, limited)
    social_links: List[str] = Field(default_factory=list, max_length=10)
    
    # Preferences
    notification_frequency: int = Field(..., ge=0, le=24, description="Hours between notifications")


# ============================================================================
# EXTRACTION RESULT CONSTRAINTS
# ============================================================================

class ExtractedEntity(BaseModel):
    """Single extracted entity with constraints."""
    text: str = Field(..., min_length=1, max_length=500)
    entity_type: str = Field(..., min_length=1, max_length=50)
    confidence: float = Field(..., ge=0.0, le=1.0)
    
    # Optional position info
    start_pos: Optional[int] = Field(None, ge=0)
    end_pos: Optional[int] = Field(None, ge=0)
    
    # Optional context
    context: Optional[str] = Field(None, max_length=1000)


class ExtractionResult(BaseModel):
    """Complete extraction result with constraints."""
    # Required metadata
    extraction_id: str = Field(..., pattern=r'^[a-z0-9\-]+$')
    source_text: str = Field(..., min_length=1, max_length=100000)
    
    # Extracted entities (at least one)
    entities: List[ExtractedEntity] = Field(..., min_length=1, max_length=1000)
    
    # Summary (optional, limited)
    summary: Optional[str] = Field(None, min_length=10, max_length=5000)
    
    # Metadata with constraints
    processing_time_ms: int = Field(..., ge=0, le=300000)  # Max 5 minutes
    model_version: str = Field(..., pattern=r'^\d+\.\d+\.\d+$')  # Semantic versioning
    
    # Quality metrics
    avg_confidence: float = Field(..., ge=0.0, le=1.0)
    entity_count: int = Field(..., ge=1, le=1000)


class BatchExtractionJob(BaseModel):
    """Batch extraction job with operational constraints."""
    # Job metadata
    job_id: str = Field(..., pattern=r'^job-[0-9]+$')
    created_at: datetime
    
    # Batch configuration
    batch_size: int = Field(..., ge=1, le=1000, description="Items per batch")
    max_retries: int = Field(..., ge=0, le=5, description="Maximum retry attempts")
    timeout_seconds: int = Field(..., ge=10, le=3600, description="Timeout per item")
    
    # Input constraints
    input_texts: List[str] = Field(..., min_length=1, max_length=10000)
    
    # Progress tracking
    processed_count: int = Field(default=0, ge=0)
    failed_count: int = Field(default=0, ge=0)
    success_rate: float = Field(default=0.0, ge=0.0, le=100.0)


# ============================================================================
# QUALITY CONSTRAINTS
# ============================================================================

class QualityConstraints(BaseModel):
    """Extraction quality constraints."""
    # Minimum requirements
    min_confidence: float = Field(..., ge=0.0, le=1.0)
    min_entity_count: int = Field(..., ge=1, le=100)
    max_entity_count: int = Field(..., ge=1, le=1000)
    
    # Text quality
    min_text_length: int = Field(..., ge=10, le=10000)
    max_text_length: int = Field(..., ge=100, le=1000000)
    
    # Processing limits
    max_processing_time_ms: int = Field(..., ge=100, le=60000)
    
    # Thresholds
    acceptable_error_rate: float = Field(..., ge=0.0, le=50.0, description="Max error rate %")


class ValidatedExtraction(BaseModel):
    """Extraction that meets quality constraints."""
    text: str = Field(..., min_length=10, max_length=10000)
    confidence: float = Field(..., ge=0.7, description="Minimum 70% confidence")
    entity_count: int = Field(..., ge=1, le=100)
    
    # Must be high quality
    quality_score: float = Field(..., ge=0.8, le=1.0, description="Minimum 80% quality")
    
    # Processing must be efficient
    processing_time_ms: int = Field(..., ge=0, le=5000, description="Max 5 seconds")


# ============================================================================
# DOCUMENTATION CONSTRAINTS
# ============================================================================

class DocumentMetadata(BaseModel):
    """Document metadata with extensive constraints."""
    # Required fields with constraints
    title: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Document title",
        examples=["Introduction to Pydantic AI"]
    )
    
    author: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Author name",
        examples=["John Doe"]
    )
    
    word_count: int = Field(
        ...,
        ge=100,
        le=1000000,
        description="Total word count",
        examples=[1500]
    )
    
    # Quality indicators with constraints
    readability_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Flesch reading ease score (0-100)",
        examples=[65.5]
    )
    
    # Metadata collections with limits
    keywords: List[str] = Field(
        ...,
        min_length=3,
        max_length=20,
        description="Document keywords for search",
        examples=[["pydantic", "ai", "python"]]
    )
    
    # Optional fields with constraints
    abstract: Optional[str] = Field(
        None,
        min_length=50,
        max_length=500,
        description="Document abstract/summary"
    )
```

**File: `src/examples/constraint_demo.py`**

```python
"""Demonstration of field constraints and limits."""

from datetime import datetime
from decimal import Decimal
from pydantic import ValidationError
from src.models.constraint_models import (
    NumericConstraints,
    RangeValidation,
    StringConstraints,
    TextContent,
    CollectionConstraints,
    ProductListing,
    UserProfile,
    ExtractedEntity,
    ExtractionResult,
    QualityConstraints,
    ValidatedExtraction,
    DocumentMetadata,
)


def demo_numeric_constraints():
    """Demonstrate numeric field constraints."""
    print("=" * 70)
    print("NUMERIC CONSTRAINTS")
    print("=" * 70)
    
    # Valid numeric values
    print("✅ Valid numeric constraints:")
    data = NumericConstraints(
        age=25,
        score=75,
        count=10,
        price=99.99,
        discount_rate=15.5,
        confidence=0.95,
        precise_price=Decimal("19.99"),
        temperature_f=72.5
    )
    print(f"  Age: {data.age} (valid: 0-150)")
    print(f"  Confidence: {data.confidence} (valid: 0.0-1.0)")
    print(f"  Price: ${data.precise_price} (decimal precision)")
    
    # Out of range
    print("\n❌ Age out of range:")
    try:
        data = NumericConstraints(
            age=200,  # > 150
            score=50,
            count=1,
            price=10.0,
            discount_rate=10.0,
            confidence=0.5,
            precise_price=Decimal("10.00"),
            temperature_f=70.0
        )
    except ValidationError as e:
        print(f"  Error: {e.errors()[0]['msg']}")
    
    # Negative where positive required
    print("\n❌ Negative price:")
    try:
        data = NumericConstraints(
            age=25,
            score=50,
            count=1,
            price=-10.0,  # Must be > 0
            discount_rate=10.0,
            confidence=0.5,
            precise_price=Decimal("10.00"),
            temperature_f=70.0
        )
    except ValidationError as e:
        print(f"  Error: {e.errors()[0]['msg']}")
    
    print()


def demo_range_validation():
    """Demonstrate common range patterns."""
    print("=" * 70)
    print("RANGE VALIDATION PATTERNS")
    print("=" * 70)
    
    # Valid ranges
    ranges = RangeValidation(
        percentage=75.5,
        probability=0.85,
        rating=4,
        year=2024,
        page=42,
        index=0
    )
    
    print(f"Percentage: {ranges.percentage}% (0-100)")
    print(f"Probability: {ranges.probability} (0-1)")
    print(f"Rating: {ranges.rating}/5 (1-5)")
    print(f"Year: {ranges.year} (1900-2100)")
    
    print()


def demo_string_constraints():
    """Demonstrate string field constraints."""
    print("=" * 70)
    print("STRING CONSTRAINTS")
    print("=" * 70)
    
    # Valid strings
    print("✅ Valid string constraints:")
    data = StringConstraints(
        username="alice",
        password="SecurePass123!",
        title="Introduction to Pydantic AI",
        description="Learn type-safe AI development",
        email="alice@example.com",
        phone="+1-555-0100",
        zip_code="94105",
        state_code="CA",
        website="https://example.com"
    )
    print(f"  Username: {data.username} (3-20 chars)")
    print(f"  Email: {data.email} (pattern match)")
    print(f"  ZIP: {data.zip_code} (pattern: \\d{{5}}(-\\d{{4}})?)")
    
    # Too short
    print("\n❌ Username too short:")
    try:
        data = StringConstraints(
            username="ab",  # < 3 chars
            password="SecurePass123!",
            title="Test",
            description="Test description",
            email="test@example.com",
            phone="555-0100",
            zip_code="94105",
            state_code="CA",
            website="https://example.com"
        )
    except ValidationError as e:
        print(f"  Error: {e.errors()[0]['msg']}")
    
    # Pattern mismatch
    print("\n❌ Invalid email pattern:")
    try:
        data = StringConstraints(
            username="alice",
            password="SecurePass123!",
            title="Test",
            description="Test description",
            email="invalid-email",  # No @ symbol
            phone="555-0100",
            zip_code="94105",
            state_code="CA",
            website="https://example.com"
        )
    except ValidationError as e:
        print(f"  Error: {e.errors()[0]['msg']}")
    
    print()


def demo_collection_constraints():
    """Demonstrate collection field constraints."""
    print("=" * 70)
    print("COLLECTION CONSTRAINTS")
    print("=" * 70)
    
    # Valid collections
    print("✅ Valid collection constraints:")
    data = CollectionConstraints(
        tags=["python", "ai", "typing"],
        authors=["Alice Johnson"],
        keywords=["pydantic", "type-safety", "validation"],
        scores=[0.95, 0.87, 0.92]
    )
    print(f"  Tags: {len(data.tags)} items (1-10 allowed)")
    print(f"  Keywords: {len(data.keywords)} items (3-20 required)")
    
    # Too few items
    print("\n❌ Too few keywords:")
    try:
        data = CollectionConstraints(
            tags=["python"],
            authors=["Alice"],
            keywords=["pydantic"],  # Need 3-20, only have 1
            scores=[0.95]
        )
    except ValidationError as e:
        print(f"  Error: {e.errors()[0]['msg']}")
    
    # Too many items
    print("\n❌ Too many tags:")
    try:
        data = CollectionConstraints(
            tags=["tag" + str(i) for i in range(15)],  # > 10
            authors=["Alice"],
            keywords=["k1", "k2", "k3"],
            scores=[0.5]
        )
    except ValidationError as e:
        print(f"  Error: {e.errors()[0]['msg']}")
    
    print()


def demo_combined_constraints():
    """Demonstrate models with multiple constraint types."""
    print("=" * 70)
    print("COMBINED CONSTRAINTS (Product)")
    print("=" * 70)
    
    # Valid product
    print("✅ Valid product listing:")
    product = ProductListing(
        name="Wireless Mouse",
        description="High-precision wireless mouse with ergonomic design",
        sku="MOUSE-2024-BLK",
        price=Decimal("29.99"),
        quantity=150,
        weight_kg=0.125,
        categories=["Electronics", "Accessories"],
        images=["image1.jpg", "image2.jpg"],
        tags=["wireless", "mouse", "ergonomic"],
        discount_percentage=10.0,
        rating=4.5
    )
    print(f"  Name: {product.name}")
    print(f"  SKU: {product.sku}")
    print(f"  Price: ${product.price}")
    print(f"  Categories: {product.categories}")
    
    # Invalid: negative quantity
    print("\n❌ Negative quantity:")
    try:
        product = ProductListing(
            name="Test Product",
            description="Test description for product",
            sku="TEST-001",
            price=Decimal("10.00"),
            quantity=-5,  # Can't be negative
            weight_kg=1.0,
            categories=["Test"],
            images=["image.jpg"]
        )
    except ValidationError as e:
        print(f"  Error: {e.errors()[0]['msg']}")
    
    print()


def demo_extraction_constraints():
    """Demonstrate extraction result constraints."""
    print("=" * 70)
    print("EXTRACTION RESULT CONSTRAINTS")
    print("=" * 70)
    
    # Valid extraction
    print("✅ Valid extraction result:")
    entity1 = ExtractedEntity(
        text="Apple Inc.",
        entity_type="ORGANIZATION",
        confidence=0.99,
        start_pos=0,
        end_pos=10
    )
    
    entity2 = ExtractedEntity(
        text="Tim Cook",
        entity_type="PERSON",
        confidence=0.98,
        start_pos=28,
        end_pos=36
    )
    
    result = ExtractionResult(
        extraction_id="ext-2024-001",
        source_text="Apple Inc. announced that Tim Cook will visit New York.",
        entities=[entity1, entity2],
        summary="Apple CEO announcement",
        processing_time_ms=250,
        model_version="1.0.0",
        avg_confidence=0.985,
        entity_count=2
    )
    
    print(f"  Extraction ID: {result.extraction_id}")
    print(f"  Entities: {result.entity_count}")
    print(f"  Avg confidence: {result.avg_confidence:.1%}")
    print(f"  Processing time: {result.processing_time_ms}ms")
    
    # Invalid: empty entities list
    print("\n❌ No entities extracted:")
    try:
        result = ExtractionResult(
            extraction_id="ext-002",
            source_text="Some text",
            entities=[],  # Need at least 1
            processing_time_ms=100,
            model_version="1.0.0",
            avg_confidence=0.0,
            entity_count=0
        )
    except ValidationError as e:
        print(f"  Error: {e.errors()[0]['msg']}")
    
    print()


def demo_quality_constraints():
    """Demonstrate quality validation constraints."""
    print("=" * 70)
    print("QUALITY CONSTRAINTS")
    print("=" * 70)
    
    # High quality extraction
    print("✅ High quality extraction:")
    extraction = ValidatedExtraction(
        text="Apple Inc. is a technology company.",
        confidence=0.95,
        entity_count=1,
        quality_score=0.92,
        processing_time_ms=150
    )
    print(f"  Confidence: {extraction.confidence:.0%} (min 70%)")
    print(f"  Quality: {extraction.quality_score:.0%} (min 80%)")
    print(f"  Processing: {extraction.processing_time_ms}ms (max 5000ms)")
    
    # Low quality (confidence too low)
    print("\n❌ Low confidence extraction:")
    try:
        extraction = ValidatedExtraction(
            text="Some text",
            confidence=0.60,  # < 0.7 minimum
            entity_count=1,
            quality_score=0.85,
            processing_time_ms=100
        )
    except ValidationError as e:
        print(f"  Error: {e.errors()[0]['msg']}")
    
    # Took too long
    print("\n❌ Processing too slow:")
    try:
        extraction = ValidatedExtraction(
            text="Some text",
            confidence=0.95,
            entity_count=1,
            quality_score=0.92,
            processing_time_ms=6000  # > 5000ms max
        )
    except ValidationError as e:
        print(f"  Error: {e.errors()[0]['msg']}")
    
    print()


def demo_documentation_constraints():
    """Demonstrate comprehensive field documentation."""
    print("=" * 70)
    print("DOCUMENTATION WITH CONSTRAINTS")
    print("=" * 70)
    
    # Well-documented model
    doc = DocumentMetadata(
        title="Introduction to Pydantic AI with Gemini",
        author="AI Development Team",
        word_count=2500,
        readability_score=68.5,
        keywords=["pydantic", "ai", "gemini", "type-safety", "validation"],
        abstract="Learn how to build type-safe AI applications using Pydantic AI framework with Google Gemini models."
    )
    
    print(f"Document: {doc.title}")
    print(f"Author: {doc.author}")
    print(f"Word count: {doc.word_count}")
    print(f"Readability: {doc.readability_score}/100")
    print(f"Keywords: {doc.keywords}")
    print(f"Abstract: {doc.abstract[:60]}...")
    
    print()


if __name__ == "__main__":
    print("\n🎯 FIELD CONSTRAINTS AND LIMITS DEMONSTRATION\n")
    
    demo_numeric_constraints()
    demo_range_validation()
    demo_string_constraints()
    demo_collection_constraints()
    demo_combined_constraints()
    demo_extraction_constraints()
    demo_quality_constraints()
    demo_documentation_constraints()
    
    print("=" * 70)
    print("✅ All demonstrations completed!")
    print("=" * 70)
```

### Line-by-Line Explanation

**Constraint Types:**

1. **Numeric**: `ge` (>=), `le` (<=), `gt` (>), `lt` (<)
2. **String**: `min_length`, `max_length`, `pattern` (regex)
3. **Collection**: `min_length`, `max_length` (for lists/sets)
4. **Decimal**: `max_digits`, `decimal_places` for precise numbers

**Field() Parameters:**

```python
field_name: type = Field(
    ...,  # Required (or provide default)
    ge=0,  # Greater than or equal
    le=100,  # Less than or equal
    min_length=1,  # Minimum length
    max_length=50,  # Maximum length
    pattern=r'^[A-Z]+$',  # Regex pattern
    description="Human-readable description",
    examples=["example1", "example2"]
)
```

**Common Patterns:**

- **Percentage**: `ge=0.0, le=100.0`
- **Probability**: `ge=0.0, le=1.0`
- **Rating**: `ge=1, le=5`
- **Required non-empty list**: `min_length=1`
- **Username**: `min_length=3, max_length=20, pattern=r'^[a-zA-Z0-9_]+$'`

### The "Why" Behind the Pattern

**Declarative > Imperative:**
Constraints in field definitions are clearer than validation code. Compare:
```python
# Declarative (clear, self-documenting)
age: int = Field(..., ge=0, le=150)

# vs Imperative (requires reading validator)
age: int
@field_validator('age')
def validate_age(cls, v):
    if v < 0 or v > 150:
        raise ValueError(...)
    return v
```

**Automatic Documentation:**
Constraints appear in generated API docs, IDE tooltips, and JSON Schema automatically.

**Performance:**
Built-in constraints are optimized C code (in Pydantic), faster than Python validators.

**Clear Error Messages:**
Pydantic generates precise error messages showing the constraint and actual value.

---

## C. Test & Apply

### How to Test It

```bash
cd data_extraction_pipeline
touch src/models/constraint_models.py
touch src/examples/constraint_demo.py
python -m src.examples.constraint_demo
```

### Expected Result

Comprehensive demonstrations of:
- Numeric constraints (ranges, bounds)
- String constraints (length, patterns)
- Collection constraints (size limits)
- Combined constraints
- Quality thresholds
- Documentation

---

## D. Common Stumbling Blocks

### Proactive Debugging

**Mistake 1: Confusing ge/le with gt/lt**

```python
# ge = Greater than or Equal (>=)
age: int = Field(..., ge=18)  # 18 is valid

# gt = Greater Than (>)
age: int = Field(..., gt=18)  # 18 is NOT valid, must be 19+
```

**Mistake 2: Pattern without raw string**

```python
# ❌ WRONG - Escape hell
pattern="\\d{5}"

# ✅ CORRECT - Raw string
pattern=r"\d{5}"
```

**Mistake 3: Contradictory constraints**

```python
# ❌ WRONG - Impossible to satisfy
value: int = Field(..., ge=100, le=50)  # Can't be >= 100 AND <= 50!
```

### Type Safety Gotchas

1. **Exclusive vs Inclusive**: `ge`/`le` are inclusive, `gt`/`lt` are exclusive
2. **String patterns**: Use raw strings `r"pattern"` to avoid escape issues
3. **Collection lengths**: Apply to the collection, not items inside
4. **Decimal precision**: Use `Decimal` type for exact monetary calculations
5. **Pattern validation**: Patterns are matched against the entire string

---

## 🎯 Next Steps

Excellent! You now understand:
- ✅ How to use numeric constraints (ge, le, gt, lt)
- ✅ How to use string constraints (length, patterns)
- ✅ How to use collection constraints (size limits)
- ✅ How to combine multiple constraints
- ✅ How constraints provide automatic documentation

In the next lesson, we'll explore **Model Composition Patterns**—learning how to build complex models from reusable components.

**Ready for Lesson 9?** 🚀
