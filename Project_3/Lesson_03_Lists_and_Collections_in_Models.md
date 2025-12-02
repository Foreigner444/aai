# Lesson 3: Lists and Collections in Models

## A. Concept Overview

### What & Why
**Lists and collections are essential for modeling real-world data extraction where you have multiple items of the same type.** A company has multiple employees, a document has multiple paragraphs, a resume has multiple skills. Pydantic's typed collections ensure that every item in a list is validated, giving you complete type safety for array data structures.

### Analogy
Think of typed lists like a specialized container warehouse:
- **Untyped list (`list`)** is like a messy warehouse that accepts any items—boxes, furniture, food, anything
- **Typed list (`List[Person]`)** is like a temperature-controlled pharmaceutical warehouse that only accepts medicine—automated scanners verify each item before it enters
- **Constrained list (`List[Person] = Field(min_length=1, max_length=10)`)** is like that pharmaceutical warehouse with limited capacity and minimum stock requirements

When Gemini extracts multiple entities from text, Pydantic's typed lists validate every single item automatically, ensuring consistency across your entire collection.

### Type Safety Benefit
Typed collections provide **element-level validation**:
- Every item in the list is validated against the specified type
- Your IDE knows exactly what type each list contains
- Iteration is type-safe: `for employee in employees` knows `employee` is an `Employee`
- List methods are type-checked: `employees[0].name` is fully typed
- Constraints apply to the collection: min/max length, unique items
- Empty collections, duplicates, and invalid items are caught immediately

---

## B. Code Implementation

### File Structure
```
data_extraction_pipeline/
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── company.py
│   │   └── collections.py      # New: Collection models
│   └── examples/
│       ├── __init__.py
│       └── collections_demo.py  # New: This lesson
```

### Complete Code Implementation

**File: `src/models/collections.py`**

```python
"""Models demonstrating lists and collections with validation."""

from pydantic import BaseModel, Field, field_validator
from typing import List, Set, Dict, Tuple, Optional
from datetime import datetime
from enum import Enum


class Skill(str, Enum):
    """Professional skills."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    RUST = "rust"
    GO = "go"
    SQL = "sql"
    AWS = "aws"
    DOCKER = "docker"
    KUBERNETES = "kubernetes"


class SkillLevel(str, Enum):
    """Skill proficiency levels."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class SkillEntry(BaseModel):
    """A single skill with proficiency level."""
    skill: Skill
    level: SkillLevel
    years_experience: int = Field(..., ge=0, le=50)


class Resume(BaseModel):
    """
    Resume with various collection types.
    Demonstrates: List, Set, Dict, and constraints.
    """
    name: str = Field(..., min_length=1)
    email: str
    
    # Simple list with constraints
    skills: List[Skill] = Field(
        ...,
        min_length=1,  # Must have at least 1 skill
        max_length=20,  # Cannot have more than 20 skills
        description="List of technical skills"
    )
    
    # List of complex objects
    detailed_skills: List[SkillEntry] = Field(
        default_factory=list,
        description="Detailed skill information"
    )
    
    # Set for unique items (no duplicates)
    certifications: Set[str] = Field(
        default_factory=set,
        description="Unique certifications"
    )
    
    # List of strings with validation
    languages: List[str] = Field(
        default_factory=lambda: ["English"],
        min_length=1,
        description="Spoken languages"
    )
    
    # List of years (integers)
    years_of_experience: List[int] = Field(
        default_factory=list,
        description="Years of experience per role"
    )
    
    @field_validator('skills')
    @classmethod
    def validate_unique_skills(cls, v: List[Skill]) -> List[Skill]:
        """Ensure skills are unique."""
        if len(v) != len(set(v)):
            raise ValueError("Skills must be unique")
        return v
    
    @field_validator('languages')
    @classmethod
    def validate_languages(cls, v: List[str]) -> List[str]:
        """Ensure languages are capitalized."""
        return [lang.capitalize() for lang in v]
    
    @property
    def total_years_experience(self) -> int:
        """Calculate total years of experience."""
        return sum(self.years_of_experience) if self.years_of_experience else 0
    
    @property
    def skill_count(self) -> int:
        """Count total number of skills."""
        return len(self.skills)


class Document(BaseModel):
    """
    Document with paragraphs and metadata.
    Demonstrates nested lists and tuple validation.
    """
    title: str = Field(..., min_length=1, max_length=200)
    author: str
    created_at: datetime = Field(default_factory=datetime.now)
    
    # List of paragraphs
    paragraphs: List[str] = Field(
        ...,
        min_length=1,
        description="Document paragraphs"
    )
    
    # List of tags
    tags: List[str] = Field(
        default_factory=list,
        max_length=10,
        description="Document tags"
    )
    
    # List of tuples (word, count)
    word_frequencies: List[Tuple[str, int]] = Field(
        default_factory=list,
        description="Word frequency pairs"
    )
    
    # Dict of metadata
    metadata: Dict[str, str] = Field(
        default_factory=dict,
        description="Additional metadata"
    )
    
    @field_validator('paragraphs')
    @classmethod
    def validate_paragraphs(cls, v: List[str]) -> List[str]:
        """Ensure paragraphs are not empty."""
        for para in v:
            if not para.strip():
                raise ValueError("Paragraphs cannot be empty")
        return v
    
    @field_validator('tags')
    @classmethod
    def normalize_tags(cls, v: List[str]) -> List[str]:
        """Normalize tags to lowercase and remove duplicates."""
        return list(dict.fromkeys(tag.lower() for tag in v))
    
    @property
    def word_count(self) -> int:
        """Calculate total word count."""
        return sum(len(para.split()) for para in self.paragraphs)
    
    @property
    def paragraph_count(self) -> int:
        """Get number of paragraphs."""
        return len(self.paragraphs)


class EmailMessage(BaseModel):
    """Email message model."""
    sender: str
    subject: str
    body: str
    timestamp: datetime


class EmailThread(BaseModel):
    """
    Email thread with multiple messages.
    Demonstrates ordered collections and chronological validation.
    """
    thread_id: str
    subject: str
    participants: List[str] = Field(..., min_length=1)
    messages: List[EmailMessage] = Field(..., min_length=1)
    
    @field_validator('messages')
    @classmethod
    def validate_chronological(cls, v: List[EmailMessage]) -> List[EmailMessage]:
        """Ensure messages are in chronological order."""
        for i in range(1, len(v)):
            if v[i].timestamp < v[i-1].timestamp:
                raise ValueError("Messages must be in chronological order")
        return v
    
    @property
    def message_count(self) -> int:
        """Get number of messages in thread."""
        return len(self.messages)
    
    @property
    def duration(self) -> Optional[float]:
        """Get thread duration in hours."""
        if len(self.messages) < 2:
            return None
        duration = self.messages[-1].timestamp - self.messages[0].timestamp
        return duration.total_seconds() / 3600


class Product(BaseModel):
    """Product model."""
    name: str
    price: float = Field(..., gt=0)
    quantity: int = Field(..., ge=0)


class Order(BaseModel):
    """
    Order with multiple products.
    Demonstrates aggregation over collections.
    """
    order_id: str
    customer_name: str
    products: List[Product] = Field(..., min_length=1)
    discounts: List[float] = Field(default_factory=list)
    
    @field_validator('discounts')
    @classmethod
    def validate_discounts(cls, v: List[float]) -> List[float]:
        """Ensure discounts are valid percentages."""
        for discount in v:
            if not 0 <= discount <= 100:
                raise ValueError("Discounts must be between 0 and 100")
        return v
    
    @property
    def subtotal(self) -> float:
        """Calculate subtotal (before discounts)."""
        return sum(p.price * p.quantity for p in self.products)
    
    @property
    def total_discount(self) -> float:
        """Calculate total discount amount."""
        discount_multiplier = sum(d / 100 for d in self.discounts)
        return self.subtotal * min(discount_multiplier, 1.0)  # Cap at 100%
    
    @property
    def total(self) -> float:
        """Calculate final total after discounts."""
        return max(0, self.subtotal - self.total_discount)
    
    @property
    def item_count(self) -> int:
        """Get total number of items."""
        return sum(p.quantity for p in self.products)


class MatrixData(BaseModel):
    """
    Matrix/nested list structure.
    Demonstrates validation of nested collections.
    """
    name: str
    data: List[List[float]] = Field(..., min_length=1)
    
    @field_validator('data')
    @classmethod
    def validate_matrix(cls, v: List[List[float]]) -> List[List[float]]:
        """Ensure all rows have the same length."""
        if not v:
            raise ValueError("Matrix cannot be empty")
        
        row_length = len(v[0])
        for i, row in enumerate(v):
            if len(row) != row_length:
                raise ValueError(f"Row {i} has {len(row)} columns, expected {row_length}")
        
        return v
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Get matrix dimensions (rows, cols)."""
        return (len(self.data), len(self.data[0]) if self.data else 0)
    
    @property
    def is_square(self) -> bool:
        """Check if matrix is square."""
        rows, cols = self.shape
        return rows == cols


class BatchExtraction(BaseModel):
    """
    Batch extraction result containing multiple extracted entities.
    Demonstrates real-world data extraction use case.
    """
    extraction_id: str
    source_text: str
    extracted_at: datetime = Field(default_factory=datetime.now)
    
    # Multiple collections of different types
    people: List[str] = Field(default_factory=list)
    organizations: List[str] = Field(default_factory=list)
    locations: List[str] = Field(default_factory=list)
    dates: List[datetime] = Field(default_factory=list)
    amounts: List[float] = Field(default_factory=list)
    
    # Nested structures
    relationships: List[Tuple[str, str, str]] = Field(
        default_factory=list,
        description="Relationships as (entity1, relation, entity2)"
    )
    
    # Confidence scores
    confidence_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Confidence scores for each extraction"
    )
    
    @property
    def total_entities(self) -> int:
        """Count total extracted entities."""
        return (
            len(self.people) +
            len(self.organizations) +
            len(self.locations) +
            len(self.dates) +
            len(self.amounts)
        )
    
    @property
    def has_entities(self) -> bool:
        """Check if any entities were extracted."""
        return self.total_entities > 0
    
    @property
    def average_confidence(self) -> Optional[float]:
        """Calculate average confidence score."""
        if not self.confidence_scores:
            return None
        return sum(self.confidence_scores.values()) / len(self.confidence_scores)
```

**File: `src/examples/collections_demo.py`**

```python
"""Demonstration of lists and collections in Pydantic models."""

from datetime import datetime, timedelta
from src.models.collections import (
    Resume,
    Skill,
    SkillLevel,
    SkillEntry,
    Document,
    EmailThread,
    EmailMessage,
    Order,
    Product,
    MatrixData,
    BatchExtraction,
)


def demo_basic_lists():
    """Demonstrate basic list validation."""
    print("=" * 70)
    print("BASIC LIST VALIDATION")
    print("=" * 70)
    
    resume = Resume(
        name="Alice Developer",
        email="alice@example.com",
        skills=[Skill.PYTHON, Skill.TYPESCRIPT, Skill.AWS, Skill.DOCKER],
        detailed_skills=[
            SkillEntry(skill=Skill.PYTHON, level=SkillLevel.EXPERT, years_experience=8),
            SkillEntry(skill=Skill.TYPESCRIPT, level=SkillLevel.ADVANCED, years_experience=5),
            SkillEntry(skill=Skill.AWS, level=SkillLevel.INTERMEDIATE, years_experience=3),
        ],
        certifications={"AWS Certified", "Python Expert", "Docker Certified"},
        languages=["english", "spanish", "french"],  # Will be capitalized
        years_of_experience=[3, 2, 5]
    )
    
    print(f"Name: {resume.name}")
    print(f"Skills ({resume.skill_count}): {[s.value for s in resume.skills]}")
    print(f"Total experience: {resume.total_years_experience} years")
    print(f"Certifications: {resume.certifications}")
    print(f"Languages: {resume.languages}")  # Capitalized by validator
    
    print(f"\nDetailed Skills:")
    for skill_entry in resume.detailed_skills:
        print(f"  - {skill_entry.skill.value}: {skill_entry.level.value} "
              f"({skill_entry.years_experience} years)")
    
    print()


def demo_list_constraints():
    """Demonstrate list constraints (min/max length)."""
    print("=" * 70)
    print("LIST CONSTRAINTS")
    print("=" * 70)
    
    # Valid: 1-20 skills
    resume1 = Resume(
        name="Bob",
        email="bob@example.com",
        skills=[Skill.PYTHON]  # Minimum 1 skill
    )
    print(f"✅ Resume with {len(resume1.skills)} skill(s): Valid")
    
    # Valid: Multiple skills
    resume2 = Resume(
        name="Charlie",
        email="charlie@example.com",
        skills=[Skill.PYTHON, Skill.JAVASCRIPT, Skill.GO, Skill.RUST]
    )
    print(f"✅ Resume with {len(resume2.skills)} skills: Valid")
    
    # Demonstrate constraint violation (will be caught by Pydantic)
    try:
        resume3 = Resume(
            name="Invalid",
            email="invalid@example.com",
            skills=[]  # ❌ min_length=1 violated
        )
    except Exception as e:
        print(f"❌ Empty skills list rejected: {type(e).__name__}")
    
    print()


def demo_nested_lists():
    """Demonstrate lists of complex objects."""
    print("=" * 70)
    print("NESTED LISTS (Lists of Objects)")
    print("=" * 70)
    
    document = Document(
        title="Introduction to Pydantic AI",
        author="AI Mentor",
        paragraphs=[
            "Pydantic AI is a powerful framework for building type-safe AI applications.",
            "It combines Pydantic's validation with modern AI model integration.",
            "With structured outputs, you can ensure AI responses match your exact requirements."
        ],
        tags=["AI", "Pydantic", "Python", "Type-Safety"],
        word_frequencies=[
            ("Pydantic", 3),
            ("AI", 3),
            ("type-safe", 1),
            ("validation", 1)
        ],
        metadata={
            "category": "tutorial",
            "difficulty": "beginner",
            "version": "1.0"
        }
    )
    
    print(f"Title: {document.title}")
    print(f"Author: {document.author}")
    print(f"Paragraphs: {document.paragraph_count}")
    print(f"Word count: {document.word_count}")
    print(f"Tags: {document.tags}")
    
    print(f"\nTop words:")
    for word, count in document.word_frequencies[:3]:
        print(f"  - '{word}': {count} occurrences")
    
    print(f"\nMetadata:")
    for key, value in document.metadata.items():
        print(f"  - {key}: {value}")
    
    print()


def demo_chronological_validation():
    """Demonstrate validation of ordered collections."""
    print("=" * 70)
    print("CHRONOLOGICAL VALIDATION")
    print("=" * 70)
    
    base_time = datetime.now()
    
    thread = EmailThread(
        thread_id="thread-001",
        subject="Project Discussion",
        participants=["alice@example.com", "bob@example.com", "charlie@example.com"],
        messages=[
            EmailMessage(
                sender="alice@example.com",
                subject="Project Discussion",
                body="Let's discuss the new project.",
                timestamp=base_time
            ),
            EmailMessage(
                sender="bob@example.com",
                subject="Re: Project Discussion",
                body="Great idea! I'm in.",
                timestamp=base_time + timedelta(hours=1)
            ),
            EmailMessage(
                sender="charlie@example.com",
                subject="Re: Project Discussion",
                body="Count me in too!",
                timestamp=base_time + timedelta(hours=2)
            ),
        ]
    )
    
    print(f"Thread: {thread.subject}")
    print(f"Participants: {len(thread.participants)}")
    print(f"Messages: {thread.message_count}")
    print(f"Duration: {thread.duration:.2f} hours")
    
    print(f"\nMessage timeline:")
    for i, msg in enumerate(thread.messages, 1):
        print(f"  {i}. [{msg.timestamp.strftime('%H:%M')}] {msg.sender}: {msg.body[:50]}...")
    
    # Try creating thread with non-chronological messages
    try:
        invalid_thread = EmailThread(
            thread_id="thread-002",
            subject="Invalid",
            participants=["alice@example.com"],
            messages=[
                EmailMessage(
                    sender="alice@example.com",
                    subject="First",
                    body="First message",
                    timestamp=base_time + timedelta(hours=2)  # Later timestamp
                ),
                EmailMessage(
                    sender="alice@example.com",
                    subject="Second",
                    body="Second message",
                    timestamp=base_time  # ❌ Earlier timestamp
                ),
            ]
        )
    except Exception as e:
        print(f"\n❌ Non-chronological messages rejected: {e}")
    
    print()


def demo_aggregations():
    """Demonstrate aggregations over collections."""
    print("=" * 70)
    print("AGGREGATIONS OVER COLLECTIONS")
    print("=" * 70)
    
    order = Order(
        order_id="ORD-12345",
        customer_name="Alice Johnson",
        products=[
            Product(name="Laptop", price=1200.00, quantity=1),
            Product(name="Mouse", price=25.00, quantity=2),
            Product(name="Keyboard", price=75.00, quantity=1),
        ],
        discounts=[10, 5]  # 10% + 5% = 15% total discount
    )
    
    print(f"Order ID: {order.order_id}")
    print(f"Customer: {order.customer_name}")
    print(f"Items: {order.item_count}")
    
    print(f"\nProducts:")
    for product in order.products:
        line_total = product.price * product.quantity
        print(f"  - {product.name}: ${product.price:.2f} x {product.quantity} = ${line_total:.2f}")
    
    print(f"\nSubtotal: ${order.subtotal:.2f}")
    print(f"Discounts: {order.discounts} = ${order.total_discount:.2f}")
    print(f"Total: ${order.total:.2f}")
    
    print()


def demo_nested_collections():
    """Demonstrate nested collections (lists of lists)."""
    print("=" * 70)
    print("NESTED COLLECTIONS (Matrix)")
    print("=" * 70)
    
    matrix = MatrixData(
        name="Sample Matrix",
        data=[
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ]
    )
    
    print(f"Matrix: {matrix.name}")
    print(f"Shape: {matrix.shape[0]}x{matrix.shape[1]}")
    print(f"Is square: {matrix.is_square}")
    
    print(f"\nData:")
    for row in matrix.data:
        print(f"  {row}")
    
    # Try creating invalid matrix (inconsistent row lengths)
    try:
        invalid_matrix = MatrixData(
            name="Invalid",
            data=[
                [1.0, 2.0, 3.0],
                [4.0, 5.0],  # ❌ Different length
            ]
        )
    except Exception as e:
        print(f"\n❌ Inconsistent matrix rejected: {e}")
    
    print()


def demo_batch_extraction():
    """Demonstrate real-world batch extraction use case."""
    print("=" * 70)
    print("BATCH EXTRACTION (Real-World Use Case)")
    print("=" * 70)
    
    extraction = BatchExtraction(
        extraction_id="ext-001",
        source_text="Apple Inc. announced that Tim Cook will visit New York on December 15, 2024, "
                    "to discuss a $2.5 billion investment in renewable energy.",
        people=["Tim Cook"],
        organizations=["Apple Inc."],
        locations=["New York"],
        dates=[datetime(2024, 12, 15)],
        amounts=[2.5e9],
        relationships=[
            ("Tim Cook", "CEO_OF", "Apple Inc."),
            ("Tim Cook", "WILL_VISIT", "New York"),
            ("Apple Inc.", "INVESTS_IN", "renewable energy")
        ],
        confidence_scores={
            "Tim Cook": 0.98,
            "Apple Inc.": 0.99,
            "New York": 0.95,
            "investment": 0.92
        }
    )
    
    print(f"Extraction ID: {extraction.extraction_id}")
    print(f"Source: {extraction.source_text[:80]}...")
    print(f"Total entities: {extraction.total_entities}")
    print(f"Average confidence: {extraction.average_confidence:.2%}")
    
    print(f"\nExtracted entities:")
    print(f"  People: {extraction.people}")
    print(f"  Organizations: {extraction.organizations}")
    print(f"  Locations: {extraction.locations}")
    print(f"  Dates: {[d.strftime('%Y-%m-%d') for d in extraction.dates]}")
    print(f"  Amounts: {[f'${a:,.0f}' for a in extraction.amounts]}")
    
    print(f"\nRelationships:")
    for entity1, relation, entity2 in extraction.relationships:
        print(f"  - {entity1} --[{relation}]--> {entity2}")
    
    print()


if __name__ == "__main__":
    print("\n🎯 LISTS AND COLLECTIONS DEMONSTRATION\n")
    
    demo_basic_lists()
    demo_list_constraints()
    demo_nested_lists()
    demo_chronological_validation()
    demo_aggregations()
    demo_nested_collections()
    demo_batch_extraction()
    
    print("=" * 70)
    print("✅ All demonstrations completed!")
    print("=" * 70)
```

### Line-by-Line Explanation

**Collection Types:**

1. **`List[T]`**: Ordered collection where every item must be type `T`. Allows duplicates.
2. **`Set[T]`**: Unordered collection with unique items only (no duplicates).
3. **`Dict[K, V]`**: Key-value mapping where keys are type `K` and values are type `V`.
4. **`Tuple[T1, T2, ...]`**: Fixed-length sequence with specific types at each position.

**Field Constraints:**

1. **`min_length`**: Minimum number of items required in the collection
2. **`max_length`**: Maximum number of items allowed
3. **`default_factory=list`**: Creates a new empty list for each instance (avoids mutable default issues)

**Custom Validators:**

1. **`@field_validator('field_name')`**: Validates and transforms field values
2. **Applied to every item**: Validators run after Pydantic's type validation
3. **Can modify values**: Return transformed collection (e.g., capitalize strings)

**Key Patterns:**

- **Unique validation**: Check `len(v) != len(set(v))` to detect duplicates
- **Chronological validation**: Compare timestamps of adjacent items
- **Matrix validation**: Ensure all rows have same length
- **Aggregation**: Use list comprehensions and `sum()` for calculations

### The "Why" Behind the Pattern

**Type Safety for Collections:**
When you write `skills: List[Skill]`, Pydantic validates that:
- The field is a list (not a string, dict, etc.)
- Every item in the list is a valid `Skill` enum
- List constraints (min/max length) are satisfied
- Custom validators pass

Your IDE knows `for skill in resume.skills` has `skill` as type `Skill`, enabling autocomplete.

**Validation at Every Level:**
Collections cascade validation:
1. Validate the collection type (is it a list?)
2. Validate collection constraints (length, etc.)
3. Validate each item in the collection (is it a valid `Product`?)
4. Run custom validators on the collection
5. Validate nested models inside each item

**Preventing Common Bugs:**
- **Empty lists where data is required**: `min_length=1` catches this
- **Duplicates where uniqueness is needed**: Use `Set` or custom validator
- **Oversized collections**: `max_length` prevents memory issues
- **Type mismatches in items**: `List[Product]` rejects `List[str]`

---

## C. Test & Apply

### How to Test It

**Step 1: Create the files**
```bash
cd data_extraction_pipeline
touch src/models/collections.py
touch src/examples/collections_demo.py
```

**Step 2: Copy the code above into each file**

**Step 3: Update `src/models/__init__.py`**
```python
from .collections import (
    Resume, Skill, SkillLevel, SkillEntry,
    Document, EmailThread, EmailMessage,
    Order, Product, MatrixData, BatchExtraction
)

__all__ = [
    # ... existing exports ...
    "Resume", "Skill", "SkillLevel", "SkillEntry",
    "Document", "EmailThread", "EmailMessage",
    "Order", "Product", "MatrixData", "BatchExtraction",
]
```

**Step 4: Run the demonstration**
```bash
python -m src.examples.collections_demo
```

### Expected Result

You should see comprehensive output demonstrating:
- Basic list validation with enums
- List constraints (min/max length)
- Nested lists (lists of objects)
- Chronological validation
- Aggregations (sums, calculations)
- Matrix validation (nested lists)
- Real-world batch extraction example

### Validation Examples

**Create `src/examples/collection_validation_demo.py`:**

```python
"""Demonstrate collection validation errors."""

from pydantic import ValidationError
from src.models.collections import Resume, Skill, Order, Product, MatrixData


def demo_collection_errors():
    print("🚫 COLLECTION VALIDATION ERRORS\n")
    
    # Error 1: Empty list when min_length=1
    print("Error 1: Empty list violates min_length")
    try:
        resume = Resume(
            name="Test",
            email="test@example.com",
            skills=[]  # ❌ min_length=1 violated
        )
    except ValidationError as e:
        print(f"❌ {e}\n")
    
    # Error 2: Wrong type in list
    print("Error 2: Wrong type in list items")
    try:
        order = Order(
            order_id="ORD-001",
            customer_name="Test",
            products=["Not a product object"]  # ❌ Should be Product objects
        )
    except ValidationError as e:
        print(f"❌ {e}\n")
    
    # Error 3: Invalid nested list structure
    print("Error 3: Inconsistent matrix dimensions")
    try:
        matrix = MatrixData(
            name="Invalid",
            data=[
                [1.0, 2.0, 3.0],
                [4.0, 5.0]  # ❌ Different length
            ]
        )
    except ValidationError as e:
        print(f"❌ {e}\n")
    
    # Error 4: Too many items (max_length)
    print("Error 4: Exceeding max_length")
    try:
        resume = Resume(
            name="Test",
            email="test@example.com",
            skills=[Skill.PYTHON] * 21  # ❌ max_length=20
        )
    except ValidationError as e:
        print(f"❌ {e}\n")
    
    print("✅ Validation error demonstration complete!")


if __name__ == "__main__":
    demo_collection_errors()
```

**Run it:**
```bash
python -m src.examples.collection_validation_demo
```

### Type Checking

**Type-safe iteration:**
```python
from src.models.collections import Resume, Skill

resume = Resume(
    name="Alice",
    email="alice@example.com",
    skills=[Skill.PYTHON, Skill.TYPESCRIPT]
)

# IDE knows 'skill' is type 'Skill'
for skill in resume.skills:
    skill_name: str = skill.value  # ✅ Type safe
    # skill.invalid_method()  # ❌ IDE error

# IDE knows list methods return correct types
first_skill: Skill = resume.skills[0]  # ✅ Type safe
skill_count: int = len(resume.skills)  # ✅ Type safe
```

---

## D. Common Stumbling Blocks

### Proactive Debugging

**Mistake 1: Using mutable default values**

```python
# ❌ WRONG - Mutable default shared across all instances
class Resume(BaseModel):
    skills: List[Skill] = []  # BUG: Same list shared!

# ✅ CORRECT - Use default_factory
class Resume(BaseModel):
    skills: List[Skill] = Field(default_factory=list)
```

**Mistake 2: Forgetting type parameters**

```python
# ❌ WRONG - No type safety for items
class Resume(BaseModel):
    skills: List  # What type are the items?

# ✅ CORRECT - Specify item type
class Resume(BaseModel):
    skills: List[Skill]  # Every item is a Skill
```

**Mistake 3: Using List instead of Set for unique items**

```python
# ❌ WRONG - Allows duplicates, need custom validator
class Resume(BaseModel):
    certifications: List[str]  # Can have duplicates

# ✅ CORRECT - Set automatically ensures uniqueness
class Resume(BaseModel):
    certifications: Set[str]  # No duplicates possible
```

### Show the Error

**Error 1: Empty list violates min_length**

```python
resume = Resume(
    name="Test",
    email="test@example.com",
    skills=[]  # ❌ min_length=1
)
```

**Error message:**
```
ValidationError: 1 validation error for Resume
skills
  List should have at least 1 item after validation, not 0 [type=too_short, input_value=[], input_type=list]
```

**Error 2: Wrong type in list**

```python
order = Order(
    order_id="ORD-001",
    customer_name="Test",
    products=["Laptop", "Mouse"]  # ❌ Strings instead of Product objects
)
```

**Error message:**
```
ValidationError: 2 validation errors for Order
products.0
  Input should be a valid dictionary or instance of Product [type=model_type, input_value='Laptop', input_type=str]
products.1
  Input should be a valid dictionary or instance of Product [type=model_type, input_value='Mouse', input_type=str]
```

Notice how Pydantic shows errors for specific indices (`.0`, `.1`).

**Error 3: Validation failure in nested collection**

```python
order = Order(
    order_id="ORD-001",
    customer_name="Test",
    products=[
        Product(name="Laptop", price=1000, quantity=1),
        Product(name="Mouse", price=-25, quantity=1)  # ❌ Negative price
    ]
)
```

**Error message:**
```
ValidationError: 1 validation error for Order
products.1.price
  Input should be greater than 0 [type=greater_than, input_value=-25, input_type=int]
```

The error path `products.1.price` tells you: in the `products` list, at index 1, the `price` field is invalid.

### Explain the Fix

**For Empty List Errors:**
- If `min_length=1`, you must provide at least one item
- Check if the data source is missing required items
- Consider making the field Optional if empty lists are valid

**For Wrong Type in List:**
- Ensure every item matches the list's type parameter
- If passing dicts, ensure they have all required fields for the model
- Pydantic can convert dicts to models, but not arbitrary strings

**For List Length Constraints:**
- Check `min_length` and `max_length` in the model definition
- Adjust your data or the constraints
- Consider pagination if dealing with very large lists

**For Nested Validation Failures:**
- Read the error path: `products.1.price` means index 1 in products, field price
- Fix the specific item at that index
- Error messages show both the path and what constraint was violated

### Type Safety Gotchas

1. **Mutable Defaults**: Always use `default_factory=list/set/dict`, never `= []`

2. **Type Parameters**: `List` without `[T]` loses type safety—always specify item type

3. **Set Ordering**: Sets are unordered—don't rely on insertion order

4. **Dict Keys**: `Dict[str, int]` validates both keys and values—ensure keys are correct type

5. **Nested Validation**: Errors show the full path—read from left to right to locate the issue

6. **Empty Collections**: `List[T] = Field(default_factory=list)` allows empty lists. Use `min_length=1` to require items.

---

## 🎯 Next Steps

Great progress! You now understand:
- ✅ How to use typed lists with validation
- ✅ How to apply constraints to collections (min/max length)
- ✅ How to validate nested collections (lists of objects)
- ✅ How to use Sets for unique items
- ✅ How to work with Dicts and Tuples
- ✅ How to create custom validators for collections
- ✅ How validation errors show exact paths through nested collections

In the next lesson, we'll explore **Optional Fields and Defaults**—learning how to handle missing data, default values, and nullable fields in your extraction pipeline.

**Ready for Lesson 4, or would you like to practice with more collection patterns?** 🚀
