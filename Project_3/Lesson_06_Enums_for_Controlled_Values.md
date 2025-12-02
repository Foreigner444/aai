# Lesson 6: Enums for Controlled Values

## A. Concept Overview

### What & Why
**Enums (Enumerations) restrict a field to a specific set of predefined values, ensuring data consistency and preventing invalid states.** When extracting data from text with Gemini, you often need fields that can only be certain values—status can be "pending", "completed", or "failed"; priority can be "low", "medium", or "high". Enums enforce these constraints at the type level, making invalid values impossible to create.

### Analogy
Think of enums like a multiple-choice question on a test:
- **Without enum**: Essay question—students can write anything (risk of invalid or inconsistent answers)
- **With enum**: Multiple choice—students must choose A, B, C, or D (guaranteed valid answer, easy to grade, consistent format)

When Gemini extracts a status field, enums are like those multiple-choice options—the AI must choose from the predefined valid options, and Pydantic validates that the choice is correct.

### Type Safety Benefit
Enums provide **compile-time and runtime value validation**:
- Only predefined values are valid—typos are caught immediately
- Your IDE autocompletes valid options—no memorizing strings
- Refactoring is safe—change the enum, find all uses
- Type narrowing works—if statements on enums are precise
- Case-insensitive matching available—"PENDING", "pending", "Pending" all work
- Invalid values raise clear errors—"Got 'invalid', expected one of ['pending', 'active', 'completed']"

---

## B. Code Implementation

### File Structure
```
data_extraction_pipeline/
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── enum_models.py      # New: Enum patterns
│   └── examples/
│       ├── __init__.py
│       └── enum_demo.py         # New: This lesson
```

### Complete Code Implementation

**File: `src/models/enum_models.py`**

```python
"""Models demonstrating enums for controlled values."""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from datetime import datetime
from enum import Enum, IntEnum


# ============================================================================
# BASIC STRING ENUMS
# ============================================================================

class TaskStatus(str, Enum):
    """Task status options."""
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    IN_REVIEW = "in_review"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class Priority(str, Enum):
    """Priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TaskCategory(str, Enum):
    """Task categories."""
    FEATURE = "feature"
    BUG = "bug"
    DOCUMENTATION = "documentation"
    REFACTORING = "refactoring"
    TESTING = "testing"


class Task(BaseModel):
    """Task with enum-controlled fields."""
    title: str = Field(..., min_length=1)
    description: str
    status: TaskStatus = TaskStatus.TODO  # Default to TODO
    priority: Priority = Priority.MEDIUM  # Default to MEDIUM
    category: TaskCategory
    assigned_to: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    
    @property
    def is_completed(self) -> bool:
        """Check if task is completed."""
        return self.status == TaskStatus.COMPLETED
    
    @property
    def is_active(self) -> bool:
        """Check if task is actively being worked on."""
        return self.status in (TaskStatus.IN_PROGRESS, TaskStatus.IN_REVIEW)


# ============================================================================
# INTEGER ENUMS (For ordering)
# ============================================================================

class SkillLevel(IntEnum):
    """Skill proficiency levels with numeric ordering."""
    BEGINNER = 1
    INTERMEDIATE = 2
    ADVANCED = 3
    EXPERT = 4
    MASTER = 5


class ExperienceLevel(IntEnum):
    """Years of experience categories."""
    ENTRY = 0  # 0-2 years
    JUNIOR = 2  # 2-5 years
    MID = 5  # 5-8 years
    SENIOR = 8  # 8+ years
    PRINCIPAL = 12  # 12+ years


class SkillAssessment(BaseModel):
    """Skill assessment with ordered levels."""
    skill_name: str
    level: SkillLevel
    years_practiced: int = Field(..., ge=0)
    
    @property
    def is_expert(self) -> bool:
        """Check if expert level or above."""
        return self.level >= SkillLevel.EXPERT
    
    @property
    def can_mentor(self) -> bool:
        """Check if experienced enough to mentor."""
        return self.level >= SkillLevel.ADVANCED and self.years_practiced >= 3


# ============================================================================
# ENTITY TYPE ENUMS
# ============================================================================

class EntityType(str, Enum):
    """Types of entities that can be extracted."""
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    DATE = "date"
    MONEY = "money"
    PERCENTAGE = "percentage"
    PRODUCT = "product"
    EVENT = "event"


class Sentiment(str, Enum):
    """Sentiment analysis results."""
    VERY_NEGATIVE = "very_negative"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    VERY_POSITIVE = "very_positive"


class Language(str, Enum):
    """Supported languages."""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    CHINESE = "zh"
    JAPANESE = "ja"
    PORTUGUESE = "pt"
    RUSSIAN = "ru"


class ExtractedEntity(BaseModel):
    """Entity extracted from text with controlled types."""
    text: str
    entity_type: EntityType  # Must be one of the defined types
    confidence: float = Field(..., ge=0.0, le=1.0)
    start_pos: Optional[int] = None
    end_pos: Optional[int] = None
    
    @property
    def is_high_confidence(self) -> bool:
        """Check if confidence is above 0.9."""
        return self.confidence >= 0.9


class TextAnalysis(BaseModel):
    """Text analysis with sentiment and language detection."""
    text: str
    detected_language: Language
    sentiment: Sentiment
    entities: List[ExtractedEntity] = Field(default_factory=list)
    
    @property
    def is_positive(self) -> bool:
        """Check if sentiment is positive."""
        return self.sentiment in (Sentiment.POSITIVE, Sentiment.VERY_POSITIVE)
    
    @property
    def entity_types_found(self) -> set:
        """Get unique entity types found."""
        return {entity.entity_type for entity in self.entities}


# ============================================================================
# STATUS AND LIFECYCLE ENUMS
# ============================================================================

class UserStatus(str, Enum):
    """User account status."""
    PENDING = "pending"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    BANNED = "banned"
    DELETED = "deleted"


class SubscriptionTier(str, Enum):
    """Subscription tiers."""
    FREE = "free"
    BASIC = "basic"
    PRO = "pro"
    ENTERPRISE = "enterprise"


class PaymentStatus(str, Enum):
    """Payment transaction status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    REFUNDED = "refunded"
    CANCELLED = "cancelled"


class UserAccount(BaseModel):
    """User account with status and subscription."""
    user_id: str
    email: str
    status: UserStatus = UserStatus.PENDING
    subscription_tier: SubscriptionTier = SubscriptionTier.FREE
    created_at: datetime = Field(default_factory=datetime.now)
    
    @property
    def is_active(self) -> bool:
        """Check if user account is active."""
        return self.status == UserStatus.ACTIVE
    
    @property
    def is_paid_user(self) -> bool:
        """Check if user has paid subscription."""
        return self.subscription_tier != SubscriptionTier.FREE
    
    @property
    def can_access_premium_features(self) -> bool:
        """Check if user can access premium features."""
        return self.is_active and self.subscription_tier in (
            SubscriptionTier.PRO,
            SubscriptionTier.ENTERPRISE
        )


# ============================================================================
# GEOGRAPHIC ENUMS
# ============================================================================

class USState(str, Enum):
    """US state codes."""
    AL = "AL"  # Alabama
    AK = "AK"  # Alaska
    AZ = "AZ"  # Arizona
    CA = "CA"  # California
    CO = "CO"  # Colorado
    FL = "FL"  # Florida
    NY = "NY"  # New York
    TX = "TX"  # Texas
    WA = "WA"  # Washington
    # Add more as needed...


class Country(str, Enum):
    """Country codes (ISO 3166-1 alpha-2)."""
    US = "US"  # United States
    UK = "UK"  # United Kingdom
    CA = "CA"  # Canada
    FR = "FR"  # France
    DE = "DE"  # Germany
    JP = "JP"  # Japan
    CN = "CN"  # China
    IN = "IN"  # India
    # Add more as needed...


class Region(str, Enum):
    """Geographic regions."""
    NORTH_AMERICA = "north_america"
    SOUTH_AMERICA = "south_america"
    EUROPE = "europe"
    ASIA = "asia"
    AFRICA = "africa"
    OCEANIA = "oceania"
    ANTARCTICA = "antarctica"


class Location(BaseModel):
    """Location with controlled geographic values."""
    name: str
    country: Country
    region: Region
    state: Optional[USState] = None  # Only for US locations
    
    @field_validator('state')
    @classmethod
    def validate_us_state(cls, v: Optional[USState], values) -> Optional[USState]:
        """Ensure state is only provided for US locations."""
        # Note: In Pydantic v2, use info.data instead of values
        # This is simplified for demonstration
        return v


# ============================================================================
# CONTENT TYPE ENUMS
# ============================================================================

class FileType(str, Enum):
    """File types for upload/processing."""
    PDF = "pdf"
    DOC = "doc"
    DOCX = "docx"
    TXT = "txt"
    CSV = "csv"
    JSON = "json"
    XML = "xml"
    HTML = "html"


class MediaType(str, Enum):
    """Media content types."""
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    DOCUMENT = "document"


class ContentStatus(str, Enum):
    """Content publication status."""
    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    PUBLISHED = "published"
    ARCHIVED = "archived"


class Article(BaseModel):
    """Article with content status."""
    title: str
    content: str
    status: ContentStatus = ContentStatus.DRAFT
    file_type: FileType = FileType.HTML
    language: Language = Language.ENGLISH
    published_at: Optional[datetime] = None
    
    @property
    def is_published(self) -> bool:
        """Check if article is published."""
        return self.status == ContentStatus.PUBLISHED
    
    @property
    def can_be_viewed(self) -> bool:
        """Check if article can be viewed by public."""
        return self.status in (ContentStatus.APPROVED, ContentStatus.PUBLISHED)


# ============================================================================
# EXTRACTION CONFIDENCE ENUMS
# ============================================================================

class ConfidenceLevel(str, Enum):
    """Human-readable confidence levels."""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class ValidationStatus(str, Enum):
    """Validation status for extracted data."""
    UNVALIDATED = "unvalidated"
    VALIDATING = "validating"
    VALIDATED = "validated"
    REJECTED = "rejected"
    NEEDS_REVIEW = "needs_review"


class ExtractionQuality(str, Enum):
    """Quality assessment of extraction."""
    POOR = "poor"
    FAIR = "fair"
    GOOD = "good"
    EXCELLENT = "excellent"


class ValidatedExtraction(BaseModel):
    """Extraction result with validation status."""
    extraction_id: str
    entity_text: str
    entity_type: EntityType
    confidence: float = Field(..., ge=0.0, le=1.0)
    validation_status: ValidationStatus = ValidationStatus.UNVALIDATED
    quality: Optional[ExtractionQuality] = None
    
    @property
    def confidence_level(self) -> ConfidenceLevel:
        """Convert numeric confidence to level."""
        if self.confidence < 0.3:
            return ConfidenceLevel.VERY_LOW
        elif self.confidence < 0.5:
            return ConfidenceLevel.LOW
        elif self.confidence < 0.7:
            return ConfidenceLevel.MEDIUM
        elif self.confidence < 0.9:
            return ConfidenceLevel.HIGH
        else:
            return ConfidenceLevel.VERY_HIGH
    
    @property
    def needs_manual_review(self) -> bool:
        """Check if extraction needs manual review."""
        return (
            self.validation_status == ValidationStatus.NEEDS_REVIEW or
            self.confidence < 0.7
        )


# ============================================================================
# BATCH PROCESSING ENUMS
# ============================================================================

class ProcessingStatus(str, Enum):
    """Batch processing status."""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ErrorSeverity(str, Enum):
    """Error severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class BatchJob(BaseModel):
    """Batch processing job with status."""
    job_id: str
    status: ProcessingStatus = ProcessingStatus.QUEUED
    total_items: int = Field(..., ge=0)
    processed_items: int = Field(default=0, ge=0)
    failed_items: int = Field(default=0, ge=0)
    error_severity: Optional[ErrorSeverity] = None
    
    @property
    def is_completed(self) -> bool:
        """Check if job is completed."""
        return self.status == ProcessingStatus.COMPLETED
    
    @property
    def is_running(self) -> bool:
        """Check if job is currently running."""
        return self.status == ProcessingStatus.PROCESSING
    
    @property
    def progress_percentage(self) -> float:
        """Calculate progress percentage."""
        if self.total_items == 0:
            return 0.0
        return (self.processed_items / self.total_items) * 100
    
    @property
    def has_errors(self) -> bool:
        """Check if job has errors."""
        return self.failed_items > 0 or self.error_severity is not None
```

**File: `src/examples/enum_demo.py`**

```python
"""Demonstration of enums for controlled values."""

from datetime import datetime
from src.models.enum_models import (
    Task,
    TaskStatus,
    Priority,
    TaskCategory,
    SkillAssessment,
    SkillLevel,
    ExtractedEntity,
    EntityType,
    TextAnalysis,
    Sentiment,
    Language,
    UserAccount,
    UserStatus,
    SubscriptionTier,
    Location,
    Country,
    Region,
    USState,
    Article,
    ContentStatus,
    FileType,
    ValidatedExtraction,
    ValidationStatus,
    ConfidenceLevel,
    ExtractionQuality,
    BatchJob,
    ProcessingStatus,
    ErrorSeverity,
)


def demo_basic_enums():
    """Demonstrate basic enum usage."""
    print("=" * 70)
    print("BASIC ENUM USAGE")
    print("=" * 70)
    
    # Create tasks with enum values
    task1 = Task(
        title="Implement user authentication",
        description="Add JWT authentication to API",
        status=TaskStatus.IN_PROGRESS,
        priority=Priority.HIGH,
        category=TaskCategory.FEATURE,
        assigned_to="alice@example.com"
    )
    
    task2 = Task(
        title="Fix login bug",
        description="Users can't log in with special characters",
        status=TaskStatus.TODO,
        priority=Priority.CRITICAL,
        category=TaskCategory.BUG
    )
    
    task3 = Task(
        title="Update README",
        description="Add installation instructions",
        status=TaskStatus.COMPLETED,
        priority=Priority.LOW,
        category=TaskCategory.DOCUMENTATION
    )
    
    tasks = [task1, task2, task3]
    
    print("Tasks:")
    for task in tasks:
        print(f"\n  [{task.status.value}] {task.title}")
        print(f"  Priority: {task.priority.value}")
        print(f"  Category: {task.category.value}")
        print(f"  Is completed: {task.is_completed}")
        print(f"  Is active: {task.is_active}")
    
    # Show all valid enum values
    print(f"\nValid task statuses:")
    for status in TaskStatus:
        print(f"  - {status.value}")
    
    print()


def demo_integer_enums():
    """Demonstrate integer enums with ordering."""
    print("=" * 70)
    print("INTEGER ENUMS (With Ordering)")
    print("=" * 70)
    
    skills = [
        SkillAssessment(skill_name="Python", level=SkillLevel.EXPERT, years_practiced=8),
        SkillAssessment(skill_name="JavaScript", level=SkillLevel.ADVANCED, years_practiced=5),
        SkillAssessment(skill_name="Rust", level=SkillLevel.INTERMEDIATE, years_practiced=2),
        SkillAssessment(skill_name="Go", level=SkillLevel.BEGINNER, years_practiced=1),
    ]
    
    print("Skills assessment:")
    for skill in skills:
        print(f"\n  {skill.skill_name}: Level {skill.level.value} ({skill.level.name})")
        print(f"  Years practiced: {skill.years_practiced}")
        print(f"  Is expert: {skill.is_expert}")
        print(f"  Can mentor: {skill.can_mentor}")
    
    # Demonstrate ordering
    print(f"\nSkill level ordering:")
    sorted_skills = sorted(skills, key=lambda s: s.level, reverse=True)
    for skill in sorted_skills:
        print(f"  {skill.skill_name}: {skill.level.name}")
    
    # Demonstrate comparison
    python_skill = skills[0]
    rust_skill = skills[2]
    print(f"\nComparison:")
    print(f"  Python level ({python_skill.level.name}) > Rust level ({rust_skill.level.name}): "
          f"{python_skill.level > rust_skill.level}")
    
    print()


def demo_entity_extraction():
    """Demonstrate enums in entity extraction."""
    print("=" * 70)
    print("ENTITY EXTRACTION WITH ENUMS")
    print("=" * 70)
    
    text = "Apple Inc. announced that Tim Cook will visit New York on December 15, 2024, " \
           "to discuss a $2.5 billion investment."
    
    analysis = TextAnalysis(
        text=text,
        detected_language=Language.ENGLISH,
        sentiment=Sentiment.POSITIVE,
        entities=[
            ExtractedEntity(
                text="Apple Inc.",
                entity_type=EntityType.ORGANIZATION,
                confidence=0.99,
                start_pos=0,
                end_pos=10
            ),
            ExtractedEntity(
                text="Tim Cook",
                entity_type=EntityType.PERSON,
                confidence=0.98,
                start_pos=28,
                end_pos=36
            ),
            ExtractedEntity(
                text="New York",
                entity_type=EntityType.LOCATION,
                confidence=0.95,
                start_pos=48,
                end_pos=56
            ),
            ExtractedEntity(
                text="December 15, 2024",
                entity_type=EntityType.DATE,
                confidence=0.93,
                start_pos=60,
                end_pos=77
            ),
            ExtractedEntity(
                text="$2.5 billion",
                entity_type=EntityType.MONEY,
                confidence=0.97,
                start_pos=92,
                end_pos=104
            ),
        ]
    )
    
    print(f"Text: {analysis.text[:60]}...")
    print(f"Language: {analysis.detected_language.name} ({analysis.detected_language.value})")
    print(f"Sentiment: {analysis.sentiment.value}")
    print(f"Is positive: {analysis.is_positive}")
    
    print(f"\nExtracted entities ({len(analysis.entities)}):")
    for entity in analysis.entities:
        print(f"  [{entity.entity_type.name}] {entity.text} "
              f"(confidence: {entity.confidence:.0%}, high: {entity.is_high_confidence})")
    
    print(f"\nEntity types found: {[et.name for et in analysis.entity_types_found]}")
    
    print()


def demo_user_status():
    """Demonstrate user status and subscription enums."""
    print("=" * 70)
    print("USER STATUS AND SUBSCRIPTIONS")
    print("=" * 70)
    
    users = [
        UserAccount(
            user_id="user-001",
            email="alice@example.com",
            status=UserStatus.ACTIVE,
            subscription_tier=SubscriptionTier.PRO
        ),
        UserAccount(
            user_id="user-002",
            email="bob@example.com",
            status=UserStatus.ACTIVE,
            subscription_tier=SubscriptionTier.FREE
        ),
        UserAccount(
            user_id="user-003",
            email="charlie@example.com",
            status=UserStatus.SUSPENDED,
            subscription_tier=SubscriptionTier.BASIC
        ),
        UserAccount(
            user_id="user-004",
            email="diana@example.com",
            status=UserStatus.PENDING,
            subscription_tier=SubscriptionTier.ENTERPRISE
        ),
    ]
    
    print("User accounts:")
    for user in users:
        print(f"\n  {user.email}")
        print(f"  Status: {user.status.value}")
        print(f"  Subscription: {user.subscription_tier.value}")
        print(f"  Is active: {user.is_active}")
        print(f"  Is paid user: {user.is_paid_user}")
        print(f"  Can access premium: {user.can_access_premium_features}")
    
    print()


def demo_geographic_enums():
    """Demonstrate geographic enums."""
    print("=" * 70)
    print("GEOGRAPHIC ENUMS")
    print("=" * 70)
    
    locations = [
        Location(
            name="San Francisco Office",
            country=Country.US,
            region=Region.NORTH_AMERICA,
            state=USState.CA
        ),
        Location(
            name="London Office",
            country=Country.UK,
            region=Region.EUROPE
        ),
        Location(
            name="Tokyo Office",
            country=Country.JP,
            region=Region.ASIA
        ),
    ]
    
    print("Office locations:")
    for loc in locations:
        print(f"\n  {loc.name}")
        print(f"  Country: {loc.country.value}")
        print(f"  Region: {loc.region.value}")
        if loc.state:
            print(f"  State: {loc.state.value}")
    
    print()


def demo_content_status():
    """Demonstrate content status enums."""
    print("=" * 70)
    print("CONTENT STATUS")
    print("=" * 70)
    
    articles = [
        Article(
            title="Getting Started with Pydantic AI",
            content="...",
            status=ContentStatus.PUBLISHED,
            file_type=FileType.HTML,
            language=Language.ENGLISH,
            published_at=datetime.now()
        ),
        Article(
            title="Advanced Type Safety Patterns",
            content="...",
            status=ContentStatus.PENDING_REVIEW,
            file_type=FileType.DOCX,
            language=Language.ENGLISH
        ),
        Article(
            title="Guide Complet de Pydantic AI",
            content="...",
            status=ContentStatus.DRAFT,
            file_type=FileType.PDF,
            language=Language.FRENCH
        ),
    ]
    
    print("Articles:")
    for article in articles:
        print(f"\n  {article.title}")
        print(f"  Status: {article.status.value}")
        print(f"  Format: {article.file_type.value}")
        print(f"  Language: {article.language.value}")
        print(f"  Is published: {article.is_published}")
        print(f"  Can be viewed: {article.can_be_viewed}")
    
    print()


def demo_validation_levels():
    """Demonstrate confidence and validation enums."""
    print("=" * 70)
    print("VALIDATION AND CONFIDENCE LEVELS")
    print("=" * 70)
    
    extractions = [
        ValidatedExtraction(
            extraction_id="ext-001",
            entity_text="Apple Inc.",
            entity_type=EntityType.ORGANIZATION,
            confidence=0.99,
            validation_status=ValidationStatus.VALIDATED,
            quality=ExtractionQuality.EXCELLENT
        ),
        ValidatedExtraction(
            extraction_id="ext-002",
            entity_text="Tim Cook",
            entity_type=EntityType.PERSON,
            confidence=0.85,
            validation_status=ValidationStatus.VALIDATED,
            quality=ExtractionQuality.GOOD
        ),
        ValidatedExtraction(
            extraction_id="ext-003",
            entity_text="somewhere",
            entity_type=EntityType.LOCATION,
            confidence=0.45,
            validation_status=ValidationStatus.NEEDS_REVIEW,
            quality=ExtractionQuality.POOR
        ),
    ]
    
    print("Extraction results:")
    for ext in extractions:
        print(f"\n  {ext.entity_text} [{ext.entity_type.name}]")
        print(f"  Confidence: {ext.confidence:.0%} ({ext.confidence_level.value})")
        print(f"  Validation: {ext.validation_status.value}")
        print(f"  Quality: {ext.quality.value if ext.quality else 'N/A'}")
        print(f"  Needs review: {ext.needs_manual_review}")
    
    print()


def demo_batch_processing():
    """Demonstrate batch processing status."""
    print("=" * 70)
    print("BATCH PROCESSING STATUS")
    print("=" * 70)
    
    jobs = [
        BatchJob(
            job_id="job-001",
            status=ProcessingStatus.COMPLETED,
            total_items=1000,
            processed_items=1000,
            failed_items=0
        ),
        BatchJob(
            job_id="job-002",
            status=ProcessingStatus.PROCESSING,
            total_items=500,
            processed_items=350,
            failed_items=5,
            error_severity=ErrorSeverity.WARNING
        ),
        BatchJob(
            job_id="job-003",
            status=ProcessingStatus.FAILED,
            total_items=200,
            processed_items=50,
            failed_items=50,
            error_severity=ErrorSeverity.CRITICAL
        ),
    ]
    
    print("Batch jobs:")
    for job in jobs:
        print(f"\n  Job {job.job_id}")
        print(f"  Status: {job.status.value}")
        print(f"  Progress: {job.progress_percentage:.1f}% "
              f"({job.processed_items}/{job.total_items})")
        print(f"  Failed items: {job.failed_items}")
        if job.error_severity:
            print(f"  Error severity: {job.error_severity.value}")
        print(f"  Is completed: {job.is_completed}")
        print(f"  Is running: {job.is_running}")
        print(f"  Has errors: {job.has_errors}")
    
    print()


if __name__ == "__main__":
    print("\n🎯 ENUMS FOR CONTROLLED VALUES DEMONSTRATION\n")
    
    demo_basic_enums()
    demo_integer_enums()
    demo_entity_extraction()
    demo_user_status()
    demo_geographic_enums()
    demo_content_status()
    demo_validation_levels()
    demo_batch_processing()
    
    print("=" * 70)
    print("✅ All demonstrations completed!")
    print("=" * 70)
```

### Line-by-Line Explanation

**Enum Types:**

1. **`class Status(str, Enum)`**: String enum—values are strings
2. **`class Level(IntEnum)`**: Integer enum—values are integers, can be compared/ordered
3. **`VALUE = "value"`**: Enum members are UPPERCASE by convention

**Enum Benefits:**

1. **Validation**: Only defined values are valid
2. **Autocomplete**: IDE shows all valid options
3. **Refactoring**: Change enum definition, find all uses
4. **Comparison**: IntEnum supports `<`, `>`, `<=`, `>=`
5. **Iteration**: `for status in TaskStatus:` lists all options

**Access Patterns:**

```python
# Create with enum value
task = Task(status=TaskStatus.TODO)

# Access the value
print(task.status.value)  # "todo"

# Access the name
print(task.status.name)  # "TODO"

# Compare
if task.status == TaskStatus.COMPLETED:
    ...

# Check membership
if task.status in (TaskStatus.IN_PROGRESS, TaskStatus.IN_REVIEW):
    ...
```

### The "Why" Behind the Pattern

**Prevent Invalid States:**
Without enums, someone could set `status = "todoo"` (typo). With enums, invalid values are rejected immediately.

**Self-Documenting:**
`status: TaskStatus` tells you exactly what values are valid. `status: str` doesn't.

**Refactor-Safe:**
Change `TaskStatus.TODO` to `TaskStatus.PENDING`, and your IDE can find and update all uses.

**Type Narrowing:**
```python
if task.status == TaskStatus.COMPLETED:
    # IDE knows this branch only runs when status is COMPLETED
```

---

## C. Test & Apply

### How to Test It

**Step 1: Create files**
```bash
cd data_extraction_pipeline
touch src/models/enum_models.py
touch src/examples/enum_demo.py
```

**Step 2: Run demonstration**
```bash
python -m src.examples.enum_demo
```

### Expected Result

Comprehensive output showing various enum patterns and their benefits.

---

## D. Common Stumbling Blocks

### Proactive Debugging

**Mistake 1: Using strings instead of enum values**

```python
# ❌ WRONG - String instead of enum
task = Task(..., status="completed")  # Works, but not type-safe

# ✅ CORRECT - Use enum value
task = Task(..., status=TaskStatus.COMPLETED)
```

**Mistake 2: Forgetting `str` or `int` in enum definition**

```python
# ❌ WRONG - Plain Enum doesn't work well with Pydantic
class Status(Enum):
    TODO = "todo"

# ✅ CORRECT - Inherit from str
class Status(str, Enum):
    TODO = "todo"
```

### Show the Error

**Error: Invalid enum value**

```python
task = Task(
    title="Test",
    description="Test task",
    status="invalid_status",  # Not a valid TaskStatus
    category=TaskCategory.FEATURE
)
```

**Error message:**
```
ValidationError: 1 validation error for Task
status
  Input should be 'todo', 'in_progress', 'in_review', 'completed' or 'cancelled' [type=enum, input_value='invalid_status', input_type=str]
```

### Type Safety Gotchas

1. **Case sensitivity**: Enum values are case-sensitive unless you add validation
2. **String vs Enum**: Pydantic accepts string that matches enum value, converts automatically
3. **Comparison**: Use `==` for equality, `is` won't work for enums
4. **IntEnum ordering**: Only IntEnum supports `<`, `>` comparisons
5. **Iteration**: `for status in TaskStatus:` iterates over enum members

---

## 🎯 Next Steps

Fantastic! You now understand:
- ✅ How to create string and integer enums
- ✅ How to use enums for validation
- ✅ How enums provide IDE autocomplete
- ✅ How to iterate and compare enum values
- ✅ How enums prevent invalid states

In the next lesson, we'll explore **Custom Validators**—learning how to add custom validation logic beyond basic types and constraints.

**Ready for Lesson 7?** 🚀
