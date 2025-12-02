# Lesson 7: Custom Validators

## A. Concept Overview

### What & Why
**Custom validators allow you to add business logic and complex validation rules beyond basic type checking.** While Pydantic handles type validation automatically, real-world data extraction needs domain-specific rules—email domains must be allowed, dates must be in range, extracted entities must pass quality checks. Custom validators let you encode these rules directly in your models.

### Analogy
Think of custom validators like a security checkpoint with specialized inspectors:
- **Basic type validation** is like checking IDs—verify everyone has the right document type
- **Field constraints** are like checking ID expiration—verify values are in valid ranges
- **Custom validators** are like specialized background checks—apply domain-specific rules (is this person on the approved list? does their story make sense? are all their documents consistent?)

When Gemini extracts data, custom validators are those specialized inspectors ensuring the data meets your exact business requirements.

### Type Safety Benefit
Custom validators provide **domain-specific validation**:
- Validate relationships between fields—end_date must be after start_date
- Normalize and clean data—capitalize names, format phone numbers
- Enforce business rules—user must be 18+, price must be positive
- Cross-field validation—password and confirm_password must match
- Computed validation—check external data sources, validate checksums
- Clear error messages—explain exactly what rule was violated

---

## B. Code Implementation

### File Structure
```
data_extraction_pipeline/
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── validator_models.py  # New: Custom validator patterns
│   └── examples/
│       ├── __init__.py
│       └── validator_demo.py     # New: This lesson
```

### Complete Code Implementation

**File: `src/models/validator_models.py`**

```python
"""Models demonstrating custom validators."""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Optional
from datetime import datetime, date
from enum import Enum
import re


# ============================================================================
# FIELD VALIDATORS (Single Field Validation)
# ============================================================================

class UserRegistration(BaseModel):
    """User registration with field validators."""
    username: str = Field(..., min_length=3, max_length=20)
    email: str
    password: str = Field(..., min_length=8)
    age: int = Field(..., ge=0, le=150)
    website: Optional[str] = None
    
    @field_validator('username')
    @classmethod
    def username_alphanumeric(cls, v: str) -> str:
        """Ensure username is alphanumeric."""
        if not v.isalnum():
            raise ValueError('Username must be alphanumeric')
        return v.lower()  # Normalize to lowercase
    
    @field_validator('email')
    @classmethod
    def email_domain_allowed(cls, v: str) -> str:
        """Ensure email is from allowed domains."""
        allowed_domains = ['example.com', 'company.com', 'test.org']
        domain = v.split('@')[-1].lower()
        if domain not in allowed_domains:
            raise ValueError(f'Email domain {domain} not allowed. Allowed: {allowed_domains}')
        return v.lower()
    
    @field_validator('password')
    @classmethod
    def password_strength(cls, v: str) -> str:
        """Ensure password has sufficient strength."""
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        if not any(c in '!@#$%^&*()_+-=[]{}|;:,.<>?' for c in v):
            raise ValueError('Password must contain at least one special character')
        return v
    
    @field_validator('age')
    @classmethod
    def age_adult(cls, v: int) -> int:
        """Ensure user is at least 18 years old."""
        if v < 18:
            raise ValueError('User must be at least 18 years old')
        return v
    
    @field_validator('website')
    @classmethod
    def website_format(cls, v: Optional[str]) -> Optional[str]:
        """Validate and normalize website URL."""
        if v is None:
            return v
        
        # Add https:// if no protocol
        if not v.startswith(('http://', 'https://')):
            v = f'https://{v}'
        
        # Basic URL validation
        url_pattern = r'https?://[^\s/$.?#].[^\s]*'
        if not re.match(url_pattern, v):
            raise ValueError('Invalid website URL')
        
        return v


# ============================================================================
# DATA NORMALIZATION VALIDATORS
# ============================================================================

class ContactInfo(BaseModel):
    """Contact information with normalization."""
    name: str
    phone: str
    email: str
    
    @field_validator('name')
    @classmethod
    def normalize_name(cls, v: str) -> str:
        """Normalize name to title case."""
        return v.strip().title()
    
    @field_validator('phone')
    @classmethod
    def normalize_phone(cls, v: str) -> str:
        """Normalize phone number to standard format."""
        # Remove all non-digit characters
        digits = re.sub(r'\D', '', v)
        
        # Validate length
        if len(digits) < 10:
            raise ValueError('Phone number must have at least 10 digits')
        
        # Format as (XXX) XXX-XXXX for US numbers
        if len(digits) == 10:
            return f'({digits[:3]}) {digits[3:6]}-{digits[6:]}'
        elif len(digits) == 11 and digits[0] == '1':
            return f'+1 ({digits[1:4]}) {digits[4:7]}-{digits[7:]}'
        else:
            # Return as-is for international numbers
            return f'+{digits}'
    
    @field_validator('email')
    @classmethod
    def normalize_email(cls, v: str) -> str:
        """Normalize email to lowercase."""
        return v.lower().strip()


# ============================================================================
# DATE RANGE VALIDATORS
# ============================================================================

class DateRange(BaseModel):
    """Date range with validation."""
    start_date: date
    end_date: date
    
    @field_validator('end_date')
    @classmethod
    def end_after_start(cls, v: date, info) -> date:
        """Ensure end_date is after start_date."""
        # Access other field values via info.data
        if 'start_date' in info.data and v < info.data['start_date']:
            raise ValueError('end_date must be after start_date')
        return v


class Event(BaseModel):
    """Event with date range validation."""
    name: str
    start_datetime: datetime
    end_datetime: datetime
    registration_deadline: Optional[datetime] = None
    
    @field_validator('end_datetime')
    @classmethod
    def end_after_start(cls, v: datetime, info) -> datetime:
        """Ensure event ends after it starts."""
        if 'start_datetime' in info.data and v <= info.data['start_datetime']:
            raise ValueError('Event must end after it starts')
        return v
    
    @field_validator('registration_deadline')
    @classmethod
    def deadline_before_start(cls, v: Optional[datetime], info) -> Optional[datetime]:
        """Ensure registration deadline is before event start."""
        if v is None:
            return v
        
        if 'start_datetime' in info.data and v >= info.data['start_datetime']:
            raise ValueError('Registration deadline must be before event start')
        
        return v


# ============================================================================
# LIST VALIDATORS
# ============================================================================

class TaggedContent(BaseModel):
    """Content with validated and normalized tags."""
    title: str
    content: str
    tags: List[str] = Field(default_factory=list)
    
    @field_validator('tags')
    @classmethod
    def normalize_tags(cls, v: List[str]) -> List[str]:
        """Normalize tags: lowercase, no duplicates, sorted."""
        if not v:
            return []
        
        # Convert to lowercase and strip whitespace
        normalized = [tag.lower().strip() for tag in v]
        
        # Remove empty strings
        normalized = [tag for tag in normalized if tag]
        
        # Remove duplicates while preserving order
        seen = set()
        unique = []
        for tag in normalized:
            if tag not in seen:
                seen.add(tag)
                unique.append(tag)
        
        # Sort alphabetically
        return sorted(unique)


class ExtractedEntities(BaseModel):
    """Extracted entities with validation."""
    entities: List[str] = Field(..., min_length=1)
    confidence_scores: List[float] = Field(..., min_length=1)
    
    @field_validator('entities')
    @classmethod
    def validate_entities(cls, v: List[str]) -> List[str]:
        """Validate extracted entities."""
        if not v:
            raise ValueError('At least one entity required')
        
        # Remove duplicates
        unique_entities = list(dict.fromkeys(v))
        
        # Remove empty or whitespace-only entities
        valid_entities = [e.strip() for e in unique_entities if e and e.strip()]
        
        if not valid_entities:
            raise ValueError('At least one non-empty entity required')
        
        return valid_entities
    
    @field_validator('confidence_scores')
    @classmethod
    def validate_confidence_scores(cls, v: List[float]) -> List[float]:
        """Validate confidence scores are in [0, 1] range."""
        for score in v:
            if not 0.0 <= score <= 1.0:
                raise ValueError(f'Confidence score {score} must be between 0.0 and 1.0')
        return v


# ============================================================================
# MODEL VALIDATORS (Multi-Field Validation)
# ============================================================================

class PasswordChange(BaseModel):
    """Password change with confirmation."""
    old_password: str
    new_password: str = Field(..., min_length=8)
    confirm_password: str
    
    @field_validator('new_password')
    @classmethod
    def password_strength(cls, v: str) -> str:
        """Validate password strength."""
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain uppercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain digit')
        return v
    
    @model_validator(mode='after')
    def check_passwords_match(self):
        """Ensure new password and confirmation match."""
        if self.new_password != self.confirm_password:
            raise ValueError('Passwords do not match')
        return self
    
    @model_validator(mode='after')
    def check_password_different(self):
        """Ensure new password is different from old."""
        if self.old_password == self.new_password:
            raise ValueError('New password must be different from old password')
        return self


class ExtractedPerson(BaseModel):
    """Extracted person with consistency validation."""
    full_name: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    age: Optional[int] = Field(None, ge=0, le=150)
    birth_year: Optional[int] = Field(None, ge=1900, le=2024)
    
    @model_validator(mode='after')
    def check_name_consistency(self):
        """Ensure first_name and last_name are consistent with full_name."""
        if self.first_name and self.last_name:
            expected_full = f"{self.first_name} {self.last_name}"
            if self.full_name.lower() != expected_full.lower():
                raise ValueError(
                    f'full_name "{self.full_name}" does not match '
                    f'first_name "{self.first_name}" and last_name "{self.last_name}"'
                )
        return self
    
    @model_validator(mode='after')
    def check_age_birth_year_consistency(self):
        """Ensure age and birth_year are consistent."""
        if self.age is not None and self.birth_year is not None:
            current_year = datetime.now().year
            expected_age = current_year - self.birth_year
            
            # Allow 1 year difference for birthdays
            if abs(self.age - expected_age) > 1:
                raise ValueError(
                    f'age {self.age} is inconsistent with birth_year {self.birth_year} '
                    f'(expected age ~{expected_age})'
                )
        return self


class ExtractedAmount(BaseModel):
    """Extracted monetary amount with format validation."""
    raw_text: str
    amount: float
    currency: str
    
    @field_validator('currency')
    @classmethod
    def validate_currency(cls, v: str) -> str:
        """Validate currency code."""
        valid_currencies = ['USD', 'EUR', 'GBP', 'JPY', 'CNY']
        v_upper = v.upper()
        if v_upper not in valid_currencies:
            raise ValueError(f'Invalid currency: {v}. Valid: {valid_currencies}')
        return v_upper
    
    @model_validator(mode='after')
    def check_consistency(self):
        """Ensure raw_text and parsed amount are consistent."""
        # Extract numbers from raw_text
        numbers = re.findall(r'[\d,]+\.?\d*', self.raw_text)
        if numbers:
            # Parse the first number found
            parsed = float(numbers[0].replace(',', ''))
            
            # Check if parsed amount is close to extracted amount
            if abs(parsed - self.amount) > 0.01:
                raise ValueError(
                    f'Extracted amount {self.amount} does not match '
                    f'raw text "{self.raw_text}" (parsed as {parsed})'
                )
        
        return self


# ============================================================================
# QUALITY VALIDATORS
# ============================================================================

class QualityThreshold(str, Enum):
    """Quality thresholds."""
    MINIMUM = "minimum"
    STANDARD = "standard"
    HIGH = "high"


class ValidatedExtraction(BaseModel):
    """Extraction with quality validation."""
    text: str
    entity_type: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    context: Optional[str] = None
    quality_threshold: QualityThreshold = QualityThreshold.STANDARD
    
    @field_validator('text')
    @classmethod
    def text_not_empty(cls, v: str) -> str:
        """Ensure extracted text is not empty."""
        if not v or not v.strip():
            raise ValueError('Extracted text cannot be empty')
        return v.strip()
    
    @model_validator(mode='after')
    def check_quality(self):
        """Validate extraction meets quality threshold."""
        thresholds = {
            QualityThreshold.MINIMUM: 0.5,
            QualityThreshold.STANDARD: 0.7,
            QualityThreshold.HIGH: 0.9,
        }
        
        required_confidence = thresholds[self.quality_threshold]
        
        if self.confidence < required_confidence:
            raise ValueError(
                f'Confidence {self.confidence:.2f} below {self.quality_threshold.value} '
                f'threshold ({required_confidence:.2f})'
            )
        
        return self


# ============================================================================
# BATCH VALIDATION
# ============================================================================

class BatchExtraction(BaseModel):
    """Batch extraction with overall validation."""
    extraction_id: str
    source_text: str
    entities: List[str] = Field(default_factory=list)
    confidence_scores: List[float] = Field(default_factory=list)
    minimum_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    
    @model_validator(mode='after')
    def check_lists_match(self):
        """Ensure entities and scores have same length."""
        if len(self.entities) != len(self.confidence_scores):
            raise ValueError(
                f'entities ({len(self.entities)}) and confidence_scores '
                f'({len(self.confidence_scores)}) must have same length'
            )
        return self
    
    @model_validator(mode='after')
    def filter_low_confidence(self):
        """Filter out low-confidence extractions."""
        if not self.entities:
            return self
        
        # Filter entities below minimum confidence
        filtered = [
            (entity, score)
            for entity, score in zip(self.entities, self.confidence_scores)
            if score >= self.minimum_confidence
        ]
        
        if filtered:
            self.entities, self.confidence_scores = zip(*filtered)
            self.entities = list(self.entities)
            self.confidence_scores = list(self.confidence_scores)
        else:
            self.entities = []
            self.confidence_scores = []
        
        return self
```

**File: `src/examples/validator_demo.py`**

```python
"""Demonstration of custom validators."""

from datetime import datetime, date, timedelta
from pydantic import ValidationError
from src.models.validator_models import (
    UserRegistration,
    ContactInfo,
    DateRange,
    Event,
    TaggedContent,
    ExtractedEntities,
    PasswordChange,
    ExtractedPerson,
    ExtractedAmount,
    ValidatedExtraction,
    QualityThreshold,
    BatchExtraction,
)


def demo_field_validators():
    """Demonstrate field validators."""
    print("=" * 70)
    print("FIELD VALIDATORS")
    print("=" * 70)
    
    # Valid registration
    print("✅ Valid registration:")
    user = UserRegistration(
        username="Alice123",
        email="alice@example.com",
        password="SecurePass123!",
        age=25,
        website="alice.dev"
    )
    print(f"  Username: {user.username} (normalized to lowercase)")
    print(f"  Email: {user.email}")
    print(f"  Website: {user.website} (added https://)")
    
    # Invalid username (non-alphanumeric)
    print("\n❌ Invalid username:")
    try:
        user = UserRegistration(
            username="alice_123",  # Contains underscore
            email="alice@example.com",
            password="SecurePass123!",
            age=25
        )
    except ValidationError as e:
        print(f"  Error: {e.errors()[0]['msg']}")
    
    # Invalid email domain
    print("\n❌ Invalid email domain:")
    try:
        user = UserRegistration(
            username="alice",
            email="alice@invalid.com",  # Not in allowed domains
            password="SecurePass123!",
            age=25
        )
    except ValidationError as e:
        print(f"  Error: {e.errors()[0]['msg']}")
    
    # Weak password
    print("\n❌ Weak password:")
    try:
        user = UserRegistration(
            username="alice",
            email="alice@example.com",
            password="password",  # No uppercase, digit, or special char
            age=25
        )
    except ValidationError as e:
        print(f"  Error: {e.errors()[0]['msg']}")
    
    # Underage user
    print("\n❌ Underage user:")
    try:
        user = UserRegistration(
            username="bob",
            email="bob@example.com",
            password="SecurePass123!",
            age=16  # Under 18
        )
    except ValidationError as e:
        print(f"  Error: {e.errors()[0]['msg']}")
    
    print()


def demo_normalization():
    """Demonstrate data normalization."""
    print("=" * 70)
    print("DATA NORMALIZATION")
    print("=" * 70)
    
    # Various input formats
    contacts = [
        ContactInfo(
            name="  alice johnson  ",
            phone="5551234567",
            email="Alice@Example.COM"
        ),
        ContactInfo(
            name="bob smith",
            phone="(555) 987-6543",
            email="BOB@COMPANY.COM"
        ),
        ContactInfo(
            name="charlie BROWN",
            phone="+1-555-111-2222",
            email="charlie@test.org"
        ),
    ]
    
    print("Normalized contacts:")
    for contact in contacts:
        print(f"\n  Name: {contact.name}")
        print(f"  Phone: {contact.phone}")
        print(f"  Email: {contact.email}")
    
    print()


def demo_date_validators():
    """Demonstrate date range validation."""
    print("=" * 70)
    print("DATE RANGE VALIDATION")
    print("=" * 70)
    
    # Valid date range
    print("✅ Valid date range:")
    date_range = DateRange(
        start_date=date(2024, 1, 1),
        end_date=date(2024, 12, 31)
    )
    print(f"  Start: {date_range.start_date}")
    print(f"  End: {date_range.end_date}")
    
    # Invalid date range
    print("\n❌ Invalid date range:")
    try:
        date_range = DateRange(
            start_date=date(2024, 12, 31),
            end_date=date(2024, 1, 1)  # Before start
        )
    except ValidationError as e:
        print(f"  Error: {e.errors()[0]['msg']}")
    
    # Valid event
    print("\n✅ Valid event:")
    now = datetime.now()
    event = Event(
        name="Tech Conference",
        start_datetime=now + timedelta(days=30),
        end_datetime=now + timedelta(days=32),
        registration_deadline=now + timedelta(days=15)
    )
    print(f"  Event: {event.name}")
    print(f"  Registration deadline: {event.registration_deadline}")
    
    # Invalid event (registration after start)
    print("\n❌ Invalid event:")
    try:
        event = Event(
            name="Tech Conference",
            start_datetime=now + timedelta(days=30),
            end_datetime=now + timedelta(days=32),
            registration_deadline=now + timedelta(days=35)  # After start!
        )
    except ValidationError as e:
        print(f"  Error: {e.errors()[0]['msg']}")
    
    print()


def demo_list_validators():
    """Demonstrate list validators."""
    print("=" * 70)
    print("LIST VALIDATORS")
    print("=" * 70)
    
    # Tags normalization
    print("Tags normalization:")
    content = TaggedContent(
        title="Pydantic AI Tutorial",
        content="Learn about type-safe AI development",
        tags=["Python", "AI", "python", "Pydantic", "ai", "Type-Safety", "pydantic"]
    )
    print(f"  Original tags: Python, AI, python, Pydantic, ai, Type-Safety, pydantic")
    print(f"  Normalized tags: {content.tags}")
    
    # Entity validation
    print("\n✅ Valid extracted entities:")
    entities = ExtractedEntities(
        entities=["Apple Inc.", "Tim Cook", "New York", "Apple Inc."],  # Has duplicate
        confidence_scores=[0.99, 0.98, 0.95, 0.99]
    )
    print(f"  Entities: {entities.entities}")  # Duplicate removed
    print(f"  Scores: {entities.confidence_scores}")
    
    print()


def demo_model_validators():
    """Demonstrate model validators."""
    print("=" * 70)
    print("MODEL VALIDATORS (Multi-Field)")
    print("=" * 70)
    
    # Valid password change
    print("✅ Valid password change:")
    pwd_change = PasswordChange(
        old_password="OldPass123!",
        new_password="NewPass456!",
        confirm_password="NewPass456!"
    )
    print(f"  Password change validated")
    
    # Passwords don't match
    print("\n❌ Passwords don't match:")
    try:
        pwd_change = PasswordChange(
            old_password="OldPass123!",
            new_password="NewPass456!",
            confirm_password="DifferentPass789!"
        )
    except ValidationError as e:
        print(f"  Error: {e.errors()[0]['msg']}")
    
    # New password same as old
    print("\n❌ New password same as old:")
    try:
        pwd_change = PasswordChange(
            old_password="OldPass123!",
            new_password="OldPass123!",
            confirm_password="OldPass123!"
        )
    except ValidationError as e:
        print(f"  Error: {e.errors()[0]['msg']}")
    
    # Valid extracted person
    print("\n✅ Valid extracted person (consistent data):")
    person = ExtractedPerson(
        full_name="Alice Johnson",
        first_name="Alice",
        last_name="Johnson",
        age=32,
        birth_year=1992
    )
    print(f"  {person.full_name}, age {person.age}")
    
    # Inconsistent names
    print("\n❌ Inconsistent names:")
    try:
        person = ExtractedPerson(
            full_name="Alice Johnson",
            first_name="Alice",
            last_name="Smith",  # Doesn't match full_name
            age=32
        )
    except ValidationError as e:
        print(f"  Error: {e.errors()[0]['msg']}")
    
    # Inconsistent age/birth year
    print("\n❌ Inconsistent age and birth year:")
    try:
        person = ExtractedPerson(
            full_name="Bob Smith",
            age=50,
            birth_year=2000  # Would make them ~24, not 50
        )
    except ValidationError as e:
        print(f"  Error: {e.errors()[0]['msg']}")
    
    print()


def demo_quality_validators():
    """Demonstrate quality validation."""
    print("=" * 70)
    print("QUALITY VALIDATORS")
    print("=" * 70)
    
    # High quality extraction
    print("✅ High quality extraction:")
    extraction = ValidatedExtraction(
        text="Apple Inc.",
        entity_type="ORGANIZATION",
        confidence=0.95,
        quality_threshold=QualityThreshold.STANDARD
    )
    print(f"  Entity: {extraction.text}")
    print(f"  Confidence: {extraction.confidence:.0%}")
    print(f"  Threshold: {extraction.quality_threshold.value}")
    
    # Low quality extraction
    print("\n❌ Low quality extraction:")
    try:
        extraction = ValidatedExtraction(
            text="something",
            entity_type="LOCATION",
            confidence=0.60,
            quality_threshold=QualityThreshold.HIGH  # Requires 0.9
        )
    except ValidationError as e:
        print(f"  Error: {e.errors()[0]['msg']}")
    
    print()


def demo_batch_validation():
    """Demonstrate batch validation."""
    print("=" * 70)
    print("BATCH VALIDATION")
    print("=" * 70)
    
    # Batch with filtering
    print("Batch extraction with confidence filtering:")
    batch = BatchExtraction(
        extraction_id="batch-001",
        source_text="Sample text",
        entities=["Entity1", "Entity2", "Entity3", "Entity4"],
        confidence_scores=[0.95, 0.60, 0.40, 0.85],
        minimum_confidence=0.7  # Filter below 0.7
    )
    
    print(f"  Original: 4 entities")
    print(f"  After filtering (min 0.7): {len(batch.entities)} entities")
    print(f"  Entities: {batch.entities}")
    print(f"  Scores: {[f'{s:.2f}' for s in batch.confidence_scores]}")
    
    # Mismatched lists
    print("\n❌ Mismatched entity and score counts:")
    try:
        batch = BatchExtraction(
            extraction_id="batch-002",
            source_text="Sample text",
            entities=["Entity1", "Entity2"],
            confidence_scores=[0.95]  # Wrong length!
        )
    except ValidationError as e:
        print(f"  Error: {e.errors()[0]['msg']}")
    
    print()


if __name__ == "__main__":
    print("\n🎯 CUSTOM VALIDATORS DEMONSTRATION\n")
    
    demo_field_validators()
    demo_normalization()
    demo_date_validators()
    demo_list_validators()
    demo_model_validators()
    demo_quality_validators()
    demo_batch_validation()
    
    print("=" * 70)
    print("✅ All demonstrations completed!")
    print("=" * 70)
```

### Line-by-Line Explanation

**Validator Types:**

1. **`@field_validator('field_name')`**: Validates/transforms a single field
2. **`@model_validator(mode='after')`**: Validates multiple fields together after all fields are set
3. **`@classmethod`**: Required decorator for validators

**Validator Patterns:**

```python
@field_validator('field_name')
@classmethod
def validate_field(cls, v: FieldType) -> FieldType:
    """Validate and return the field value."""
    if not_valid(v):
        raise ValueError('Explanation of why invalid')
    return transform(v)  # Can transform the value

@model_validator(mode='after')
def validate_model(self):
    """Validate relationships between fields."""
    if self.field1 conflicts_with self.field2:
        raise ValueError('Explanation')
    return self  # Must return self
```

**Key Concepts:**

1. **Field validators**: Receive field value, return transformed value or raise ValueError
2. **Model validators**: Receive full model instance, return self or raise ValueError
3. **Access other fields**: In field validators use `info.data`, in model validators use `self.field`
4. **Transformation**: Validators can modify values (normalize, clean, format)
5. **Error messages**: Clear, specific error messages help debugging

### The "Why" Behind the Pattern

**Business Logic Validation:**
Type checking validates types, but business rules need custom logic. "User must be 18+" is a business rule, not a type constraint.

**Data Normalization:**
Extract data from text often needs cleaning—capitalize names, format phones, lowercase emails. Validators handle this automatically.

**Cross-Field Validation:**
Some rules span multiple fields—password confirmation, date ranges, consistency checks. Model validators handle these relationships.

**Quality Gates:**
When Gemini extracts data, you need quality thresholds. Validators reject low-confidence extractions before they enter your system.

---

## C. Test & Apply

### How to Test It

**Step 1: Create files**
```bash
cd data_extraction_pipeline
touch src/models/validator_models.py
touch src/examples/validator_demo.py
```

**Step 2: Run demonstration**
```bash
python -m src.examples.validator_demo
```

### Expected Result

Comprehensive output showing:
- Field validators for individual fields
- Data normalization
- Date range validation
- List validation
- Multi-field validation
- Quality validation
- Batch validation with filtering

---

## D. Common Stumbling Blocks

### Proactive Debugging

**Mistake 1: Forgetting @classmethod on field validators**

```python
# ❌ WRONG
@field_validator('username')
def validate_username(cls, v: str) -> str:  # Missing @classmethod
    ...

# ✅ CORRECT
@field_validator('username')
@classmethod
def validate_username(cls, v: str) -> str:
    ...
```

**Mistake 2: Not returning value from validator**

```python
# ❌ WRONG
@field_validator('email')
@classmethod
def validate_email(cls, v: str):
    if '@' not in v:
        raise ValueError('Invalid email')
    # Forgot to return v!

# ✅ CORRECT
@field_validator('email')
@classmethod
def validate_email(cls, v: str) -> str:
    if '@' not in v:
        raise ValueError('Invalid email')
    return v  # Must return!
```

**Mistake 3: Not returning self from model validator**

```python
# ❌ WRONG
@model_validator(mode='after')
def validate(self):
    if self.field1 != self.field2:
        raise ValueError('Mismatch')
    # Forgot to return self!

# ✅ CORRECT
@model_validator(mode='after')
def validate(self):
    if self.field1 != self.field2:
        raise ValueError('Mismatch')
    return self  # Must return!
```

### Type Safety Gotchas

1. **Validator order**: Field validators run before model validators
2. **Type hints**: Add type hints to validator parameters and return values
3. **Accessing other fields**: Use `info.data` in field validators, `self.field` in model validators
4. **Mutation**: Validators can modify values, but be careful with mutable types
5. **Exceptions**: Only raise `ValueError` or `AssertionError` in validators

---

## 🎯 Next Steps

Excellent work! You now understand:
- ✅ How to create field validators for single-field validation
- ✅ How to create model validators for multi-field validation
- ✅ How to normalize and transform data in validators
- ✅ How to validate relationships between fields
- ✅ How to implement quality thresholds
- ✅ How to provide clear error messages

In the next lesson, we'll explore **Field Constraints and Limits**—learning Pydantic's built-in constraint system for common validation patterns.

**Ready for Lesson 8?** 🚀
