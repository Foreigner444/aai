# Lesson 4: Optional Fields and Defaults

## A. Concept Overview

### What & Why
**Optional fields and default values are essential for handling real-world data where information might be missing or incomplete.** When extracting data from unstructured text with Gemini, not every piece of information is always present. Optional fields let you model this reality while maintaining type safety—you explicitly define what can be missing and what defaults to use.

### Analogy
Think of optional fields like a job application form:
- **Required fields** (name, email) are marked with a red asterisk (*) — you can't submit without them
- **Optional fields** (phone number, LinkedIn) can be left blank — the form still submits
- **Fields with defaults** (country = "USA") are pre-filled — you can change them or leave the default

When Gemini extracts data, it's like filling out this form. Required fields must be found, optional fields can be missing, and defaulted fields use fallback values when not specified.

### Type Safety Benefit
Optional fields provide **explicit nullable types**:
- `name: str` — must be provided, cannot be None
- `phone: Optional[str]` — can be None or a string, but you must handle both cases
- `country: str = "USA"` — defaults to "USA" if not provided, but cannot be None
- `phone: Optional[str] = None` — can be None, defaults to None if not provided

Your IDE knows when values might be None, forcing you to check before use: `if person.phone:`. This prevents null reference errors at runtime.

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
│   │   ├── collections.py
│   │   └── optional_models.py  # New: Optional field patterns
│   └── examples/
│       ├── __init__.py
│       └── optional_demo.py     # New: This lesson
```

### Complete Code Implementation

**File: `src/models/optional_models.py`**

```python
"""Models demonstrating optional fields and default values."""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime, date
from enum import Enum


class ContactPreference(str, Enum):
    """Preferred contact method."""
    EMAIL = "email"
    PHONE = "phone"
    SMS = "sms"
    ANY = "any"


class PersonProfile(BaseModel):
    """
    Person profile with optional fields.
    Demonstrates: required, optional, and default values.
    """
    # Required fields - must be provided
    first_name: str = Field(..., min_length=1)
    last_name: str = Field(..., min_length=1)
    email: str
    
    # Optional fields - can be None
    middle_name: Optional[str] = None
    phone: Optional[str] = None
    date_of_birth: Optional[date] = None
    bio: Optional[str] = None
    website: Optional[str] = None
    
    # Fields with defaults - uses default if not provided
    country: str = "USA"
    contact_preference: ContactPreference = ContactPreference.EMAIL
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.now)
    
    # Optional field with a non-None default
    timezone: Optional[str] = "UTC"
    
    @property
    def full_name(self) -> str:
        """Construct full name, including middle name if present."""
        if self.middle_name:
            return f"{self.first_name} {self.middle_name} {self.last_name}"
        return f"{self.first_name} {self.last_name}"
    
    @property
    def age(self) -> Optional[int]:
        """Calculate age if date_of_birth is provided."""
        if not self.date_of_birth:
            return None
        today = date.today()
        return today.year - self.date_of_birth.year - (
            (today.month, today.day) < (self.date_of_birth.month, self.date_of_birth.day)
        )


class Address(BaseModel):
    """Address with optional fields."""
    # Required
    city: str
    country: str
    
    # Optional - not all addresses have these
    street: Optional[str] = None
    apartment: Optional[str] = None
    state: Optional[str] = None
    zip_code: Optional[str] = None
    
    # Default
    is_primary: bool = False
    
    def __str__(self) -> str:
        """Format address, handling optional fields."""
        parts = []
        if self.street:
            parts.append(self.street)
        if self.apartment:
            parts.append(f"Apt {self.apartment}")
        parts.append(f"{self.city}, {self.country}")
        if self.state:
            parts[-1] = f"{self.city}, {self.state}, {self.country}"
        if self.zip_code:
            parts.append(self.zip_code)
        return ", ".join(parts)


class CompanyInfo(BaseModel):
    """Company information with extensive optional fields."""
    # Required
    name: str = Field(..., min_length=1)
    
    # Optional metadata
    industry: Optional[str] = None
    founded_year: Optional[int] = Field(None, ge=1800, le=2100)
    employee_count: Optional[int] = Field(None, ge=0)
    revenue: Optional[float] = Field(None, ge=0)
    website: Optional[str] = None
    description: Optional[str] = None
    
    # Optional nested objects
    headquarters: Optional[Address] = None
    
    # Optional collections
    office_locations: List[Address] = Field(default_factory=list)
    products: List[str] = Field(default_factory=list)
    
    # Defaults
    is_public: bool = False
    is_verified: bool = False
    
    @property
    def has_detailed_info(self) -> bool:
        """Check if company has detailed information."""
        return all([
            self.industry,
            self.founded_year,
            self.employee_count,
            self.website
        ])


class ArticleMetadata(BaseModel):
    """Article metadata with many optional fields."""
    # Required
    title: str = Field(..., min_length=1)
    
    # Optional author info
    author: Optional[str] = None
    author_email: Optional[str] = None
    
    # Optional dates
    published_date: Optional[datetime] = None
    updated_date: Optional[datetime] = None
    
    # Optional content metadata
    word_count: Optional[int] = Field(None, ge=0)
    reading_time_minutes: Optional[int] = Field(None, ge=0)
    excerpt: Optional[str] = Field(None, max_length=500)
    
    # Optional categorization
    category: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    
    # Optional SEO
    meta_description: Optional[str] = Field(None, max_length=160)
    keywords: List[str] = Field(default_factory=list)
    
    # Defaults
    is_published: bool = False
    is_featured: bool = False
    language: str = "en"
    created_at: datetime = Field(default_factory=datetime.now)


class ExtractedEntity(BaseModel):
    """
    Entity extracted from text with confidence and optional metadata.
    Real-world extraction use case.
    """
    # Required
    text: str = Field(..., min_length=1, description="Extracted text")
    entity_type: str = Field(..., description="Type of entity (person, org, etc)")
    
    # Confidence and position
    confidence: float = Field(..., ge=0.0, le=1.0, description="Extraction confidence")
    start_pos: Optional[int] = Field(None, ge=0, description="Start position in source")
    end_pos: Optional[int] = Field(None, ge=0, description="End position in source")
    
    # Optional metadata
    canonical_form: Optional[str] = Field(None, description="Normalized form")
    aliases: List[str] = Field(default_factory=list, description="Alternative names")
    
    # Optional linked data
    external_id: Optional[str] = Field(None, description="ID in external knowledge base")
    wikidata_id: Optional[str] = Field(None, description="Wikidata identifier")
    
    # Optional context
    context_before: Optional[str] = Field(None, max_length=200)
    context_after: Optional[str] = Field(None, max_length=200)
    
    # Defaults
    is_validated: bool = False
    extracted_at: datetime = Field(default_factory=datetime.now)


class DataExtractionResult(BaseModel):
    """
    Complete extraction result with optional fields throughout.
    Demonstrates handling incomplete extractions.
    """
    # Required
    extraction_id: str
    source_text: str
    
    # Optional source metadata
    source_url: Optional[str] = None
    source_title: Optional[str] = None
    source_author: Optional[str] = None
    source_date: Optional[datetime] = None
    
    # Extracted entities (can be empty)
    people: List[ExtractedEntity] = Field(default_factory=list)
    organizations: List[ExtractedEntity] = Field(default_factory=list)
    locations: List[ExtractedEntity] = Field(default_factory=list)
    
    # Optional analysis
    summary: Optional[str] = None
    sentiment: Optional[str] = None
    language: Optional[str] = None
    
    # Optional quality metrics
    extraction_confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    processing_time_ms: Optional[int] = Field(None, ge=0)
    
    # Defaults
    extracted_at: datetime = Field(default_factory=datetime.now)
    status: str = "completed"
    
    @property
    def total_entities(self) -> int:
        """Count all extracted entities."""
        return len(self.people) + len(self.organizations) + len(self.locations)
    
    @property
    def has_entities(self) -> bool:
        """Check if any entities were extracted."""
        return self.total_entities > 0
    
    @property
    def has_metadata(self) -> bool:
        """Check if source metadata is present."""
        return any([self.source_url, self.source_title, self.source_author])


class UserPreferences(BaseModel):
    """
    User preferences with all optional settings.
    Demonstrates a settings/config pattern.
    """
    user_id: str
    
    # All preferences are optional with defaults
    theme: str = "light"
    language: str = "en"
    timezone: str = "UTC"
    
    # Email preferences
    email_notifications: bool = True
    email_frequency: str = "daily"
    
    # Privacy preferences
    profile_visibility: str = "public"
    show_email: bool = False
    show_phone: bool = False
    
    # Optional contact info
    notification_email: Optional[str] = None
    backup_email: Optional[str] = None
    
    # Feature flags
    beta_features: bool = False
    analytics_enabled: bool = True
    
    # Metadata
    updated_at: datetime = Field(default_factory=datetime.now)


class PartialPerson(BaseModel):
    """
    Person model where almost everything is optional.
    Useful for incremental data extraction or partial updates.
    """
    # Only ID is required
    person_id: str
    
    # Everything else is optional
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    date_of_birth: Optional[date] = None
    
    # Optional nested
    address: Optional[Address] = None
    company: Optional[str] = None
    
    # Optional collections
    skills: List[str] = Field(default_factory=list)
    interests: List[str] = Field(default_factory=list)
    
    # Metadata
    data_completeness: Optional[float] = Field(None, ge=0.0, le=1.0)
    last_updated: datetime = Field(default_factory=datetime.now)
    
    @property
    def has_basic_info(self) -> bool:
        """Check if basic information is present."""
        return all([self.first_name, self.last_name, self.email])
    
    @property
    def completeness_score(self) -> float:
        """Calculate what percentage of fields are filled."""
        total_fields = 10  # Total optional fields we care about
        filled_fields = sum([
            bool(self.first_name),
            bool(self.last_name),
            bool(self.email),
            bool(self.phone),
            bool(self.date_of_birth),
            bool(self.address),
            bool(self.company),
            bool(self.skills),
            bool(self.interests),
            self.data_completeness is not None
        ])
        return (filled_fields / total_fields) * 100
```

**File: `src/examples/optional_demo.py`**

```python
"""Demonstration of optional fields and default values."""

from datetime import datetime, date
from src.models.optional_models import (
    PersonProfile,
    ContactPreference,
    Address,
    CompanyInfo,
    ArticleMetadata,
    ExtractedEntity,
    DataExtractionResult,
    UserPreferences,
    PartialPerson,
)


def demo_optional_fields():
    """Demonstrate optional fields vs required fields."""
    print("=" * 70)
    print("OPTIONAL FIELDS")
    print("=" * 70)
    
    # Profile with all optional fields filled
    profile_full = PersonProfile(
        first_name="Alice",
        last_name="Johnson",
        email="alice@example.com",
        middle_name="Marie",
        phone="+1-555-0100",
        date_of_birth=date(1992, 5, 15),
        bio="Software engineer passionate about AI and type safety.",
        website="https://alice.dev"
    )
    
    print(f"Full profile:")
    print(f"  Name: {profile_full.full_name}")
    print(f"  Email: {profile_full.email}")
    print(f"  Phone: {profile_full.phone}")
    print(f"  Age: {profile_full.age}")
    print(f"  Bio: {profile_full.bio[:50]}...")
    
    # Profile with minimal information (only required fields)
    profile_minimal = PersonProfile(
        first_name="Bob",
        last_name="Smith",
        email="bob@example.com"
        # All optional fields omitted
    )
    
    print(f"\nMinimal profile:")
    print(f"  Name: {profile_minimal.full_name}")
    print(f"  Email: {profile_minimal.email}")
    print(f"  Phone: {profile_minimal.phone}")  # None
    print(f"  Age: {profile_minimal.age}")  # None
    print(f"  Country: {profile_minimal.country}")  # Default: USA
    
    print()


def demo_default_values():
    """Demonstrate fields with default values."""
    print("=" * 70)
    print("DEFAULT VALUES")
    print("=" * 70)
    
    # Using all defaults
    profile1 = PersonProfile(
        first_name="Charlie",
        last_name="Developer",
        email="charlie@example.com"
    )
    
    print(f"Profile with defaults:")
    print(f"  Country: {profile1.country}")  # Default: USA
    print(f"  Contact Preference: {profile1.contact_preference.value}")  # Default: email
    print(f"  Is Active: {profile1.is_active}")  # Default: True
    print(f"  Timezone: {profile1.timezone}")  # Default: UTC
    
    # Overriding defaults
    profile2 = PersonProfile(
        first_name="Diana",
        last_name="International",
        email="diana@example.com",
        country="UK",
        contact_preference=ContactPreference.PHONE,
        is_active=False,
        timezone="Europe/London"
    )
    
    print(f"\nProfile with custom values:")
    print(f"  Country: {profile2.country}")  # Overridden: UK
    print(f"  Contact Preference: {profile2.contact_preference.value}")  # Overridden: phone
    print(f"  Is Active: {profile2.is_active}")  # Overridden: False
    print(f"  Timezone: {profile2.timezone}")  # Overridden: Europe/London
    
    print()


def demo_optional_nested():
    """Demonstrate optional nested objects."""
    print("=" * 70)
    print("OPTIONAL NESTED OBJECTS")
    print("=" * 70)
    
    # Company without headquarters
    company1 = CompanyInfo(
        name="Startup Inc",
        industry="Technology"
    )
    
    print(f"Company without headquarters:")
    print(f"  Name: {company1.name}")
    print(f"  Headquarters: {company1.headquarters}")  # None
    
    # Company with headquarters
    company2 = CompanyInfo(
        name="TechCorp",
        industry="Software",
        founded_year=2015,
        employee_count=500,
        headquarters=Address(
            street="100 Tech Boulevard",
            city="San Francisco",
            state="CA",
            country="USA",
            zip_code="94105"
        )
    )
    
    print(f"\nCompany with headquarters:")
    print(f"  Name: {company2.name}")
    print(f"  Headquarters: {company2.headquarters}")
    print(f"  Has detailed info: {company2.has_detailed_info}")
    
    print()


def demo_safe_access():
    """Demonstrate safe access to optional fields."""
    print("=" * 70)
    print("SAFE ACCESS TO OPTIONAL FIELDS")
    print("=" * 70)
    
    profile = PersonProfile(
        first_name="Eve",
        last_name="Developer",
        email="eve@example.com"
        # phone is None
    )
    
    # ❌ UNSAFE: Assuming phone exists
    # phone_upper = profile.phone.upper()  # Would crash if None!
    
    # ✅ SAFE: Check before accessing
    if profile.phone:
        phone_upper = profile.phone.upper()
        print(f"Phone (uppercase): {phone_upper}")
    else:
        print(f"Phone not provided")
    
    # ✅ SAFE: Use or operator for fallback
    phone_display = profile.phone or "Not provided"
    print(f"Phone display: {phone_display}")
    
    # ✅ SAFE: Use walrus operator for checking and using
    if (phone := profile.phone):
        print(f"Phone length: {len(phone)}")
    else:
        print("Cannot calculate phone length (no phone)")
    
    print()


def demo_partial_extraction():
    """Demonstrate partial data extraction (real-world use case)."""
    print("=" * 70)
    print("PARTIAL DATA EXTRACTION")
    print("=" * 70)
    
    # Extraction with minimal data found
    extraction1 = DataExtractionResult(
        extraction_id="ext-001",
        source_text="A short text with minimal information."
        # Most fields are optional and not provided
    )
    
    print(f"Minimal extraction:")
    print(f"  ID: {extraction1.extraction_id}")
    print(f"  Has entities: {extraction1.has_entities}")
    print(f"  Has metadata: {extraction1.has_metadata}")
    print(f"  Summary: {extraction1.summary}")  # None
    
    # Extraction with rich data
    extraction2 = DataExtractionResult(
        extraction_id="ext-002",
        source_text="Apple Inc. announced that Tim Cook will visit New York.",
        source_url="https://example.com/article",
        source_title="Apple CEO Announcement",
        people=[
            ExtractedEntity(
                text="Tim Cook",
                entity_type="PERSON",
                confidence=0.98,
                canonical_form="Tim Cook",
                aliases=["Timothy Cook"],
                start_pos=32,
                end_pos=40
            )
        ],
        organizations=[
            ExtractedEntity(
                text="Apple Inc.",
                entity_type="ORGANIZATION",
                confidence=0.99,
                canonical_form="Apple Inc.",
                start_pos=0,
                end_pos=10
            )
        ],
        locations=[
            ExtractedEntity(
                text="New York",
                entity_type="LOCATION",
                confidence=0.95,
                start_pos=52,
                end_pos=60
            )
        ],
        summary="Article about Apple CEO visiting New York",
        extraction_confidence=0.97,
        processing_time_ms=250
    )
    
    print(f"\nRich extraction:")
    print(f"  ID: {extraction2.extraction_id}")
    print(f"  Source: {extraction2.source_title}")
    print(f"  Total entities: {extraction2.total_entities}")
    print(f"  People: {[e.text for e in extraction2.people]}")
    print(f"  Organizations: {[e.text for e in extraction2.organizations]}")
    print(f"  Locations: {[e.text for e in extraction2.locations]}")
    print(f"  Summary: {extraction2.summary}")
    print(f"  Confidence: {extraction2.extraction_confidence:.0%}")
    
    print()


def demo_incremental_data():
    """Demonstrate incremental data building."""
    print("=" * 70)
    print("INCREMENTAL DATA BUILDING")
    print("=" * 70)
    
    # Start with minimal info
    person = PartialPerson(
        person_id="person-001",
        first_name="Frank"
    )
    
    print(f"Initial data (completeness: {person.completeness_score:.0f}%):")
    print(f"  ID: {person.person_id}")
    print(f"  Name: {person.first_name} {person.last_name}")
    print(f"  Has basic info: {person.has_basic_info}")
    
    # Update with more info (simulating incremental extraction)
    person.last_name = "Developer"
    person.email = "frank@example.com"
    person.phone = "+1-555-0100"
    
    print(f"\nAfter update (completeness: {person.completeness_score:.0f}%):")
    print(f"  Name: {person.first_name} {person.last_name}")
    print(f"  Email: {person.email}")
    print(f"  Phone: {person.phone}")
    print(f"  Has basic info: {person.has_basic_info}")
    
    # Add even more info
    person.address = Address(city="San Francisco", state="CA", country="USA")
    person.company = "TechCorp"
    person.skills = ["Python", "TypeScript", "AI"]
    
    print(f"\nFully enriched (completeness: {person.completeness_score:.0f}%):")
    print(f"  Address: {person.address}")
    print(f"  Company: {person.company}")
    print(f"  Skills: {person.skills}")
    
    print()


def demo_settings_pattern():
    """Demonstrate settings/preferences pattern with defaults."""
    print("=" * 70)
    print("SETTINGS PATTERN (All Optional with Defaults)")
    print("=" * 70)
    
    # User with default preferences
    prefs1 = UserPreferences(user_id="user-001")
    
    print(f"Default preferences:")
    print(f"  Theme: {prefs1.theme}")
    print(f"  Language: {prefs1.language}")
    print(f"  Email notifications: {prefs1.email_notifications}")
    print(f"  Profile visibility: {prefs1.profile_visibility}")
    
    # User with custom preferences
    prefs2 = UserPreferences(
        user_id="user-002",
        theme="dark",
        language="es",
        email_notifications=False,
        profile_visibility="private",
        notification_email="custom@example.com",
        beta_features=True
    )
    
    print(f"\nCustom preferences:")
    print(f"  Theme: {prefs2.theme}")
    print(f"  Language: {prefs2.language}")
    print(f"  Email notifications: {prefs2.email_notifications}")
    print(f"  Profile visibility: {prefs2.profile_visibility}")
    print(f"  Notification email: {prefs2.notification_email}")
    print(f"  Beta features: {prefs2.beta_features}")
    
    print()


if __name__ == "__main__":
    print("\n🎯 OPTIONAL FIELDS AND DEFAULTS DEMONSTRATION\n")
    
    demo_optional_fields()
    demo_default_values()
    demo_optional_nested()
    demo_safe_access()
    demo_partial_extraction()
    demo_incremental_data()
    demo_settings_pattern()
    
    print("=" * 70)
    print("✅ All demonstrations completed!")
    print("=" * 70)
```

### Line-by-Line Explanation

**Optional Field Patterns:**

1. **`Optional[T] = None`**: Field can be None or type T, defaults to None if not provided
2. **`Optional[T]`** (no default): Field can be None but must be explicitly provided
3. **`T = default_value`**: Field cannot be None, uses default if not provided
4. **`Optional[T] = default_value`**: Field can be None, uses default if not provided

**Default Value Strategies:**

1. **Simple defaults**: `country: str = "USA"`
2. **Enum defaults**: `contact_preference: ContactPreference = ContactPreference.EMAIL`
3. **Factory defaults**: `created_at: datetime = Field(default_factory=datetime.now)`
4. **Collection defaults**: `tags: List[str] = Field(default_factory=list)`

**Safe Access Patterns:**

1. **If check**: `if person.phone: ...`
2. **Or operator**: `phone_display = profile.phone or "Not provided"`
3. **Walrus operator**: `if (phone := profile.phone): ...`
4. **Computed properties**: Check for None and return appropriate value

### The "Why" Behind the Pattern

**Explicit Nullability:**
`Optional[str]` makes it clear the field can be None. Your IDE will warn you if you try to call `.upper()` without checking for None first. This prevents null reference errors at runtime.

**Incremental Data Extraction:**
When Gemini extracts data from text, it might not find everything. Optional fields let you model partial data: start with what you found, enrich later.

**Backward Compatibility:**
Adding new optional fields doesn't break existing code. Adding new required fields would break all existing data. Optional fields allow schema evolution.

**Configuration Patterns:**
Settings and preferences are almost always optional with defaults. Users customize what they care about, everything else uses sensible defaults.

---

## C. Test & Apply

### How to Test It

**Step 1: Create files**
```bash
cd data_extraction_pipeline
touch src/models/optional_models.py
touch src/examples/optional_demo.py
```

**Step 2: Copy the code**

**Step 3: Run demonstration**
```bash
python -m src.examples.optional_demo
```

### Expected Result

Comprehensive output showing:
- Optional fields can be omitted
- Default values are used when fields not provided
- Safe access patterns for optional fields
- Partial data extraction
- Incremental data building
- Settings pattern with defaults

### Validation Examples

**Create `src/examples/optional_validation_demo.py`:**

```python
"""Demonstrate validation with optional fields."""

from pydantic import ValidationError
from src.models.optional_models import PersonProfile, CompanyInfo


def demo_optional_validation():
    print("🚫 OPTIONAL FIELD VALIDATION\n")
    
    # ✅ Valid: All required fields provided
    print("✅ Valid: Required fields only")
    profile = PersonProfile(
        first_name="Alice",
        last_name="Smith",
        email="alice@example.com"
    )
    print(f"Created: {profile.full_name}\n")
    
    # ❌ Invalid: Missing required field
    print("❌ Invalid: Missing required field")
    try:
        profile = PersonProfile(
            first_name="Bob",
            # last_name missing!
            email="bob@example.com"
        )
    except ValidationError as e:
        print(f"Error: {e}\n")
    
    # ✅ Valid: Optional field can be None
    print("✅ Valid: Optional field explicitly set to None")
    profile = PersonProfile(
        first_name="Charlie",
        last_name="Dev",
        email="charlie@example.com",
        phone=None  # Explicitly None is valid
    )
    print(f"Phone: {profile.phone}\n")
    
    # ❌ Invalid: Wrong type for optional field
    print("❌ Invalid: Wrong type for optional field")
    try:
        profile = PersonProfile(
            first_name="Diana",
            last_name="Test",
            email="diana@example.com",
            phone=12345  # Should be string, not int
        )
    except ValidationError as e:
        print(f"Error: {e}\n")
    
    print("✅ Validation demonstration complete!")


if __name__ == "__main__":
    demo_optional_validation()
```

**Run it:**
```bash
python -m src.examples.optional_validation_demo
```

### Type Checking

**Type-safe handling of optional fields:**

```python
from src.models.optional_models import PersonProfile

profile = PersonProfile(
    first_name="Alice",
    last_name="Smith",
    email="alice@example.com"
)

# IDE knows phone is Optional[str] (can be None)
phone: Optional[str] = profile.phone  # ✅ Type safe

# IDE warns about calling methods without checking for None
# phone_upper = profile.phone.upper()  # ⚠️ IDE warning: phone might be None

# Type-safe access
if profile.phone:
    phone_upper: str = profile.phone.upper()  # ✅ IDE knows phone is str here

# Or operator maintains type
phone_display: str = profile.phone or "N/A"  # ✅ Always str
```

---

## D. Common Stumbling Blocks

### Proactive Debugging

**Mistake 1: Forgetting Optional wrapper**

```python
# ❌ WRONG - Field can't be None but has None default
class Profile(BaseModel):
    phone: str = None  # Type error! str can't be None

# ✅ CORRECT - Use Optional
class Profile(BaseModel):
    phone: Optional[str] = None  # Can be None
```

**Mistake 2: Calling methods on None values**

```python
profile = PersonProfile(...)

# ❌ WRONG - Crashes if phone is None
length = len(profile.phone)  # AttributeError if None!

# ✅ CORRECT - Check first
if profile.phone:
    length = len(profile.phone)
```

**Mistake 3: Confusing None with empty string**

```python
# These are DIFFERENT:
phone1: Optional[str] = None  # No value
phone2: str = ""  # Empty string (has a value)

# Optional allows None, not empty string (unless validated)
```

### Show the Error

**Error 1: Missing required field**

```python
profile = PersonProfile(
    first_name="Alice",
    # last_name missing!
    email="alice@example.com"
)
```

**Error message:**
```
ValidationError: 1 validation error for PersonProfile
last_name
  Field required [type=missing, input_value={'first_name': 'Alice', 'email': 'alice@example.com'}, input_type=dict]
```

**Error 2: Wrong type for optional field**

```python
profile = PersonProfile(
    first_name="Bob",
    last_name="Smith",
    email="bob@example.com",
    phone=12345  # Should be string
)
```

**Error message:**
```
ValidationError: 1 validation error for PersonProfile
phone
  Input should be a valid string [type=string_type, input_value=12345, input_type=int]
```

### Explain the Fix

**For Missing Required Fields:**
- Check the model definition—fields without Optional and without defaults are required
- Ensure you provide all required fields when creating the model
- Consider making the field optional if it's not always available

**For None Value Errors:**
- Check if the field is Optional before calling methods on it
- Use if checks, or operators, or the walrus operator
- Consider using computed properties that handle None gracefully

**For Type Errors on Optional Fields:**
- Optional changes what values are allowed (adds None), not the base type
- `Optional[str]` can be None or str, but not int
- Ensure you're passing the correct type or None

### Type Safety Gotchas

1. **Optional vs Default**: `Optional[str]` means can be None. `str = "default"` means uses default. They're independent.

2. **None ≠ Empty**: `None` is "no value", `""` is "empty string value". They're different.

3. **Checking for None**: Use `if field:` which is False for None and empty strings. Use `if field is not None:` to specifically check for None.

4. **Optional Collections**: `List[str] = Field(default_factory=list)` is an empty list, not None. `Optional[List[str]] = None` allows None or a list.

5. **IDE Warnings**: Your IDE will warn about calling methods on Optional fields—listen to these warnings!

---

## 🎯 Next Steps

Excellent progress! You now understand:
- ✅ How to use Optional fields for nullable values
- ✅ How to set default values for fields
- ✅ How to safely access optional fields
- ✅ How to model partial/incomplete data
- ✅ How to use optional fields for incremental extraction
- ✅ How to create settings patterns with defaults

In the next lesson, we'll explore **Union Types and Discriminators**—learning how to handle fields that can be multiple different types and how to validate them correctly.

**Ready for Lesson 5, or would you like to practice with optional fields?** 🚀
