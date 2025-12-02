# Lesson 5: Union Types and Discriminators

## A. Concept Overview

### What & Why
**Union types allow a field to accept multiple different types, while discriminators help Pydantic choose the correct type automatically.** In real-world data extraction, entities can have different shapes—a contact might be a person or a company, an address might be US or international format, a value might be a number or a string. Union types with discriminators make this type-safe and validated.

### Analogy
Think of union types like a parking lot with different vehicle sections:
- **Without discriminator**: You show up with a vehicle, and the parking attendant has to try each section (car? motorcycle? truck?) until something fits
- **With discriminator**: Your vehicle has a clear label (type: "car"), and the attendant immediately knows which section and which rules apply

When Gemini extracts data that could be different types, discriminators are like that label—they tell Pydantic exactly which model to use without guessing.

### Type Safety Benefit
Union types with discriminators provide **precise type narrowing**:
- `value: Union[int, str]` — can be an int or a string, validated as either
- Discriminators eliminate ambiguity — no guessing which type to validate against
- Type narrowing works — after checking the discriminator, your IDE knows the exact type
- Validation is precise — each type validates with its own rules
- No silent failures — if data doesn't match any type, you get a clear error

---

## B. Code Implementation

### File Structure
```
data_extraction_pipeline/
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── union_models.py     # New: Union type patterns
│   └── examples/
│       ├── __init__.py
│       └── union_demo.py        # New: This lesson
```

### Complete Code Implementation

**File: `src/models/union_models.py`**

```python
"""Models demonstrating union types and discriminated unions."""

from pydantic import BaseModel, Field, field_validator
from typing import Union, Literal, List, Optional
from datetime import datetime
from enum import Enum


# ============================================================================
# BASIC UNION TYPES (Without Discriminators)
# ============================================================================

class SimpleValue(BaseModel):
    """Demonstrates simple union types."""
    # Can be int or string
    value: Union[int, str]
    
    # Can be float or None
    confidence: Union[float, None] = None


class FlexibleData(BaseModel):
    """Demonstrates multiple union patterns."""
    # Can be string or int
    identifier: Union[str, int]
    
    # Can be single item or list
    tags: Union[str, List[str]]
    
    # Can be bool or string representation
    is_active: Union[bool, str]
    
    @field_validator('is_active')
    @classmethod
    def validate_is_active(cls, v: Union[bool, str]) -> bool:
        """Convert string representations to bool."""
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            if v.lower() in ('true', 'yes', '1'):
                return True
            if v.lower() in ('false', 'no', '0'):
                return False
            raise ValueError(f"Cannot convert '{v}' to bool")
        raise ValueError(f"Expected bool or str, got {type(v)}")


# ============================================================================
# DISCRIMINATED UNIONS (Tagged Unions)
# ============================================================================

class PersonContact(BaseModel):
    """Contact information for a person."""
    contact_type: Literal["person"] = "person"  # Discriminator field
    first_name: str
    last_name: str
    email: str
    phone: Optional[str] = None
    
    @property
    def display_name(self) -> str:
        return f"{self.first_name} {self.last_name}"


class CompanyContact(BaseModel):
    """Contact information for a company."""
    contact_type: Literal["company"] = "company"  # Discriminator field
    company_name: str
    industry: Optional[str] = None
    contact_email: str
    contact_phone: Optional[str] = None
    
    @property
    def display_name(self) -> str:
        return self.company_name


class GovernmentContact(BaseModel):
    """Contact information for a government entity."""
    contact_type: Literal["government"] = "government"  # Discriminator field
    agency_name: str
    department: Optional[str] = None
    official_email: str
    public_phone: str
    
    @property
    def display_name(self) -> str:
        if self.department:
            return f"{self.agency_name} - {self.department}"
        return self.agency_name


# Discriminated union using the contact_type field
Contact = Union[PersonContact, CompanyContact, GovernmentContact]


class ContactBook(BaseModel):
    """Collection of contacts with different types."""
    contacts: List[Contact] = Field(default_factory=list)
    
    @property
    def person_count(self) -> int:
        return sum(1 for c in self.contacts if isinstance(c, PersonContact))
    
    @property
    def company_count(self) -> int:
        return sum(1 for c in self.contacts if isinstance(c, CompanyContact))
    
    @property
    def government_count(self) -> int:
        return sum(1 for c in self.contacts if isinstance(c, GovernmentContact))


# ============================================================================
# ADDRESS VARIANTS (Different formats)
# ============================================================================

class USAddress(BaseModel):
    """US address format."""
    address_type: Literal["us"] = "us"
    street: str
    city: str
    state: str = Field(..., min_length=2, max_length=2)
    zip_code: str = Field(..., pattern=r"^\d{5}(-\d{4})?$")
    country: Literal["USA"] = "USA"


class InternationalAddress(BaseModel):
    """International address format."""
    address_type: Literal["international"] = "international"
    street: str
    city: str
    region: Optional[str] = None
    postal_code: str
    country: str = Field(..., min_length=2)


class POBoxAddress(BaseModel):
    """PO Box address format."""
    address_type: Literal["po_box"] = "po_box"
    po_box_number: str
    city: str
    state: str
    zip_code: str
    country: str = "USA"


Address = Union[USAddress, InternationalAddress, POBoxAddress]


class Location(BaseModel):
    """Location with various address formats."""
    name: str
    address: Address  # Can be any address type
    is_primary: bool = False


# ============================================================================
# MEASUREMENT TYPES (Different units)
# ============================================================================

class MetricMeasurement(BaseModel):
    """Metric system measurement."""
    unit_system: Literal["metric"] = "metric"
    value: float
    unit: Literal["meters", "kilometers", "grams", "kilograms"]


class ImperialMeasurement(BaseModel):
    """Imperial system measurement."""
    unit_system: Literal["imperial"] = "imperial"
    value: float
    unit: Literal["feet", "miles", "ounces", "pounds"]


Measurement = Union[MetricMeasurement, ImperialMeasurement]


class Product(BaseModel):
    """Product with measurements in different systems."""
    name: str
    weight: Measurement
    dimensions: List[Measurement] = Field(default_factory=list)


# ============================================================================
# EVENT TYPES (Different event structures)
# ============================================================================

class MessageEvent(BaseModel):
    """Message sent event."""
    event_type: Literal["message"] = "message"
    timestamp: datetime = Field(default_factory=datetime.now)
    sender: str
    recipient: str
    content: str
    message_id: str


class FileUploadEvent(BaseModel):
    """File uploaded event."""
    event_type: Literal["file_upload"] = "file_upload"
    timestamp: datetime = Field(default_factory=datetime.now)
    uploader: str
    filename: str
    file_size: int
    mime_type: str


class UserLoginEvent(BaseModel):
    """User login event."""
    event_type: Literal["user_login"] = "user_login"
    timestamp: datetime = Field(default_factory=datetime.now)
    user_id: str
    ip_address: str
    user_agent: str
    success: bool


class PaymentEvent(BaseModel):
    """Payment processed event."""
    event_type: Literal["payment"] = "payment"
    timestamp: datetime = Field(default_factory=datetime.now)
    transaction_id: str
    amount: float
    currency: str
    status: Literal["pending", "completed", "failed"]


Event = Union[MessageEvent, FileUploadEvent, UserLoginEvent, PaymentEvent]


class EventLog(BaseModel):
    """Log of various event types."""
    log_id: str
    events: List[Event] = Field(default_factory=list)
    
    @property
    def event_count_by_type(self) -> dict:
        """Count events by type."""
        counts = {}
        for event in self.events:
            event_type = event.event_type
            counts[event_type] = counts.get(event_type, 0) + 1
        return counts


# ============================================================================
# EXTRACTION RESULT VARIANTS
# ============================================================================

class TextExtraction(BaseModel):
    """Text extracted from source."""
    result_type: Literal["text"] = "text"
    content: str
    language: Optional[str] = None
    word_count: int


class EntityExtraction(BaseModel):
    """Entities extracted from source."""
    result_type: Literal["entity"] = "entity"
    entities: List[str]
    entity_type: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class RelationshipExtraction(BaseModel):
    """Relationships extracted from source."""
    result_type: Literal["relationship"] = "relationship"
    subject: str
    predicate: str
    object: str
    confidence: float = Field(..., ge=0.0, le=1.0)


class SummaryExtraction(BaseModel):
    """Summary generated from source."""
    result_type: Literal["summary"] = "summary"
    summary: str
    original_length: int
    summary_length: int
    
    @property
    def compression_ratio(self) -> float:
        return self.summary_length / self.original_length if self.original_length > 0 else 0


ExtractionResult = Union[TextExtraction, EntityExtraction, RelationshipExtraction, SummaryExtraction]


class BatchExtractionResult(BaseModel):
    """Batch extraction with multiple result types."""
    extraction_id: str
    source_text: str
    results: List[ExtractionResult] = Field(default_factory=list)
    extracted_at: datetime = Field(default_factory=datetime.now)
    
    @property
    def result_count_by_type(self) -> dict:
        """Count results by type."""
        counts = {}
        for result in self.results:
            result_type = result.result_type
            counts[result_type] = counts.get(result_type, 0) + 1
        return counts
    
    def get_text_results(self) -> List[TextExtraction]:
        """Filter for text extraction results."""
        return [r for r in self.results if isinstance(r, TextExtraction)]
    
    def get_entity_results(self) -> List[EntityExtraction]:
        """Filter for entity extraction results."""
        return [r for r in self.results if isinstance(r, EntityExtraction)]
    
    def get_relationship_results(self) -> List[RelationshipExtraction]:
        """Filter for relationship extraction results."""
        return [r for r in self.results if isinstance(r, RelationshipExtraction)]
    
    def get_summary_results(self) -> List[SummaryExtraction]:
        """Filter for summary extraction results."""
        return [r for r in self.results if isinstance(r, SummaryExtraction)]


# ============================================================================
# RESPONSE TYPES (Success or Error)
# ============================================================================

class SuccessResponse(BaseModel):
    """Successful operation response."""
    status: Literal["success"] = "success"
    data: dict
    message: Optional[str] = None


class ErrorResponse(BaseModel):
    """Error operation response."""
    status: Literal["error"] = "error"
    error_code: str
    error_message: str
    details: Optional[dict] = None


Response = Union[SuccessResponse, ErrorResponse]


class APIResult(BaseModel):
    """API operation result."""
    request_id: str
    response: Response
    timestamp: datetime = Field(default_factory=datetime.now)
    
    @property
    def is_success(self) -> bool:
        return isinstance(self.response, SuccessResponse)
    
    @property
    def is_error(self) -> bool:
        return isinstance(self.response, ErrorResponse)
```

**File: `src/examples/union_demo.py`**

```python
"""Demonstration of union types and discriminated unions."""

from datetime import datetime
from src.models.union_models import (
    SimpleValue,
    FlexibleData,
    PersonContact,
    CompanyContact,
    GovernmentContact,
    ContactBook,
    USAddress,
    InternationalAddress,
    POBoxAddress,
    Location,
    MetricMeasurement,
    ImperialMeasurement,
    Product,
    MessageEvent,
    FileUploadEvent,
    UserLoginEvent,
    PaymentEvent,
    EventLog,
    TextExtraction,
    EntityExtraction,
    RelationshipExtraction,
    SummaryExtraction,
    BatchExtractionResult,
    SuccessResponse,
    ErrorResponse,
    APIResult,
)


def demo_basic_unions():
    """Demonstrate basic union types."""
    print("=" * 70)
    print("BASIC UNION TYPES")
    print("=" * 70)
    
    # Value can be int or string
    value1 = SimpleValue(value=42, confidence=0.95)
    value2 = SimpleValue(value="hello", confidence=0.87)
    value3 = SimpleValue(value=100)  # confidence defaults to None
    
    print(f"Value 1: {value1.value} (type: {type(value1.value).__name__})")
    print(f"Value 2: {value2.value} (type: {type(value2.value).__name__})")
    print(f"Value 3: {value3.value}, confidence: {value3.confidence}")
    
    # Flexible data with multiple unions
    data1 = FlexibleData(
        identifier=12345,
        tags="python",
        is_active=True
    )
    
    data2 = FlexibleData(
        identifier="user-001",
        tags=["python", "ai", "typing"],
        is_active="yes"  # String converted to bool
    )
    
    print(f"\nData 1 - ID: {data1.identifier}, Tags: {data1.tags}")
    print(f"Data 2 - ID: {data2.identifier}, Tags: {data2.tags}")
    print()


def demo_discriminated_contacts():
    """Demonstrate discriminated union for different contact types."""
    print("=" * 70)
    print("DISCRIMINATED UNIONS: Contact Types")
    print("=" * 70)
    
    # Create different contact types
    person = PersonContact(
        first_name="Alice",
        last_name="Johnson",
        email="alice@example.com",
        phone="+1-555-0100"
    )
    
    company = CompanyContact(
        company_name="TechCorp Industries",
        industry="Software",
        contact_email="info@techcorp.com",
        contact_phone="+1-800-TECH"
    )
    
    government = GovernmentContact(
        agency_name="Department of Technology",
        department="Innovation Division",
        official_email="innovation@dot.gov",
        public_phone="+1-800-DOT-TECH"
    )
    
    # All contacts in one collection
    contact_book = ContactBook(
        contacts=[person, company, government]
    )
    
    print(f"Total contacts: {len(contact_book.contacts)}")
    print(f"  People: {contact_book.person_count}")
    print(f"  Companies: {contact_book.company_count}")
    print(f"  Government: {contact_book.government_count}")
    
    print(f"\nContact details:")
    for contact in contact_book.contacts:
        print(f"  [{contact.contact_type}] {contact.display_name}")
        
        # Type narrowing with isinstance
        if isinstance(contact, PersonContact):
            print(f"    Email: {contact.email}")
        elif isinstance(contact, CompanyContact):
            print(f"    Industry: {contact.industry}")
        elif isinstance(contact, GovernmentContact):
            print(f"    Agency: {contact.agency_name}")
    
    print()


def demo_address_variants():
    """Demonstrate address format variants."""
    print("=" * 70)
    print("ADDRESS FORMAT VARIANTS")
    print("=" * 70)
    
    # Different address formats
    location1 = Location(
        name="San Francisco Office",
        address=USAddress(
            street="100 Tech Boulevard",
            city="San Francisco",
            state="CA",
            zip_code="94105"
        ),
        is_primary=True
    )
    
    location2 = Location(
        name="London Office",
        address=InternationalAddress(
            street="10 Downing Street",
            city="London",
            postal_code="SW1A 2AA",
            country="United Kingdom"
        )
    )
    
    location3 = Location(
        name="Mail Center",
        address=POBoxAddress(
            po_box_number="12345",
            city="New York",
            state="NY",
            zip_code="10001"
        )
    )
    
    for loc in [location1, location2, location3]:
        print(f"Location: {loc.name}")
        print(f"  Address type: {loc.address.address_type}")
        
        if isinstance(loc.address, USAddress):
            print(f"  {loc.address.street}, {loc.address.city}, {loc.address.state} {loc.address.zip_code}")
        elif isinstance(loc.address, InternationalAddress):
            print(f"  {loc.address.street}, {loc.address.city}, {loc.address.country}")
        elif isinstance(loc.address, POBoxAddress):
            print(f"  PO Box {loc.address.po_box_number}, {loc.address.city}, {loc.address.state}")
        
        print()


def demo_measurement_units():
    """Demonstrate measurements in different unit systems."""
    print("=" * 70)
    print("MEASUREMENT UNIT VARIANTS")
    print("=" * 70)
    
    product1 = Product(
        name="Laptop",
        weight=MetricMeasurement(value=1.5, unit="kilograms"),
        dimensions=[
            MetricMeasurement(value=35.0, unit="centimeters"),
            MetricMeasurement(value=24.0, unit="centimeters"),
            MetricMeasurement(value=2.0, unit="centimeters"),
        ]
    )
    
    product2 = Product(
        name="Monitor",
        weight=ImperialMeasurement(value=15.5, unit="pounds"),
        dimensions=[
            ImperialMeasurement(value=24.0, unit="inches"),
            ImperialMeasurement(value=16.0, unit="inches"),
            ImperialMeasurement(value=2.0, unit="inches"),
        ]
    )
    
    print(f"Product: {product1.name}")
    print(f"  Weight: {product1.weight.value} {product1.weight.unit} ({product1.weight.unit_system})")
    
    print(f"\nProduct: {product2.name}")
    print(f"  Weight: {product2.weight.value} {product2.weight.unit} ({product2.weight.unit_system})")
    
    print()


def demo_event_types():
    """Demonstrate different event types in a log."""
    print("=" * 70)
    print("EVENT TYPE VARIANTS")
    print("=" * 70)
    
    base_time = datetime.now()
    
    event_log = EventLog(
        log_id="log-001",
        events=[
            UserLoginEvent(
                timestamp=base_time,
                user_id="user-123",
                ip_address="192.168.1.1",
                user_agent="Mozilla/5.0",
                success=True
            ),
            MessageEvent(
                timestamp=base_time,
                sender="user-123",
                recipient="user-456",
                content="Hello!",
                message_id="msg-001"
            ),
            FileUploadEvent(
                timestamp=base_time,
                uploader="user-123",
                filename="report.pdf",
                file_size=1024000,
                mime_type="application/pdf"
            ),
            PaymentEvent(
                timestamp=base_time,
                transaction_id="txn-001",
                amount=99.99,
                currency="USD",
                status="completed"
            ),
        ]
    )
    
    print(f"Event Log: {event_log.log_id}")
    print(f"Total events: {len(event_log.events)}")
    print(f"Events by type: {event_log.event_count_by_type}")
    
    print(f"\nEvent timeline:")
    for event in event_log.events:
        print(f"  [{event.event_type}] ", end="")
        
        if isinstance(event, MessageEvent):
            print(f"{event.sender} → {event.recipient}: {event.content}")
        elif isinstance(event, FileUploadEvent):
            print(f"{event.uploader} uploaded {event.filename} ({event.file_size} bytes)")
        elif isinstance(event, UserLoginEvent):
            status = "✅" if event.success else "❌"
            print(f"{status} User {event.user_id} from {event.ip_address}")
        elif isinstance(event, PaymentEvent):
            print(f"${event.amount} {event.currency} - {event.status}")
    
    print()


def demo_extraction_results():
    """Demonstrate extraction result variants (real-world use case)."""
    print("=" * 70)
    print("EXTRACTION RESULT VARIANTS (Real-World)")
    print("=" * 70)
    
    source = (
        "Apple Inc. announced that Tim Cook will visit New York. "
        "The company plans to invest $2.5 billion in renewable energy."
    )
    
    extraction = BatchExtractionResult(
        extraction_id="ext-001",
        source_text=source,
        results=[
            TextExtraction(
                content=source,
                language="en",
                word_count=len(source.split())
            ),
            EntityExtraction(
                entities=["Apple Inc.", "Tim Cook"],
                entity_type="PERSON",
                confidence=0.98
            ),
            EntityExtraction(
                entities=["New York"],
                entity_type="LOCATION",
                confidence=0.95
            ),
            RelationshipExtraction(
                subject="Tim Cook",
                predicate="CEO_OF",
                object="Apple Inc.",
                confidence=0.92
            ),
            RelationshipExtraction(
                subject="Tim Cook",
                predicate="WILL_VISIT",
                object="New York",
                confidence=0.88
            ),
            SummaryExtraction(
                summary="Apple CEO to visit NY for renewable energy investment.",
                original_length=len(source),
                summary_length=58
            ),
        ]
    )
    
    print(f"Extraction ID: {extraction.extraction_id}")
    print(f"Source: {extraction.source_text[:60]}...")
    print(f"Total results: {len(extraction.results)}")
    print(f"Results by type: {extraction.result_count_by_type}")
    
    print(f"\nExtracted entities:")
    for entity_result in extraction.get_entity_results():
        print(f"  [{entity_result.entity_type}] {entity_result.entities} (confidence: {entity_result.confidence:.0%})")
    
    print(f"\nExtracted relationships:")
    for rel in extraction.get_relationship_results():
        print(f"  {rel.subject} --[{rel.predicate}]--> {rel.object} (confidence: {rel.confidence:.0%})")
    
    print(f"\nSummary:")
    for summary in extraction.get_summary_results():
        print(f"  {summary.summary}")
        print(f"  Compression: {summary.compression_ratio:.1%}")
    
    print()


def demo_response_variants():
    """Demonstrate success/error response variants."""
    print("=" * 70)
    print("API RESPONSE VARIANTS")
    print("=" * 70)
    
    # Success response
    success_result = APIResult(
        request_id="req-001",
        response=SuccessResponse(
            data={"user_id": "123", "status": "active"},
            message="User retrieved successfully"
        )
    )
    
    # Error response
    error_result = APIResult(
        request_id="req-002",
        response=ErrorResponse(
            error_code="NOT_FOUND",
            error_message="User not found",
            details={"user_id": "999"}
        )
    )
    
    for result in [success_result, error_result]:
        print(f"Request ID: {result.request_id}")
        print(f"Status: {result.response.status}")
        print(f"Is success: {result.is_success}")
        
        if isinstance(result.response, SuccessResponse):
            print(f"Data: {result.response.data}")
            print(f"Message: {result.response.message}")
        elif isinstance(result.response, ErrorResponse):
            print(f"Error: [{result.response.error_code}] {result.response.error_message}")
            print(f"Details: {result.response.details}")
        
        print()


if __name__ == "__main__":
    print("\n🎯 UNION TYPES AND DISCRIMINATORS DEMONSTRATION\n")
    
    demo_basic_unions()
    demo_discriminated_contacts()
    demo_address_variants()
    demo_measurement_units()
    demo_event_types()
    demo_extraction_results()
    demo_response_variants()
    
    print("=" * 70)
    print("✅ All demonstrations completed!")
    print("=" * 70)
```

### Line-by-Line Explanation

**Union Type Patterns:**

1. **Basic Union**: `Union[int, str]` — field accepts int or str
2. **Discriminated Union**: Uses `Literal` field to identify which type
3. **Type Narrowing**: Use `isinstance()` to narrow to specific type

**Discriminator Pattern:**

```python
class TypeA(BaseModel):
    type_field: Literal["a"] = "a"  # Discriminator
    # ... fields specific to TypeA

class TypeB(BaseModel):
    type_field: Literal["b"] = "b"  # Discriminator
    # ... fields specific to TypeB

UnionType = Union[TypeA, TypeB]
```

When Pydantic sees `type_field: "a"`, it knows to use TypeA and validates against TypeA's schema.

**Key Concepts:**

1. **Literal types**: `Literal["person"]` means the field can only be that exact value
2. **Discriminator field**: A field with Literal type that identifies which variant to use
3. **Type narrowing**: After `isinstance()` check, Python/your IDE knows the exact type
4. **Validation order**: Pydantic tries each type in the union until one validates

### The "Why" Behind the Pattern

**Type Safety with Multiple Shapes:**
Without discriminators, Pydantic must try each type until one fits. With discriminators, it knows immediately which type to validate, making errors clearer and validation faster.

**Real-World Data Variation:**
Extracted data has natural variations:
- Contacts can be people, companies, or government entities
- Addresses have different formats by country
- Events have different structures based on event type
- API responses can be success or error

Discriminated unions model this variation with full type safety.

**Clear Error Messages:**
With discriminators, validation errors are specific: "Expected contact_type to be one of ['person', 'company', 'government'], got 'invalid'". Without discriminators, you get vague "no variants matched" errors.

---

## C. Test & Apply

### How to Test It

**Step 1: Create files**
```bash
cd data_extraction_pipeline
touch src/models/union_models.py
touch src/examples/union_demo.py
```

**Step 2: Copy code**

**Step 3: Run demonstration**
```bash
python -m src.examples.union_demo
```

### Expected Result

Comprehensive output showing:
- Basic unions accepting multiple types
- Discriminated unions with type identification
- Address format variants
- Measurement unit variants
- Event type variants
- Extraction result variants
- Success/error response variants

### Validation Examples

Type narrowing and validation work automatically with discriminators!

---

## D. Common Stumbling Blocks

### Proactive Debugging

**Mistake 1: Forgetting discriminator field**

```python
# ❌ Without discriminator - ambiguous
class PersonContact(BaseModel):
    name: str

class CompanyContact(BaseModel):
    name: str

# How does Pydantic know which one to use?

# ✅ With discriminator - explicit
class PersonContact(BaseModel):
    contact_type: Literal["person"] = "person"
    name: str

class CompanyContact(BaseModel):
    contact_type: Literal["company"] = "company"
    name: str
```

**Mistake 2: Type narrowing without isinstance**

```python
contact: Contact = ...  # Could be any contact type

# ❌ IDE doesn't know which type
# email = contact.email  # Error: not all types have email

# ✅ Narrow the type first
if isinstance(contact, PersonContact):
    email = contact.email  # OK: IDE knows it's PersonContact
```

### Show the Error

**Error: Invalid discriminator value**

```python
contact = PersonContact(
    contact_type="invalid",  # ❌ Must be "person"
    first_name="Alice",
    last_name="Smith",
    email="alice@example.com"
)
```

**Error message:**
```
ValidationError: 1 validation error for PersonContact
contact_type
  Input should be 'person' [type=literal_error, input_value='invalid', input_type=str]
```

### Type Safety Gotchas

1. **Discriminator must be Literal**: Use `Literal["type"]` not just `str`
2. **Union order matters**: Pydantic tries types in order (but discriminators override this)
3. **Type narrowing**: Use `isinstance()` to narrow Union types
4. **Unique discriminators**: Each type in union needs unique discriminator value

---

## 🎯 Next Steps

Amazing work! You now understand:
- ✅ How to use union types for fields that accept multiple types
- ✅ How to use discriminators for clear type identification
- ✅ How to model data with natural variations
- ✅ How to narrow types with isinstance checks
- ✅ How to create type-safe extraction results with multiple shapes

In the next lesson, we'll explore **Enums for Controlled Values**—learning how to restrict fields to a specific set of valid options.

**Ready for Lesson 6?** 🚀
