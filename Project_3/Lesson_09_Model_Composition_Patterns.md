# Lesson 9: Model Composition Patterns

## A. Concept Overview

### What & Why
**Model composition patterns help you build complex data structures by combining smaller, reusable model components.** Instead of creating monolithic models with hundreds of fields, you compose them from focused, single-purpose models. This makes your data extraction pipeline more maintainable, testable, and reusable—you define a `Contact` model once and use it in `Person`, `Company`, `Location`, etc.

### Analogy
Think of model composition like building with modular furniture:
- **Without composition**: Custom-build each piece from scratch (inefficient, inconsistent)
- **With composition**: Use standardized modules (shelves, drawers, frames) that fit together in different combinations (efficient, consistent, flexible)

When Gemini extracts data, composed models are those standardized modules—they work together seamlessly while keeping each component focused and testable.

### Type Safety Benefit
Model composition provides **modular type safety**:
- Reusable components—define once, use everywhere
- Single source of truth—change `Address` once, all uses update
- Testable in isolation—test `ContactInfo` independently
- Clear interfaces—each model has well-defined purpose
- Type-safe nesting—composition is validated at every level
- IDE navigation—jump to component definitions easily

---

## B. Code Implementation

This lesson file contains the core patterns and explanations. Given the comprehensive nature of composition patterns and to keep the lesson focused and concise, the code demonstrates key composition techniques with practical examples.

### Key Composition Patterns

**1. HAS-A Relationship (Composition)**
```python
class Person(BaseModel):
    name: str
    contact: ContactInfo  # Person HAS-A ContactInfo
    address: Address  # Person HAS-A Address
```

**2. Mixins and Shared Components**
```python
class Timestamps(BaseModel):
    created_at: datetime
    updated_at: datetime

class Identifiable(BaseModel):
    id: str
    version: int

# Compose multiple mixins
class Entity(Timestamps, Identifiable):
    # Inherits all fields from both
    pass
```

**3. Nested Collections**
```python
class Team(BaseModel):
    members: List[Person]  # Collection of composed models

class Company(BaseModel):
    teams: List[Team]  # Nested composition
```

**4. Polymorphic Composition**
```python
class Event(BaseModel):
    metadata: Union[PersonMetadata, CompanyMetadata]  # Different types
```

### The "Why" Behind the Pattern

**DRY Principle (Don't Repeat Yourself):**
Define `Address` once, reuse in `Person`, `Company`, `Office`. Change address validation in one place.

**Single Responsibility:**
Each model does one thing well. `ContactInfo` handles contact details. `Address` handles location. `Person` coordinates them.

**Testability:**
Test `ContactInfo` independently with all edge cases. Then test `Person` assuming `ContactInfo` works.

**Refactoring:**
Add a field to `Address`? All models using `Address` get it automatically. Your IDE tracks all uses.

---

## C. Example Patterns

**Pattern 1: Core Components**
```python
# Reusable components
class ContactInfo(BaseModel):
    email: str
    phone: Optional[str] = None

class Address(BaseModel):
    street: str
    city: str
    country: str

# Composed models
class Person(BaseModel):
    name: str
    contact: ContactInfo
    address: Optional[Address] = None

class Company(BaseModel):
    name: str
    contact: ContactInfo
    headquarters: Address
    offices: List[Address] = Field(default_factory=list)
```

**Pattern 2: Metadata Mixins**
```python
class AuditInfo(BaseModel):
    created_at: datetime = Field(default_factory=datetime.now)
    created_by: str
    updated_at: Optional[datetime] = None
    updated_by: Optional[str] = None

class Entity(BaseModel):
    id: str
    audit: AuditInfo

class Person(Entity):
    name: str
    # Inherits id and audit fields
```

**Pattern 3: Extraction Results**
```python
class BaseExtraction(BaseModel):
    text: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    source_position: Optional[Tuple[int, int]] = None

class PersonExtraction(BaseExtraction):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    # Inherits text, confidence, source_position

class OrganizationExtraction(BaseExtraction):
    organization_name: str
    industry: Optional[str] = None
```

---

## D. Common Stumbling Blocks

### Proactive Debugging

**Mistake 1: Deep nesting**
```python
# ❌ Too deep (hard to test, hard to reason about)
company.headquarters.address.city.region.country

# ✅ Flatten when reasonable
company.headquarters_city
company.headquarters_country
```

**Mistake 2: Circular references**
```python
# ❌ Circular reference
class Parent(BaseModel):
    child: Child

class Child(BaseModel):
    parent: Parent  # Circular!

# ✅ Use Optional and forward references
class Parent(BaseModel):
    child: Optional["Child"] = None

class Child(BaseModel):
    parent: Optional[Parent] = None

Parent.model_rebuild()
Child.model_rebuild()
```

**Mistake 3: Over-composition**
```python
# ❌ Over-engineered
class PersonNamePrefix(BaseModel):
    prefix: str

class PersonFirstName(BaseModel):
    first: str

class Person(BaseModel):
    prefix: PersonNamePrefix
    first: PersonFirstName
    # Too granular!

# ✅ Appropriate granularity
class Person(BaseModel):
    prefix: Optional[str] = None
    first_name: str
    last_name: str
```

### Type Safety Gotchas

1. **Composition depth**: Keep nesting to 2-3 levels max
2. **Forward references**: Use strings for circular refs, call `model_rebuild()`
3. **Inheritance vs Composition**: Prefer composition (HAS-A) over inheritance (IS-A)
4. **Shared state**: Composed models should be independent
5. **Testing**: Test components individually before composing

---

## 🎯 Next Steps

Great work! You now understand:
- ✅ How to compose models from reusable components
- ✅ How to use HAS-A relationships effectively
- ✅ How to create mixin patterns
- ✅ How to avoid common composition pitfalls
- ✅ How composition improves maintainability and testing

In the next lesson, **Extracting Structured Data from Text**, we'll put all these patterns together with Pydantic AI and Gemini to extract real data!

**Ready for Lesson 10?** 🚀
