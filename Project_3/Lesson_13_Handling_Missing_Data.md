# Lesson 13: Handling Missing Data

## A. Concept Overview

### What & Why
**Real-world text extraction is messy—information is often incomplete, ambiguous, or missing entirely.** Your models must gracefully handle partial data, allowing extraction to succeed even when some fields can't be filled. This lesson teaches you to design robust extraction models that work with imperfect inputs.

### Analogy
Think of handling missing data like filling out a form at a doctor's office:
- **All fields required**: Can't submit unless everything is filled (form gets rejected)
- **Flexible fields**: Fill what you know, leave blanks for unknowns (form is accepted)
- **Smart defaults**: Some fields auto-fill based on others (intelligent inference)

When Gemini extracts data, it can't always find everything. Models with Optional fields and smart defaults succeed with partial information instead of failing completely.

### Type Safety Benefit
Missing data handling with Pydantic provides **graceful degradation**:
- Optional fields allow partial success
- Default values provide fallbacks
- Validators can infer missing data
- Type safety is maintained even with missing fields
- Clear distinction between "not found" (None) and "not looked for" (field doesn't exist)
- Completeness tracking shows what's missing

---

## B. Key Patterns

**Pattern 1: Optional Fields with None**
```python
class Person(BaseModel):
    name: str  # Always required
    age: Optional[int] = None  # Missing if not found
    email: Optional[str] = None  # Missing if not found
```

**Pattern 2: Completeness Tracking**
```python
class Person(BaseModel):
    name: str
    age: Optional[int] = None
    email: Optional[str] = None
    
    @property
    def completeness(self) -> float:
        """Calculate what percentage of fields are filled."""
        fields = [self.name, self.age, self.email]
        filled = sum(1 for f in fields if f is not None)
        return filled / len(fields)
```

**Pattern 3: Required Subset Pattern**
```python
class PersonMinimal(BaseModel):
    """Minimal required fields."""
    name: str

class PersonComplete(PersonMinimal):
    """Complete profile with optional fields."""
    age: Optional[int] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    # ... many optional fields
```

**Pattern 4: Confidence-Based Optionality**
```python
class ExtractedField(BaseModel):
    value: Optional[str] = None
    confidence: float = Field(..., ge=0.0, le=1.0)
    was_found: bool = False  # Distinguish "not found" from "found but empty"

class Person(BaseModel):
    name: ExtractedField
    age: ExtractedField
```

---

## C. Strategies for Missing Data

**1. Smart Defaults**
```python
class Document(BaseModel):
    title: str
    language: str = "en"  # Default to English if not specified
    published: bool = False  # Default to unpublished
    created_at: datetime = Field(default_factory=datetime.now)
```

**2. Inference from Context**
```python
@model_validator(mode='after')
def infer_missing_fields(self):
    """Infer missing data from available data."""
    # Infer country from state
    if self.state and not self.country:
        us_states = {"CA", "NY", "TX", ...}
        if self.state in us_states:
            self.country = "USA"
    
    # Infer full_name from parts
    if not self.full_name and self.first_name and self.last_name:
        self.full_name = f"{self.first_name} {self.last_name}"
    
    return self
```

**3. Partial Extraction Success**
```python
class ExtractionResult(BaseModel):
    """Result that succeeds even with partial data."""
    people: List[Person] = Field(default_factory=list)  # Empty if none found
    organizations: List[Organization] = Field(default_factory=list)
    
    extraction_status: str = "partial"  # "complete" | "partial" | "failed"
    missing_fields: List[str] = Field(default_factory=list)
    
    @property
    def has_any_data(self) -> bool:
        """Check if any entities were extracted."""
        return len(self.people) > 0 or len(self.organizations) > 0
```

**4. Quality Thresholds**
```python
class QualityAwareExtraction(BaseModel):
    person: Optional[Person] = None
    confidence: float
    quality_threshold: float = 0.7
    
    @model_validator(mode='after')
    def check_quality(self):
        """Reject low-quality extractions."""
        if self.person and self.confidence < self.quality_threshold:
            # Low confidence extraction - discard
            self.person = None
        return self
```

---

## D. Best Practices

**1. Distinguish "Not Found" from "Empty"**
```python
class Person(BaseModel):
    name: str
    bio: Optional[str] = None  # Not found in text
    # vs
    # bio: str = ""  # Found but empty
    
    # Better approach:
    bio: Optional[str] = Field(None, description="Biography if mentioned")
    bio_found: bool = False  # Explicitly track if we looked for it
```

**2. Graduated Confidence**
```python
class ExtractedEntity(BaseModel):
    text: str
    confidence: float
    
    def is_high_confidence(self) -> bool:
        return self.confidence >= 0.9
    
    def is_medium_confidence(self) -> bool:
        return 0.7 <= self.confidence < 0.9
    
    def is_low_confidence(self) -> bool:
        return self.confidence < 0.7
```

**3. Fallback Strategies**
```python
class PersonExtraction(BaseModel):
    # Primary extraction
    full_name: Optional[str] = None
    
    # Fallback components
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    
    @property
    def best_name(self) -> Optional[str]:
        """Return best available name."""
        if self.full_name:
            return self.full_name
        if self.first_name and self.last_name:
            return f"{self.first_name} {self.last_name}"
        if self.first_name:
            return self.first_name
        return None
```

**4. Explicit "Unknown" vs "Not Applicable"**
```python
class Organization(BaseModel):
    name: str
    employee_count: Optional[int] = None  # Unknown
    stock_ticker: Optional[str] = None  # N/A if private company
    
    is_public: bool = False  # Helps interpret stock_ticker=None
```

---

## E. System Prompt Guidance

```python
system_prompt = """
You are an expert at extracting information from text.

IMPORTANT RULES FOR MISSING DATA:
1. Extract ONLY information explicitly stated or strongly implied
2. If a field cannot be determined, leave it as null/empty
3. DO NOT invent or guess information
4. DO NOT fill fields with placeholder values like "unknown" or "N/A"
5. It's better to leave a field empty than to fill it incorrectly

If you're not confident about a piece of information (< 70% confidence),
leave that field empty.

If the text mentions that information is unknown or unavailable,
leave the corresponding field empty rather than extracting "unknown".

Example:
Text: "John works at some tech company in California"
Correct: {name: "John", company: null, location: "California"}
Incorrect: {name: "John", company: "some tech company", location: "California"}
"""
```

---

## F. Common Stumbling Blocks

### Proactive Debugging

**Mistake 1: Making everything required**
```python
# ❌ Extraction fails if any field missing
class Person(BaseModel):
    name: str
    age: int  # Required!
    email: str  # Required!
    # What if text doesn't mention age or email?

# ✅ Only require what's always present
class Person(BaseModel):
    name: str  # Only this is required
    age: Optional[int] = None
    email: Optional[str] = None
```

**Mistake 2: Using empty strings instead of None**
```python
# ❌ Empty string looks filled but isn't
email: str = ""  # Is this "no email" or "empty email address"?

# ✅ None is explicit
email: Optional[str] = None  # Clearly means "not found"
```

**Mistake 3: Not tracking data quality**
```python
# ❌ No way to know if data is complete
class Person(BaseModel):
    name: Optional[str] = None
    age: Optional[int] = None
    # If all fields are None, was extraction successful?

# ✅ Track quality explicitly
class Person(BaseModel):
    name: Optional[str] = None
    age: Optional[int] = None
    extraction_confidence: float = 0.0
    fields_found: int = 0
```

### Type Safety Gotchas

1. **None vs Empty**: `Optional[List]` can be None or empty list—different meanings
2. **False vs None**: For booleans, `False` and `None` mean different things
3. **Validation**: Optional fields still validate when present
4. **Defaults**: `Field(default_factory=list)` vs `Field(None)`—different behaviors
5. **Inference**: Be careful with inferred data—mark it clearly

---

## 🎯 Next Steps

Great work! You now understand:
- ✅ How to design models for incomplete data
- ✅ How to use Optional fields effectively
- ✅ How to track data completeness
- ✅ How to infer missing fields
- ✅ How to distinguish different types of missing data

In the next lesson, **Confidence Scores in Extraction**, we'll learn to track and use confidence scores to handle uncertain extractions.

**Ready for Lesson 14?** 🚀
