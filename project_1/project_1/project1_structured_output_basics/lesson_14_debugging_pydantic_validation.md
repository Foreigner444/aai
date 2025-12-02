# 📘 Lesson 14: Debugging Pydantic Validation

When validation fails, you need to know exactly what went wrong and how to fix it! 🔍

---

## A. Concept Overview

### What & Why

Debugging validation errors is crucial because:
- The AI might return data in an unexpected format
- Your model constraints might be too strict
- Field types might not match what the AI produces
- Nested structures can have subtle issues

Understanding how to read and fix validation errors will save you hours of frustration!

### The Analogy 🔬

Think of validation debugging like being a detective:
- **The crime** (validation error) = Data doesn't match expectations
- **The evidence** (error details) = Field name, expected type, actual value
- **The investigation** (debugging) = Understanding what went wrong
- **The solution** (fix) = Adjust model, prompt, or data

### Type Safety Benefit

Good debugging helps you:
- Create more robust models
- Write better system prompts
- Understand AI output patterns
- Build more reliable applications

---

## B. Code Implementation

### File Structure
```
pydantic-ai-project/
├── debug/
│   └── validation_debugging.py    # Debugging examples
└── ...
```

### Reading Validation Errors

```python
"""
How to read and understand Pydantic validation errors.
"""
from pydantic import BaseModel, Field, ValidationError
import json


class Product(BaseModel):
    """A product with various constraints."""
    name: str = Field(min_length=1, max_length=100)
    price: float = Field(gt=0)
    quantity: int = Field(ge=0)
    category: str
    tags: list[str] = Field(default_factory=list)


def analyze_validation_error(error: ValidationError) -> None:
    """Analyze a validation error in detail."""
    print("=" * 60)
    print("VALIDATION ERROR ANALYSIS")
    print("=" * 60)
    
    print(f"\nTotal errors: {error.error_count()}")
    print(f"Model: {error.title}")
    
    print("\n--- Detailed Errors ---")
    for i, err in enumerate(error.errors(), 1):
        print(f"\nError #{i}:")
        print(f"  Location: {' -> '.join(str(loc) for loc in err['loc'])}")
        print(f"  Message: {err['msg']}")
        print(f"  Type: {err['type']}")
        print(f"  Input value: {err['input']!r}")
        if 'ctx' in err:
            print(f"  Context: {err['ctx']}")
    
    print("\n--- JSON Format ---")
    print(json.dumps(error.errors(), indent=2))
    
    print("\n--- Human Readable ---")
    print(str(error))


# Example 1: Multiple errors
print("\n### EXAMPLE 1: Multiple validation errors ###")
try:
    product = Product(
        name="",           # Too short
        price=-10,         # Not > 0
        quantity="five",   # Not an integer
        category=None,     # None not allowed
        tags="not-a-list"  # Not a list
    )
except ValidationError as e:
    analyze_validation_error(e)


# Example 2: Type coercion failure
print("\n### EXAMPLE 2: Type coercion failure ###")
try:
    product = Product(
        name="Widget",
        price="free",  # Can't convert to float
        quantity=10,
        category="Electronics"
    )
except ValidationError as e:
    analyze_validation_error(e)
```

### Common Validation Error Types

```python
"""
Reference guide for common validation error types.
"""
from pydantic import BaseModel, Field, ValidationError


class Example(BaseModel):
    name: str = Field(min_length=1, max_length=10)
    age: int = Field(ge=0, le=150)
    email: str


# Dictionary of common error types and their meanings
ERROR_TYPES = {
    # Type errors
    "string_type": "Expected a string, got something else",
    "int_type": "Expected an integer, got something else",
    "float_type": "Expected a float, got something else",
    "bool_type": "Expected a boolean, got something else",
    "list_type": "Expected a list, got something else",
    "dict_type": "Expected a dictionary, got something else",
    
    # Parsing errors
    "int_parsing": "Could not parse value as integer",
    "float_parsing": "Could not parse value as float",
    "date_parsing": "Could not parse value as date",
    "datetime_parsing": "Could not parse value as datetime",
    "json_invalid": "Invalid JSON format",
    
    # Constraint errors
    "string_too_short": "String is shorter than minimum length",
    "string_too_long": "String is longer than maximum length",
    "greater_than": "Value must be greater than X",
    "greater_than_equal": "Value must be greater than or equal to X",
    "less_than": "Value must be less than X",
    "less_than_equal": "Value must be less than or equal to X",
    
    # Missing/extra errors
    "missing": "Required field is missing",
    "extra_forbidden": "Extra field not allowed",
    
    # Literal/Enum errors
    "literal_error": "Value not in allowed literals",
    "enum": "Value not in allowed enum values",
    
    # Pattern errors
    "string_pattern_mismatch": "String doesn't match required pattern",
}


def explain_error(error_dict: dict) -> str:
    """Explain a validation error in plain English."""
    error_type = error_dict.get('type', 'unknown')
    field = ' -> '.join(str(loc) for loc in error_dict.get('loc', []))
    input_value = error_dict.get('input')
    
    explanation = ERROR_TYPES.get(error_type, f"Unknown error type: {error_type}")
    
    return f"""
Field: {field}
Error: {explanation}
Got: {input_value!r}
Fix: {get_fix_suggestion(error_type, field)}
"""


def get_fix_suggestion(error_type: str, field: str) -> str:
    """Suggest how to fix common errors."""
    suggestions = {
        "missing": f"Add the '{field}' field to your input, or make it optional in the model",
        "string_too_short": f"Ensure '{field}' has enough characters, or reduce min_length in model",
        "string_too_long": f"Truncate '{field}' or increase max_length in model",
        "int_parsing": f"Ensure '{field}' contains only digits (no letters or symbols)",
        "greater_than": f"Increase the value of '{field}'",
        "less_than": f"Decrease the value of '{field}'",
        "literal_error": f"Use one of the allowed values for '{field}'",
    }
    return suggestions.get(error_type, "Check the field value and model constraints")


# Demo
try:
    Example(name="", age=200, email="test@test.com")
except ValidationError as e:
    for error in e.errors():
        print(explain_error(error))
```

### Debugging Nested Model Errors

```python
"""
Debugging validation errors in nested models.
"""
from pydantic import BaseModel, Field, ValidationError


class Address(BaseModel):
    street: str = Field(min_length=1)
    city: str = Field(min_length=1)
    zip_code: str = Field(pattern=r'^\d{5}$')


class Person(BaseModel):
    name: str
    age: int = Field(ge=0)
    address: Address


class Company(BaseModel):
    name: str
    employees: list[Person]


def debug_nested_error(error: ValidationError) -> None:
    """Debug errors in nested structures."""
    print("Nested Validation Error Debug")
    print("=" * 50)
    
    for err in error.errors():
        # Build the path to the error
        path_parts = []
        for loc in err['loc']:
            if isinstance(loc, int):
                path_parts.append(f"[{loc}]")
            else:
                if path_parts:
                    path_parts.append(f".{loc}")
                else:
                    path_parts.append(str(loc))
        
        full_path = "".join(path_parts)
        
        print(f"\n❌ Error at: {full_path}")
        print(f"   Message: {err['msg']}")
        print(f"   Type: {err['type']}")
        print(f"   Value: {err['input']!r}")


# Example with deeply nested error
try:
    company = Company(
        name="TechCorp",
        employees=[
            Person(
                name="Alice",
                age=30,
                address=Address(
                    street="123 Main St",
                    city="Boston",
                    zip_code="02101"  # Valid
                )
            ),
            Person(
                name="Bob",
                age=-5,  # Invalid!
                address=Address(
                    street="",        # Invalid!
                    city="NYC",
                    zip_code="abc"    # Invalid!
                )
            ),
        ]
    )
except ValidationError as e:
    debug_nested_error(e)
```

### Interactive Debugging Tool

```python
"""
An interactive tool for debugging validation issues.
"""
from pydantic import BaseModel, Field, ValidationError
from typing import Any
import json


def interactive_debug(model_class: type[BaseModel], data: dict[str, Any]) -> None:
    """
    Interactively debug validation issues.
    Tries to validate and shows detailed feedback.
    """
    print(f"\n{'='*60}")
    print(f"Validating data against {model_class.__name__}")
    print(f"{'='*60}")
    
    # Show the model's expected fields
    print("\n📋 Expected Schema:")
    for name, field in model_class.model_fields.items():
        required = "required" if field.is_required() else "optional"
        print(f"  - {name}: {field.annotation} ({required})")
    
    # Show the input data
    print("\n📦 Input Data:")
    print(json.dumps(data, indent=2, default=str))
    
    # Try to validate
    try:
        result = model_class.model_validate(data)
        print("\n✅ Validation PASSED!")
        print(f"Result: {result}")
        return
    except ValidationError as e:
        print(f"\n❌ Validation FAILED with {e.error_count()} error(s)")
    
    # Detailed error analysis
    print("\n🔍 Error Analysis:")
    for i, error in enumerate(e.errors(), 1):
        field = ".".join(str(x) for x in error['loc'])
        print(f"\n  Error #{i}: {field}")
        print(f"    Expected: {error['msg']}")
        print(f"    Received: {error['input']!r}")
        print(f"    Error Type: {error['type']}")
        
        # Suggest fixes
        if error['type'] == 'missing':
            print(f"    💡 Fix: Add '{field}' to your data")
        elif 'parsing' in error['type']:
            print(f"    💡 Fix: Check the format of '{field}'")
        elif 'too_short' in error['type']:
            print(f"    💡 Fix: '{field}' needs more characters")
        elif 'too_long' in error['type']:
            print(f"    💡 Fix: '{field}' is too long, truncate it")


# Usage example
class UserProfile(BaseModel):
    username: str = Field(min_length=3, max_length=20)
    email: str
    age: int = Field(ge=13, le=120)
    bio: str = Field(default="", max_length=500)


# Test with problematic data
test_data = {
    "username": "ab",      # Too short
    "email": "test@test.com",
    "age": 10,             # Too young
    "bio": "Hello!"
}

interactive_debug(UserProfile, test_data)
```

### Logging Validation Errors

```python
"""
Proper logging of validation errors for production.
"""
from pydantic import BaseModel, ValidationError
import logging
import json
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ValidationErrorLogger:
    """Log validation errors with context."""
    
    @staticmethod
    def log_error(
        error: ValidationError,
        context: dict | None = None,
        input_data: Any = None
    ) -> None:
        """Log a validation error with full context."""
        error_record = {
            "timestamp": datetime.now().isoformat(),
            "error_count": error.error_count(),
            "model": error.title,
            "errors": [
                {
                    "field": ".".join(str(x) for x in err['loc']),
                    "type": err['type'],
                    "message": err['msg'],
                    "input": str(err['input'])[:100]  # Truncate for safety
                }
                for err in error.errors()
            ],
            "context": context or {},
        }
        
        # Log as JSON for easy parsing
        logger.error(f"ValidationError: {json.dumps(error_record)}")
        
        # Also log human-readable version
        logger.debug(f"Validation failed:\n{error}")


# Usage
class Product(BaseModel):
    name: str
    price: float


try:
    Product(name="Widget", price="free")
except ValidationError as e:
    ValidationErrorLogger.log_error(
        e,
        context={"source": "api_endpoint", "user_id": "123"},
        input_data={"name": "Widget", "price": "free"}
    )
```

---

## C. Test & Apply

### Debugging Checklist

When you encounter a validation error:

1. **Read the error message** - It tells you exactly what's wrong
2. **Identify the field** - Look at the `loc` (location)
3. **Check the type** - What error type occurred?
4. **Compare input vs expected** - What did you send vs what was expected?
5. **Fix the source** - Either the data, the model, or the prompt

### Common Fixes Table

| Error Type | Common Cause | Fix |
|------------|--------------|-----|
| `missing` | Required field not provided | Add field or make it Optional |
| `string_too_short` | String shorter than min_length | Increase string or reduce min_length |
| `int_parsing` | Non-numeric string | Clean input or use str type |
| `literal_error` | Invalid enum/literal value | Use allowed value or expand choices |
| `extra_forbidden` | Unknown field in input | Remove field or allow extras |

---

## D. Common Stumbling Blocks

### "I keep getting validation errors from the AI"

Improve your prompt and model:
```python
# Add detailed descriptions
class Response(BaseModel):
    """The AI will follow these descriptions."""
    count: int = Field(description="A whole number, not a string")
    status: str = Field(description="Must be exactly 'active' or 'inactive'")
```

### "The error location is confusing"

Location is a tuple showing the path:
- `('name',)` → `name` field
- `('items', 0, 'price')` → `items[0].price`
- `('address', 'zip')` → `address.zip`

### "I can't reproduce the error"

Log the exact input:
```python
import json

try:
    result = Model.model_validate(data)
except ValidationError as e:
    # Log exactly what was passed
    print(f"Failed input: {json.dumps(data)}")
    raise
```

### "The AI sometimes returns wrong types"

Make your model more forgiving:
```python
from typing import Union

class FlexibleModel(BaseModel):
    # Accept either int or string that looks like int
    count: int  # Pydantic will try to convert "5" to 5
    
    # Or be explicit about accepting multiple types
    value: Union[int, str]
```

---

## ✅ Lesson 14 Complete!

### Key Takeaways

1. **Read error messages carefully** - they contain all the info you need
2. **Use `.errors()`** to get structured error information
3. **Check the `loc`** to find which field failed
4. **Check the `type`** to understand why it failed
5. **Log errors properly** in production
6. **Add field descriptions** to help the AI

### Debugging Checklist

- [ ] Can read and understand ValidationError messages
- [ ] Know how to access error details with .errors()
- [ ] Can debug nested model errors
- [ ] Have proper error logging in place
- [ ] Know common fixes for common errors

### What's Next?

In Lesson 15, we'll experiment with different prompts to improve AI output quality!

---

*Debugging mastered! Let's test prompts in Lesson 15!* 🚀
