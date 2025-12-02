# 📘 Lesson 6: Field Types and Validation

Now let's supercharge your models with advanced field types and validation rules! 🔧

---

## A. Concept Overview

### What & Why

Basic types like `str` and `int` are great, but real applications need more:
- Numbers within a specific range (age between 0-150)
- Strings matching a pattern (email, phone number)
- Lists of items (tags, skills)
- Constrained values (rating between 1-5)

Pydantic's `Field()` function and special types let you add these constraints. When Gemini returns data, these constraints are validated automatically!

### The Analogy 🎯

Think of field validation like a bouncer at a club:

**Basic type (`int`):** "Are you a number? Okay, you can enter."

**Field with constraints (`Field(ge=21, le=100)`):** "Are you a number? Great. Are you at least 21? Are you under 100? Only then can you enter!"

The more specific your constraints, the better your data quality.

### Type Safety Benefit

Field constraints ensure:
- AI outputs realistic values (not age=-500)
- Strings match expected formats
- Lists have appropriate lengths
- Data makes sense for your domain

---

## B. Code Implementation

### File Structure
```
pydantic-ai-project/
├── models/
│   ├── __init__.py
│   ├── basic_models.py
│   └── validated_models.py    # New file with validation!
└── ...
```

### Field Constraints with `Field()`

Create `models/validated_models.py`:

```python
"""
Advanced models with field constraints and validation.
"""
from pydantic import BaseModel, Field
from typing import Optional


class Person(BaseModel):
    """A person with validated fields."""
    
    name: str = Field(
        min_length=1,           # At least 1 character
        max_length=100,         # At most 100 characters
        description="Person's full name"
    )
    
    age: int = Field(
        ge=0,                   # Greater than or equal to 0
        le=150,                 # Less than or equal to 150
        description="Age in years"
    )
    
    email: str = Field(
        pattern=r'^[\w\.-]+@[\w\.-]+\.\w+$',  # Email regex pattern
        description="Email address"
    )


class Product(BaseModel):
    """A product with price and rating constraints."""
    
    name: str = Field(min_length=1, max_length=200)
    
    price: float = Field(
        gt=0,                   # Greater than 0 (must be positive)
        le=1000000,            # Max 1 million
        description="Price in USD"
    )
    
    rating: float = Field(
        ge=0.0,                # Min 0
        le=5.0,                # Max 5
        description="Rating from 0 to 5 stars"
    )
    
    quantity: int = Field(
        ge=0,                  # Can't have negative quantity
        default=0,
        description="Items in stock"
    )


# Test the validation
if __name__ == "__main__":
    # ✅ Valid person
    person = Person(
        name="Alice Smith",
        age=30,
        email="alice@example.com"
    )
    print(f"✅ Valid person: {person}")
    
    # ❌ Invalid age (negative)
    try:
        bad_person = Person(name="Bob", age=-5, email="bob@test.com")
    except Exception as e:
        print(f"\n❌ Invalid age: {e}")
    
    # ❌ Invalid email (no @ symbol)
    try:
        bad_email = Person(name="Charlie", age=25, email="not-an-email")
    except Exception as e:
        print(f"\n❌ Invalid email: {e}")
    
    # ❌ Name too long
    try:
        long_name = Person(name="A" * 200, age=25, email="test@test.com")
    except Exception as e:
        print(f"\n❌ Name too long: {e}")
```

### Field Constraint Reference

| Constraint | Meaning | Example |
|------------|---------|---------|
| `gt` | Greater than | `Field(gt=0)` → must be > 0 |
| `ge` | Greater than or equal | `Field(ge=0)` → must be >= 0 |
| `lt` | Less than | `Field(lt=100)` → must be < 100 |
| `le` | Less than or equal | `Field(le=100)` → must be <= 100 |
| `min_length` | Minimum string length | `Field(min_length=1)` → at least 1 char |
| `max_length` | Maximum string length | `Field(max_length=50)` → at most 50 chars |
| `pattern` | Regex pattern | `Field(pattern=r'^\d{5}$')` → 5 digits |
| `default` | Default value | `Field(default=0)` → defaults to 0 |
| `description` | Field description | Used in JSON schema for AI |

### Lists and Collections

```python
from pydantic import BaseModel, Field
from typing import Optional


class Article(BaseModel):
    """An article with tags and categories."""
    
    title: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=10)
    
    # List of strings with constraints
    tags: list[str] = Field(
        min_length=1,              # At least 1 tag
        max_length=10,             # At most 10 tags
        description="Article tags"
    )
    
    # Optional list
    categories: Optional[list[str]] = None


class Recipe(BaseModel):
    """A recipe with ingredients list."""
    
    name: str
    prep_time_minutes: int = Field(ge=0)
    cook_time_minutes: int = Field(ge=0)
    servings: int = Field(ge=1, le=100)
    
    # List must have at least one ingredient
    ingredients: list[str] = Field(min_length=1)
    
    # List of steps
    steps: list[str] = Field(min_length=1)


if __name__ == "__main__":
    article = Article(
        title="Introduction to Pydantic",
        content="This is a comprehensive guide to using Pydantic...",
        tags=["python", "pydantic", "validation"]
    )
    print(f"Article tags: {article.tags}")
    
    recipe = Recipe(
        name="Chocolate Chip Cookies",
        prep_time_minutes=15,
        cook_time_minutes=12,
        servings=24,
        ingredients=["flour", "sugar", "butter", "chocolate chips"],
        steps=["Mix dry ingredients", "Add wet ingredients", "Bake"]
    )
    print(f"Recipe: {recipe.name}, {len(recipe.ingredients)} ingredients")
```

### Special Types

Pydantic provides special types for common validations:

```python
from pydantic import BaseModel, Field, EmailStr, HttpUrl
from datetime import date, datetime
from typing import Literal
from enum import Enum


# Enum for controlled values
class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ContactInfo(BaseModel):
    """Contact with special validated types."""
    
    email: EmailStr                    # Validates email format
    website: Optional[HttpUrl] = None  # Validates URL format


class Task(BaseModel):
    """A task with various field types."""
    
    title: str = Field(min_length=1)
    description: Optional[str] = None
    
    # Enum ensures only valid values
    priority: Priority = Priority.MEDIUM
    
    # Literal type for fixed options
    status: Literal["pending", "in_progress", "completed"] = "pending"
    
    # Date types
    due_date: Optional[date] = None
    created_at: datetime = Field(default_factory=datetime.now)


class Score(BaseModel):
    """A score with precise constraints."""
    
    # Use multiple constraints
    value: float = Field(ge=0, le=100, description="Score from 0-100")
    
    # Literal for letter grades
    letter_grade: Literal["A", "B", "C", "D", "F"]


if __name__ == "__main__":
    # Contact with email validation
    contact = ContactInfo(
        email="user@example.com",
        website="https://example.com"
    )
    print(f"Contact: {contact}")
    
    # Invalid email raises error
    try:
        bad_contact = ContactInfo(email="not-an-email")
    except Exception as e:
        print(f"❌ Bad email: {e}")
    
    # Task with enum
    task = Task(
        title="Learn Pydantic",
        priority=Priority.HIGH,
        status="in_progress",
        due_date="2024-12-31"
    )
    print(f"Task: {task.title}, Priority: {task.priority}")
    
    # Invalid status
    try:
        bad_task = Task(title="Test", status="invalid_status")
    except Exception as e:
        print(f"❌ Bad status: {e}")
```

### Nested Models with Validation

```python
from pydantic import BaseModel, Field, EmailStr
from typing import Optional


class Address(BaseModel):
    """Validated address model."""
    street: str = Field(min_length=1)
    city: str = Field(min_length=1)
    state: str = Field(min_length=2, max_length=2)  # 2-letter state code
    zip_code: str = Field(pattern=r'^\d{5}(-\d{4})?$')  # US ZIP code


class Company(BaseModel):
    """Company with validated nested address."""
    name: str = Field(min_length=1, max_length=200)
    industry: str
    employee_count: int = Field(ge=1)
    headquarters: Address  # Nested model with its own validation!


class Employee(BaseModel):
    """Employee with all validations."""
    first_name: str = Field(min_length=1, max_length=50)
    last_name: str = Field(min_length=1, max_length=50)
    email: EmailStr
    age: int = Field(ge=18, le=100)  # Working age
    salary: float = Field(gt=0)
    department: str
    company: Company  # Nested with nested Address


if __name__ == "__main__":
    # Valid employee with nested structures
    employee = Employee(
        first_name="Jane",
        last_name="Doe",
        email="jane@techcorp.com",
        age=35,
        salary=95000,
        department="Engineering",
        company=Company(
            name="TechCorp",
            industry="Technology",
            employee_count=500,
            headquarters=Address(
                street="123 Tech Blvd",
                city="San Francisco",
                state="CA",
                zip_code="94105"
            )
        )
    )
    
    print(f"Employee: {employee.first_name} {employee.last_name}")
    print(f"Company: {employee.company.name}")
    print(f"City: {employee.company.headquarters.city}")
    
    # Invalid nested data
    try:
        bad_employee = Employee(
            first_name="John",
            last_name="Smith",
            email="john@test.com",
            age=25,
            salary=50000,
            department="Sales",
            company=Company(
                name="BadCorp",
                industry="Unknown",
                employee_count=10,
                headquarters=Address(
                    street="456 Main St",
                    city="Boston",
                    state="Massachusetts",  # Too long! Should be "MA"
                    zip_code="02101"
                )
            )
        )
    except Exception as e:
        print(f"\n❌ Nested validation error: {e}")
```

---

## C. Test & Apply

### Why Constraints Matter for AI

When you use these constrained models with Pydantic AI, the AI is instructed to return valid values. If it doesn't, you get an immediate error:

```python
from pydantic import BaseModel, Field
from pydantic_ai import Agent


class ProductReview(BaseModel):
    """A validated product review."""
    product_name: str = Field(min_length=1)
    rating: float = Field(ge=1.0, le=5.0)  # Must be 1-5
    summary: str = Field(min_length=10, max_length=500)
    pros: list[str] = Field(min_length=1, max_length=5)
    cons: list[str] = Field(min_length=0, max_length=5)


# The agent MUST return data matching these constraints
agent = Agent('gemini-1.5-flash', result_type=ProductReview)

# If Gemini tries to return rating=10 or rating=-1, 
# Pydantic catches it immediately!
```

### Practice: Build a Constrained Model

Create a `Job` model for a job posting with these requirements:
- `title`: 1-100 characters
- `company`: 1-100 characters
- `salary_min`: positive number
- `salary_max`: positive number, must be >= salary_min
- `experience_years`: 0-50
- `skills`: list of 1-20 strings
- `remote`: boolean with default False

**Solution:**

```python
from pydantic import BaseModel, Field, model_validator
from typing import Self


class Job(BaseModel):
    """A job posting with validated fields."""
    
    title: str = Field(min_length=1, max_length=100)
    company: str = Field(min_length=1, max_length=100)
    salary_min: float = Field(gt=0)
    salary_max: float = Field(gt=0)
    experience_years: int = Field(ge=0, le=50)
    skills: list[str] = Field(min_length=1, max_length=20)
    remote: bool = False
    
    @model_validator(mode='after')
    def check_salary_range(self) -> Self:
        """Ensure salary_max >= salary_min."""
        if self.salary_max < self.salary_min:
            raise ValueError('salary_max must be >= salary_min')
        return self
```

---

## D. Common Stumbling Blocks

### "Value error, X is less than minimum"

Your value doesn't meet the constraint:
```python
# Field has ge=0 (must be >= 0)
Product(name="Test", price=-10, rating=3.0)  
# ❌ Input should be greater than 0
```

### "String should have at least X characters"

String is too short:
```python
# Field has min_length=1
Person(name="", age=30, email="test@test.com")
# ❌ String should have at least 1 character
```

### "String should match pattern"

String doesn't match the regex:
```python
# Field has pattern for email
Person(name="Test", age=30, email="not-an-email")
# ❌ String should match pattern '^[\w\.-]+@[\w\.-]+\.\w+$'
```

### "Input should be 'X', 'Y' or 'Z'"

Using Literal but provided invalid value:
```python
status: Literal["pending", "done"] = "pending"
Task(title="Test", status="completed")  # "completed" not in Literal!
# ❌ Input should be 'pending' or 'done'
```

### "How do I validate across multiple fields?"

Use `@model_validator`:

```python
from pydantic import BaseModel, Field, model_validator
from typing import Self


class DateRange(BaseModel):
    start_date: date
    end_date: date
    
    @model_validator(mode='after')
    def check_dates(self) -> Self:
        if self.end_date < self.start_date:
            raise ValueError('end_date must be after start_date')
        return self
```

---

## ✅ Lesson 6 Complete!

### Key Takeaways

1. **`Field()`** adds constraints to model fields
2. **Numeric constraints:** `gt`, `ge`, `lt`, `le`
3. **String constraints:** `min_length`, `max_length`, `pattern`
4. **Lists support:** `min_length` and `max_length` for item count
5. **Special types:** `EmailStr`, `HttpUrl`, `Literal`, `Enum`
6. **Nested models** validate their own constraints too
7. **`@model_validator`** for cross-field validation

### What's Next?

In Lesson 7, we'll set up your Google Gemini API key so you can start making real AI calls!

---

*Models are validated! Let's get your API key in Lesson 7!* 🚀
