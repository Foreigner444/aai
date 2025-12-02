# Lesson 1: Understanding Complex Data Structures

## A. Concept Overview

### What & Why
**Understanding complex data structures is the foundation of building type-safe data extraction systems.** In the real world, data isn't just simple strings and numbers—it's nested objects, lists of items, relationships between entities, and optional fields. Learning how to model this complexity with Pydantic ensures that every piece of extracted data is validated and structured correctly before it enters your application.

### Analogy
Think of complex data structures like a filing cabinet system in a company:
- **Simple data** is like a single sticky note with one piece of info
- **Complex data** is like an entire filing cabinet with drawers (categories), folders (entities), documents (records), and nested sub-folders (relationships)
- **Pydantic models** are like the organizational system that defines exactly where everything belongs and what format it must be in

When you extract data from text with Gemini, you're asking the AI to organize messy information into this perfect filing system automatically!

### Type Safety Benefit
Complex data structures with Pydantic provide **compile-time and runtime guarantees** that:
- Every nested object has the correct structure
- Lists contain only the expected types
- Required fields are never missing
- Relationships between entities are valid
- Your IDE can autocomplete every field at every nesting level

This means bugs are caught immediately during validation, not days later when corrupted data causes a production failure.

---

## B. Code Implementation

### Project Structure
```
data_extraction_pipeline/
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── structures.py    # We'll create this
│   └── examples/
│       ├── __init__.py
│       └── complex_data_demo.py
├── requirements.txt
└── README.md
```

### Complete Code Example

**File: `src/models/structures.py`**

```python
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from enum import Enum


# Simple model - single level
class Person(BaseModel):
    """A simple person model with basic fields."""
    name: str
    age: int
    email: str


# Nested model - one object inside another
class Address(BaseModel):
    """Physical address information."""
    street: str
    city: str
    state: str
    zip_code: str
    country: str = "USA"  # Default value


class PersonWithAddress(BaseModel):
    """Person model with nested address."""
    name: str
    age: int
    email: str
    address: Address  # Nested model - another Pydantic model inside


# Collection model - lists of items
class Company(BaseModel):
    """Company with multiple employees."""
    name: str
    founded_year: int
    employees: List[Person]  # List of Person objects
    headquarters: Address  # Nested single object


# Complex nested model - multiple levels deep
class Department(BaseModel):
    """Department within a company."""
    name: str
    budget: float
    head: Person  # Single person
    team_members: List[Person]  # Multiple people


class Organization(BaseModel):
    """Complete organization structure - deeply nested."""
    name: str
    founded: datetime
    headquarters: Address
    departments: List[Department]  # List of nested objects
    total_employees: int
    is_public: bool


# Optional fields - data that might not exist
class ContactInfo(BaseModel):
    """Contact information with optional fields."""
    email: str  # Required
    phone: Optional[str] = None  # Optional - can be missing
    linkedin: Optional[str] = None
    twitter: Optional[str] = None
    website: Optional[str] = None


class ProfessionalProfile(BaseModel):
    """Professional profile with optional data."""
    name: str
    title: str
    contact: ContactInfo
    skills: List[str]
    certifications: Optional[List[str]] = None  # Entire list can be missing
    years_experience: int
    current_company: Optional[str] = None


# Enum for controlled values
class ProjectStatus(str, Enum):
    """Valid project statuses."""
    PLANNING = "planning"
    IN_PROGRESS = "in_progress"
    ON_HOLD = "on_hold"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class Project(BaseModel):
    """Project with controlled status values."""
    name: str
    status: ProjectStatus  # Can only be one of the enum values
    start_date: datetime
    end_date: Optional[datetime] = None
    team_size: int
    budget: float
```

**File: `src/examples/complex_data_demo.py`**

```python
from datetime import datetime
from src.models.structures import (
    Person,
    Address,
    PersonWithAddress,
    Company,
    Organization,
    Department,
    ProfessionalProfile,
    ContactInfo,
    Project,
    ProjectStatus,
)


def demo_simple_model():
    """Demonstrate simple single-level model."""
    print("=" * 50)
    print("SIMPLE MODEL DEMO")
    print("=" * 50)
    
    person = Person(
        name="Alice Johnson",
        age=32,
        email="alice@example.com"
    )
    
    print(f"Person: {person.name}, Age: {person.age}")
    print(f"Full model: {person.model_dump()}")
    print()


def demo_nested_model():
    """Demonstrate one level of nesting."""
    print("=" * 50)
    print("NESTED MODEL DEMO")
    print("=" * 50)
    
    address = Address(
        street="123 Main St",
        city="San Francisco",
        state="CA",
        zip_code="94105",
        country="USA"
    )
    
    person = PersonWithAddress(
        name="Bob Smith",
        age=28,
        email="bob@example.com",
        address=address
    )
    
    print(f"Person: {person.name}")
    print(f"Lives in: {person.address.city}, {person.address.state}")
    print(f"Full address: {person.address.street}")
    print(f"Full model: {person.model_dump()}")
    print()


def demo_collections():
    """Demonstrate lists of objects."""
    print("=" * 50)
    print("COLLECTIONS DEMO")
    print("=" * 50)
    
    employees = [
        Person(name="Alice Johnson", age=32, email="alice@company.com"),
        Person(name="Bob Smith", age=28, email="bob@company.com"),
        Person(name="Carol White", age=35, email="carol@company.com"),
    ]
    
    hq = Address(
        street="100 Tech Boulevard",
        city="San Francisco",
        state="CA",
        zip_code="94105",
        country="USA"
    )
    
    company = Company(
        name="TechCorp",
        founded_year=2015,
        employees=employees,
        headquarters=hq
    )
    
    print(f"Company: {company.name}")
    print(f"Number of employees: {len(company.employees)}")
    print("Employee names:")
    for emp in company.employees:
        print(f"  - {emp.name} ({emp.email})")
    print()


def demo_deep_nesting():
    """Demonstrate multiple levels of nesting."""
    print("=" * 50)
    print("DEEP NESTING DEMO")
    print("=" * 50)
    
    eng_dept = Department(
        name="Engineering",
        budget=2000000.0,
        head=Person(name="Sarah Tech", age=40, email="sarah@org.com"),
        team_members=[
            Person(name="Dev One", age=25, email="dev1@org.com"),
            Person(name="Dev Two", age=27, email="dev2@org.com"),
        ]
    )
    
    sales_dept = Department(
        name="Sales",
        budget=1500000.0,
        head=Person(name="John Sell", age=38, email="john@org.com"),
        team_members=[
            Person(name="Sales One", age=30, email="sales1@org.com"),
        ]
    )
    
    org = Organization(
        name="MegaCorp",
        founded=datetime(2010, 1, 15),
        headquarters=Address(
            street="500 Business Ave",
            city="New York",
            state="NY",
            zip_code="10001",
            country="USA"
        ),
        departments=[eng_dept, sales_dept],
        total_employees=50,
        is_public=True
    )
    
    print(f"Organization: {org.name}")
    print(f"Founded: {org.founded.year}")
    print(f"Departments: {len(org.departments)}")
    for dept in org.departments:
        print(f"\n  Department: {dept.name}")
        print(f"  Head: {dept.head.name}")
        print(f"  Team size: {len(dept.team_members)}")
        print(f"  Budget: ${dept.budget:,.2f}")
    print()


def demo_optional_fields():
    """Demonstrate optional fields."""
    print("=" * 50)
    print("OPTIONAL FIELDS DEMO")
    print("=" * 50)
    
    # Profile with all optional fields filled
    contact_full = ContactInfo(
        email="jane@example.com",
        phone="+1-555-0100",
        linkedin="linkedin.com/in/jane",
        twitter="@jane",
        website="jane.dev"
    )
    
    profile_full = ProfessionalProfile(
        name="Jane Developer",
        title="Senior Software Engineer",
        contact=contact_full,
        skills=["Python", "TypeScript", "AWS"],
        certifications=["AWS Certified", "Python Expert"],
        years_experience=8,
        current_company="TechCorp"
    )
    
    print("Profile with all fields:")
    print(f"  Name: {profile_full.name}")
    print(f"  Phone: {profile_full.contact.phone}")
    print(f"  Certifications: {profile_full.certifications}")
    
    # Profile with minimal optional fields
    contact_minimal = ContactInfo(
        email="john@example.com"
        # All other fields are optional and omitted
    )
    
    profile_minimal = ProfessionalProfile(
        name="John Starter",
        title="Junior Developer",
        contact=contact_minimal,
        skills=["Python", "JavaScript"],
        # certifications is optional - not provided
        years_experience=1,
        # current_company is optional - not provided
    )
    
    print("\nProfile with minimal fields:")
    print(f"  Name: {profile_minimal.name}")
    print(f"  Phone: {profile_minimal.contact.phone}")  # Will be None
    print(f"  Certifications: {profile_minimal.certifications}")  # Will be None
    print()


def demo_enums():
    """Demonstrate enum-controlled values."""
    print("=" * 50)
    print("ENUM DEMO")
    print("=" * 50)
    
    project = Project(
        name="Website Redesign",
        status=ProjectStatus.IN_PROGRESS,  # Must be one of the enum values
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 6, 30),
        team_size=5,
        budget=250000.0
    )
    
    print(f"Project: {project.name}")
    print(f"Status: {project.status.value}")  # Access the enum value
    print(f"Team size: {project.team_size}")
    
    # Show all valid statuses
    print("\nValid project statuses:")
    for status in ProjectStatus:
        print(f"  - {status.value}")
    print()


if __name__ == "__main__":
    print("\n🎯 COMPLEX DATA STRUCTURES DEMONSTRATION\n")
    
    demo_simple_model()
    demo_nested_model()
    demo_collections()
    demo_deep_nesting()
    demo_optional_fields()
    demo_enums()
    
    print("=" * 50)
    print("✅ All demonstrations completed!")
    print("=" * 50)
```

### Line-by-Line Explanation

**Model Definitions:**

1. **Simple Model (Person)**: Basic Pydantic model with primitive types (str, int). Every field is required.

2. **Nested Model (Address, PersonWithAddress)**: One Pydantic model contains another as a field. The `address: Address` field means this field must be a valid Address object.

3. **Collection Model (Company)**: Uses `List[Person]` to indicate a list where every item must be a valid Person object. Python validates each item.

4. **Deep Nesting (Organization)**: Multiple levels: Organization contains List[Department], each Department contains a Person (head) and List[Person] (team_members).

5. **Optional Fields**: `Optional[str] = None` means the field can be missing or None. Required fields have no Optional wrapper.

6. **Enums**: `class ProjectStatus(str, Enum)` defines a fixed set of valid values. Pydantic validates that only these values are used.

**Demo Functions:**

Each function shows how to create and access data at different complexity levels, demonstrating type-safe access to nested data.

### The "Why" Behind the Pattern

**Type Safety at Every Level:**
- When you write `person.address.city`, your IDE knows `address` is an Address object and `city` is a string
- If you try `person.address.invalid_field`, your IDE warns you immediately
- At runtime, if Gemini returns data that doesn't match, Pydantic catches it before it reaches your code

**Validation Cascades:**
- When you validate an Organization, Pydantic automatically validates every Department inside it
- Each Department validates its head (Person) and all team_members (List[Person])
- Every Person validates its own fields
- One validation call checks the entire nested structure

**No Silent Failures:**
- Missing required fields? ValidationError before assignment
- Wrong type in a list? ValidationError with the specific index
- Invalid enum value? ValidationError with valid options shown
- Malformed nested object? ValidationError with the exact path to the error

---

## C. Test & Apply

### How to Test It

**Step 1: Create the project structure**
```bash
mkdir -p data_extraction_pipeline/src/models
mkdir -p data_extraction_pipeline/src/examples
cd data_extraction_pipeline
```

**Step 2: Create empty `__init__.py` files**
```bash
touch src/__init__.py
touch src/models/__init__.py
touch src/examples/__init__.py
```

**Step 3: Create a requirements.txt**
```bash
echo "pydantic>=2.0.0" > requirements.txt
```

**Step 4: Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 5: Copy the code above into the files**
- Create `src/models/structures.py` with the models code
- Create `src/examples/complex_data_demo.py` with the demo code

**Step 6: Run the demonstration**
```bash
python -m src.examples.complex_data_demo
```

### Expected Result

You should see output like this:

```
🎯 COMPLEX DATA STRUCTURES DEMONSTRATION

==================================================
SIMPLE MODEL DEMO
==================================================
Person: Alice Johnson, Age: 32
Full model: {'name': 'Alice Johnson', 'age': 32, 'email': 'alice@example.com'}

==================================================
NESTED MODEL DEMO
==================================================
Person: Bob Smith
Lives in: San Francisco, CA
Full address: 123 Main St
Full model: {'name': 'Bob Smith', 'age': 28, 'email': 'bob@example.com', 'address': {'street': '123 Main St', 'city': 'San Francisco', 'state': 'CA', 'zip_code': '94105', 'country': 'USA'}}

==================================================
COLLECTIONS DEMO
==================================================
Company: TechCorp
Number of employees: 3
Employee names:
  - Alice Johnson (alice@company.com)
  - Bob Smith (bob@company.com)
  - Carol White (carol@company.com)

==================================================
DEEP NESTING DEMO
==================================================
Organization: MegaCorp
Founded: 2010
Departments: 2

  Department: Engineering
  Head: Sarah Tech
  Team size: 2
  Budget: $2,000,000.00

  Department: Sales
  Head: John Sell
  Team size: 1
  Budget: $1,500,000.00

==================================================
OPTIONAL FIELDS DEMO
==================================================
Profile with all fields:
  Name: Jane Developer
  Phone: +1-555-0100
  Certifications: ['AWS Certified', 'Python Expert']

Profile with minimal fields:
  Name: John Starter
  Phone: None
  Certifications: None

==================================================
ENUM DEMO
==================================================
Project: Website Redesign
Status: in_progress
Team size: 5

Valid project statuses:
  - planning
  - in_progress
  - on_hold
  - completed
  - cancelled

==================================================
✅ All demonstrations completed!
==================================================
```

### Validation Examples

**Try this experiment** - Add this to the bottom of `complex_data_demo.py`:

```python
# This will FAIL - demonstrating validation
def demo_validation_errors():
    print("\n🚫 VALIDATION ERROR DEMOS\n")
    
    # Error 1: Wrong type
    try:
        person = Person(
            name="Test",
            age="thirty-two",  # Should be int, not str
            email="test@example.com"
        )
    except Exception as e:
        print(f"Error 1 - Wrong type: {e}\n")
    
    # Error 2: Missing required field
    try:
        person = Person(
            name="Test",
            # age is missing!
            email="test@example.com"
        )
    except Exception as e:
        print(f"Error 2 - Missing field: {e}\n")
    
    # Error 3: Invalid enum value
    try:
        project = Project(
            name="Test Project",
            status="invalid_status",  # Not a valid ProjectStatus
            start_date=datetime.now(),
            team_size=3,
            budget=10000.0
        )
    except Exception as e:
        print(f"Error 3 - Invalid enum: {e}\n")

# Uncomment to see validation errors:
# demo_validation_errors()
```

### Type Checking

**Create a file `type_check_demo.py`:**

```python
from src.models.structures import Person, PersonWithAddress, Address

# Your IDE will give you autocomplete for all these!
person = Person(name="Alice", age=30, email="alice@example.com")

# IDE knows 'name' is a string
uppercase_name: str = person.name.upper()  # ✅ Type safe

# IDE knows 'age' is an int
next_year_age: int = person.age + 1  # ✅ Type safe

# This will show an error in your IDE:
# result = person.age.upper()  # ❌ int has no method 'upper'

# Nested access is also type-safe
person_with_addr = PersonWithAddress(
    name="Bob",
    age=25,
    email="bob@example.com",
    address=Address(
        street="123 Main",
        city="NYC",
        state="NY",
        zip_code="10001"
    )
)

# IDE knows address is Address, and city is str
city_name: str = person_with_addr.address.city.upper()  # ✅ Type safe
```

**Run mypy to verify types:**
```bash
pip install mypy
mypy src/
```

You should see no errors if all types are correct!

---

## D. Common Stumbling Blocks

### Proactive Debugging

**Mistake 1: Forgetting that nested models need to be instantiated**

Many beginners try to pass a dictionary where a Pydantic model is expected:

```python
# ❌ WRONG - passing a dict instead of Address object
person = PersonWithAddress(
    name="Bob",
    age=25,
    email="bob@example.com",
    address={  # This is a dict, not an Address object
        "street": "123 Main",
        "city": "NYC",
        "state": "NY",
        "zip_code": "10001"
    }
)
```

**The Fix:** Pydantic actually handles this automatically! It will convert the dict to an Address object. But if you're constructing manually, create the object explicitly:

```python
# ✅ CORRECT - explicit object creation
address = Address(
    street="123 Main",
    city="NYC",
    state="NY",
    zip_code="10001"
)

person = PersonWithAddress(
    name="Bob",
    age=25,
    email="bob@example.com",
    address=address
)
```

**Mistake 2: Confusing Optional fields with default values**

```python
# These are DIFFERENT:
class Example1(BaseModel):
    field: Optional[str] = None  # Can be None, defaults to None if not provided

class Example2(BaseModel):
    field: str = "default"  # Cannot be None, defaults to "default" if not provided

class Example3(BaseModel):
    field: Optional[str]  # Can be None, but MUST be provided explicitly
```

### Show the Error

**Error 1: Type mismatch in nested structure**

```python
person = Person(
    name="Alice",
    age="thirty",  # Wrong type
    email="alice@example.com"
)
```

**Error message:**
```
ValidationError: 1 validation error for Person
age
  Input should be a valid integer, unable to parse string as an integer [type=int_parsing, input_value='thirty', input_type=str]
```

**Error 2: Missing required nested field**

```python
person = PersonWithAddress(
    name="Bob",
    age=25,
    email="bob@example.com",
    address=Address(
        street="123 Main",
        city="NYC",
        state="NY"
        # zip_code is missing!
    )
)
```

**Error message:**
```
ValidationError: 1 validation error for Address
zip_code
  Field required [type=missing, input_value={'street': '123 Main', 'city': 'NYC', 'state': 'NY'}, input_type=dict]
```

**Error 3: Wrong type in a list**

```python
company = Company(
    name="TechCorp",
    founded_year=2015,
    employees=[
        Person(name="Alice", age=30, email="alice@company.com"),
        "Bob Smith",  # ❌ String instead of Person object
    ],
    headquarters=Address(...)
)
```

**Error message:**
```
ValidationError: 1 validation error for Company
employees.1
  Input should be a valid dictionary or instance of Person [type=model_type, input_value='Bob Smith', input_type=str]
```

Notice how Pydantic tells you exactly which item in the list failed (`.1` means index 1).

### Explain the Fix

**For Type Mismatches:**
- Check the model definition to see what type is expected
- Ensure you're passing the correct Python type (int not str, etc.)
- If converting from strings, parse them first: `age=int("30")`

**For Missing Fields:**
- Check if the field is Optional - if not, it MUST be provided
- Look at the model definition to see all required fields
- If the data might be missing, make the field Optional in your model

**For List Type Errors:**
- Ensure every item in the list matches the expected type
- The error message shows the index of the problematic item
- Lists are homogeneous - all items must be the same type

### Type Safety Gotchas

1. **Optional vs Default Values**: `Optional[str]` means it can be None. `str = "default"` means it has a default. `Optional[str] = None` means both.

2. **List vs List[T]**: Always specify what's in the list: `List[Person]` not just `List`. Without the type parameter, you lose all type safety for list items.

3. **Nested Validation Order**: Validation happens from the inside out. If a nested model fails validation, the parent model won't even try to validate.

4. **Mutable Defaults**: Never use `field: List[str] = []` - use `field: List[str] = Field(default_factory=list)` instead to avoid shared mutable default issues.

---

## 🎯 Next Steps

You now understand the foundation of complex data structures! You know:
- ✅ How to create nested Pydantic models
- ✅ How to use lists and collections with type safety
- ✅ How to work with optional fields
- ✅ How to use enums for controlled values
- ✅ How validation cascades through nested structures

In the next lesson, we'll dive deeper into **Nested Pydantic Models** and learn advanced patterns for creating maintainable, reusable model hierarchies.

**Ready for Lesson 2, or would you like to practice building some complex structures first?** 🚀
