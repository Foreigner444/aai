# 📘 Lesson 5: Your First Pydantic Model

Now for the exciting part - you're about to create your first Pydantic model! 🎉 This is the foundation of everything you'll build with Pydantic AI.

---

## A. Concept Overview

### What & Why

A **Pydantic model** is a Python class that defines the exact shape of your data. It specifies:
- What fields exist
- What type each field must be
- What values are valid

When you use this model with Pydantic AI, it becomes a "contract" that the AI must follow. If Gemini returns data that doesn't match your model, Pydantic catches the error immediately.

### The Analogy 📝

Think of a Pydantic model like a form template:

**A paper form might say:**
```
Name: _____________ (text)
Age: ______________ (number)
Email: ____________ (email format)
```

**A Pydantic model says the same thing in Python:**
```python
class Person(BaseModel):
    name: str           # text
    age: int            # number  
    email: EmailStr     # email format
```

If someone fills out the form wrong (puts text in the age field), the form is rejected. Same with Pydantic!

### Type Safety Benefit

Pydantic models give you:
- **IDE autocomplete** for every field
- **Type checking** with mypy
- **Automatic validation** of all data
- **Clear error messages** when something's wrong
- **Documentation** - the model IS the spec

---

## B. Code Implementation

### File Structure
```
pydantic-ai-project/
├── venv/
├── models/
│   └── basic_models.py    # Your first models!
├── .env
├── .gitignore
└── requirements.txt
```

Create the models directory:
```bash
mkdir models
touch models/__init__.py
```

### Your First Model

Create `models/basic_models.py`:

```python
"""
Your first Pydantic models!
These define the shape of data that AI will return.
"""
from pydantic import BaseModel


class Person(BaseModel):
    """A simple model representing a person."""
    name: str
    age: int


# Let's test it!
if __name__ == "__main__":
    # ✅ Valid data - this works!
    person1 = Person(name="Alice", age=30)
    print(f"Created: {person1}")
    print(f"Name: {person1.name}")
    print(f"Age: {person1.age}")
    
    # ✅ Pydantic converts types when possible
    person2 = Person(name="Bob", age="25")  # String "25" becomes int 25
    print(f"\nType conversion: age={person2.age}, type={type(person2.age)}")
    
    # ❌ Invalid data - this raises an error!
    try:
        person3 = Person(name="Charlie", age="not a number")
    except Exception as e:
        print(f"\n❌ Validation Error: {e}")
```

Run it:
```bash
python models/basic_models.py
```

**Expected output:**
```
Created: name='Alice' age=30
Name: Alice
Age: 30

Type conversion: age=25, type=<class 'int'>

❌ Validation Error: 1 validation error for Person
age
  Input should be a valid integer, unable to parse string as an integer 
  [type=int_parsing, input_value='not a number', input_type=str]
```

### Understanding the Code

```python
from pydantic import BaseModel  # Import Pydantic's base class

class Person(BaseModel):        # Inherit from BaseModel
    """Docstring explains the model."""
    name: str                   # Field 'name' must be a string
    age: int                    # Field 'age' must be an integer
```

**Key points:**
| Element | Purpose |
|---------|---------|
| `BaseModel` | Makes this class a Pydantic model with validation |
| `name: str` | Type annotation - tells Pydantic the expected type |
| `age: int` | Pydantic validates AND converts to this type if possible |

### More Examples

```python
from pydantic import BaseModel
from typing import Optional
from datetime import date


class Product(BaseModel):
    """A product with name, price, and optional description."""
    name: str
    price: float
    description: Optional[str] = None  # Optional with default None
    in_stock: bool = True              # Required bool with default


class Address(BaseModel):
    """A physical address."""
    street: str
    city: str
    state: str
    zip_code: str
    country: str = "USA"  # Default value


class Event(BaseModel):
    """An event with a date."""
    title: str
    event_date: date  # Pydantic handles date parsing!
    attendees: int


# Testing these models
if __name__ == "__main__":
    # Product with optional field omitted
    laptop = Product(name="MacBook Pro", price=1999.99)
    print(f"Product: {laptop}")
    print(f"Description: {laptop.description}")  # None
    print(f"In Stock: {laptop.in_stock}")  # True (default)
    
    # Product with all fields
    phone = Product(
        name="iPhone", 
        price=999.99, 
        description="Latest model",
        in_stock=False
    )
    print(f"\nProduct with all fields: {phone}")
    
    # Address with default country
    home = Address(
        street="123 Main St",
        city="Springfield",
        state="IL",
        zip_code="62701"
    )
    print(f"\nAddress: {home}")
    print(f"Country: {home.country}")  # USA (default)
    
    # Event with date parsing
    party = Event(
        title="Birthday Party",
        event_date="2024-06-15",  # String gets parsed to date!
        attendees=50
    )
    print(f"\nEvent: {party}")
    print(f"Date type: {type(party.event_date)}")  # <class 'datetime.date'>
```

### Model Methods and Features

Pydantic models come with useful built-in methods:

```python
from pydantic import BaseModel


class User(BaseModel):
    username: str
    email: str
    age: int


# Create a user
user = User(username="alice", email="alice@example.com", age=28)

# Convert to dictionary
user_dict = user.model_dump()
print(f"As dict: {user_dict}")
# {'username': 'alice', 'email': 'alice@example.com', 'age': 28}

# Convert to JSON string
user_json = user.model_dump_json()
print(f"As JSON: {user_json}")
# '{"username":"alice","email":"alice@example.com","age":28}'

# Create from dictionary
data = {"username": "bob", "email": "bob@example.com", "age": 35}
user2 = User.model_validate(data)
print(f"From dict: {user2}")

# Get the JSON schema (this is what Pydantic AI sends to Gemini!)
schema = User.model_json_schema()
print(f"JSON Schema: {schema}")
```

---

## C. Test & Apply

### Practice Exercise

Create a file called `practice_models.py` and try building these models:

```python
"""
Practice Exercise: Create these models yourself!
"""
from pydantic import BaseModel
from typing import Optional


# TODO: Create a Book model with:
# - title (string)
# - author (string)
# - pages (integer)
# - isbn (optional string)


# TODO: Create a Movie model with:
# - title (string)
# - director (string)
# - year (integer)
# - rating (float)
# - genre (string with default "Unknown")


# Test your models!
if __name__ == "__main__":
    # Test Book
    book = Book(title="Python Crash Course", author="Eric Matthes", pages=544)
    print(f"Book: {book}")
    
    # Test Movie
    movie = Movie(
        title="Inception",
        director="Christopher Nolan",
        year=2010,
        rating=8.8
    )
    print(f"Movie: {movie}")
```

### Solution

```python
from pydantic import BaseModel
from typing import Optional


class Book(BaseModel):
    """A book with optional ISBN."""
    title: str
    author: str
    pages: int
    isbn: Optional[str] = None


class Movie(BaseModel):
    """A movie with rating and genre."""
    title: str
    director: str
    year: int
    rating: float
    genre: str = "Unknown"


if __name__ == "__main__":
    book = Book(title="Python Crash Course", author="Eric Matthes", pages=544)
    print(f"Book: {book}")
    # Book: title='Python Crash Course' author='Eric Matthes' pages=544 isbn=None
    
    movie = Movie(
        title="Inception",
        director="Christopher Nolan",
        year=2010,
        rating=8.8
    )
    print(f"Movie: {movie}")
    # Movie: title='Inception' director='Christopher Nolan' year=2010 rating=8.8 genre='Unknown'
```

---

## D. Common Stumbling Blocks

### "TypeError: BaseModel.__init__() takes 1 positional argument"

You're using positional arguments instead of keyword arguments:

```python
# ❌ Wrong - positional arguments
person = Person("Alice", 30)

# ✅ Correct - keyword arguments
person = Person(name="Alice", age=30)
```

### "Field required" error

You forgot to provide a required field:

```python
class Person(BaseModel):
    name: str
    age: int

# ❌ Missing 'age' field
person = Person(name="Alice")
# ValidationError: Field required [type=missing, input_value={'name': 'Alice'}]

# ✅ Provide all required fields
person = Person(name="Alice", age=30)
```

### "I want a field to be optional"

Use `Optional` and provide a default:

```python
from typing import Optional

class Person(BaseModel):
    name: str
    age: int
    nickname: Optional[str] = None  # Optional with None default
```

### "Input should be a valid integer"

The value can't be converted to the expected type:

```python
# ✅ Works - "25" can be converted to 25
Person(name="Alice", age="25")  

# ❌ Fails - "twenty-five" cannot be converted
Person(name="Alice", age="twenty-five")  
# ValidationError: Input should be a valid integer
```

### "Extra fields are being ignored!"

By default, Pydantic ignores extra fields. To forbid them:

```python
from pydantic import BaseModel, ConfigDict


class StrictPerson(BaseModel):
    model_config = ConfigDict(extra='forbid')
    
    name: str
    age: int


# ❌ This now raises an error
StrictPerson(name="Alice", age=30, height=170)
# ValidationError: Extra inputs are not permitted
```

---

## ✅ Lesson 5 Complete!

### Key Takeaways

1. **Pydantic models** define the shape of your data
2. **Type annotations** (`name: str`) tell Pydantic what to expect
3. **Validation is automatic** - wrong types raise clear errors
4. **Type conversion** happens when possible (string "25" → int 25)
5. **Optional fields** use `Optional[type] = None`
6. **Defaults** make fields optional: `field: type = default_value`

### Your Model Checklist

- [ ] Understand `BaseModel` inheritance
- [ ] Can define required fields with types
- [ ] Can define optional fields with defaults
- [ ] Understand validation errors
- [ ] Can convert models to dict/JSON

### What's Next?

In Lesson 6, we'll explore more field types and validation options - integers with min/max, strings with patterns, and more!

---

*First model created! Let's explore more types in Lesson 6!* 🚀
