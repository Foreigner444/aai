# 📘 Lesson 2: Understanding Structured Outputs vs Raw Text

Now that you understand why type safety matters, let's dive deeper into what "structured outputs" actually means and see real examples of the chaos that raw text responses can cause.

---

## A. Concept Overview

### What & Why

**Structured output** means the AI returns data in a predictable, machine-readable format that maps directly to a Python object. Every field has a known name, a known type, and a known location.

**Raw text output** means the AI returns free-form text that might contain the information you need, but in an unpredictable format that requires parsing.

**Why does this matter for production applications?**

In production, your AI application might:
- Process thousands of requests per hour
- Feed data into databases
- Trigger automated workflows
- Make business decisions based on AI output

If even 1% of those responses are formatted unexpectedly, you have:
- Crashed pipelines
- Corrupted data
- Failed transactions
- Angry users

Structured outputs eliminate this entire class of problems.

### The Analogy 📋

Imagine you're collecting survey responses:

**Raw Text (Unstructured):**
People write whatever they want in a text box:
- "I'm John, 25, from NYC"
- "Name: Jane. I am thirty years old. Living in LA."
- "Bob here! Age is 42. City? Chicago I guess"
- "lol idk maybe sarah? 28? somewhere in texas"

Good luck putting that in a spreadsheet!

**Structured Output:**
People fill out a form with specific fields:
- Name: [___John___]
- Age: [___25___]
- City: [___NYC___]

Every response has the same fields in the same places. Easy to process!

### Type Safety Benefit

Structured outputs give you:

| Feature | Benefit |
|---------|---------|
| **Predictable schema** | You know exactly what fields exist |
| **Consistent types** | Numbers are numbers, strings are strings |
| **Complete data** | Required fields are always present |
| **Parseable format** | No regex or string manipulation needed |
| **Validatable** | You can verify correctness automatically |

---

## B. Code Implementation

Let's see real examples of raw text vs structured outputs.

### File Structure
```
project1_structured_output_basics/
└── examples/
    └── structured_vs_raw.py
```

### ❌ Raw Text: The Unpredictable Nightmare

When you ask Gemini a question without structure, you get creative but inconsistent responses:

```python
# Prompt: "What are the details of this product: iPhone 15 Pro Max 256GB Blue"

# Response 1 (Paragraph style):
"""
The iPhone 15 Pro Max is Apple's flagship smartphone. It comes with 256GB 
of storage and is available in a beautiful Blue Titanium color. The price 
typically ranges from $1,199 to $1,299 depending on the retailer.
"""

# Response 2 (List style):
"""
Product Details:
- Name: iPhone 15 Pro Max
- Storage: 256GB
- Color: Blue Titanium
- Price: $1,199
"""

# Response 3 (Conversational style):
"""
Sure! The iPhone 15 Pro Max you're asking about has 256GB storage. 
It's the blue variant, which Apple calls "Blue Titanium." You're 
looking at around $1,199 for this configuration.
"""

# Response 4 (JSON-ish but not quite):
"""
Here's the information:
{
  name: "iPhone 15 Pro Max",
  storage: "256GB",
  color: "Blue",
  price: "$1,199"
}
Note: Prices may vary by region.
"""

# Response 5 (Markdown table):
"""
| Attribute | Value |
|-----------|-------|
| Name | iPhone 15 Pro Max |
| Storage | 256GB |
| Color | Blue Titanium |
| Price | $1,199 |
"""
```

**The Problem:** All five responses contain the same information, but extracting it programmatically requires five different parsing strategies!

### ✅ Structured Output: Predictable Every Time

```python
from pydantic import BaseModel
from pydantic_ai import Agent

class ProductDetails(BaseModel):
    """Structured product information."""
    name: str
    storage_gb: int
    color: str
    price_usd: float

agent = Agent('gemini-1.5-flash', result_type=ProductDetails)

result = agent.run_sync(
    "What are the details of this product: iPhone 15 Pro Max 256GB Blue"
)

# EVERY response looks like this:
print(result.data)
# ProductDetails(name='iPhone 15 Pro Max', storage_gb=256, color='Blue Titanium', price_usd=1199.0)

# Access fields directly:
print(f"Product: {result.data.name}")
print(f"Storage: {result.data.storage_gb}GB")
print(f"Color: {result.data.color}")
print(f"Price: ${result.data.price_usd}")
```

**The Magic:** No matter how Gemini "thinks" about the answer internally, Pydantic AI ensures you always get a `ProductDetails` object with exactly these four fields.

### Real-World Comparison

Here's a side-by-side comparison for a common task - extracting contact information:

```python
# ============================================================
# TASK: Extract contact info from "Call John at 555-1234 or 
#       email john@example.com. He's the CTO."
# ============================================================

# ❌ RAW TEXT APPROACH
# ------------------------------------------------------------
import re

raw_response = """
Contact Information:
- Name: John
- Role: CTO  
- Phone: 555-1234
- Email: john@example.com
"""

# Now you have to parse this mess:
def parse_contact(text):
    # Hope the format doesn't change!
    name_match = re.search(r'Name:\s*(\w+)', text)
    role_match = re.search(r'Role:\s*(\w+)', text)
    phone_match = re.search(r'Phone:\s*([\d-]+)', text)
    email_match = re.search(r'Email:\s*(\S+)', text)
    
    return {
        'name': name_match.group(1) if name_match else None,
        'role': role_match.group(1) if role_match else None,
        'phone': phone_match.group(1) if phone_match else None,
        'email': email_match.group(1) if email_match else None,
    }

# What if the AI returns it differently next time? 💥


# ✅ STRUCTURED OUTPUT APPROACH
# ------------------------------------------------------------
from pydantic import BaseModel, EmailStr
from pydantic_ai import Agent

class ContactInfo(BaseModel):
    name: str
    role: str
    phone: str
    email: str

agent = Agent('gemini-1.5-flash', result_type=ContactInfo)
result = agent.run_sync(
    "Extract contact info from: Call John at 555-1234 or "
    "email john@example.com. He's the CTO."
)

# Clean, validated, guaranteed:
contact = result.data
print(contact.name)   # "John"
print(contact.role)   # "CTO"
print(contact.phone)  # "555-1234"
print(contact.email)  # "john@example.com"
```

### How Pydantic AI Achieves This

Behind the scenes, Pydantic AI:

1. **Converts your Pydantic model to a JSON schema**
2. **Includes the schema in the prompt** to Gemini
3. **Instructs Gemini** to respond in that exact format
4. **Parses Gemini's response** as JSON
5. **Validates against your model** using Pydantic
6. **Converts types automatically** (strings to ints, etc.)
7. **Raises clear errors** if anything doesn't match

```
Your Pydantic Model
       ↓
   JSON Schema
       ↓
Gemini (with schema instructions)
       ↓
   JSON Response
       ↓
 Pydantic Validation
       ↓
 Typed Python Object ✅
```

---

## C. Test & Apply

### Why This Matters for Your Projects

Consider these real scenarios:

**Scenario 1: E-commerce Product Extraction**
```python
# You're building a price comparison tool
# Raw text could give you prices as:
# "$1,199", "1199 USD", "1199.00", "around $1200", "~$1.2k"

# Structured output guarantees:
class Product(BaseModel):
    name: str
    price: float  # Always a number: 1199.0
    currency: str  # Always explicit: "USD"
```

**Scenario 2: Resume Parser**
```python
# Raw text might format skills as:
# "Python, JavaScript, SQL" or "- Python\n- JavaScript\n- SQL"
# or "Proficient in Python and JavaScript; familiar with SQL"

# Structured output guarantees:
class Resume(BaseModel):
    name: str
    skills: list[str]  # Always a list: ["Python", "JavaScript", "SQL"]
    years_experience: int
```

**Scenario 3: Sentiment Analysis**
```python
# Raw text might say:
# "The sentiment is positive" or "POSITIVE" or "positive sentiment detected"
# or "This text has a positive tone"

# Structured output guarantees:
from enum import Enum

class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

class SentimentResult(BaseModel):
    sentiment: Sentiment  # Always exactly one of three values
    confidence: float     # Always a number between 0 and 1
```

---

## D. Common Stumbling Blocks

### "Can't I just ask the AI to return JSON?"

You can, but there are problems:

```python
# ❌ Just asking for JSON
prompt = "Return this as JSON: John is 25 years old"

# Gemini might return:
'{"name": "John", "age": 25}'           # Valid JSON ✓
'{"name": "John", "age": "25"}'         # age is string, not int!
"```json\n{\"name\": \"John\"}\n```"    # Wrapped in markdown!
'Here is the JSON: {"name": "John"}'    # Extra text!
'{name: "John", age: 25}'               # Invalid JSON (unquoted keys)!
```

Pydantic AI handles all of these edge cases for you.

### "What about very complex data structures?"

Pydantic AI handles nested structures beautifully:

```python
from pydantic import BaseModel

class Address(BaseModel):
    street: str
    city: str
    country: str

class Company(BaseModel):
    name: str
    address: Address  # Nested model!

class Person(BaseModel):
    name: str
    age: int
    employer: Company  # Nested model with its own nested model!

# Pydantic AI will validate the ENTIRE structure,
# including all nested objects
```

### "Is there a performance cost to structured outputs?"

Minimal! The overhead is:
- Generating JSON schema from your model (done once)
- Adding schema to prompt (a few extra tokens)
- Parsing and validating response (microseconds with Pydantic)

The reliability benefits far outweigh this tiny cost.

---

## ✅ Lesson 2 Complete!

### Key Takeaways

1. **Raw text** is unpredictable and requires complex parsing
2. **Structured outputs** are consistent and machine-readable
3. **Pydantic AI** converts your models to JSON schemas automatically
4. **Validation** catches format errors before they reach your code
5. **Nested structures** are fully supported

### What's Next?

In Lesson 3, we'll set up your Python development environment so you can start running code!

---

*Let's get your environment ready in Lesson 3!* 🚀
