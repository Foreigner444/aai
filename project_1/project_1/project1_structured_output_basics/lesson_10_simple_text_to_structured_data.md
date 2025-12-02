# 📘 Lesson 10: Simple Text to Structured Data

Now let's explore various text-to-structured-data transformations - the bread and butter of Pydantic AI! 🍞

---

## A. Concept Overview

### What & Why

**Text-to-structured-data** is the process of extracting organized information from unstructured text. This is one of the most common and valuable AI applications because:

- Humans write in free-form text (emails, documents, chat)
- Computers need structured data (databases, APIs, forms)
- AI bridges this gap perfectly

With Pydantic AI, you get **guaranteed** structured output, not best-effort parsing.

### The Analogy 📋

Think of this like a medical transcriptionist:

- **Patient says:** "I've had this headache for three days, and I feel nauseous. I took some aspirin yesterday but it didn't help."
- **Structured record:**
  ```
  Symptoms: ["headache", "nausea"]
  Duration: 3 days
  Medications: [{"name": "aspirin", "timing": "yesterday", "effective": false}]
  ```

Pydantic AI is your AI transcriptionist that always outputs in the exact format you need.

### Type Safety Benefit

Structured data extraction ensures:
- Consistent data format for downstream processing
- No manual parsing or regex needed
- Validation catches extraction errors
- Perfect integration with databases and APIs

---

## B. Code Implementation

### File Structure
```
pydantic-ai-project/
├── agents/
│   └── extractors/
│       ├── __init__.py
│       ├── contact_extractor.py
│       ├── receipt_extractor.py
│       └── review_extractor.py
└── ...
```

### Contact Information Extractor

```python
"""
Extract contact information from text.
"""
from pydantic import BaseModel, Field, EmailStr
from pydantic_ai import Agent
from typing import Optional


class ContactInfo(BaseModel):
    """Extracted contact information."""
    name: str = Field(description="Full name of the person")
    email: Optional[str] = Field(default=None, description="Email address")
    phone: Optional[str] = Field(default=None, description="Phone number")
    company: Optional[str] = Field(default=None, description="Company or organization")
    role: Optional[str] = Field(default=None, description="Job title or role")


contact_extractor = Agent(
    'gemini-1.5-flash',
    result_type=ContactInfo,
    system_prompt=(
        "Extract contact information from the given text. "
        "Look for names, email addresses, phone numbers, companies, and job titles. "
        "Only include information that is explicitly mentioned or clearly implied."
    )
)


if __name__ == "__main__":
    test_texts = [
        "Hi, I'm Sarah Johnson from TechCorp. You can reach me at sarah.j@techcorp.com or 555-123-4567.",
        "Contact Dr. Michael Chen, Head of Research, at m.chen@university.edu",
        "For inquiries, email support@example.com or call our CEO John Williams at (800) 555-0199",
    ]
    
    for text in test_texts:
        result = contact_extractor.run_sync(text)
        print(f"\nInput: {text[:60]}...")
        print(f"  Name: {result.data.name}")
        print(f"  Email: {result.data.email}")
        print(f"  Phone: {result.data.phone}")
        print(f"  Company: {result.data.company}")
        print(f"  Role: {result.data.role}")
```

### Receipt/Transaction Extractor

```python
"""
Extract transaction details from receipt text.
"""
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from typing import Optional
from datetime import date


class TransactionItem(BaseModel):
    """A single item in a transaction."""
    description: str = Field(description="Item name or description")
    quantity: int = Field(ge=1, default=1, description="Number of items")
    unit_price: float = Field(ge=0, description="Price per unit")
    total: float = Field(ge=0, description="Total price for this line item")


class Receipt(BaseModel):
    """Extracted receipt information."""
    vendor: str = Field(description="Store or vendor name")
    transaction_date: Optional[date] = Field(default=None, description="Date of purchase")
    items: list[TransactionItem] = Field(description="List of purchased items")
    subtotal: float = Field(ge=0, description="Subtotal before tax")
    tax: float = Field(ge=0, default=0, description="Tax amount")
    total: float = Field(ge=0, description="Final total including tax")
    payment_method: Optional[str] = Field(default=None, description="How payment was made")


receipt_extractor = Agent(
    'gemini-1.5-flash',
    result_type=Receipt,
    system_prompt=(
        "Extract receipt information from the given text. "
        "Identify the vendor, date, individual items with prices, subtotal, tax, and total. "
        "Calculate totals if not explicitly stated."
    )
)


if __name__ == "__main__":
    receipt_text = """
    COFFEE SHOP EXPRESS
    Date: 03/15/2024
    
    Latte (Large)         x2    $5.50    $11.00
    Croissant             x1    $3.25    $3.25
    Blueberry Muffin      x1    $3.50    $3.50
    
    Subtotal:                           $17.75
    Tax (8%):                           $1.42
    Total:                              $19.17
    
    Paid with: Credit Card
    """
    
    result = receipt_extractor.run_sync(receipt_text)
    
    print(f"Vendor: {result.data.vendor}")
    print(f"Date: {result.data.transaction_date}")
    print(f"\nItems:")
    for item in result.data.items:
        print(f"  - {item.description}: {item.quantity} x ${item.unit_price} = ${item.total}")
    print(f"\nSubtotal: ${result.data.subtotal}")
    print(f"Tax: ${result.data.tax}")
    print(f"Total: ${result.data.total}")
    print(f"Payment: {result.data.payment_method}")
```

### Product Review Analyzer

```python
"""
Extract structured insights from product reviews.
"""
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from typing import Literal, Optional


class ReviewAnalysis(BaseModel):
    """Structured analysis of a product review."""
    product_name: str = Field(description="Name of the product being reviewed")
    sentiment: Literal["positive", "negative", "mixed", "neutral"] = Field(
        description="Overall sentiment of the review"
    )
    rating_mentioned: Optional[float] = Field(
        default=None, ge=0, le=5,
        description="Explicit rating if mentioned (out of 5)"
    )
    pros: list[str] = Field(description="Positive aspects mentioned")
    cons: list[str] = Field(description="Negative aspects mentioned")
    would_recommend: Optional[bool] = Field(
        default=None,
        description="Whether the reviewer would recommend the product"
    )
    key_quote: str = Field(description="Most representative quote from the review")


review_analyzer = Agent(
    'gemini-1.5-flash',
    result_type=ReviewAnalysis,
    system_prompt=(
        "Analyze the given product review and extract structured insights. "
        "Identify pros and cons mentioned, overall sentiment, and key quotes. "
        "Be objective in your analysis."
    )
)


if __name__ == "__main__":
    reviews = [
        """
        I absolutely love this laptop! The battery lasts all day, the screen is 
        gorgeous, and it's so lightweight. My only complaint is that it only has 
        2 USB ports. 4.5 stars, would definitely recommend to anyone!
        """,
        """
        Disappointed with this purchase. The Bluetooth headphones have great sound 
        quality but the comfort is terrible - my ears hurt after 30 minutes. Also, 
        the battery only lasts 3 hours, not the 8 hours advertised. Returning these.
        """,
        """
        Decent coffee maker for the price. Makes good coffee, easy to clean. 
        However, it's quite loud and the carafe could be bigger. Overall, it does 
        the job but nothing special. 3 out of 5 stars.
        """
    ]
    
    for review in reviews:
        result = review_analyzer.run_sync(review)
        analysis = result.data
        
        print(f"\n{'='*60}")
        print(f"Product: {analysis.product_name}")
        print(f"Sentiment: {analysis.sentiment}")
        print(f"Rating: {analysis.rating_mentioned or 'Not mentioned'}")
        print(f"Pros: {', '.join(analysis.pros)}")
        print(f"Cons: {', '.join(analysis.cons)}")
        print(f"Would Recommend: {analysis.would_recommend}")
        print(f"Key Quote: \"{analysis.key_quote}\"")
```

### Address Parser

```python
"""
Parse addresses from free-form text.
"""
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from typing import Optional


class ParsedAddress(BaseModel):
    """A parsed street address."""
    street_number: Optional[str] = Field(default=None, description="House/building number")
    street_name: str = Field(description="Street name")
    unit: Optional[str] = Field(default=None, description="Apartment/suite number")
    city: str = Field(description="City name")
    state: str = Field(description="State/province")
    postal_code: str = Field(description="ZIP or postal code")
    country: str = Field(default="USA", description="Country")


address_parser = Agent(
    'gemini-1.5-flash',
    result_type=ParsedAddress,
    system_prompt=(
        "Parse the given address into its components. "
        "Handle various address formats including international addresses. "
        "If country is not specified, assume USA."
    )
)


if __name__ == "__main__":
    addresses = [
        "123 Main Street, Apt 4B, New York, NY 10001",
        "456 Oak Avenue, San Francisco, California 94102",
        "789 Maple Drive, Suite 100, Austin, TX 78701-1234",
    ]
    
    for addr in addresses:
        result = address_parser.run_sync(addr)
        parsed = result.data
        
        print(f"\nInput: {addr}")
        print(f"  Street: {parsed.street_number} {parsed.street_name}")
        if parsed.unit:
            print(f"  Unit: {parsed.unit}")
        print(f"  City: {parsed.city}")
        print(f"  State: {parsed.state}")
        print(f"  Postal Code: {parsed.postal_code}")
        print(f"  Country: {parsed.country}")
```

### Multi-Entity Extraction

```python
"""
Extract multiple entities of different types from text.
"""
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from typing import Optional


class Person(BaseModel):
    """A person mentioned in the text."""
    name: str
    role: Optional[str] = None


class Organization(BaseModel):
    """An organization mentioned in the text."""
    name: str
    org_type: Optional[str] = None


class Location(BaseModel):
    """A location mentioned in the text."""
    name: str
    location_type: str = Field(description="city, country, building, etc.")


class ExtractedEntities(BaseModel):
    """All entities extracted from text."""
    people: list[Person] = Field(default_factory=list, description="People mentioned")
    organizations: list[Organization] = Field(default_factory=list, description="Organizations mentioned")
    locations: list[Location] = Field(default_factory=list, description="Locations mentioned")
    dates: list[str] = Field(default_factory=list, description="Dates mentioned")
    monetary_values: list[str] = Field(default_factory=list, description="Money amounts mentioned")


entity_extractor = Agent(
    'gemini-1.5-flash',
    result_type=ExtractedEntities,
    system_prompt=(
        "Extract all named entities from the given text. "
        "Identify people (with their roles if mentioned), organizations, "
        "locations, dates, and monetary values."
    )
)


if __name__ == "__main__":
    text = """
    Apple CEO Tim Cook announced at WWDC 2024 in San Jose that the company 
    will invest $1 billion in AI research. The announcement was made alongside 
    Google's Sundar Pichai at a joint conference on June 15th. Both companies 
    plan to establish new research centers in London and Tokyo.
    """
    
    result = entity_extractor.run_sync(text)
    entities = result.data
    
    print("Extracted Entities:")
    print(f"\nPeople:")
    for p in entities.people:
        print(f"  - {p.name}" + (f" ({p.role})" if p.role else ""))
    
    print(f"\nOrganizations:")
    for o in entities.organizations:
        print(f"  - {o.name}" + (f" ({o.org_type})" if o.org_type else ""))
    
    print(f"\nLocations:")
    for l in entities.locations:
        print(f"  - {l.name} ({l.location_type})")
    
    print(f"\nDates: {', '.join(entities.dates)}")
    print(f"Monetary Values: {', '.join(entities.monetary_values)}")
```

---

## C. Test & Apply

### Designing Good Extraction Models

**Tips for effective extraction:**

1. **Be specific with field descriptions:**
   ```python
   # ❌ Vague
   value: float
   
   # ✅ Specific
   value: float = Field(description="Price in USD, excluding tax")
   ```

2. **Use Optional for uncertain fields:**
   ```python
   # The AI might not find this in every text
   middle_name: Optional[str] = None
   ```

3. **Use Literal for constrained choices:**
   ```python
   status: Literal["open", "closed", "pending"]
   ```

4. **Use lists for multiple items:**
   ```python
   skills: list[str] = Field(description="Technical skills mentioned")
   ```

### Common Extraction Patterns

| Data Type | Model Pattern |
|-----------|---------------|
| Optional info | `Optional[str] = None` |
| Constrained values | `Literal["a", "b", "c"]` |
| Multiple items | `list[str]` or `list[SubModel]` |
| Numeric ranges | `Field(ge=0, le=100)` |
| Dates | `Optional[date]` |
| Nested data | Separate Pydantic model |

---

## D. Common Stumbling Blocks

### "Some fields are always None"

The AI might not find the information. Make sure:
1. The text actually contains the information
2. Your field description is clear
3. Consider making the field optional if it's not always present

### "Extracted values seem wrong"

Improve your system prompt:
```python
# ❌ Too vague
system_prompt="Extract information from text."

# ✅ More specific
system_prompt=(
    "Extract contact information from business correspondence. "
    "Look for: full names, email addresses, phone numbers. "
    "Phone numbers should include area codes if present."
)
```

### "List fields are empty"

Make sure your model handles the empty case:
```python
# Use default_factory for mutable defaults
items: list[Item] = Field(default_factory=list)
```

### "Dates are in wrong format"

The AI might return dates inconsistently. Use type coercion:
```python
from datetime import date

class Event(BaseModel):
    event_date: date  # Pydantic will try to parse various formats
```

---

## ✅ Lesson 10 Complete!

### Key Takeaways

1. **Text-to-structured-data** is a core Pydantic AI use case
2. **Field descriptions** help the AI understand what to extract
3. **Optional fields** handle missing information gracefully
4. **Nested models** represent complex, hierarchical data
5. **Lists** capture multiple instances of the same type
6. **Clear system prompts** improve extraction accuracy

### What's Next?

In Lesson 11, we'll dive deeper into `run_sync()` and understand all the options for running agents!

---

*Extracting data like a pro! Let's learn about running agents in Lesson 11!* 🚀
