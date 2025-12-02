# Lesson 10: Extracting Structured Data from Text

## A. Concept Overview

### What & Why
**Now we bring everything together: using Pydantic AI with Gemini to extract structured, validated data from unstructured text.** All the models you've built—with nested structures, optional fields, enums, validators, and constraints—now become the target schema that Gemini fills with extracted data. Pydantic AI handles the prompt engineering, API calls, and validation automatically.

### Analogy
Think of structured data extraction like a smart form-filling system:
- **Unstructured text** is like a conversation where someone tells you information
- **Your Pydantic models** are like a detailed form with specific fields
- **Gemini** is like an intelligent assistant listening to the conversation
- **Pydantic AI** is the system that shows the form to the assistant and validates each answer as it's filled in

When someone says "I'm Alice Johnson, 32 years old, living in San Francisco," Gemini extracts this into your `Person` model: `name="Alice Johnson", age=32, location="San Francisco"`, and Pydantic validates everything immediately.

### Type Safety Benefit
Pydantic AI extraction provides **guaranteed structure**:
- Define your schema with Pydantic models—that's your contract
- Gemini extracts data matching your schema
- Pydantic validates every field automatically
- Invalid data is rejected with clear errors
- Your application receives type-safe, validated objects
- No manual parsing, no null checks, no type casting

---

## B. Code Implementation

### File Structure
```
data_extraction_pipeline/
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── extraction_models.py  # New: Extraction schemas
│   ├── agents/
│   │   ├── __init__.py
│   │   └── extractor.py          # New: Pydantic AI agents
│   └── examples/
│       ├── __init__.py
│       └── extraction_demo.py     # New: This lesson
├── requirements.txt
└── .env  # For API keys
```

### Complete Code Implementation

**File: `requirements.txt`** (Updated)
```
pydantic>=2.0.0
pydantic-ai>=0.0.1
python-dotenv>=1.0.0
```

**File: `.env`**
```
GEMINI_API_KEY=your_gemini_api_key_here
```

**File: `src/models/extraction_models.py`**

```python
"""Models for structured data extraction."""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from enum import Enum


class EntityType(str, Enum):
    """Entity types we can extract."""
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    DATE = "date"
    MONEY = "money"
    PRODUCT = "product"


class ExtractedPerson(BaseModel):
    """Extracted person information."""
    full_name: str = Field(..., description="Person's full name")
    first_name: Optional[str] = Field(None, description="First name")
    last_name: Optional[str] = Field(None, description="Last name")
    age: Optional[int] = Field(None, ge=0, le=150, description="Age in years")
    occupation: Optional[str] = Field(None, description="Job title or occupation")
    email: Optional[str] = Field(None, description="Email address")
    phone: Optional[str] = Field(None, description="Phone number")


class ExtractedOrganization(BaseModel):
    """Extracted organization information."""
    name: str = Field(..., description="Organization name")
    industry: Optional[str] = Field(None, description="Industry or sector")
    location: Optional[str] = Field(None, description="Headquarters location")
    founded_year: Optional[int] = Field(None, ge=1800, le=2100, description="Year founded")
    employee_count: Optional[int] = Field(None, ge=0, description="Number of employees")


class ExtractedLocation(BaseModel):
    """Extracted location information."""
    name: str = Field(..., description="Location name")
    city: Optional[str] = Field(None, description="City name")
    state: Optional[str] = Field(None, description="State or region")
    country: Optional[str] = Field(None, description="Country name")
    location_type: Optional[str] = Field(None, description="Type of location (city, building, landmark, etc)")


class ExtractedAmount(BaseModel):
    """Extracted monetary amount."""
    amount: float = Field(..., description="Numeric amount")
    currency: str = Field(..., description="Currency code (USD, EUR, etc)")
    context: Optional[str] = Field(None, description="Context (revenue, investment, price, etc)")


class SimpleExtraction(BaseModel):
    """Simple extraction: list of entities by type."""
    people: List[str] = Field(default_factory=list, description="Names of people mentioned")
    organizations: List[str] = Field(default_factory=list, description="Names of organizations mentioned")
    locations: List[str] = Field(default_factory=list, description="Names of locations mentioned")


class DetailedExtraction(BaseModel):
    """Detailed extraction: structured entity data."""
    people: List[ExtractedPerson] = Field(default_factory=list, description="Detailed person information")
    organizations: List[ExtractedOrganization] = Field(default_factory=list, description="Detailed organization information")
    locations: List[ExtractedLocation] = Field(default_factory=list, description="Detailed location information")
    amounts: List[ExtractedAmount] = Field(default_factory=list, description="Monetary amounts mentioned")


class ArticleSummary(BaseModel):
    """Extracted article summary and metadata."""
    title: str = Field(..., description="Article title")
    summary: str = Field(..., min_length=50, max_length=500, description="Brief summary")
    key_points: List[str] = Field(..., min_length=1, max_length=10, description="Key points or takeaways")
    people_mentioned: List[str] = Field(default_factory=list, description="People mentioned in article")
    companies_mentioned: List[str] = Field(default_factory=list, description="Companies mentioned")
    sentiment: str = Field(..., description="Overall sentiment (positive, negative, neutral)")


class ProductReview(BaseModel):
    """Extracted product review information."""
    product_name: str = Field(..., description="Name of the product")
    rating: int = Field(..., ge=1, le=5, description="Rating out of 5")
    pros: List[str] = Field(default_factory=list, description="Positive aspects mentioned")
    cons: List[str] = Field(default_factory=list, description="Negative aspects mentioned")
    reviewer_sentiment: str = Field(..., description="Overall sentiment (positive, negative, neutral, mixed)")
    would_recommend: bool = Field(..., description="Whether reviewer recommends the product")


class ContactExtraction(BaseModel):
    """Extracted contact information."""
    name: str = Field(..., description="Contact name")
    email: Optional[str] = Field(None, description="Email address")
    phone: Optional[str] = Field(None, description="Phone number")
    company: Optional[str] = Field(None, description="Company name")
    job_title: Optional[str] = Field(None, description="Job title")
    linkedin: Optional[str] = Field(None, description="LinkedIn profile URL")
```

**File: `src/agents/extractor.py`**

```python
"""Pydantic AI agents for data extraction."""

from pydantic_ai import Agent
from src.models.extraction_models import (
    SimpleExtraction,
    DetailedExtraction,
    ArticleSummary,
    ProductReview,
    ContactExtraction,
)
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables. Please set it in .env file.")


# ============================================================================
# SIMPLE EXTRACTION AGENT
# ============================================================================

simple_extractor = Agent(
    "gemini-1.5-flash",  # Fast, cost-effective model
    result_type=SimpleExtraction,
    system_prompt=(
        "You are an expert at extracting named entities from text. "
        "Extract all people, organizations, and locations mentioned. "
        "Return simple lists of entity names."
    ),
)


# ============================================================================
# DETAILED EXTRACTION AGENT
# ============================================================================

detailed_extractor = Agent(
    "gemini-1.5-pro",  # More capable model for detailed extraction
    result_type=DetailedExtraction,
    system_prompt=(
        "You are an expert at extracting and structuring entity information from text. "
        "Extract detailed information about people, organizations, locations, and monetary amounts. "
        "Include as much detail as available in the text. "
        "If information is not available, leave fields as null/empty."
    ),
)


# ============================================================================
# ARTICLE SUMMARY AGENT
# ============================================================================

article_summarizer = Agent(
    "gemini-1.5-flash",
    result_type=ArticleSummary,
    system_prompt=(
        "You are an expert at analyzing and summarizing articles. "
        "Extract the title, create a concise summary, identify key points, "
        "list people and companies mentioned, and determine the overall sentiment. "
        "Be objective and factual in your analysis."
    ),
)


# ============================================================================
# PRODUCT REVIEW AGENT
# ============================================================================

review_extractor = Agent(
    "gemini-1.5-flash",
    result_type=ProductReview,
    system_prompt=(
        "You are an expert at analyzing product reviews. "
        "Extract the product name, rating, pros, cons, sentiment, and recommendation. "
        "Be accurate and capture all positive and negative points mentioned."
    ),
)


# ============================================================================
# CONTACT EXTRACTION AGENT
# ============================================================================

contact_extractor = Agent(
    "gemini-1.5-flash",
    result_type=ContactExtraction,
    system_prompt=(
        "You are an expert at extracting contact information from text. "
        "Extract names, emails, phone numbers, companies, job titles, and LinkedIn URLs. "
        "Ensure email and phone formats are correct."
    ),
)
```

**File: `src/examples/extraction_demo.py`**

```python
"""Demonstration of structured data extraction with Pydantic AI and Gemini."""

import asyncio
from src.agents.extractor import (
    simple_extractor,
    detailed_extractor,
    article_summarizer,
    review_extractor,
    contact_extractor,
)


async def demo_simple_extraction():
    """Demonstrate simple entity extraction."""
    print("=" * 70)
    print("SIMPLE ENTITY EXTRACTION")
    print("=" * 70)
    
    text = """
    Apple Inc. announced that Tim Cook will visit New York and London next month
    to meet with Microsoft executives including Satya Nadella. The meetings will
    take place at their Seattle headquarters.
    """
    
    print(f"Text: {text.strip()[:100]}...")
    print(f"\nExtracting entities...")
    
    result = await simple_extractor.run(text)
    
    print(f"\n✅ Extraction complete!")
    print(f"\nPeople: {result.data.people}")
    print(f"Organizations: {result.data.organizations}")
    print(f"Locations: {result.data.locations}")


async def demo_detailed_extraction():
    """Demonstrate detailed entity extraction."""
    print("\n" + "=" * 70)
    print("DETAILED ENTITY EXTRACTION")
    print("=" * 70)
    
    text = """
    Sarah Johnson, 35, is the Chief Technology Officer at TechCorp, a software
    company based in San Francisco with over 500 employees. The company, founded
    in 2015, recently announced a $50 million Series B funding round. Sarah can
    be reached at sarah.johnson@techcorp.com or +1-555-0100.
    """
    
    print(f"Text: {text.strip()}")
    print(f"\nExtracting detailed information...")
    
    result = await detailed_extractor.run(text)
    
    print(f"\n✅ Extraction complete!")
    
    if result.data.people:
        print(f"\nPeople:")
        for person in result.data.people:
            print(f"  Name: {person.full_name}")
            print(f"  Age: {person.age}")
            print(f"  Occupation: {person.occupation}")
            print(f"  Email: {person.email}")
            print(f"  Phone: {person.phone}")
    
    if result.data.organizations:
        print(f"\nOrganizations:")
        for org in result.data.organizations:
            print(f"  Name: {org.name}")
            print(f"  Industry: {org.industry}")
            print(f"  Location: {org.location}")
            print(f"  Founded: {org.founded_year}")
            print(f"  Employees: {org.employee_count}")
    
    if result.data.amounts:
        print(f"\nAmounts:")
        for amount in result.data.amounts:
            print(f"  {amount.currency} ${amount.amount:,.2f} ({amount.context})")


async def demo_article_summary():
    """Demonstrate article summarization."""
    print("\n" + "=" * 70)
    print("ARTICLE SUMMARIZATION")
    print("=" * 70)
    
    article = """
    # AI Revolution in Healthcare
    
    Artificial intelligence is transforming healthcare at an unprecedented pace.
    Dr. Emily Chen, Chief of AI Innovation at Stanford Medical Center, reports
    that AI-powered diagnostic tools have improved detection rates by 30% in the
    past year alone.
    
    Major technology companies including Google, Microsoft, and IBM are investing
    billions in healthcare AI. Google's DeepMind has developed algorithms that can
    detect eye diseases with 94% accuracy, matching expert ophthalmologists.
    
    "We're seeing AI become an indispensable partner to clinicians," says Dr. Chen.
    "The technology doesn't replace doctors but augments their capabilities,
    allowing them to focus on patient care while AI handles routine analysis."
    
    Hospitals across the United States are implementing AI systems for various
    applications, from predicting patient deterioration to optimizing treatment plans.
    Early results are overwhelmingly positive, with improved patient outcomes and
    reduced healthcare costs.
    """
    
    print(f"Article length: {len(article.split())} words")
    print(f"\nSummarizing...")
    
    result = await article_summarizer.run(article)
    
    print(f"\n✅ Summary complete!")
    print(f"\nTitle: {result.data.title}")
    print(f"\nSummary: {result.data.summary}")
    print(f"\nKey Points:")
    for i, point in enumerate(result.data.key_points, 1):
        print(f"  {i}. {point}")
    print(f"\nPeople: {result.data.people_mentioned}")
    print(f"Companies: {result.data.companies_mentioned}")
    print(f"Sentiment: {result.data.sentiment}")


async def demo_review_extraction():
    """Demonstrate product review extraction."""
    print("\n" + "=" * 70)
    print("PRODUCT REVIEW EXTRACTION")
    print("=" * 70)
    
    review = """
    I've been using the XYZ Wireless Headphones for three months now, and I'm
    generally impressed. The sound quality is excellent, with deep bass and clear
    highs. Battery life is outstanding - easily lasting 30+ hours on a single charge.
    
    However, there are some downsides. The ear cups are a bit too tight for long
    listening sessions, causing discomfort after 2-3 hours. Also, the Bluetooth
    connection occasionally drops when I'm far from my device.
    
    Overall, I'd rate these 4 out of 5 stars. Despite the comfort issues, the sound
    quality and battery life make these worth recommending, especially at the current
    price point.
    """
    
    print(f"Review: {review.strip()[:150]}...")
    print(f"\nExtracting review data...")
    
    result = await review_extractor.run(review)
    
    print(f"\n✅ Extraction complete!")
    print(f"\nProduct: {result.data.product_name}")
    print(f"Rating: {result.data.rating}/5 stars")
    print(f"\nPros:")
    for pro in result.data.pros:
        print(f"  + {pro}")
    print(f"\nCons:")
    for con in result.data.cons:
        print(f"  - {con}")
    print(f"\nSentiment: {result.data.reviewer_sentiment}")
    print(f"Recommends: {'Yes' if result.data.would_recommend else 'No'}")


async def demo_contact_extraction():
    """Demonstrate contact information extraction."""
    print("\n" + "=" * 70)
    print("CONTACT INFORMATION EXTRACTION")
    print("=" * 70)
    
    text = """
    Hi, I'm Alice Johnson, Senior Software Engineer at DataTech Solutions.
    You can reach me at alice.johnson@datatech.com or call my office at
    +1-555-0199. Feel free to connect with me on LinkedIn at
    linkedin.com/in/alicejohnson.
    """
    
    print(f"Text: {text.strip()}")
    print(f"\nExtracting contact information...")
    
    result = await contact_extractor.run(text)
    
    print(f"\n✅ Extraction complete!")
    print(f"\nName: {result.data.name}")
    print(f"Email: {result.data.email}")
    print(f"Phone: {result.data.phone}")
    print(f"Company: {result.data.company}")
    print(f"Title: {result.data.job_title}")
    print(f"LinkedIn: {result.data.linkedin}")


async def main():
    """Run all demonstrations."""
    print("\n🎯 STRUCTURED DATA EXTRACTION DEMONSTRATION\n")
    print("Using Pydantic AI + Google Gemini\n")
    
    try:
        await demo_simple_extraction()
        await demo_detailed_extraction()
        await demo_article_summary()
        await demo_review_extraction()
        await demo_contact_extraction()
        
        print("\n" + "=" * 70)
        print("✅ All demonstrations completed!")
        print("=" * 70)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have:")
        print("1. Set GEMINI_API_KEY in .env file")
        print("2. Installed requirements: pip install -r requirements.txt")


if __name__ == "__main__":
    asyncio.run(main())
```

### Line-by-Line Explanation

**Agent Configuration:**

```python
agent = Agent(
    "gemini-1.5-flash",  # Model selection
    result_type=YourPydanticModel,  # Output schema
    system_prompt="Instructions for the AI"  # Behavior guidance
)
```

**Key Concepts:**

1. **`result_type`**: Pydantic model defining the structure Gemini must return
2. **`system_prompt`**: Instructions that guide Gemini's extraction behavior
3. **`agent.run(text)`**: Extracts data from text, validates against `result_type`
4. **`result.data`**: The validated Pydantic model instance

**Gemini Model Selection:**

- **gemini-1.5-flash**: Fast, cheap, great for simple extractions
- **gemini-1.5-pro**: More capable, better for complex/detailed extractions

### The "Why" Behind the Pattern

**Automatic Validation:**
Gemini returns JSON that Pydantic validates. Invalid data raises clear errors before reaching your code.

**Type Safety:**
`result.data` is a validated Pydantic model. Your IDE knows all fields and types.

**No Manual Parsing:**
No regex, no string manipulation, no error-prone parsing. Define the schema, get validated objects.

**Iterative Refinement:**
Start with simple models, run extractions, see what Gemini returns, refine your models.

---

## C. Test & Apply

### How to Test It

**Step 1: Install dependencies**
```bash
cd data_extraction_pipeline
pip install pydantic-ai python-dotenv google-generativeai
```

**Step 2: Get Gemini API key**
- Go to https://makersuite.google.com/app/apikey
- Create an API key
- Copy it

**Step 3: Create .env file**
```bash
echo "GEMINI_API_KEY=your_actual_api_key_here" > .env
```

**Step 4: Create the files and copy code**

**Step 5: Run demonstration**
```bash
python -m src.examples.extraction_demo
```

### Expected Result

You'll see:
- Simple entity lists extracted
- Detailed structured entity data
- Article summaries with key points
- Product review analysis
- Contact information extracted

All output is validated Pydantic models!

---

## D. Common Stumbling Blocks

### Proactive Debugging

**Mistake 1: Overly complex models for first extraction**

Start simple, iterate:
```python
# ❌ First attempt: too complex
class Person(BaseModel):
    full_name: str
    first_name: str
    middle_name: Optional[str]
    last_name: str
    age: int
    date_of_birth: date
    # ... 20 more fields

# ✅ First attempt: simple
class Person(BaseModel):
    name: str
    age: Optional[int] = None

# Then refine based on results
```

**Mistake 2: Not using Optional for extracted fields**

Gemini can't always find everything:
```python
# ❌ All required
class Person(BaseModel):
    name: str
    age: int  # What if age not mentioned?
    email: str  # What if email not mentioned?

# ✅ Only name required
class Person(BaseModel):
    name: str
    age: Optional[int] = None
    email: Optional[str] = None
```

**Mistake 3: Vague field descriptions**

```python
# ❌ Vague
name: str = Field(..., description="name")

# ✅ Clear
name: str = Field(..., description="Full name of the person mentioned in the text")
```

### Type Safety Gotchas

1. **Async/await**: Pydantic AI agents are async—use `await agent.run(text)`
2. **API key**: Must be set in environment variables
3. **Rate limits**: Gemini has rate limits—handle them in production
4. **Cost**: Track API usage, especially with large texts and expensive models
5. **Field descriptions**: Help Gemini understand what to extract

---

## 🎯 Next Steps

Congratulations! You're now extracting structured data with Pydantic AI and Gemini! You've learned:
- ✅ How to configure Pydantic AI agents
- ✅ How to use Gemini models for extraction
- ✅ How to define extraction schemas
- ✅ How to validate extracted data automatically
- ✅ How to iterate and refine your models

In the next lesson, we'll explore **Multi-Entity Extraction**—handling multiple entities of different types in a single pass.

**Ready for Lesson 11?** 🚀
