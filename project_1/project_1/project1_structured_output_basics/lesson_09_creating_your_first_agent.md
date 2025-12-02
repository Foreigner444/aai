# 📘 Lesson 9: Creating Your First Agent

This is it - you're about to create your first Pydantic AI Agent! 🎉 Everything you've learned comes together here.

---

## A. Concept Overview

### What & Why

A **Pydantic AI Agent** is the core abstraction that connects:
- Your **Pydantic model** (what data you want)
- A **Gemini model** (the AI that generates it)
- A **system prompt** (instructions for the AI)

The Agent handles all the complexity of:
- Sending your request to Gemini
- Instructing Gemini to return structured data
- Parsing and validating the response
- Converting to your Pydantic model

### The Analogy 🤖

Think of an Agent like a skilled translator:

- **You speak Python** (your code)
- **Gemini speaks natural language/JSON** (the AI)
- **The Agent translates between you** perfectly

You say "Get me a Person with name and age" in Python, the Agent explains this to Gemini, and returns a perfect `Person` object.

### Type Safety Benefit

With Pydantic AI Agents:
- Return types are enforced at the Agent level
- Invalid responses raise errors immediately
- IDE autocomplete works for all results
- Type checkers verify your code

---

## B. Code Implementation

### File Structure
```
pydantic-ai-project/
├── models/
│   ├── __init__.py
│   └── basic_models.py
├── agents/
│   ├── __init__.py
│   └── first_agent.py    # Your first agent!
├── config.py
├── .env
└── ...
```

Create the agents directory:
```bash
mkdir agents
touch agents/__init__.py
```

### Your First Agent

Create `agents/first_agent.py`:

```python
"""
Your first Pydantic AI Agent!
This agent extracts person information from text.
"""
from pydantic import BaseModel, Field
from pydantic_ai import Agent


# Step 1: Define the output model
class PersonInfo(BaseModel):
    """Information about a person extracted from text."""
    name: str = Field(description="The person's full name")
    age: int = Field(ge=0, le=150, description="The person's age in years")
    occupation: str = Field(description="The person's job or profession")


# Step 2: Create the Agent
person_agent = Agent(
    'gemini-1.5-flash',              # The AI model to use
    result_type=PersonInfo,           # What the agent must return
    system_prompt=(
        "You are an expert at extracting person information from text. "
        "Extract the name, age, and occupation from the given text. "
        "If information is not explicitly stated, make a reasonable inference."
    )
)


# Step 3: Use the Agent
if __name__ == "__main__":
    # Test the agent with sample text
    test_text = "John Smith is a 35-year-old software engineer from San Francisco."
    
    # Run the agent synchronously
    result = person_agent.run_sync(test_text)
    
    # Access the validated data
    print("Extracted Person Information:")
    print(f"  Name: {result.data.name}")
    print(f"  Age: {result.data.age}")
    print(f"  Occupation: {result.data.occupation}")
    
    # The result object has more info
    print(f"\nFull result object: {result.data}")
```

Run it:
```bash
python agents/first_agent.py
```

**Expected output:**
```
Extracted Person Information:
  Name: John Smith
  Age: 35
  Occupation: software engineer

Full result object: name='John Smith' age=35 occupation='software engineer'
```

### Understanding the Agent

Let's break down each part:

```python
# PART 1: Import the essentials
from pydantic import BaseModel, Field
from pydantic_ai import Agent

# PART 2: Define your data contract
class PersonInfo(BaseModel):
    """This docstring helps the AI understand the model."""
    name: str = Field(description="The person's full name")
    # ^-- The description helps Gemini understand what to extract

# PART 3: Create the Agent
person_agent = Agent(
    'gemini-1.5-flash',     # Model: which Gemini variant to use
    result_type=PersonInfo,  # Contract: what shape must the output be
    system_prompt="..."      # Instructions: how should the AI behave
)

# PART 4: Run the Agent
result = person_agent.run_sync(user_input)
#         ^-- Sends request, gets response, validates, returns

# PART 5: Use the Result
result.data  # The validated PersonInfo object
result.data.name  # Access individual fields with autocomplete!
```

### Agent Configuration Options

```python
from pydantic_ai import Agent

agent = Agent(
    # Required: Which model to use
    'gemini-1.5-flash',
    
    # Required for structured output: The result type
    result_type=MyModel,
    
    # Optional: Instructions for the AI
    system_prompt="You are a helpful assistant...",
    
    # Optional: Retry configuration
    retries=2,  # How many times to retry on failure
)
```

### More Agent Examples

**Product Information Extractor:**

```python
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from typing import Optional


class ProductInfo(BaseModel):
    """Extracted product information."""
    name: str = Field(description="Product name")
    brand: Optional[str] = Field(default=None, description="Brand name if mentioned")
    price: float = Field(ge=0, description="Price in USD")
    category: str = Field(description="Product category")


product_agent = Agent(
    'gemini-1.5-flash',
    result_type=ProductInfo,
    system_prompt=(
        "Extract product information from the given text. "
        "Identify the product name, brand (if mentioned), price, and category."
    )
)

# Test it
result = product_agent.run_sync(
    "The new Apple MacBook Pro 16-inch costs $2,499 and is perfect for professionals."
)
print(f"Product: {result.data.name}")
print(f"Brand: {result.data.brand}")
print(f"Price: ${result.data.price}")
print(f"Category: {result.data.category}")
```

**Event Information Extractor:**

```python
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from datetime import date
from typing import Optional


class EventInfo(BaseModel):
    """Extracted event information."""
    title: str = Field(description="Event title or name")
    event_date: Optional[date] = Field(default=None, description="Date of the event")
    location: Optional[str] = Field(default=None, description="Event location")
    description: str = Field(description="Brief description of the event")


event_agent = Agent(
    'gemini-1.5-flash',
    result_type=EventInfo,
    system_prompt=(
        "Extract event information from the text. "
        "Identify the event name, date, location, and provide a brief description."
    )
)

# Test it
result = event_agent.run_sync(
    "Join us for the Annual Tech Conference 2024 on March 15th at the "
    "Convention Center. Learn about the latest in AI and cloud computing!"
)
print(f"Event: {result.data.title}")
print(f"Date: {result.data.event_date}")
print(f"Location: {result.data.location}")
print(f"Description: {result.data.description}")
```

### Agent with Custom Model from Config

```python
"""
Agent that uses model from configuration.
"""
import os
from pydantic import BaseModel
from pydantic_ai import Agent
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class SimpleResponse(BaseModel):
    """A simple response model."""
    answer: str
    confidence: float


# Get model from environment variable with default
model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# Create agent with configurable model
flexible_agent = Agent(
    model_name,
    result_type=SimpleResponse,
    system_prompt="Answer questions concisely and rate your confidence from 0-1."
)
```

---

## C. Test & Apply

### Running Your First Agent

1. **Make sure your environment is set up:**
   ```bash
   source venv/bin/activate
   ```

2. **Verify your API key is configured:**
   ```bash
   cat .env | grep GEMINI
   ```

3. **Run the agent:**
   ```bash
   python agents/first_agent.py
   ```

### Practice: Create Your Own Agent

Try creating an agent that extracts restaurant information:

```python
"""
Practice: Create a restaurant information extractor.
"""
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from typing import Literal, Optional


class RestaurantInfo(BaseModel):
    """Information about a restaurant."""
    name: str = Field(description="Restaurant name")
    cuisine: str = Field(description="Type of cuisine")
    price_range: Literal["$", "$$", "$$$", "$$$$"] = Field(
        description="Price range from $ (cheap) to $$$$ (expensive)"
    )
    rating: Optional[float] = Field(
        default=None, ge=0, le=5,
        description="Rating out of 5 stars"
    )


# Create your agent here
restaurant_agent = Agent(
    'gemini-1.5-flash',
    result_type=RestaurantInfo,
    system_prompt=(
        "Extract restaurant information from the text. "
        "Identify the restaurant name, cuisine type, price range, and rating if mentioned."
    )
)


# Test it
if __name__ == "__main__":
    test_texts = [
        "Sakura Japanese Restaurant serves amazing sushi. It's a bit pricey but worth it. 4.5 stars!",
        "Check out Mario's Italian Bistro for affordable pasta dishes. Great value for money!",
        "The Golden Dragon is an upscale Chinese restaurant with a 5-star rating.",
    ]
    
    for text in test_texts:
        result = restaurant_agent.run_sync(text)
        print(f"\nInput: {text[:50]}...")
        print(f"  Name: {result.data.name}")
        print(f"  Cuisine: {result.data.cuisine}")
        print(f"  Price Range: {result.data.price_range}")
        print(f"  Rating: {result.data.rating}")
```

---

## D. Common Stumbling Blocks

### "Agent object has no attribute 'run_sync'"

Make sure you're importing from `pydantic_ai`, not `pydantic`:

```python
# ✅ Correct
from pydantic_ai import Agent

# ❌ Wrong - this would be a different Agent class
from some_other_package import Agent
```

### "result_type must be a Pydantic model"

Your result type must inherit from `BaseModel`:

```python
from pydantic import BaseModel

# ❌ Wrong - plain class
class MyResponse:
    message: str

# ✅ Correct - inherits from BaseModel
class MyResponse(BaseModel):
    message: str
```

### "API key not configured"

The agent can't find your API key. Ensure:

1. `.env` file exists with `GEMINI_API_KEY=...`
2. You've loaded dotenv before creating the agent:
   ```python
   from dotenv import load_dotenv
   load_dotenv()
   ```

### "The model returned invalid JSON"

This can happen occasionally. Solutions:

1. Make your system prompt clearer
2. Add field descriptions to help the AI
3. Use retries:
   ```python
   agent = Agent(..., retries=2)
   ```

### "Type of result.data is wrong in my IDE"

If your IDE isn't recognizing the type, you might need a type annotation:

```python
from pydantic_ai import Agent, RunResult

agent = Agent('gemini-1.5-flash', result_type=PersonInfo)
result: RunResult[PersonInfo] = agent.run_sync(text)
# Now result.data is typed as PersonInfo
```

---

## ✅ Lesson 9 Complete!

### Key Takeaways

1. **Agent** connects your model, Gemini, and instructions
2. **result_type** enforces the output structure
3. **system_prompt** guides the AI's behavior
4. **run_sync()** sends the request and returns validated results
5. **result.data** is your typed, validated Pydantic model

### Agent Creation Checklist

- [ ] Import `Agent` from `pydantic_ai`
- [ ] Define a Pydantic model for your output
- [ ] Choose your Gemini model (`gemini-1.5-flash`)
- [ ] Write a clear system prompt
- [ ] Create the Agent with all components
- [ ] Use `run_sync()` to get results

### What's Next?

In Lesson 10, we'll build more complex text-to-structured-data transformations!

---

*First agent working? Let's do more in Lesson 10!* 🚀
