# 📘 Lesson 11: Running Agent with run_sync

Let's master the `run_sync()` method - your primary way to execute Pydantic AI agents! 🏃

---

## A. Concept Overview

### What & Why

`run_sync()` is the **synchronous** method to run an agent. "Synchronous" means:
- Your code waits for the AI response before continuing
- Simple to use and understand
- Perfect for scripts, simple APIs, and learning

Pydantic AI also offers `run()` (async) and `run_stream()` (streaming), but `run_sync()` is where you should start.

### The Analogy ☎️

Think of `run_sync()` like making a phone call:
- You dial (send request)
- You wait on the line (synchronous)
- You get the answer (response)
- Then you hang up and continue (next line of code)

Contrast with async (leaving a voicemail and getting called back) or streaming (getting the answer word by word).

### Type Safety Benefit

`run_sync()` returns a `RunResult` object with:
- `data`: Your validated Pydantic model
- Type information for IDE autocomplete
- Full type safety throughout

---

## B. Code Implementation

### File Structure
```
pydantic-ai-project/
├── agents/
│   └── run_examples.py    # Examples of running agents
└── ...
```

### Basic run_sync() Usage

```python
"""
Examples of using run_sync() to execute agents.
"""
from pydantic import BaseModel, Field
from pydantic_ai import Agent


class Summary(BaseModel):
    """A text summary."""
    title: str = Field(description="A short title for the content")
    summary: str = Field(description="2-3 sentence summary")
    key_points: list[str] = Field(description="Main takeaways")


summarizer = Agent(
    'gemini-1.5-flash',
    result_type=Summary,
    system_prompt="Summarize the given text concisely."
)


# Basic usage
text = """
Artificial intelligence has transformed how businesses operate. Companies 
are using AI for customer service, data analysis, and automation. The 
technology continues to evolve rapidly, with new breakthroughs announced 
almost daily. Experts predict AI will be integral to most industries 
within the next decade.
"""

# Run the agent and wait for result
result = summarizer.run_sync(text)

# Access the validated data
print(f"Title: {result.data.title}")
print(f"Summary: {result.data.summary}")
print(f"Key Points:")
for point in result.data.key_points:
    print(f"  - {point}")
```

### Understanding RunResult

```python
"""
Understanding the RunResult object returned by run_sync().
"""
from pydantic import BaseModel
from pydantic_ai import Agent


class Answer(BaseModel):
    """A simple answer model."""
    response: str
    confidence: float


agent = Agent('gemini-1.5-flash', result_type=Answer)
result = agent.run_sync("What is the capital of France?")

# The result object contains several useful properties:

# 1. data - Your validated Pydantic model
print(f"Data: {result.data}")
print(f"Response: {result.data.response}")
print(f"Confidence: {result.data.confidence}")

# 2. Type is inferred - IDE knows result.data is Answer
answer: Answer = result.data  # Type checker is happy!

# 3. You can serialize the result
print(f"As dict: {result.data.model_dump()}")
print(f"As JSON: {result.data.model_dump_json()}")
```

### Passing Different Input Types

```python
"""
run_sync() can accept different input types.
"""
from pydantic import BaseModel
from pydantic_ai import Agent


class Response(BaseModel):
    message: str


agent = Agent('gemini-1.5-flash', result_type=Response)

# 1. Simple string input
result = agent.run_sync("Hello!")

# 2. Multi-line string
result = agent.run_sync("""
    This is a longer input
    that spans multiple lines.
    The AI will process all of it.
""")

# 3. Formatted string with variables
name = "Alice"
question = "weather"
result = agent.run_sync(f"Hi, I'm {name}. What's the {question} like today?")

# 4. String built from data
items = ["apple", "banana", "orange"]
result = agent.run_sync(f"Tell me about these fruits: {', '.join(items)}")
```

### Running Multiple Requests

```python
"""
Processing multiple inputs with run_sync().
"""
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from typing import Literal


class Sentiment(BaseModel):
    """Sentiment analysis result."""
    text: str = Field(description="The analyzed text")
    sentiment: Literal["positive", "negative", "neutral"]
    score: float = Field(ge=-1, le=1, description="-1 to 1 score")


analyzer = Agent(
    'gemini-1.5-flash',
    result_type=Sentiment,
    system_prompt="Analyze the sentiment of the given text."
)

# Process multiple texts
texts = [
    "I love this product! Best purchase ever!",
    "Terrible experience. Would not recommend.",
    "It's okay. Nothing special.",
    "Absolutely fantastic! 5 stars!",
    "Disappointed with the quality.",
]

results = []
for text in texts:
    result = analyzer.run_sync(text)
    results.append(result.data)
    print(f"{result.data.sentiment:>10}: {text[:40]}...")

# Aggregate results
positive = sum(1 for r in results if r.sentiment == "positive")
negative = sum(1 for r in results if r.sentiment == "negative")
neutral = sum(1 for r in results if r.sentiment == "neutral")

print(f"\nSummary: {positive} positive, {negative} negative, {neutral} neutral")
```

### Using run_sync() in Functions

```python
"""
Wrapping run_sync() in reusable functions.
"""
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from typing import Optional


# Define models
class ProductInfo(BaseModel):
    """Extracted product information."""
    name: str
    category: str
    price: Optional[float] = None
    features: list[str]


class ComparisonResult(BaseModel):
    """Product comparison result."""
    winner: str
    reason: str
    scores: dict[str, float]


# Create agents
extractor = Agent(
    'gemini-1.5-flash',
    result_type=ProductInfo,
    system_prompt="Extract product information from the description."
)

comparator = Agent(
    'gemini-1.5-flash',
    result_type=ComparisonResult,
    system_prompt="Compare the given products and determine a winner."
)


# Wrapper functions for clean API
def extract_product(description: str) -> ProductInfo:
    """Extract product info from a description."""
    result = extractor.run_sync(description)
    return result.data


def compare_products(product1: str, product2: str) -> ComparisonResult:
    """Compare two products."""
    prompt = f"Compare these products:\n1. {product1}\n2. {product2}"
    result = comparator.run_sync(prompt)
    return result.data


# Usage
if __name__ == "__main__":
    # Extract product info
    desc = "iPhone 15 Pro - 256GB, titanium design, A17 chip, $999"
    product = extract_product(desc)
    print(f"Product: {product.name}")
    print(f"Category: {product.category}")
    print(f"Price: ${product.price}")
    print(f"Features: {product.features}")
    
    # Compare products
    comparison = compare_products(
        "MacBook Pro 16-inch - M3 chip, 36GB RAM, $3499",
        "Dell XPS 15 - Intel i9, 32GB RAM, $2199"
    )
    print(f"\nWinner: {comparison.winner}")
    print(f"Reason: {comparison.reason}")
```

### Error Handling with run_sync()

```python
"""
Proper error handling when using run_sync().
"""
from pydantic import BaseModel, ValidationError
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelRetry, UnexpectedModelBehavior


class Result(BaseModel):
    answer: str
    confidence: float


agent = Agent('gemini-1.5-flash', result_type=Result)


def safe_run(prompt: str) -> Result | None:
    """Run agent with comprehensive error handling."""
    try:
        result = agent.run_sync(prompt)
        return result.data
    
    except ValidationError as e:
        # Pydantic validation failed - AI returned wrong format
        print(f"Validation Error: {e}")
        print("The AI returned data that doesn't match the expected format.")
        return None
    
    except ModelRetry as e:
        # The model requested a retry (usually a tool error)
        print(f"Model Retry: {e}")
        return None
    
    except UnexpectedModelBehavior as e:
        # Something unexpected happened
        print(f"Unexpected Behavior: {e}")
        return None
    
    except Exception as e:
        # Catch-all for other errors (network, API key, etc.)
        print(f"Error: {type(e).__name__}: {e}")
        return None


# Usage
result = safe_run("What is 2+2?")
if result:
    print(f"Answer: {result.answer}")
else:
    print("Failed to get a valid response")
```

---

## C. Test & Apply

### When to Use run_sync()

| Use Case | run_sync() Good For |
|----------|---------------------|
| Scripts | ✅ Perfect |
| CLI tools | ✅ Perfect |
| Simple APIs | ✅ Good |
| Jupyter notebooks | ✅ Perfect |
| Learning/prototyping | ✅ Perfect |
| High-concurrency APIs | ❌ Use async instead |
| Real-time streaming | ❌ Use run_stream() |

### Performance Considerations

```python
"""
Understanding run_sync() performance.
"""
import time
from pydantic import BaseModel
from pydantic_ai import Agent


class Response(BaseModel):
    message: str


agent = Agent('gemini-1.5-flash', result_type=Response)


# Single request timing
start = time.time()
result = agent.run_sync("Say hello")
elapsed = time.time() - start
print(f"Single request: {elapsed:.2f} seconds")

# Multiple sequential requests (NOT parallel)
texts = ["Hello", "How are you?", "What's up?"]
start = time.time()
for text in texts:
    agent.run_sync(text)
total = time.time() - start
print(f"3 sequential requests: {total:.2f} seconds")
print(f"Average per request: {total/3:.2f} seconds")

# Note: For parallel requests, you'd use async (covered in later project)
```

### Testing Agents with run_sync()

```python
"""
Testing pattern for agents using run_sync().
"""
from pydantic import BaseModel, Field
from pydantic_ai import Agent


class MathResult(BaseModel):
    """Result of a math operation."""
    expression: str
    result: float
    explanation: str


math_agent = Agent(
    'gemini-1.5-flash',
    result_type=MathResult,
    system_prompt="Evaluate the math expression and explain your work."
)


def test_math_agent():
    """Test the math agent with known inputs."""
    test_cases = [
        ("2 + 2", 4.0),
        ("10 * 5", 50.0),
        ("100 / 4", 25.0),
    ]
    
    for expression, expected in test_cases:
        result = math_agent.run_sync(f"Calculate: {expression}")
        
        # Check the result is close to expected (allowing for float precision)
        if abs(result.data.result - expected) < 0.01:
            print(f"✅ {expression} = {result.data.result}")
        else:
            print(f"❌ {expression}: expected {expected}, got {result.data.result}")


if __name__ == "__main__":
    test_math_agent()
```

---

## D. Common Stumbling Blocks

### "run_sync() is blocking my application"

That's expected! `run_sync()` waits for the response. For non-blocking behavior:
- Use `run()` (async) in async applications
- Run in a separate thread for sync applications
- Use background tasks/queues

### "I want to process many requests faster"

Sequential `run_sync()` calls are slow. Options:
1. Use async `run()` with `asyncio.gather()` (covered in later lessons)
2. Use threading (with caution)
3. Use a task queue like Celery

### "My request times out"

Gemini requests can take a few seconds. For long inputs:
```python
# Pydantic AI handles this, but be patient
# Typical response times: 1-5 seconds for simple requests
```

### "I'm hitting rate limits"

Add delays between requests:
```python
import time

for text in many_texts:
    result = agent.run_sync(text)
    time.sleep(0.5)  # Wait 500ms between requests
```

### "Return type is not recognized"

Make sure your function return type is correct:
```python
from pydantic_ai import RunResult

# The result is RunResult[YourModel]
result: RunResult[Response] = agent.run_sync(text)

# But usually you just want the data
response: Response = result.data
```

---

## ✅ Lesson 11 Complete!

### Key Takeaways

1. **run_sync()** is the synchronous way to run agents
2. **Returns RunResult** with `.data` containing your Pydantic model
3. **Blocks execution** until the AI responds
4. **Perfect for** scripts, CLI tools, and learning
5. **Handle errors** with try/except for robust code
6. **Sequential processing** - for parallel, use async

### run_sync() Checklist

- [ ] Understand that run_sync() blocks execution
- [ ] Know how to access result.data
- [ ] Can handle validation errors
- [ ] Understand when to use vs async alternatives

### What's Next?

In Lesson 12, we'll explore the full `RunResult` object and all the information it contains!

---

*Running agents like a pro! Let's explore results in Lesson 12!* 🚀
