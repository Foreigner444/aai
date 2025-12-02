# 📘 Lesson 12: Understanding Agent Results

Let's dive deep into the `RunResult` object and discover all the valuable information it contains! 📊

---

## A. Concept Overview

### What & Why

When you call `agent.run_sync()`, you get back a `RunResult` object. This object contains:
- Your validated data (the Pydantic model)
- Usage information (tokens used)
- Message history
- And more!

Understanding `RunResult` helps you:
- Build better logging and monitoring
- Track API costs
- Debug issues
- Understand what happened during the request

### The Analogy 📦

Think of `RunResult` like a delivery package:
- **The item you ordered** → `result.data` (your Pydantic model)
- **The shipping receipt** → usage info (tokens, cost)
- **Tracking history** → message history
- **Package metadata** → additional context

You ordered the item, but the packaging contains useful info too!

### Type Safety Benefit

`RunResult` is a generic type:
- `RunResult[Person]` means `.data` is type `Person`
- IDE autocomplete works perfectly
- Type checkers understand the exact return type

---

## B. Code Implementation

### File Structure
```
pydantic-ai-project/
├── agents/
│   └── result_exploration.py    # Exploring RunResult
└── ...
```

### Basic RunResult Access

```python
"""
Exploring the RunResult object in detail.
"""
from pydantic import BaseModel, Field
from pydantic_ai import Agent


class BookInfo(BaseModel):
    """Information about a book."""
    title: str
    author: str
    year: int
    genre: str


book_agent = Agent(
    'gemini-1.5-flash',
    result_type=BookInfo,
    system_prompt="Extract book information from the given text."
)

# Run the agent
text = "1984 by George Orwell, published in 1949, is a classic dystopian novel."
result = book_agent.run_sync(text)

# 1. The main data - your Pydantic model
print("=== result.data ===")
print(f"Type: {type(result.data)}")
print(f"Value: {result.data}")
print(f"  Title: {result.data.title}")
print(f"  Author: {result.data.author}")
print(f"  Year: {result.data.year}")
print(f"  Genre: {result.data.genre}")
```

### Accessing Usage Information

```python
"""
Understanding token usage from RunResult.
"""
from pydantic import BaseModel
from pydantic_ai import Agent


class Response(BaseModel):
    message: str


agent = Agent('gemini-1.5-flash', result_type=Response)
result = agent.run_sync("Write a short greeting.")

# Usage information (tokens consumed)
print("\n=== Usage Information ===")
if result.usage():
    usage = result.usage()
    print(f"Request tokens: {usage.request_tokens}")
    print(f"Response tokens: {usage.response_tokens}")
    print(f"Total tokens: {usage.total_tokens}")
    
    # Estimate cost (rough calculation)
    # Gemini Flash: ~$0.075 per 1M input, ~$0.30 per 1M output
    input_cost = (usage.request_tokens / 1_000_000) * 0.075
    output_cost = (usage.response_tokens / 1_000_000) * 0.30
    total_cost = input_cost + output_cost
    print(f"Estimated cost: ${total_cost:.6f}")
```

### Accessing Message History

```python
"""
Viewing the message history from a result.
"""
from pydantic import BaseModel
from pydantic_ai import Agent


class Answer(BaseModel):
    response: str


agent = Agent(
    'gemini-1.5-flash',
    result_type=Answer,
    system_prompt="You are a helpful assistant. Answer questions concisely."
)

result = agent.run_sync("What is the speed of light?")

# Access all messages in the conversation
print("\n=== Message History ===")
for i, message in enumerate(result.all_messages()):
    print(f"\n--- Message {i+1} ---")
    print(f"Kind: {message.kind}")
    
    # Handle different message types
    if hasattr(message, 'content'):
        content = message.content
        if isinstance(content, str):
            print(f"Content: {content[:100]}...")
        else:
            print(f"Content type: {type(content)}")
```

### Creating a Result Logger

```python
"""
A utility to log all information from RunResult.
"""
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from datetime import datetime
from typing import Any
import json


class LogEntry(BaseModel):
    """A log entry for an agent run."""
    timestamp: str
    model: str
    input_text: str
    output_data: dict[str, Any]
    request_tokens: int | None
    response_tokens: int | None
    total_tokens: int | None
    estimated_cost_usd: float | None


def log_agent_result(
    agent: Agent,
    input_text: str,
    result: Any,
    model_name: str = "gemini-1.5-flash"
) -> LogEntry:
    """Create a detailed log entry from a result."""
    
    # Get usage info
    usage = result.usage()
    request_tokens = usage.request_tokens if usage else None
    response_tokens = usage.response_tokens if usage else None
    total_tokens = usage.total_tokens if usage else None
    
    # Calculate estimated cost
    estimated_cost = None
    if usage:
        input_cost = (usage.request_tokens / 1_000_000) * 0.075
        output_cost = (usage.response_tokens / 1_000_000) * 0.30
        estimated_cost = input_cost + output_cost
    
    return LogEntry(
        timestamp=datetime.now().isoformat(),
        model=model_name,
        input_text=input_text,
        output_data=result.data.model_dump(),
        request_tokens=request_tokens,
        response_tokens=response_tokens,
        total_tokens=total_tokens,
        estimated_cost_usd=estimated_cost
    )


# Example usage
class Sentiment(BaseModel):
    sentiment: str
    confidence: float


agent = Agent(
    'gemini-1.5-flash',
    result_type=Sentiment,
    system_prompt="Analyze sentiment."
)

text = "This is absolutely wonderful!"
result = agent.run_sync(text)

log_entry = log_agent_result(agent, text, result)
print(json.dumps(log_entry.model_dump(), indent=2))
```

### Working with Result Data

```python
"""
Various ways to work with result.data.
"""
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from typing import TypeVar


class Person(BaseModel):
    name: str
    age: int
    city: str


agent = Agent(
    'gemini-1.5-flash',
    result_type=Person,
    system_prompt="Extract person information."
)

result = agent.run_sync("John is 30 years old and lives in New York.")

# 1. Direct attribute access
print(f"Name: {result.data.name}")
print(f"Age: {result.data.age}")
print(f"City: {result.data.city}")

# 2. Convert to dictionary
person_dict = result.data.model_dump()
print(f"\nAs dict: {person_dict}")
# {'name': 'John', 'age': 30, 'city': 'New York'}

# 3. Convert to JSON string
person_json = result.data.model_dump_json()
print(f"\nAs JSON: {person_json}")
# '{"name":"John","age":30,"city":"New York"}'

# 4. Iterate over fields
print("\nAll fields:")
for field_name, field_value in result.data:
    print(f"  {field_name}: {field_value}")

# 5. Create a copy with modifications
modified = result.data.model_copy(update={"city": "Boston"})
print(f"\nModified copy: {modified}")

# 6. Validate and compare
other_person = Person(name="John", age=30, city="New York")
print(f"\nSame data? {result.data == other_person}")
```

### Batch Processing with Result Tracking

```python
"""
Process multiple inputs and track all results.
"""
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from dataclasses import dataclass
from typing import Generic, TypeVar
import statistics


class Classification(BaseModel):
    """Text classification result."""
    category: str
    confidence: float = Field(ge=0, le=1)


classifier = Agent(
    'gemini-1.5-flash',
    result_type=Classification,
    system_prompt="Classify the text into a category."
)


@dataclass
class BatchStats:
    """Statistics from a batch of runs."""
    total_runs: int
    successful_runs: int
    failed_runs: int
    total_input_tokens: int
    total_output_tokens: int
    avg_confidence: float
    total_cost_estimate: float


def process_batch(texts: list[str]) -> tuple[list[Classification], BatchStats]:
    """Process a batch of texts and collect statistics."""
    results = []
    confidences = []
    total_input = 0
    total_output = 0
    failures = 0
    
    for text in texts:
        try:
            result = classifier.run_sync(text)
            results.append(result.data)
            confidences.append(result.data.confidence)
            
            if result.usage():
                total_input += result.usage().request_tokens
                total_output += result.usage().response_tokens
        except Exception as e:
            print(f"Failed to process: {text[:30]}... - {e}")
            failures += 1
    
    # Calculate stats
    stats = BatchStats(
        total_runs=len(texts),
        successful_runs=len(texts) - failures,
        failed_runs=failures,
        total_input_tokens=total_input,
        total_output_tokens=total_output,
        avg_confidence=statistics.mean(confidences) if confidences else 0,
        total_cost_estimate=(total_input / 1_000_000 * 0.075) + (total_output / 1_000_000 * 0.30)
    )
    
    return results, stats


# Example
texts = [
    "The new iPhone has amazing camera quality.",
    "Python is my favorite programming language.",
    "The weather is beautiful today.",
    "This restaurant serves delicious Italian food.",
    "The stock market crashed yesterday.",
]

results, stats = process_batch(texts)

print("\n=== Batch Results ===")
for text, result in zip(texts, results):
    print(f"{result.category:>15} ({result.confidence:.2f}): {text[:40]}...")

print(f"\n=== Batch Statistics ===")
print(f"Total runs: {stats.total_runs}")
print(f"Successful: {stats.successful_runs}")
print(f"Failed: {stats.failed_runs}")
print(f"Total tokens: {stats.total_input_tokens + stats.total_output_tokens}")
print(f"Avg confidence: {stats.avg_confidence:.2%}")
print(f"Est. cost: ${stats.total_cost_estimate:.4f}")
```

---

## C. Test & Apply

### RunResult Quick Reference

| Property/Method | What It Returns |
|-----------------|-----------------|
| `result.data` | Your validated Pydantic model |
| `result.usage()` | Token usage stats (or None) |
| `result.all_messages()` | List of all messages in conversation |

### Usage Object Properties

| Property | Description |
|----------|-------------|
| `request_tokens` | Tokens in your input |
| `response_tokens` | Tokens in AI response |
| `total_tokens` | Total tokens used |

### Cost Tracking Example

```python
"""
Track costs across multiple runs.
"""
from pydantic import BaseModel
from pydantic_ai import Agent


class CostTracker:
    """Track API costs across multiple runs."""
    
    def __init__(self, input_rate: float = 0.075, output_rate: float = 0.30):
        self.input_rate = input_rate / 1_000_000  # per token
        self.output_rate = output_rate / 1_000_000  # per token
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.run_count = 0
    
    def track(self, result) -> None:
        """Add a result to tracking."""
        if result.usage():
            self.total_input_tokens += result.usage().request_tokens
            self.total_output_tokens += result.usage().response_tokens
            self.run_count += 1
    
    @property
    def total_cost(self) -> float:
        """Calculate total cost so far."""
        input_cost = self.total_input_tokens * self.input_rate
        output_cost = self.total_output_tokens * self.output_rate
        return input_cost + output_cost
    
    def report(self) -> str:
        """Generate a cost report."""
        return f"""
Cost Report:
  Runs: {self.run_count}
  Input tokens: {self.total_input_tokens:,}
  Output tokens: {self.total_output_tokens:,}
  Total tokens: {self.total_input_tokens + self.total_output_tokens:,}
  Estimated cost: ${self.total_cost:.4f}
"""


# Usage
tracker = CostTracker()

class Response(BaseModel):
    message: str

agent = Agent('gemini-1.5-flash', result_type=Response)

# Run several requests
for prompt in ["Hello", "How are you?", "What's 2+2?"]:
    result = agent.run_sync(prompt)
    tracker.track(result)

print(tracker.report())
```

---

## D. Common Stumbling Blocks

### "usage() returns None"

Usage info might not be available in all cases:
```python
if result.usage():
    print(f"Tokens: {result.usage().total_tokens}")
else:
    print("Usage info not available")
```

### "How do I know the exact cost?"

Costs depend on the model and can change. Use approximate rates:
```python
# As of 2024 (always check current pricing!)
GEMINI_FLASH_INPUT = 0.075  # per 1M tokens
GEMINI_FLASH_OUTPUT = 0.30  # per 1M tokens
GEMINI_PRO_INPUT = 3.50     # per 1M tokens
GEMINI_PRO_OUTPUT = 10.50   # per 1M tokens
```

### "I need to store results in a database"

Convert to dict first:
```python
# Easy serialization
result_dict = result.data.model_dump()

# Store as JSON
result_json = result.data.model_dump_json()

# Recreate from dict
recreated = MyModel.model_validate(result_dict)
```

### "Type checker complains about result.data"

Add type annotations:
```python
from pydantic_ai import RunResult

# Be explicit about the type
result: RunResult[Person] = agent.run_sync(text)
person: Person = result.data  # Now properly typed
```

---

## ✅ Lesson 12 Complete!

### Key Takeaways

1. **`result.data`** is your validated Pydantic model
2. **`result.usage()`** provides token counts for cost tracking
3. **`result.all_messages()`** shows the conversation history
4. **Always check** if usage() returns None before accessing
5. **Use `model_dump()`** to convert results to dictionaries
6. **Track costs** to avoid surprise bills

### RunResult Checklist

- [ ] Can access result.data and its fields
- [ ] Understand how to get usage information
- [ ] Can convert results to dict/JSON
- [ ] Know how to estimate costs
- [ ] Can track statistics across batch runs

### What's Next?

In Lesson 13, we'll learn about error handling and what happens when things go wrong!

---

*Results mastered! Let's handle errors in Lesson 13!* 🚀
