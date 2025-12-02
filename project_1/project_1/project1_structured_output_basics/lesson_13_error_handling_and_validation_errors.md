# 📘 Lesson 13: Error Handling and Validation Errors

Things don't always go perfectly - let's learn to handle errors gracefully! 🛡️

---

## A. Concept Overview

### What & Why

When working with AI and validation, several things can go wrong:
- **Validation errors** - AI returned data that doesn't match your model
- **API errors** - Network issues, rate limits, invalid API key
- **Model errors** - The AI couldn't process your request
- **Unexpected responses** - Something completely unexpected happened

Good error handling ensures your application:
- Doesn't crash on failures
- Provides helpful feedback
- Can recover when possible
- Logs issues for debugging

### The Analogy 🎯

Think of error handling like a safety net at a circus:
- **The performer** (your code) does their act
- **The safety net** (error handling) catches them if they fall
- **Different nets** (different exception types) for different falls
- **The show goes on** (graceful degradation)

Without a net, one mistake ends everything. With a net, you recover and continue.

### Type Safety Benefit

Pydantic's validation errors are:
- Detailed and specific
- Typed and structured
- Easy to log and analyze
- Helpful for debugging

---

## B. Code Implementation

### File Structure
```
pydantic-ai-project/
├── agents/
│   └── error_handling.py    # Error handling examples
├── utils/
│   └── exceptions.py        # Custom exceptions
└── ...
```

### Understanding Pydantic ValidationError

```python
"""
Understanding Pydantic ValidationError in detail.
"""
from pydantic import BaseModel, Field, ValidationError


class Person(BaseModel):
    """A person model with validated fields."""
    name: str = Field(min_length=1)
    age: int = Field(ge=0, le=150)
    email: str


# Example 1: Missing required field
print("=== Missing Field ===")
try:
    person = Person(name="Alice", age=30)  # Missing email!
except ValidationError as e:
    print(f"Error count: {e.error_count()}")
    print(f"Errors: {e.errors()}")
    print(f"\nHuman-readable:\n{e}")

# Example 2: Wrong type
print("\n=== Wrong Type ===")
try:
    person = Person(name="Bob", age="not a number", email="bob@test.com")
except ValidationError as e:
    for error in e.errors():
        print(f"Field: {error['loc']}")
        print(f"Message: {error['msg']}")
        print(f"Type: {error['type']}")
        print(f"Input: {error['input']}")
        print()

# Example 3: Constraint violation
print("=== Constraint Violation ===")
try:
    person = Person(name="", age=-5, email="test@test.com")
except ValidationError as e:
    print(f"Number of errors: {e.error_count()}")
    for error in e.errors():
        field = error['loc'][0]
        msg = error['msg']
        print(f"  {field}: {msg}")
```

### Handling Agent Errors

```python
"""
Comprehensive error handling for Pydantic AI agents.
"""
from pydantic import BaseModel, ValidationError
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelRetry, UnexpectedModelBehavior
import httpx


class Response(BaseModel):
    answer: str
    confidence: float


agent = Agent('gemini-1.5-flash', result_type=Response)


def safe_agent_call(prompt: str) -> Response | None:
    """
    Call the agent with comprehensive error handling.
    Returns None if the call fails.
    """
    try:
        result = agent.run_sync(prompt)
        return result.data
    
    except ValidationError as e:
        # The AI returned data that doesn't match your Pydantic model
        print("❌ Validation Error: AI response didn't match expected format")
        print(f"   Errors: {e.error_count()}")
        for error in e.errors():
            print(f"   - {error['loc']}: {error['msg']}")
        return None
    
    except ModelRetry as e:
        # The model asked for a retry (usually due to tool errors)
        print(f"❌ Model Retry: {e}")
        return None
    
    except UnexpectedModelBehavior as e:
        # Something unexpected happened with the model
        print(f"❌ Unexpected Model Behavior: {e}")
        return None
    
    except httpx.HTTPStatusError as e:
        # HTTP error (rate limit, auth error, etc.)
        if e.response.status_code == 429:
            print("❌ Rate Limited: Too many requests, please wait")
        elif e.response.status_code == 401:
            print("❌ Authentication Error: Check your API key")
        elif e.response.status_code == 403:
            print("❌ Forbidden: API key may not have access")
        else:
            print(f"❌ HTTP Error {e.response.status_code}: {e}")
        return None
    
    except httpx.ConnectError:
        print("❌ Connection Error: Could not connect to API")
        return None
    
    except httpx.TimeoutException:
        print("❌ Timeout: Request took too long")
        return None
    
    except Exception as e:
        # Catch-all for unexpected errors
        print(f"❌ Unexpected Error: {type(e).__name__}: {e}")
        return None


# Usage
result = safe_agent_call("What is 2+2?")
if result:
    print(f"✅ Answer: {result.answer}")
else:
    print("Failed to get response")
```

### Creating Custom Error Types

```python
"""
Custom exceptions for your application.
"""
from enum import Enum
from dataclasses import dataclass
from pydantic import ValidationError


class ErrorSeverity(str, Enum):
    """Severity levels for errors."""
    LOW = "low"           # Recoverable, continue
    MEDIUM = "medium"     # Degraded functionality
    HIGH = "high"         # Major issue
    CRITICAL = "critical" # Cannot continue


@dataclass
class AgentError:
    """Structured error information."""
    error_type: str
    message: str
    severity: ErrorSeverity
    recoverable: bool
    details: dict | None = None
    
    def __str__(self):
        return f"[{self.severity.value.upper()}] {self.error_type}: {self.message}"


def classify_error(exception: Exception) -> AgentError:
    """Classify an exception into an AgentError."""
    
    if isinstance(exception, ValidationError):
        return AgentError(
            error_type="ValidationError",
            message="AI response didn't match expected format",
            severity=ErrorSeverity.MEDIUM,
            recoverable=True,
            details={"errors": exception.errors()}
        )
    
    if "rate limit" in str(exception).lower() or "429" in str(exception):
        return AgentError(
            error_type="RateLimitError",
            message="API rate limit exceeded",
            severity=ErrorSeverity.LOW,
            recoverable=True,
            details={"retry_after": "60 seconds"}
        )
    
    if "api key" in str(exception).lower() or "401" in str(exception):
        return AgentError(
            error_type="AuthenticationError",
            message="Invalid or missing API key",
            severity=ErrorSeverity.CRITICAL,
            recoverable=False
        )
    
    if "timeout" in str(exception).lower():
        return AgentError(
            error_type="TimeoutError",
            message="Request timed out",
            severity=ErrorSeverity.LOW,
            recoverable=True
        )
    
    # Default for unknown errors
    return AgentError(
        error_type=type(exception).__name__,
        message=str(exception),
        severity=ErrorSeverity.HIGH,
        recoverable=False,
        details={"exception_class": type(exception).__name__}
    )


# Usage example
try:
    # Some operation that might fail
    raise ValidationError.from_exception_data("test", [])
except Exception as e:
    error = classify_error(e)
    print(error)
    if error.recoverable:
        print("  → This error is recoverable, will retry")
    else:
        print("  → This error is NOT recoverable")
```

### Retry Logic

```python
"""
Implementing retry logic for transient failures.
"""
import time
from pydantic import BaseModel, ValidationError
from pydantic_ai import Agent
from functools import wraps
from typing import TypeVar, Callable


T = TypeVar('T')


class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = 3
    initial_delay: float = 1.0  # seconds
    exponential_base: float = 2.0
    max_delay: float = 60.0


def with_retry(config: RetryConfig = RetryConfig()):
    """Decorator to add retry logic to a function."""
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            delay = config.initial_delay
            
            for attempt in range(config.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except ValidationError:
                    # Don't retry validation errors - they won't fix themselves
                    raise
                except Exception as e:
                    last_exception = e
                    
                    if attempt < config.max_retries:
                        print(f"Attempt {attempt + 1} failed: {e}")
                        print(f"Retrying in {delay:.1f} seconds...")
                        time.sleep(delay)
                        delay = min(delay * config.exponential_base, config.max_delay)
                    else:
                        print(f"All {config.max_retries + 1} attempts failed")
            
            raise last_exception
        
        return wrapper
    return decorator


# Example usage
class Result(BaseModel):
    message: str


agent = Agent('gemini-1.5-flash', result_type=Result)


@with_retry(RetryConfig(max_retries=3, initial_delay=1.0))
def call_agent(prompt: str) -> Result:
    """Call agent with automatic retries."""
    result = agent.run_sync(prompt)
    return result.data


# This will retry up to 3 times on transient failures
try:
    response = call_agent("Hello!")
    print(f"Success: {response.message}")
except Exception as e:
    print(f"Final failure: {e}")
```

### Validation Error Recovery

```python
"""
Strategies for recovering from validation errors.
"""
from pydantic import BaseModel, Field, ValidationError
from pydantic_ai import Agent
from typing import Optional, Any


class StrictResponse(BaseModel):
    """A response with strict validation."""
    name: str = Field(min_length=1)
    count: int = Field(ge=0)
    tags: list[str]


class FlexibleResponse(BaseModel):
    """A more flexible version for fallback."""
    name: Optional[str] = None
    count: Optional[int] = None
    tags: list[str] = Field(default_factory=list)
    raw_data: Optional[dict[str, Any]] = None


# Create two agents: strict and flexible
strict_agent = Agent(
    'gemini-1.5-flash',
    result_type=StrictResponse,
    system_prompt="Extract structured data."
)

flexible_agent = Agent(
    'gemini-1.5-flash',
    result_type=FlexibleResponse,
    system_prompt="Extract structured data. It's okay if some fields are missing."
)


def extract_with_fallback(text: str) -> StrictResponse | FlexibleResponse:
    """Try strict extraction, fall back to flexible if it fails."""
    try:
        # Try strict extraction first
        result = strict_agent.run_sync(text)
        print("✅ Strict extraction succeeded")
        return result.data
    
    except ValidationError as e:
        print(f"⚠️ Strict extraction failed: {e.error_count()} errors")
        
        # Fall back to flexible extraction
        try:
            result = flexible_agent.run_sync(text)
            print("✅ Flexible extraction succeeded (partial data)")
            return result.data
        except Exception as e2:
            print(f"❌ Flexible extraction also failed: {e2}")
            raise


# Usage
text = "Some ambiguous text that might not have all fields..."
try:
    data = extract_with_fallback(text)
    print(f"Result: {data}")
except Exception as e:
    print(f"All extraction attempts failed: {e}")
```

---

## C. Test & Apply

### Error Handling Best Practices

| Practice | Why |
|----------|-----|
| Catch specific exceptions first | More precise handling |
| Log all errors | Debugging and monitoring |
| Provide user-friendly messages | Better UX |
| Don't swallow errors silently | Hidden bugs are dangerous |
| Use retry for transient failures | Improve reliability |
| Have a fallback strategy | Graceful degradation |

### Exception Hierarchy

```
Exception
├── ValidationError          # Pydantic validation failed
├── ModelRetry              # Model requested retry
├── UnexpectedModelBehavior # Unexpected AI behavior
├── httpx.HTTPStatusError   # HTTP errors
│   ├── 401 Unauthorized    # Bad API key
│   ├── 403 Forbidden       # No access
│   ├── 429 Rate Limited    # Too many requests
│   └── 500+ Server Error   # API issues
├── httpx.ConnectError      # Network issues
└── httpx.TimeoutException  # Request timeout
```

### Testing Error Handling

```python
"""
Test your error handling code.
"""
from pydantic import BaseModel, ValidationError
import pytest


class Person(BaseModel):
    name: str
    age: int


def test_validation_error_missing_field():
    """Test that missing fields raise ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        Person(name="Alice")  # Missing age
    
    assert exc_info.value.error_count() == 1
    assert "age" in str(exc_info.value)


def test_validation_error_wrong_type():
    """Test that wrong types raise ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        Person(name="Bob", age="not a number")
    
    errors = exc_info.value.errors()
    assert errors[0]["type"] == "int_parsing"


def test_validation_error_count():
    """Test multiple validation errors."""
    with pytest.raises(ValidationError) as exc_info:
        Person(name=123, age="bad")  # Both wrong!
    
    assert exc_info.value.error_count() == 2
```

---

## D. Common Stumbling Blocks

### "I'm catching Exception but nothing happens"

Make sure you're not swallowing errors:
```python
# ❌ Bad - swallowing errors
try:
    result = agent.run_sync(text)
except Exception:
    pass  # Silent failure!

# ✅ Good - handle and log
try:
    result = agent.run_sync(text)
except Exception as e:
    print(f"Error: {e}")
    logger.error(f"Agent failed: {e}")
    raise  # Or handle appropriately
```

### "ValidationError has no attribute 'message'"

Use the correct attributes:
```python
try:
    # ...
except ValidationError as e:
    # ❌ Wrong
    print(e.message)
    
    # ✅ Correct
    print(str(e))  # Human-readable
    print(e.errors())  # List of error dicts
    print(e.error_count())  # Number of errors
```

### "Retrying validation errors doesn't help"

Validation errors mean the AI returned the wrong format. Retrying usually gives the same result. Instead:
- Improve your system prompt
- Make your model more flexible
- Add field descriptions

### "I keep hitting rate limits"

Add delays and implement proper backoff:
```python
import time

for item in items:
    try:
        result = agent.run_sync(item)
    except Exception as e:
        if "429" in str(e):
            time.sleep(60)  # Wait a minute
            result = agent.run_sync(item)  # Retry
```

---

## ✅ Lesson 13 Complete!

### Key Takeaways

1. **ValidationError** = AI response didn't match your model
2. **Catch specific exceptions** before general ones
3. **Always log errors** for debugging
4. **Retry transient failures** (network, rate limits)
5. **Don't retry validation errors** - fix the prompt instead
6. **Have fallback strategies** for graceful degradation

### Error Handling Checklist

- [ ] Understand ValidationError structure
- [ ] Can catch and handle different exception types
- [ ] Implement retry logic for transient failures
- [ ] Log errors appropriately
- [ ] Provide user-friendly error messages

### What's Next?

In Lesson 14, we'll dive deeper into debugging Pydantic validation issues!

---

*Errors handled! Let's debug validation in Lesson 14!* 🚀
