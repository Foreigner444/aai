# Lesson 14: Retries and Fallbacks

## A. Concept Overview

### What & Why
**Retries and Fallbacks** are reliability patterns that automatically recover from transient failures. Retries attempt operations again after temporary errors (network glitches, rate limits, brief service outages), while fallbacks provide alternative approaches when primary methods consistently fail. This is crucial because production systems face constant transient failures - a single network blip shouldn't crash your agent. Smart retry logic with exponential backoff and circuit breakers transforms fragile systems into resilient ones.

### Analogy
Think of retries and fallbacks like a skilled salesperson:

**No Retry/Fallback** = Giving up immediately:
- Salesperson: "Hi, would you like to buy this product?"
- Customer: "I'm busy right now"
- Salesperson: "OK, I'll never contact you again" ❌
- Lost sale

**With Retries** = Persistent but smart:
- Salesperson: "Hi, would you like to buy this product?"
- Customer: "I'm busy right now"
- Salesperson: *Waits 1 hour, tries again*
- Customer: "Still busy"
- Salesperson: *Waits 2 hours, tries again*
- Customer: "Actually, yes! Tell me more"
- Sale made! ✅

**With Fallbacks** = Alternative approaches:
- Salesperson: "Would you like to buy the premium model?"
- Customer: "Too expensive"
- Salesperson: "How about the standard model?" (fallback #1)
- Customer: "Still too much"
- Salesperson: "We have a budget-friendly option" (fallback #2)
- Customer: "Perfect!"
- Sale made! ✅

**Circuit Breaker** = Knowing when to stop:
- Salesperson calls customer 50 times in a row
- Customer blocks their number
- Smart salesperson: "This customer isn't interested. I'll move on to others"
- Prevents wasted effort on hopeless cases

### Type Safety Benefit
Type-safe retry and fallback patterns provide:
- **Retry policy types**: Typed configuration for retry behavior
- **Attempt tracking**: Type-checked retry counters and timing
- **Fallback chains**: Type-safe alternative implementations
- **Circuit state**: Typed state machine (open/half-open/closed)
- **Timeout types**: Type-checked timeout values
- **Error classification**: Type-safe determination of retry-able errors

Your reliability layer becomes fully type-checked and predictable!

---

## B. Code Implementation

### File Structure
```
agent_system_design/
├── lesson_14_retries_fallbacks.py
├── requirements.txt
└── .env
```

### Complete Code Snippet

**lesson_14_retries_fallbacks.py**
```python
"""
Lesson 14: Retries and Fallbacks
Comprehensive reliability patterns for production agents
"""

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from dataclasses import dataclass, field
from typing import Callable, Optional, TypeVar, Generic
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import random
import time
from functools import wraps
from dotenv import load_dotenv

load_dotenv()

T = TypeVar('T')


# PATTERN 1: Exponential Backoff with Jitter

class RetryPolicy(BaseModel):
    """Configuration for retry behavior"""
    max_attempts: int = Field(ge=1, le=10, default=3)
    initial_delay: float = Field(ge=0.1, le=60.0, default=1.0)
    max_delay: float = Field(ge=1.0, le=300.0, default=60.0)
    exponential_base: float = Field(ge=1.0, le=10.0, default=2.0)
    jitter: bool = Field(default=True)
    
    def calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay for a given attempt number.
        
        Uses exponential backoff: delay = initial_delay * (base ^ attempt)
        Adds jitter to prevent thundering herd
        
        Args:
            attempt: Attempt number (0-indexed)
        
        Returns:
            Delay in seconds
        """
        # Exponential backoff
        delay = min(
            self.initial_delay * (self.exponential_base ** attempt),
            self.max_delay
        )
        
        # Add jitter (random variation)
        if self.jitter:
            # Jitter between 0% and 100% of delay
            jitter_amount = random.uniform(0, delay)
            delay = delay * 0.5 + jitter_amount * 0.5
        
        return delay


def retry_with_backoff(
    policy: Optional[RetryPolicy] = None,
    retry_on: tuple[type[Exception], ...] = (Exception,)
):
    """
    Decorator that retries a function with exponential backoff.
    
    Args:
        policy: RetryPolicy configuration (defaults to 3 attempts)
        retry_on: Tuple of exception types to retry on
    
    Example:
        @retry_with_backoff(
            policy=RetryPolicy(max_attempts=3, initial_delay=1.0),
            retry_on=(ConnectionError, TimeoutError)
        )
        def flaky_function():
            ...
    """
    if policy is None:
        policy = RetryPolicy()
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(policy.max_attempts):
                try:
                    # Attempt the operation
                    return func(*args, **kwargs)
                
                except retry_on as e:
                    last_exception = e
                    
                    if attempt == policy.max_attempts - 1:
                        # Final attempt failed
                        print(f"   ❌ All {policy.max_attempts} attempts failed")
                        raise
                    
                    # Calculate delay for next attempt
                    delay = policy.calculate_delay(attempt)
                    
                    print(f"   ⏳ Attempt {attempt + 1}/{policy.max_attempts} failed: {e}")
                    print(f"      Retrying in {delay:.2f}s...")
                    
                    time.sleep(delay)
                
                except Exception as e:
                    # Non-retryable exception
                    print(f"   ❌ Non-retryable error: {type(e).__name__}")
                    raise
            
            # Should never reach here
            raise last_exception
        
        return wrapper
    return decorator


# Async version of retry decorator
def async_retry_with_backoff(
    policy: Optional[RetryPolicy] = None,
    retry_on: tuple[type[Exception], ...] = (Exception,)
):
    """
    Async version of retry_with_backoff decorator.
    
    Args:
        policy: RetryPolicy configuration
        retry_on: Tuple of exception types to retry on
    """
    if policy is None:
        policy = RetryPolicy()
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            last_exception = None
            
            for attempt in range(policy.max_attempts):
                try:
                    return await func(*args, **kwargs)
                
                except retry_on as e:
                    last_exception = e
                    
                    if attempt == policy.max_attempts - 1:
                        print(f"   ❌ All {policy.max_attempts} attempts failed")
                        raise
                    
                    delay = policy.calculate_delay(attempt)
                    print(f"   ⏳ Attempt {attempt + 1}/{policy.max_attempts} failed: {e}")
                    print(f"      Retrying in {delay:.2f}s...")
                    
                    await asyncio.sleep(delay)
                
                except Exception as e:
                    print(f"   ❌ Non-retryable error: {type(e).__name__}")
                    raise
            
            raise last_exception
        
        return wrapper
    return decorator


# PATTERN 2: Circuit Breaker

class CircuitState(str, Enum):
    """Circuit breaker states"""
    CLOSED = "closed"        # Normal operation
    OPEN = "open"           # Failing, rejecting requests
    HALF_OPEN = "half_open" # Testing if service recovered


class CircuitBreaker:
    """
    Circuit breaker pattern to prevent cascading failures.
    
    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Too many failures, requests rejected immediately
    - HALF_OPEN: After timeout, allow one test request
    
    Example:
        breaker = CircuitBreaker(failure_threshold=5, timeout=60)
        
        result = breaker.call(risky_function, arg1, arg2)
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: int = 60,
        expected_exception: type[Exception] = Exception
    ):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = CircuitState.CLOSED
    
    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Function to execute
            *args, **kwargs: Arguments to pass to function
        
        Returns:
            Function result
        
        Raises:
            Exception: If circuit is open or function fails
        """
        # Check if circuit should transition from OPEN to HALF_OPEN
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                print(f"   🔄 Circuit HALF_OPEN - testing service")
                self.state = CircuitState.HALF_OPEN
            else:
                time_remaining = self.timeout - (datetime.now() - self.last_failure_time).total_seconds()
                raise Exception(
                    f"Circuit breaker OPEN - service unavailable. "
                    f"Retry in {time_remaining:.0f}s"
                )
        
        # Attempt the operation
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to test service recovery"""
        if self.last_failure_time is None:
            return False
        
        elapsed = (datetime.now() - self.last_failure_time).total_seconds()
        return elapsed >= self.timeout
    
    def _on_success(self):
        """Handle successful execution"""
        if self.state == CircuitState.HALF_OPEN:
            print(f"   ✅ Circuit CLOSED - service recovered")
            self.state = CircuitState.CLOSED
        
        self.failure_count = 0
    
    def _on_failure(self):
        """Handle failed execution"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            if self.state != CircuitState.OPEN:
                print(f"   ⚠️  Circuit OPEN - too many failures ({self.failure_count})")
                self.state = CircuitState.OPEN
    
    def reset(self):
        """Manually reset circuit breaker"""
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED


# PATTERN 3: Fallback Chain

class FallbackChain(Generic[T]):
    """
    Chain of fallback strategies that try alternatives on failure.
    
    Example:
        chain = FallbackChain[dict]()
        chain.add_strategy("primary", fetch_from_primary_api)
        chain.add_strategy("secondary", fetch_from_secondary_api)
        chain.add_strategy("cache", fetch_from_cache)
        
        result = chain.execute()
    """
    
    def __init__(self):
        self.strategies: list[tuple[str, Callable[[], T]]] = []
    
    def add_strategy(self, name: str, func: Callable[[], T]):
        """
        Add a fallback strategy to the chain.
        
        Args:
            name: Strategy name for logging
            func: Function to execute (should return T or raise exception)
        """
        self.strategies.append((name, func))
    
    def execute(self) -> tuple[Optional[T], list[str]]:
        """
        Execute strategies in order until one succeeds.
        
        Returns:
            Tuple of (result, errors_encountered)
        """
        errors = []
        
        for name, func in self.strategies:
            try:
                print(f"   🔄 Trying strategy: {name}")
                result = func()
                print(f"   ✅ Strategy '{name}' succeeded")
                return result, errors
            
            except Exception as e:
                error_msg = f"{name}: {str(e)}"
                errors.append(error_msg)
                print(f"   ❌ Strategy '{name}' failed: {e}")
        
        print(f"   ❌ All {len(self.strategies)} strategies failed")
        return None, errors


# PATTERN 4: Agent with Retries and Fallbacks

@dataclass
class ReliabilityDeps:
    """Dependencies for reliability examples"""
    user_id: str
    request_id: str


class ReliableResponse(BaseModel):
    """Response that tracks retry and fallback attempts"""
    answer: str
    success: bool
    primary_attempt_succeeded: bool
    total_attempts: int
    fallback_used: Optional[str] = None
    errors_encountered: list[str] = Field(default_factory=list)


reliable_agent = Agent(
    model='gemini-1.5-flash',
    result_type=ReliableResponse,
    deps_type=ReliabilityDeps,
    system_prompt="""
You are a resilient AI assistant designed for maximum reliability.

When tools fail:
1. Note which attempt/strategy succeeded
2. Acknowledge any errors encountered
3. Explain if fallback strategies were used
4. Provide the best answer possible given the data obtained

Always be transparent about what worked and what didn't.
""",
)


# Simulated flaky service
class FlakyService:
    """Service that fails randomly to demonstrate retries"""
    
    def __init__(self, failure_rate: float = 0.5):
        self.failure_rate = failure_rate
        self.call_count = 0
    
    def fetch_data(self) -> dict:
        """Fetch data (fails randomly)"""
        self.call_count += 1
        
        if random.random() < self.failure_rate:
            raise ConnectionError(f"Service unavailable (attempt {self.call_count})")
        
        return {
            "data": "Important information",
            "fetched_at": datetime.now().isoformat(),
            "attempt": self.call_count
        }
    
    def reset(self):
        """Reset call counter"""
        self.call_count = 0


# Global service instance and circuit breaker
flaky_service = FlakyService(failure_rate=0.6)
service_breaker = CircuitBreaker(failure_threshold=3, timeout=10)


@reliable_agent.tool
@retry_with_backoff(
    policy=RetryPolicy(max_attempts=3, initial_delay=0.5, max_delay=2.0),
    retry_on=(ConnectionError,)
)
def fetch_with_retry(
    ctx: RunContext[ReliabilityDeps]
) -> dict:
    """
    Fetch data with automatic retry on failure.
    
    Demonstrates:
    - Retry decorator with exponential backoff
    - Only retrying transient errors (ConnectionError)
    - Giving up after max attempts
    
    Returns:
        Data dictionary
    """
    print(f"\n🔧 fetch_with_retry() - attempting...")
    
    # This will retry automatically on ConnectionError
    data = flaky_service.fetch_data()
    
    return {
        "source": "primary_service",
        "data": data,
        "success": True
    }


@reliable_agent.tool
def fetch_with_circuit_breaker(
    ctx: RunContext[ReliabilityDeps]
) -> dict:
    """
    Fetch data with circuit breaker protection.
    
    Demonstrates:
    - Circuit breaker preventing repeated calls to failing service
    - Automatic state management (closed/open/half-open)
    - Fast-fail when circuit is open
    
    Returns:
        Data dictionary or error
    """
    print(f"\n🔧 fetch_with_circuit_breaker() - attempting...")
    
    try:
        # Circuit breaker wraps the call
        data = service_breaker.call(flaky_service.fetch_data)
        
        return {
            "source": "primary_service",
            "data": data,
            "circuit_state": service_breaker.state.value,
            "success": True
        }
    
    except Exception as e:
        return {
            "source": "primary_service",
            "error": str(e),
            "circuit_state": service_breaker.state.value,
            "success": False
        }


@reliable_agent.tool
def fetch_with_fallback(
    ctx: RunContext[ReliabilityDeps]
) -> dict:
    """
    Fetch data with fallback strategies.
    
    Demonstrates:
    - Primary, secondary, and cache fallback strategies
    - Trying alternatives when primary fails
    - Tracking which strategy succeeded
    
    Returns:
        Data dictionary with source information
    """
    print(f"\n🔧 fetch_with_fallback() - attempting...")
    
    # Create fallback chain
    chain = FallbackChain[dict]()
    
    # Strategy 1: Primary service (60% failure rate)
    def primary():
        data = flaky_service.fetch_data()
        return {"source": "primary_service", "data": data, "freshness": "real-time"}
    
    # Strategy 2: Secondary service (40% failure rate)
    def secondary():
        if random.random() < 0.4:
            raise ConnectionError("Secondary service unavailable")
        return {
            "source": "secondary_service",
            "data": {"data": "Backup information"},
            "freshness": "recent"
        }
    
    # Strategy 3: Cache (always succeeds, but stale data)
    def cache():
        return {
            "source": "cache",
            "data": {"data": "Cached information (may be stale)"},
            "freshness": "stale"
        }
    
    chain.add_strategy("primary", primary)
    chain.add_strategy("secondary", secondary)
    chain.add_strategy("cache", cache)
    
    # Execute with fallbacks
    result, errors = chain.execute()
    
    if result:
        result["errors_encountered"] = errors
        result["success"] = True
        return result
    else:
        return {
            "source": "none",
            "errors_encountered": errors,
            "success": False
        }


@reliable_agent.tool
def fetch_with_retry_and_fallback(
    ctx: RunContext[ReliabilityDeps]
) -> dict:
    """
    Combine retry and fallback strategies for maximum reliability.
    
    Demonstrates:
    - Retrying primary approach multiple times
    - Falling back to alternative if retries exhausted
    - Best-effort data retrieval
    
    Returns:
        Data dictionary
    """
    print(f"\n🔧 fetch_with_retry_and_fallback() - attempting...")
    
    # Try primary with retries
    for attempt in range(3):
        try:
            print(f"   🔄 Primary attempt {attempt + 1}/3")
            data = flaky_service.fetch_data()
            print(f"   ✅ Primary succeeded on attempt {attempt + 1}")
            return {
                "source": "primary_service",
                "data": data,
                "attempts": attempt + 1,
                "fallback_used": False,
                "success": True
            }
        except ConnectionError as e:
            print(f"   ❌ Primary attempt {attempt + 1} failed: {e}")
            if attempt < 2:
                delay = 0.5 * (2 ** attempt)
                print(f"      Waiting {delay}s before retry...")
                time.sleep(delay)
    
    # Primary failed - try fallback
    print(f"   ⚠️  Primary exhausted, trying fallback...")
    
    try:
        # Fallback: cached data
        print(f"   🔄 Fallback: using cache")
        time.sleep(0.2)  # Simulate cache lookup
        
        return {
            "source": "cache",
            "data": {"data": "Cached information (fallback)"},
            "attempts": 3,
            "fallback_used": True,
            "success": True,
            "warning": "Using cached data because primary service is unavailable"
        }
    
    except Exception as e:
        print(f"   ❌ Fallback also failed: {e}")
        return {
            "source": "none",
            "attempts": 3,
            "fallback_used": True,
            "success": False,
            "error": "All strategies failed"
        }


# Demonstration

def main():
    print("\n" + "="*70)
    print("RETRIES AND FALLBACKS - COMPREHENSIVE PATTERNS")
    print("="*70)
    print("\nThis lesson demonstrates 4 reliability patterns:\n")
    print("1. Retry with exponential backoff and jitter")
    print("2. Circuit breaker pattern (prevent cascading failures)")
    print("3. Fallback chain (try alternatives)")
    print("4. Combined retry + fallback (maximum reliability)")
    print("\nNote: Failures are simulated randomly for demonstration!")
    
    deps = ReliabilityDeps(user_id="user_123", request_id="req_456")
    
    test_cases = [
        {
            "query": "Fetch data with retry logic",
            "description": "Retries on transient failures (exponential backoff)"
        },
        {
            "query": "Fetch data with circuit breaker",
            "description": "Protects against repeated failures"
        },
        {
            "query": "Fetch data with fallback strategies",
            "description": "Tries primary, then secondary, then cache"
        },
        {
            "query": "Fetch data with retry and fallback combined",
            "description": "Retries primary, then falls back to cache"
        },
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        query = test_case["query"]
        description = test_case["description"]
        
        print(f"\n\n{'='*70}")
        print(f"TEST {i}: {query}")
        print(f"Strategy: {description}")
        print(f"{'='*70}")
        
        # Reset service for each test
        flaky_service.reset()
        
        try:
            result = reliable_agent.run_sync(query, deps=deps)
            response = result.data
            
            # Show result
            status_emoji = "✅" if response.success else "❌"
            print(f"\n{status_emoji} RESULT:")
            print(f"   Answer: {response.answer}")
            print(f"   Success: {response.success}")
            print(f"   Primary Succeeded: {response.primary_attempt_succeeded}")
            print(f"   Total Attempts: {response.total_attempts}")
            
            if response.fallback_used:
                print(f"   Fallback Used: {response.fallback_used}")
            
            if response.errors_encountered:
                print(f"   Errors: {len(response.errors_encountered)}")
                for error in response.errors_encountered[:3]:  # Show first 3
                    print(f"      • {error}")
        
        except Exception as e:
            print(f"\n❌ Agent error: {e}")
    
    print("\n\n" + "="*70)
    print("RELIABILITY PATTERNS SUMMARY")
    print("="*70)
    print("\n✅ When to Use Each Pattern:")
    print("\n1️⃣  RETRY (Exponential Backoff):")
    print("   • Transient failures (network glitches, rate limits)")
    print("   • Service is generally reliable but occasionally flaky")
    print("   • Each attempt is independent")
    print("   • Example: API call that times out occasionally")
    
    print("\n2️⃣  CIRCUIT BREAKER:")
    print("   • Service is completely down or degraded")
    print("   • Want to fail fast instead of wasting time")
    print("   • Prevent cascading failures across services")
    print("   • Example: Database is offline, stop trying for 60 seconds")
    
    print("\n3️⃣  FALLBACK:")
    print("   • Alternative data sources available")
    print("   • Graceful degradation acceptable")
    print("   • Different quality levels (real-time vs cached)")
    print("   • Example: Primary API down, use secondary or cache")
    
    print("\n4️⃣  COMBINED (Retry + Fallback):")
    print("   • Maximum reliability required")
    print("   • Multiple strategies available")
    print("   • Can tolerate degraded service")
    print("   • Example: Retry primary 3x, then fallback to cache")
    
    print("\n⚠️  Anti-Patterns to Avoid:")
    print("   ❌ Retrying forever (use max attempts)")
    print("   ❌ No backoff delay (hammers failing service)")
    print("   ❌ Retrying non-transient errors (user input errors)")
    print("   ❌ No circuit breaker (waste resources on dead service)")
    print("   ❌ No fallback for critical paths (all-or-nothing)")


if __name__ == "__main__":
    main()
```

### Line-by-Line Explanation

**Pattern 1: Exponential Backoff (Lines 24-139)**:
- `RetryPolicy`: Pydantic model for retry configuration
- `calculate_delay()`: Exponential backoff with jitter
- `retry_with_backoff()`: Decorator for automatic retries
- Jitter prevents thundering herd problem
- Type-safe retry configuration

**Pattern 2: Circuit Breaker (Lines 142-250)**:
- `CircuitState`: Enum for circuit states (closed/open/half-open)
- `CircuitBreaker`: Prevents calls to failing service
- Transitions: closed → open (too many failures) → half-open (testing) → closed (recovered)
- Fast-fail when circuit is open
- Automatic recovery testing

**Pattern 3: Fallback Chain (Lines 253-310)**:
- `FallbackChain`: Generic chain of alternative strategies
- Tries strategies in order until one succeeds
- Returns result + list of errors encountered
- Graceful degradation pattern

**Pattern 4: Combined Patterns (Lines 313-587)**:
- `fetch_with_retry`: Retry decorator in action
- `fetch_with_circuit_breaker`: Circuit breaker protecting service
- `fetch_with_fallback`: Multiple fallback strategies
- `fetch_with_retry_and_fallback`: Best of both worlds
- FlakyService simulates real-world failures

### The "Why" Behind the Pattern

**Why not just retry forever?**

❌ **Infinite Retries** (Bad):
```python
while True:
    try:
        return fetch_data()
    except:
        pass  # Try again immediately forever!
# Hammers failing service, wastes resources, never gives up
```

✅ **Exponential Backoff** (Good):
```python
for attempt in range(3):
    try:
        return fetch_data()
    except ConnectionError:
        if attempt < 2:
            delay = 0.5 * (2 ** attempt)  # 0.5s, 1s, 2s
            await asyncio.sleep(delay)
raise  # Give up after 3 attempts
```

**Benefits**:
1. **Transient failures recover**: Brief network glitches self-heal
2. **Doesn't overwhelm failing services**: Backoff gives them time to recover
3. **Prevents cascading failures**: Circuit breaker stops wasteful retries
4. **User experience**: Fallbacks provide degraded service instead of nothing
5. **Resource efficiency**: Smart retries vs infinite hammering

---

## C. Test & Apply

### How to Test It

1. **Run the reliability demo**:
```bash
python lesson_14_retries_fallbacks.py
```

2. **Observe retry behavior and fallback chains**

3. **Try your own retry pattern**:
```python
@retry_with_backoff(
    policy=RetryPolicy(max_attempts=5, initial_delay=1.0),
    retry_on=(TimeoutError, ConnectionError)
)
def my_flaky_function():
    """This will retry automatically on network errors"""
    return requests.get("https://api.example.com", timeout=5)

result = my_flaky_function()
```

### Expected Result

You should see intelligent retry and fallback behavior:

```
======================================================================
TEST 1: Fetch data with retry logic
Strategy: Retries on transient failures (exponential backoff)
======================================================================

🔧 fetch_with_retry() - attempting...
   ⏳ Attempt 1/3 failed: Service unavailable (attempt 1)
      Retrying in 0.43s...
   ⏳ Attempt 2/3 failed: Service unavailable (attempt 2)
      Retrying in 1.27s...
   ✅ Success on attempt 3

✅ RESULT:
   Answer: I successfully fetched the data after 3 attempts. The service 
           was temporarily unavailable but recovered.
   Success: True
   Primary Succeeded: True
   Total Attempts: 3

======================================================================
TEST 3: Fetch data with fallback strategies
Strategy: Tries primary, then secondary, then cache
======================================================================

🔧 fetch_with_fallback() - attempting...
   🔄 Trying strategy: primary
   ❌ Strategy 'primary' failed: Service unavailable (attempt 1)
   🔄 Trying strategy: secondary
   ❌ Strategy 'secondary' failed: Secondary service unavailable
   🔄 Trying strategy: cache
   ✅ Strategy 'cache' succeeded

✅ RESULT:
   Answer: The primary and secondary services are unavailable, so I'm 
           providing cached data. It may be slightly outdated.
   Success: True
   Primary Succeeded: False
   Total Attempts: 1
   Fallback Used: cache
   Errors: 2
      • primary: Service unavailable (attempt 1)
      • secondary: Secondary service unavailable
```

### Validation Examples

**Reliability Pattern Checklist**:

```python
✅ Use exponential backoff (not linear)
✅ Add jitter to prevent thundering herd
✅ Set max retry attempts (don't retry forever)
✅ Only retry transient errors (not validation errors)
✅ Use circuit breakers for failing services
✅ Implement fallback chains for alternatives
✅ Combine retry + fallback for critical paths
✅ Log retry attempts for debugging
✅ Track metrics (retry count, fallback usage)
✅ Test failure scenarios
```

### Type Checking

```bash
mypy lesson_14_retries_fallbacks.py
```

Expected: `Success: no issues found`

---

## D. Common Stumbling Blocks

### 1. Retrying Non-Transient Errors

**The Problem**:
```python
@retry_with_backoff(retry_on=(Exception,))  # ❌ Retries EVERYTHING
def process_data(value: int):
    if value < 0:
        raise ValueError("Value must be positive")  # User error!
    return fetch_from_api(value)  # Network error - transient

# Retries user input errors that will never succeed!
```

**The Fix**:
Only retry transient errors:
```python
@retry_with_backoff(
    retry_on=(ConnectionError, TimeoutError)  # ✅ Only transient
)
def process_data(value: int):
    if value < 0:
        raise ValueError("Value must be positive")  # ✅ Not retried
    return fetch_from_api(value)  # ✅ Retried on network issues
```

### 2. No Backoff Delay

**The Problem**:
```python
for attempt in range(100):
    try:
        return fetch_data()
    except:
        pass  # ❌ No delay - hammers service immediately!
# Overwhelms the failing service
```

**The Fix**:
Always use backoff:
```python
for attempt in range(3):
    try:
        return fetch_data()
    except ConnectionError:
        if attempt < 2:
            delay = 0.5 * (2 ** attempt)  # ✅ 0.5s, 1s
            await asyncio.sleep(delay)
```

### 3. Circuit Breaker Never Resets

**The Problem**:
```python
# ❌ No timeout - circuit stays open forever
breaker = CircuitBreaker(failure_threshold=3, timeout=0)

# Once open, never recovers even if service is back
```

**The Fix**:
Set reasonable timeout:
```python
# ✅ After 60s, try one request to test recovery
breaker = CircuitBreaker(
    failure_threshold=5,
    timeout=60  # Test recovery after 60 seconds
)
```

### 4. Fallback That Can Also Fail

**The Problem**:
```python
def fetch_data():
    try:
        return primary_api()
    except:
        return secondary_api()  # ❌ This can also fail!
# No final fallback
```

**The Fix**:
Always have a guaranteed fallback:
```python
def fetch_data():
    try:
        return primary_api()
    except Exception as e1:
        try:
            return secondary_api()
        except Exception as e2:
            # ✅ Final fallback that can't fail
            return {
                "data": None,
                "error": "All sources unavailable",
                "errors": [str(e1), str(e2)]
            }
```

### 5. Type Safety Gotcha: Retry Return Type Mismatch

**The Problem**:
```python
@retry_with_backoff()
def fetch_data() -> dict:  # Says returns dict
    if random.random() < 0.5:
        return {"data": "value"}  # ✅ dict
    else:
        return None  # ❌ Actually returns None sometimes!

result = fetch_data()
print(result["data"])  # ❌ Crashes if None returned
```

**The Fix**:
Be explicit about optional returns:
```python
@retry_with_backoff()
def fetch_data() -> Optional[dict]:  # ✅ Honest return type
    if random.random() < 0.5:
        return {"data": "value"}
    else:
        return None

result = fetch_data()
if result is not None:  # ✅ Type-safe check
    print(result["data"])
```

---

## Ready for the Next Lesson?

🎉 **Incredible work!** You now understand comprehensive reliability patterns:

✅ Retry with exponential backoff and jitter  
✅ Circuit breaker pattern for failing services  
✅ Fallback chains for alternative strategies  
✅ Combined retry + fallback for maximum reliability  
✅ Transient vs permanent error classification  
✅ Type-safe retry policies and circuit states  

**Reliability patterns transform fragile prototypes into production-ready systems!** Your agents can now handle the chaos of real-world distributed systems - network failures, service outages, rate limits - and keep working through it all.

In the next lesson, we'll explore **Agent Result Validation** - you'll learn comprehensive validation strategies for agent outputs, custom validators, cross-field validation, and ensuring outputs meet business requirements!

**Ready for Lesson 15, or would you like to practice retry and fallback patterns first?** 🚀
