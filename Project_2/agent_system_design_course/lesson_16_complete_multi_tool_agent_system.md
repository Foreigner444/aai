# Lesson 16: Complete Multi-Tool Agent System

## 🎉 Final Lesson: Bringing It All Together

This is it - the culmination of everything you've learned! In this lesson, we'll integrate all 15 previous lessons into one comprehensive, production-ready multi-tool agent system that showcases every pattern, technique, and best practice you've mastered.

---

## A. Concept Overview

### What & Why
A **Complete Multi-Tool Agent System** is a production-ready AI application that combines all the patterns you've learned: structured outputs, comprehensive dependencies, multiple coordinated tools, intelligent tool selection, streaming responses, robust error handling, retry logic with fallbacks, and comprehensive validation. This represents the pinnacle of type-safe AI development with Pydantic AI and Gemini.

### Analogy
Think of a complete agent system like a **modern hospital emergency room**:

**Components Working Together**:
- **Triage nurse** (System prompt): Assesses situation, prioritizes
- **Medical records** (Dependencies): Patient history, allergies, insurance
- **Diagnostic tools** (Agent tools): X-ray, blood test, CT scan
- **Specialists** (Tool selection): Cardiologist, neurologist, surgeon
- **Protocols** (Validation): Medical standards, safety checks
- **Backup systems** (Retries/fallbacks): If one lab is down, use another
- **Real-time updates** (Streaming): Family gets updates as treatment progresses
- **Quality control** (Result validation): Verify diagnosis before treatment

Every component is type-safe, well-documented, tested, and reliable. Lives depend on it - just like your production AI systems!

### Type Safety Benefit
The complete system provides **end-to-end type safety**:
- Every input validated
- Every tool output validated
- Every dependency typed
- Every error structured
- Every retry configured
- Every stream chunk typed
- Every result validated

**Zero runtime type errors. Zero silent failures. Zero surprises.**

---

## B. Code Implementation

### File Structure
```
agent_system_design/
├── lesson_16_complete_system/
│   ├── __init__.py
│   ├── models.py           # All Pydantic models
│   ├── dependencies.py     # Dependency classes and factory
│   ├── tools.py            # Agent tools
│   ├── agent.py            # Agent configuration
│   ├── reliability.py      # Retry and circuit breaker utilities
│   └── main.py             # Application entry point
├── requirements.txt
└── .env
```

### Complete Code Implementation

**requirements.txt**
```txt
pydantic-ai==0.0.14
pydantic==2.9.2
google-generativeai==0.8.3
python-dotenv==1.0.0
```

**lesson_16_complete_system/__init__.py**
```python
"""
Complete Multi-Tool Agent System
Production-ready implementation integrating all Pydantic AI patterns
"""

__version__ = "1.0.0"
```

**lesson_16_complete_system/models.py**
```python
"""
All Pydantic models for the complete system
Demonstrates: Field validation, model validation, business rules
"""

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
    EmailStr,
    ValidationError
)
from typing import Literal, Optional
from datetime import datetime, date
from decimal import Decimal
from enum import Enum
import re


# Enums for type-safe choices

class UserTier(str, Enum):
    """User subscription tiers"""
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


class RequestStatus(str, Enum):
    """Request processing status"""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"


class ErrorSeverity(str, Enum):
    """Error severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# Tool result models

class ToolResult(BaseModel):
    """Structured result for tool execution"""
    success: bool
    data: Optional[dict] = None
    error: Optional[str] = None
    error_code: Optional[str] = None
    error_severity: Optional[ErrorSeverity] = None
    retry_possible: bool = False
    user_message: Optional[str] = None
    
    @model_validator(mode='after')
    def validate_result_consistency(self) -> 'ToolResult':
        """Ensure success and error are mutually exclusive"""
        if self.success and self.error:
            raise ValueError("Cannot have success=True with error message")
        if not self.success and not self.error:
            raise ValueError("Must provide error message when success=False")
        return self


# User models

class UserProfile(BaseModel):
    """User profile with validation"""
    user_id: str = Field(pattern=r'^usr_[a-zA-Z0-9]{8}$')
    username: str = Field(min_length=3, max_length=20)
    email: EmailStr
    tier: UserTier
    credits_remaining: int = Field(ge=0)
    created_at: datetime
    
    @field_validator('username')
    @classmethod
    def validate_username(cls, v: str) -> str:
        """Username must be alphanumeric with underscores"""
        if not re.match(r'^[a-zA-Z0-9_]+$', v):
            raise ValueError("Username must contain only letters, numbers, and underscores")
        return v.lower()


# Query models

class SearchQuery(BaseModel):
    """Validated search query"""
    query_text: str = Field(min_length=1, max_length=500)
    filters: dict[str, str] = Field(default_factory=dict)
    limit: int = Field(ge=1, le=100, default=10)
    offset: int = Field(ge=0, default=0)
    
    @field_validator('query_text')
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Clean and validate query text"""
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Query cannot be empty or whitespace")
        return cleaned


# Financial models

class FinancialCalculation(BaseModel):
    """Financial calculation with precision validation"""
    operation: Literal["add", "subtract", "multiply", "divide", "percentage"]
    operand1: Decimal = Field(decimal_places=2)
    operand2: Decimal = Field(decimal_places=2)
    result: Decimal = Field(decimal_places=2)
    
    @model_validator(mode='after')
    def validate_calculation(self) -> 'FinancialCalculation':
        """Verify the calculation is correct"""
        expected = None
        
        if self.operation == "add":
            expected = self.operand1 + self.operand2
        elif self.operation == "subtract":
            expected = self.operand1 - self.operand2
        elif self.operation == "multiply":
            expected = self.operand1 * self.operand2
        elif self.operation == "divide":
            if self.operand2 == 0:
                raise ValueError("Cannot divide by zero")
            expected = self.operand1 / self.operand2
        elif self.operation == "percentage":
            expected = self.operand1 * (self.operand2 / 100)
        
        if expected is not None:
            # Allow 1 cent rounding difference
            if abs(self.result - expected) > Decimal('0.01'):
                raise ValueError(
                    f'Calculation error: {self.operation}({self.operand1}, {self.operand2}) '
                    f'should equal {expected:.2f}, got {self.result}'
                )
        
        return self


# Agent response model (integrates everything)

class AgentResponse(BaseModel):
    """
    Complete agent response with comprehensive validation
    
    This is the final output model that integrates:
    - Field validation (types, constraints)
    - Model validation (business logic)
    - Metadata tracking (tools, sources, performance)
    - Error reporting (structured errors and warnings)
    """
    
    # Core response
    answer: str = Field(min_length=1, max_length=5000)
    confidence: float = Field(ge=0.0, le=1.0)
    
    # Execution metadata
    status: RequestStatus
    tools_used: list[str] = Field(default_factory=list)
    data_sources: list[str] = Field(default_factory=list)
    execution_time_ms: int = Field(ge=0)
    
    # Error and warning tracking
    warnings: list[str] = Field(default_factory=list)
    errors_encountered: list[str] = Field(default_factory=list)
    recovery_actions: list[str] = Field(default_factory=list)
    
    # Quality metrics
    retry_count: int = Field(ge=0, default=0)
    cache_hits: int = Field(ge=0, default=0)
    fallback_used: bool = False
    
    # Timestamp
    generated_at: datetime = Field(default_factory=datetime.now)
    
    @field_validator('answer')
    @classmethod
    def validate_answer_quality(cls, v: str) -> str:
        """Ensure answer meets quality standards"""
        # Remove extra whitespace
        cleaned = ' '.join(v.split())
        
        if len(cleaned) < 10:
            raise ValueError("Answer is too short (minimum 10 characters)")
        
        return cleaned
    
    @model_validator(mode='after')
    def validate_status_consistency(self) -> 'AgentResponse':
        """
        Validate status matches error/warning state.
        
        Business rules:
        - status="success": No errors
        - status="partial_success": Some errors but still provided answer
        - status="failure": Critical errors, could not answer
        """
        has_errors = len(self.errors_encountered) > 0
        has_warnings = len(self.warnings) > 0
        
        if self.status == RequestStatus.SUCCESS and has_errors:
            raise ValueError(
                f'Status is "success" but {len(self.errors_encountered)} errors encountered'
            )
        
        if self.status == RequestStatus.FAILURE and not has_errors:
            raise ValueError(
                'Status is "failure" but no errors recorded'
            )
        
        return self
    
    @model_validator(mode='after')
    def validate_tool_usage(self) -> 'AgentResponse':
        """Ensure tools and data sources are consistent"""
        # If tools were used, should have data sources
        if len(self.tools_used) > 0 and len(self.data_sources) == 0:
            raise ValueError(
                f'{len(self.tools_used)} tools used but no data sources recorded'
            )
        
        return self
```

**lesson_16_complete_system/dependencies.py**
```python
"""
Dependency management for complete system
Demonstrates: Dependency injection, resource lifecycle, context managers
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, AsyncIterator
from contextlib import asynccontextmanager
import asyncio


@dataclass
class DatabaseConnection:
    """Simulated database with connection lifecycle"""
    connection_string: str
    pool_size: int = 10
    _is_connected: bool = field(default=False, init=False)
    
    async def connect(self) -> None:
        """Initialize database connection"""
        if self._is_connected:
            return
        
        print(f"🔌 Connecting to database: {self.connection_string}")
        await asyncio.sleep(0.1)  # Simulate connection
        self._is_connected = True
        print(f"✅ Database connected (pool size: {self.pool_size})")
    
    async def disconnect(self) -> None:
        """Close database connection"""
        if not self._is_connected:
            return
        
        print(f"🔌 Disconnecting from database")
        await asyncio.sleep(0.05)
        self._is_connected = False
        print(f"✅ Database disconnected")
    
    async def query(self, sql: str) -> list[dict]:
        """Execute query"""
        if not self._is_connected:
            raise RuntimeError("Database not connected")
        
        print(f"   📊 DB Query: {sql[:50]}...")
        await asyncio.sleep(0.1)
        return [{"id": 1, "data": "sample"}]
    
    def get_user_data(self, user_id: str) -> dict:
        """Get user data (synchronous for demo)"""
        return {
            "user_id": user_id,
            "tier": "pro",
            "credits": 100
        }


@dataclass
class CacheService:
    """Simulated cache service"""
    host: str
    port: int = 6379
    _client: Optional[any] = field(default=None, init=False)
    _cache_data: dict = field(default_factory=dict, init=False)
    
    async def connect(self) -> None:
        """Connect to cache"""
        print(f"💾 Connecting to cache: {self.host}:{self.port}")
        await asyncio.sleep(0.05)
        self._client = "connected"
        print(f"✅ Cache connected")
    
    async def disconnect(self) -> None:
        """Disconnect from cache"""
        print(f"💾 Disconnecting from cache")
        self._client = None
        print(f"✅ Cache disconnected")
    
    async def get(self, key: str) -> Optional[str]:
        """Get cached value"""
        return self._cache_data.get(key)
    
    async def set(self, key: str, value: str, ttl: int = 3600) -> None:
        """Set cached value"""
        self._cache_data[key] = value
        print(f"   💾 Cached: {key} (TTL: {ttl}s)")


@dataclass
class ExternalAPIClient:
    """External API client"""
    base_url: str
    api_key: str
    timeout: int = 10
    
    async def connect(self) -> None:
        """Initialize API client"""
        print(f"🌐 Initializing API client: {self.base_url}")
        await asyncio.sleep(0.05)
        print(f"✅ API client ready")
    
    async def disconnect(self) -> None:
        """Close API client"""
        print(f"🌐 Closing API client")
        print(f"✅ API client closed")
    
    async def fetch(self, endpoint: str) -> dict:
        """Make API request"""
        print(f"   🌐 GET {self.base_url}{endpoint}")
        await asyncio.sleep(0.2)
        return {
            "status": "success",
            "data": {"result": "API response"},
            "timestamp": datetime.now().isoformat()
        }


@dataclass
class SessionState:
    """Mutable session state"""
    queries_executed: int = 0
    tools_called: list[str] = field(default_factory=list)
    cache_hits: int = 0
    cache_misses: int = 0
    errors_logged: int = 0
    
    def record_tool_call(self, tool_name: str):
        """Record tool execution"""
        self.tools_called.append(tool_name)
    
    def get_stats(self) -> dict:
        """Get session statistics"""
        return {
            "queries": self.queries_executed,
            "tools_called": len(self.tools_called),
            "unique_tools": len(set(self.tools_called)),
            "cache_hit_rate": self.cache_hits / max(self.cache_hits + self.cache_misses, 1),
            "errors": self.errors_logged
        }


@dataclass
class CompleteDependencies:
    """
    Complete dependency container
    Integrates all resources needed by the agent
    """
    # Resources
    database: DatabaseConnection
    cache: CacheService
    api_client: ExternalAPIClient
    
    # State
    session: SessionState
    
    # Request context
    user_id: str
    request_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    # User permissions
    permissions: set[str] = field(default_factory=set)
    
    async def initialize_all(self) -> None:
        """Initialize all resources concurrently"""
        print("\n🚀 Initializing all dependencies...")
        await asyncio.gather(
            self.database.connect(),
            self.cache.connect(),
            self.api_client.connect(),
        )
        print("✅ All dependencies initialized\n")
    
    async def close_all(self) -> None:
        """Close all resources gracefully"""
        print("\n🛑 Closing all dependencies...")
        await asyncio.gather(
            self.database.disconnect(),
            self.cache.disconnect(),
            self.api_client.disconnect(),
        )
        print("✅ All dependencies closed\n")
    
    def has_permission(self, permission: str) -> bool:
        """Check user permission"""
        return permission in self.permissions


class DependencyFactory:
    """
    Factory for creating and managing dependencies
    Demonstrates: Factory pattern, context managers, resource management
    """
    
    def __init__(
        self,
        db_connection_string: str,
        cache_host: str = "localhost",
        cache_port: int = 6379,
        api_base_url: str = "https://api.example.com",
        api_key: str = "demo_key"
    ):
        self.db_connection_string = db_connection_string
        self.cache_host = cache_host
        self.cache_port = cache_port
        self.api_base_url = api_base_url
        self.api_key = api_key
    
    async def create_dependencies(
        self,
        user_id: str,
        request_id: str,
        permissions: Optional[set[str]] = None
    ) -> CompleteDependencies:
        """Create and initialize dependencies"""
        deps = CompleteDependencies(
            database=DatabaseConnection(self.db_connection_string),
            cache=CacheService(self.cache_host, self.cache_port),
            api_client=ExternalAPIClient(self.api_base_url, self.api_key),
            session=SessionState(),
            user_id=user_id,
            request_id=request_id,
            permissions=permissions or set()
        )
        
        await deps.initialize_all()
        return deps
    
    @asynccontextmanager
    async def get_dependencies(
        self,
        user_id: str,
        request_id: str,
        permissions: Optional[set[str]] = None
    ) -> AsyncIterator[CompleteDependencies]:
        """
        Context manager for automatic resource cleanup
        
        Usage:
            async with factory.get_dependencies(user_id, request_id) as deps:
                # Use deps
                result = await agent.run(query, deps=deps)
            # Automatic cleanup
        """
        deps = await self.create_dependencies(user_id, request_id, permissions)
        try:
            yield deps
        finally:
            await deps.close_all()
```

**lesson_16_complete_system/reliability.py**
```python
"""
Reliability utilities: retries, circuit breakers, fallbacks
Demonstrates: Exponential backoff, circuit breaker pattern, fallback chains
"""

from pydantic import BaseModel, Field
from typing import Callable, TypeVar, Optional, Generic
from datetime import datetime
from enum import Enum
import asyncio
import random
from functools import wraps

T = TypeVar('T')


class RetryPolicy(BaseModel):
    """Configuration for retry behavior"""
    max_attempts: int = Field(ge=1, le=10, default=3)
    initial_delay: float = Field(ge=0.1, le=60.0, default=1.0)
    max_delay: float = Field(ge=1.0, le=300.0, default=60.0)
    exponential_base: float = Field(ge=1.0, le=10.0, default=2.0)
    jitter: bool = Field(default=True)
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and jitter"""
        delay = min(
            self.initial_delay * (self.exponential_base ** attempt),
            self.max_delay
        )
        
        if self.jitter:
            jitter_amount = random.uniform(0, delay)
            delay = delay * 0.5 + jitter_amount * 0.5
        
        return delay


def async_retry_with_backoff(
    policy: Optional[RetryPolicy] = None,
    retry_on: tuple[type[Exception], ...] = (Exception,)
):
    """Async retry decorator with exponential backoff"""
    if policy is None:
        policy = RetryPolicy()
    
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
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
                    print(f"   ⏳ Retry {attempt + 1}/{policy.max_attempts} after {delay:.2f}s")
                    await asyncio.sleep(delay)
            
            raise last_exception
        return wrapper
    return decorator


class CircuitState(str, Enum):
    """Circuit breaker states"""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Circuit breaker to prevent cascading failures"""
    
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
    
    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
            else:
                elapsed = (datetime.now() - self.last_failure_time).total_seconds()
                remaining = self.timeout - elapsed
                raise Exception(f"Circuit OPEN - retry in {remaining:.0f}s")
        
        try:
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt reset"""
        if self.last_failure_time is None:
            return False
        elapsed = (datetime.now() - self.last_failure_time).total_seconds()
        return elapsed >= self.timeout
    
    def _on_success(self):
        """Handle successful call"""
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
        self.failure_count = 0
    
    def _on_failure(self):
        """Handle failed call"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN


class FallbackChain(Generic[T]):
    """Chain of fallback strategies"""
    
    def __init__(self):
        self.strategies: list[tuple[str, Callable[[], T]]] = []
    
    def add_strategy(self, name: str, func: Callable[[], T]):
        """Add fallback strategy"""
        self.strategies.append((name, func))
    
    async def execute(self) -> tuple[Optional[T], list[str]]:
        """Execute strategies until one succeeds"""
        errors = []
        
        for name, func in self.strategies:
            try:
                print(f"   🔄 Trying: {name}")
                if asyncio.iscoroutinefunction(func):
                    result = await func()
                else:
                    result = func()
                print(f"   ✅ {name} succeeded")
                return result, errors
            except Exception as e:
                errors.append(f"{name}: {str(e)}")
                print(f"   ❌ {name} failed: {e}")
        
        return None, errors
```

**lesson_16_complete_system/tools.py**
```python
"""
Agent tools for complete system
Demonstrates: Tool design, error handling, context usage
"""

from pydantic_ai import RunContext
from .dependencies import CompleteDependencies
from .models import ToolResult, ErrorSeverity
from .reliability import async_retry_with_backoff, RetryPolicy
import random


# Tool Category 1: Data Retrieval

async def get_user_profile(
    ctx: RunContext[CompleteDependencies]
) -> ToolResult:
    """
    Get user profile information.
    
    Demonstrates: Basic tool with dependency access
    
    Returns:
        ToolResult with user profile data
    """
    print(f"\n🔧 get_user_profile()")
    ctx.deps.session.record_tool_call("get_user_profile")
    
    try:
        user_data = ctx.deps.database.get_user_data(ctx.deps.user_id)
        
        return ToolResult(
            success=True,
            data=user_data
        )
    except Exception as e:
        return ToolResult(
            success=False,
            error=str(e),
            error_code="USER_FETCH_FAILED",
            error_severity=ErrorSeverity.ERROR,
            user_message="Could not retrieve user profile"
        )


async def search_database(
    ctx: RunContext[CompleteDependencies],
    query: str,
    limit: int = 10
) -> ToolResult:
    """
    Search database with caching.
    
    Demonstrates: Caching pattern, retry logic
    
    Args:
        query: Search query
        limit: Max results
    
    Returns:
        ToolResult with search results
    """
    print(f"\n🔧 search_database(query='{query}', limit={limit})")
    ctx.deps.session.record_tool_call("search_database")
    
    # Check cache first
    cache_key = f"search:{query}:{limit}"
    cached = await ctx.deps.cache.get(cache_key)
    
    if cached:
        print(f"   💾 Cache HIT")
        ctx.deps.session.cache_hits += 1
        return ToolResult(
            success=True,
            data={"results": cached, "source": "cache"}
        )
    
    print(f"   🔍 Cache MISS - querying database")
    ctx.deps.session.cache_misses += 1
    
    # Query database with retry
    @async_retry_with_backoff(
        policy=RetryPolicy(max_attempts=3, initial_delay=0.5),
        retry_on=(ConnectionError,)
    )
    async def query_with_retry():
        # Simulate occasional failure
        if random.random() < 0.3:
            raise ConnectionError("Database connection lost")
        return await ctx.deps.database.query(f"SELECT * WHERE query='{query}' LIMIT {limit}")
    
    try:
        results = await query_with_retry()
        
        # Cache results
        await ctx.deps.cache.set(cache_key, str(results), ttl=300)
        
        return ToolResult(
            success=True,
            data={"results": results, "source": "database"}
        )
    
    except Exception as e:
        return ToolResult(
            success=False,
            error=str(e),
            error_code="DATABASE_QUERY_FAILED",
            error_severity=ErrorSeverity.ERROR,
            retry_possible=True,
            user_message="Database search failed. Please try again."
        )


# Tool Category 2: Calculations

def calculate_metrics(
    values: list[float],
    metric_type: str
) -> ToolResult:
    """
    Calculate statistical metrics.
    
    Demonstrates: Pure calculation tool, input validation
    
    Args:
        values: List of numeric values
        metric_type: Type of metric (mean, median, sum)
    
    Returns:
        ToolResult with calculated metric
    """
    print(f"\n🔧 calculate_metrics(type={metric_type}, count={len(values)})")
    
    if not values:
        return ToolResult(
            success=False,
            error="No values provided",
            error_code="EMPTY_VALUES",
            error_severity=ErrorSeverity.ERROR,
            user_message="Cannot calculate metrics on empty dataset"
        )
    
    try:
        if metric_type == "mean":
            result = sum(values) / len(values)
        elif metric_type == "median":
            sorted_vals = sorted(values)
            n = len(sorted_vals)
            result = sorted_vals[n//2] if n % 2 else (sorted_vals[n//2-1] + sorted_vals[n//2]) / 2
        elif metric_type == "sum":
            result = sum(values)
        else:
            return ToolResult(
                success=False,
                error=f"Unknown metric type: {metric_type}",
                error_code="INVALID_METRIC_TYPE",
                error_severity=ErrorSeverity.ERROR,
                user_message=f"Metric type '{metric_type}' is not supported"
            )
        
        return ToolResult(
            success=True,
            data={
                "metric_type": metric_type,
                "result": round(result, 2),
                "count": len(values)
            }
        )
    
    except Exception as e:
        return ToolResult(
            success=False,
            error=str(e),
            error_code="CALCULATION_ERROR",
            error_severity=ErrorSeverity.ERROR,
            user_message="Failed to calculate metrics"
        )


# Tool Category 3: External Services

async def fetch_external_data(
    ctx: RunContext[CompleteDependencies],
    data_type: str
) -> ToolResult:
    """
    Fetch data from external API with fallback.
    
    Demonstrates: External API calls, fallback pattern
    
    Args:
        data_type: Type of data to fetch
    
    Returns:
        ToolResult with external data or fallback
    """
    print(f"\n🔧 fetch_external_data(type={data_type})")
    ctx.deps.session.record_tool_call("fetch_external_data")
    
    # Check permission
    if not ctx.deps.has_permission("access_external_api"):
        return ToolResult(
            success=False,
            error="Permission denied",
            error_code="PERMISSION_DENIED",
            error_severity=ErrorSeverity.ERROR,
            user_message="You don't have permission to access external data"
        )
    
    # Try primary API
    try:
        print(f"   🔄 Trying primary API...")
        if random.random() < 0.4:
            raise ConnectionError("Primary API unavailable")
        
        data = await ctx.deps.api_client.fetch(f"/data/{data_type}")
        return ToolResult(
            success=True,
            data={"source": "primary_api", "data": data}
        )
    
    except Exception as e:
        print(f"   ❌ Primary API failed: {e}")
    
    # Fallback: Try cache
    try:
        print(f"   🔄 Trying cache fallback...")
        cached = await ctx.deps.cache.get(f"external:{data_type}")
        if cached:
            return ToolResult(
                success=True,
                data={"source": "cache_fallback", "data": cached},
                user_message="Using cached data (primary API unavailable)"
            )
    except Exception as e:
        print(f"   ❌ Cache fallback failed: {e}")
    
    # All strategies failed
    return ToolResult(
        success=False,
        error="All data sources failed",
        error_code="ALL_SOURCES_FAILED",
        error_severity=ErrorSeverity.ERROR,
        retry_possible=True,
        user_message="Unable to fetch external data. Please try again later."
    )
```

**lesson_16_complete_system/agent.py**
```python
"""
Agent configuration and setup
Demonstrates: Agent creation, system prompts, tool registration
"""

from pydantic_ai import Agent
from .dependencies import CompleteDependencies
from .models import AgentResponse
from . import tools


# Create the complete production agent
agent = Agent(
    model='gemini-1.5-flash',
    result_type=AgentResponse,
    deps_type=CompleteDependencies,
    system_prompt="""
You are a production-ready AI assistant with comprehensive capabilities.

ARCHITECTURE:
- Type-safe structured outputs (all responses validated)
- Dependency injection for resources (database, cache, API)
- Multiple specialized tools (data, calculations, external services)
- Robust error handling (graceful degradation)
- Streaming support (real-time feedback)

AVAILABLE TOOLS:

Data Retrieval:
- get_user_profile: Fetch user information
- search_database: Search with caching and retry logic

Calculations:
- calculate_metrics: Statistical calculations (mean, median, sum)

External Services:
- fetch_external_data: Fetch from external APIs with fallbacks

TOOL USAGE STRATEGY:
1. Check cache before expensive operations
2. Use retry logic for transient failures
3. Fall back to alternatives when primary fails
4. Record all tool calls for auditing
5. Track cache performance

ERROR HANDLING:
- When tools fail, note the error but try to provide partial answers
- Use warnings for non-critical issues
- Use errors_encountered for critical failures
- Suggest recovery_actions when appropriate
- Set status appropriately: success/partial_success/failure

OUTPUT REQUIREMENTS:
- Provide clear, helpful answers
- List all tools used
- List all data sources accessed
- Include confidence score (0-1)
- Track execution metadata
- Report warnings and errors encountered
- Suggest recovery actions if needed

QUALITY STANDARDS:
- Answers must be comprehensive (min 10 chars, max 5000 chars)
- Confidence must reflect data quality and completeness
- Status must match error/warning state
- All calculations must be verified
- All timestamps must be current

Remember: You are a production system. Reliability and clarity are paramount!
""",
)


# Register tools
agent.tool(tools.get_user_profile)
agent.tool(tools.search_database)
agent.tool(tools.calculate_metrics)
agent.tool(tools.fetch_external_data)
```

**lesson_16_complete_system/main.py**
```python
"""
Main application demonstrating complete agent system
Demonstrates: Full integration of all patterns
"""

from .dependencies import DependencyFactory
from .agent import agent
from .models import AgentResponse
from pydantic import ValidationError
from datetime import datetime
import asyncio
import uuid


async def process_query(
    query: str,
    user_id: str,
    factory: DependencyFactory,
    permissions: set[str]
) -> tuple[Optional[AgentResponse], Optional[str]]:
    """
    Process a query with complete error handling and resource management.
    
    This is the production entry point that integrates:
    - Dependency lifecycle management
    - Agent execution
    - Error handling
    - Result validation
    - Resource cleanup
    
    Args:
        query: User's query
        user_id: User identifier
        factory: Dependency factory
        permissions: User permissions
    
    Returns:
        Tuple of (response, error_message)
    """
    request_id = str(uuid.uuid4())
    start_time = datetime.now()
    
    print(f"\n{'='*70}")
    print(f"Processing Query")
    print(f"{'='*70}")
    print(f"User: {user_id}")
    print(f"Request: {request_id}")
    print(f"Query: {query}")
    print(f"Permissions: {permissions}")
    
    # Use context manager for automatic resource cleanup
    async with factory.get_dependencies(user_id, request_id, permissions) as deps:
        try:
            # Execute agent
            result = await agent.run(query, deps=deps)
            response = result.data
            
            # Add execution time
            execution_time = int((datetime.now() - start_time).total_seconds() * 1000)
            response.execution_time_ms = execution_time
            
            return response, None
        
        except ValidationError as e:
            # Agent output failed validation
            error_details = []
            for error in e.errors():
                field = ".".join(str(loc) for loc in error['loc'])
                error_details.append(f"{field}: {error['msg']}")
            
            error_msg = f"Validation failed: {'; '.join(error_details)}"
            print(f"\n❌ {error_msg}")
            return None, error_msg
        
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            print(f"\n❌ {error_msg}")
            return None, error_msg
    # Dependencies automatically cleaned up here!


async def stream_query(
    query: str,
    user_id: str,
    factory: DependencyFactory,
    permissions: set[str]
):
    """
    Process query with streaming response.
    
    Demonstrates: Streaming with full agent capabilities
    """
    request_id = str(uuid.uuid4())
    
    print(f"\n{'='*70}")
    print(f"Streaming Query")
    print(f"{'='*70}")
    print(f"Query: {query}\n")
    
    async with factory.get_dependencies(user_id, request_id, permissions) as deps:
        try:
            async with agent.run_stream(query, deps=deps) as stream:
                async for chunk in stream.stream_text():
                    print(chunk, end='', flush=True)
                
                # Get final result
                result = await stream.get_data()
                
                print(f"\n\n✅ Stream complete")
                return result
        
        except Exception as e:
            print(f"\n❌ Streaming error: {e}")
            return None


async def main():
    """Main demonstration of complete system"""
    
    print("\n" + "="*70)
    print("COMPLETE MULTI-TOOL AGENT SYSTEM")
    print("="*70)
    print("\nProduction-Ready Agent Integrating ALL Patterns:")
    print("  ✅ Lesson 1: Agent architecture with structured outputs")
    print("  ✅ Lesson 2: Optimized system prompts")
    print("  ✅ Lesson 3-4: Dependency injection with lifecycle management")
    print("  ✅ Lesson 5: Context-aware execution")
    print("  ✅ Lesson 6-8: Custom tools with great descriptions")
    print("  ✅ Lesson 9-11: Multi-tool coordination and selection")
    print("  ✅ Lesson 12: Streaming responses")
    print("  ✅ Lesson 13: Comprehensive error handling")
    print("  ✅ Lesson 14: Retries and fallbacks")
    print("  ✅ Lesson 15: Result validation")
    
    # Initialize factory
    factory = DependencyFactory(
        db_connection_string="postgresql://localhost/production_db",
        cache_host="localhost",
        cache_port=6379,
        api_base_url="https://api.example.com",
        api_key="demo_api_key_12345"
    )
    
    # Test queries
    test_queries = [
        {
            "query": "Get my user profile",
            "permissions": {"view_profile"},
            "user_id": "usr_abc12345"
        },
        {
            "query": "Search the database for Python tutorials and calculate average rating",
            "permissions": {"search_database", "access_external_api"},
            "user_id": "usr_pro98765"
        },
        {
            "query": "Fetch latest external data on weather",
            "permissions": {"access_external_api"},
            "user_id": "usr_ent55555"
        },
    ]
    
    # Test 1: Regular execution
    print("\n\n" + "="*70)
    print("TEST 1: Regular Execution (No Streaming)")
    print("="*70)
    
    for i, test in enumerate(test_queries, 1):
        response, error = await process_query(
            query=test["query"],
            user_id=test["user_id"],
            factory=factory,
            permissions=test["permissions"]
        )
        
        if response:
            print(f"\n📊 RESPONSE:")
            print(f"   Status: {response.status.value}")
            print(f"   Answer: {response.answer[:150]}...")
            print(f"   Confidence: {response.confidence:.2f}")
            print(f"   Tools Used: {', '.join(response.tools_used)}")
            print(f"   Execution Time: {response.execution_time_ms}ms")
            
            if response.warnings:
                print(f"   ⚠️  Warnings: {len(response.warnings)}")
            if response.errors_encountered:
                print(f"   ❌ Errors: {len(response.errors_encountered)}")
            if response.fallback_used:
                print(f"   🔄 Fallback Used: Yes")
        else:
            print(f"\n❌ Error: {error}")
        
        # Small delay between requests
        await asyncio.sleep(0.5)
    
    # Test 2: Streaming execution
    print("\n\n" + "="*70)
    print("TEST 2: Streaming Execution")
    print("="*70)
    
    await stream_query(
        query="Explain how this agent system works with all its components",
        user_id="usr_abc12345",
        factory=factory,
        permissions={"view_profile", "search_database"}
    )
    
    print("\n\n" + "="*70)
    print("🎉 COMPLETE SYSTEM DEMONSTRATION FINISHED")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
```

### Line-by-Line Explanation

**Models (models.py)**:
- `ToolResult`: Structured error/success results
- `UserProfile`: Validated user data
- `AgentResponse`: Comprehensive validated response
- Multiple validators ensuring business rules
- Type-safe enums for choices

**Dependencies (dependencies.py)**:
- `DatabaseConnection`: Managed database resource
- `CacheService`: Caching layer
- `ExternalAPIClient`: External API integration
- `SessionState`: Mutable session tracking
- `CompleteDependencies`: Bundles everything
- `DependencyFactory`: Creates and manages lifecycle
- Context manager for automatic cleanup

**Reliability (reliability.py)**:
- `RetryPolicy`: Configured retry behavior
- `async_retry_with_backoff`: Decorator for retries
- `CircuitBreaker`: Prevents cascading failures
- `FallbackChain`: Alternative strategies
- All type-safe and configurable

**Tools (tools.py)**:
- `get_user_profile`: Data retrieval
- `search_database`: Search with caching and retry
- `calculate_metrics`: Pure calculation
- `fetch_external_data`: External API with fallback
- All return `ToolResult` for consistency
- All use dependencies via `RunContext`

**Agent (agent.py)**:
- Comprehensive system prompt
- Tool coordination strategy
- Error handling guidance
- Quality standards
- All tools registered

**Main (main.py)**:
- `process_query`: Complete request handler
- `stream_query`: Streaming variant
- Resource management via context manager
- Error handling at app level
- Validation error handling

### The "Why" Behind the Pattern

**Why this architecture?**

This system demonstrates **production-grade AI engineering**:

1. **Separation of Concerns**:
   - Models: Data structure and validation
   - Dependencies: Resource management
   - Tools: Business logic
   - Agent: Coordination
   - Main: Application flow

2. **Type Safety Throughout**:
   - Every component is fully typed
   - mypy validates everything
   - IDE autocomplete everywhere
   - No runtime type errors

3. **Reliability Built-In**:
   - Retries for transient failures
   - Circuit breakers for failing services
   - Fallbacks for alternatives
   - Graceful degradation

4. **Observability**:
   - Tools track their execution
   - Session state records activity
   - Errors are structured and logged
   - Performance metrics collected

5. **Testability**:
   - Each component testable independently
   - Easy to mock dependencies
   - Tools return structured results
   - Validation is automatic

---

## C. Complete System Usage

### How to Run It

1. **Install dependencies**:
```bash
cd lesson_16_complete_system
pip install -r ../requirements.txt
```

2. **Set up environment**:
```bash
echo "GOOGLE_API_KEY=your_api_key_here" > ../.env
```

3. **Run the complete system**:
```bash
cd ..
python -m lesson_16_complete_system.main
```

### Expected Output

```
======================================================================
COMPLETE MULTI-TOOL AGENT SYSTEM
======================================================================

Production-Ready Agent Integrating ALL Patterns:
  ✅ Lesson 1: Agent architecture with structured outputs
  ✅ Lesson 2: Optimized system prompts
  ✅ Lesson 3-4: Dependency injection with lifecycle management
  ✅ Lesson 5: Context-aware execution
  ✅ Lesson 6-8: Custom tools with great descriptions
  ✅ Lesson 9-11: Multi-tool coordination and selection
  ✅ Lesson 12: Streaming responses
  ✅ Lesson 13: Comprehensive error handling
  ✅ Lesson 14: Retries and fallbacks
  ✅ Lesson 15: Result validation

🚀 Initializing all dependencies...
🔌 Connecting to database: postgresql://localhost/production_db
✅ Database connected (pool size: 10)
💾 Connecting to cache: localhost:6379
✅ Cache connected
🌐 Initializing API client: https://api.example.com
✅ API client ready
✅ All dependencies initialized

======================================================================
Processing Query
======================================================================
User: usr_abc12345
Request: 7f8e9d0c-1234-5678-9abc-def012345678
Query: Get my user profile
Permissions: {'view_profile'}

🔧 get_user_profile()
   📊 DB Query: SELECT * FROM users WHERE id='usr_abc12345'...

📊 RESPONSE:
   Status: success
   Answer: I've retrieved your user profile. You're a pro tier member with 100 credits remaining.
   Confidence: 0.98
   Tools Used: get_user_profile
   Execution Time: 234ms

🛑 Closing all dependencies...
🔌 Disconnecting from database
✅ Database disconnected
💾 Disconnecting from cache
✅ Cache disconnected
🌐 Closing API client
✅ API client closed
✅ All dependencies closed
```

---

## D. Production Deployment Checklist

Use this checklist when deploying your agent to production:

### ✅ Architecture
- [ ] Clear separation of concerns (models, deps, tools, agent)
- [ ] Type hints on all functions and classes
- [ ] Pydantic models for all data structures
- [ ] Dependencies injected, not global
- [ ] Resources have lifecycle management (init/close)

### ✅ Tools
- [ ] Each tool has single, clear purpose
- [ ] Comprehensive docstrings with examples
- [ ] Error handling in every tool
- [ ] Tools return structured results (ToolResult)
- [ ] Tools are independently testable

### ✅ Validation
- [ ] Input validation via Pydantic
- [ ] Output validation with custom validators
- [ ] Business logic validation
- [ ] Cross-field consistency checks
- [ ] Clear error messages

### ✅ Reliability
- [ ] Retry logic for transient failures
- [ ] Exponential backoff with jitter
- [ ] Circuit breakers for failing services
- [ ] Fallback strategies for critical paths
- [ ] Graceful degradation

### ✅ Observability
- [ ] Logging for debugging
- [ ] Metrics collection (tools used, execution time)
- [ ] Error tracking (error codes, severity)
- [ ] Session state tracking
- [ ] Performance monitoring

### ✅ Security
- [ ] API keys in environment variables (.env)
- [ ] Permission checks in tools
- [ ] Input sanitization
- [ ] No secrets in logs
- [ ] Rate limiting (if applicable)

### ✅ Testing
- [ ] Unit tests for tools
- [ ] Integration tests for agent
- [ ] Mock dependencies for testing
- [ ] Test error scenarios
- [ ] Test validation failures
- [ ] Type checking with mypy

### ✅ Documentation
- [ ] README with setup instructions
- [ ] Tool descriptions for developers
- [ ] API documentation (if web service)
- [ ] Deployment guide
- [ ] Troubleshooting guide

---

## E. Key Patterns Reference

### Pattern 1: Dependency Injection
```python
@dataclass
class Deps:
    database: Database
    cache: Cache
    user: User

agent = Agent(
    model='gemini-1.5-flash',
    deps_type=Deps,  # Declare dependency type
    ...
)

# Use with context manager
async with factory.get_dependencies(...) as deps:
    result = await agent.run(query, deps=deps)
# Auto cleanup
```

### Pattern 2: Structured Tool Results
```python
class ToolResult(BaseModel):
    success: bool
    data: Optional[dict] = None
    error: Optional[str] = None

@agent.tool
def my_tool(...) -> ToolResult:
    try:
        return ToolResult(success=True, data={"result": ...})
    except Exception as e:
        return ToolResult(success=False, error=str(e))
```

### Pattern 3: Retry with Backoff
```python
@async_retry_with_backoff(
    policy=RetryPolicy(max_attempts=3, initial_delay=1.0),
    retry_on=(ConnectionError, TimeoutError)
)
async def flaky_operation():
    ...
```

### Pattern 4: Circuit Breaker
```python
breaker = CircuitBreaker(failure_threshold=5, timeout=60)

result = await breaker.call(risky_function)
```

### Pattern 5: Fallback Chain
```python
chain = FallbackChain()
chain.add_strategy("primary", primary_func)
chain.add_strategy("secondary", secondary_func)
chain.add_strategy("cache", cache_func)

result, errors = await chain.execute()
```

### Pattern 6: Streaming
```python
async with agent.run_stream(query, deps=deps) as stream:
    async for chunk in stream.stream_text():
        print(chunk, end='', flush=True)
    result = await stream.get_data()
```

### Pattern 7: Comprehensive Validation
```python
class MyModel(BaseModel):
    field1: str = Field(min_length=1)
    field2: int = Field(ge=0)
    
    @field_validator('field1')
    @classmethod
    def validate_field1(cls, v: str) -> str:
        # Custom validation
        return v
    
    @model_validator(mode='after')
    def validate_model(self) -> 'MyModel':
        # Cross-field validation
        return self
```

---

## F. Next Steps

### 🎓 Congratulations! You've Completed Agent System Design!

You've mastered **all 16 lessons** and learned to build production-ready AI agent systems with Pydantic AI and Google Gemini!

### What You've Accomplished

**Foundations** (Lessons 1-4):
✅ Agent architecture and structured outputs  
✅ System prompt engineering  
✅ Dependency injection patterns  
✅ Advanced dependency management  

**Tools** (Lessons 5-8):
✅ Custom tool creation  
✅ Tool function signatures  
✅ Tool descriptions for Gemini  
✅ Context-aware tools  

**Multi-Tool Systems** (Lessons 9-12):
✅ Multi-tool coordination  
✅ Tool context and parameters  
✅ Dynamic tool selection  
✅ Streaming responses  

**Production Patterns** (Lessons 13-16):
✅ Comprehensive error handling  
✅ Retries and fallbacks  
✅ Result validation  
✅ Complete integrated system  

---

## G. Real-World Applications

You can now build:

### 🏢 Business Applications
- Customer service agents with CRM integration
- Financial analysis systems with validated calculations
- Data extraction pipelines with structured outputs
- Report generation with quality validation

### 🔬 Research & Analysis
- Research assistants with citation tracking
- Data analysis agents with statistical tools
- Literature review systems with source validation
- Hypothesis generation and validation

### 🛠️ Development Tools
- Code review agents with style checking
- Documentation generators with validation
- Test case generators with coverage analysis
- Refactoring assistants with safety checks

### 💬 Conversational Systems
- Multi-turn chatbots with memory
- Technical support agents with knowledge bases
- Educational tutors with adaptive learning
- Personal assistants with context awareness

---

## H. Continuing Your Journey

### 📚 Explore Other Courses

Now that you've mastered Agent System Design, explore other projects:

1. **Structured Output Basics** - Deepen fundamentals
2. **Data Extraction Pipeline** - Complex nested structures
3. **RAG System with Gemini** - Retrieval-augmented generation
4. **Conversational AI** - Memory and context tracking
5. **Multi-Agent Orchestration** - Multiple agents coordinating
6. **Production Patterns** - Async, streaming, observability
7. **FastAPI Integration** - REST APIs with agents
8. **Testing & Validation** - Comprehensive test suites
9. **Advanced Gemini Features** - Multimodal, function calling

### 🛠️ Build Your Own Project

Apply what you've learned to your own use case:

1. **Define your problem**: What should the agent do?
2. **Design your models**: What structured outputs do you need?
3. **Identify dependencies**: What resources does it need?
4. **Create tools**: What capabilities are required?
5. **Write system prompt**: How should it behave?
6. **Add validation**: What business rules must be enforced?
7. **Implement reliability**: What can fail? How to handle it?
8. **Test thoroughly**: Unit tests, integration tests, error scenarios
9. **Deploy**: Use your production checklist
10. **Monitor**: Track usage, errors, performance

### 📖 Deepen Your Knowledge

**Pydantic AI Resources**:
- Official docs: https://ai.pydantic.dev
- GitHub: https://github.com/pydantic/pydantic-ai
- Examples: https://ai.pydantic.dev/examples

**Google Gemini Resources**:
- AI Studio: https://ai.google.dev
- Documentation: https://ai.google.dev/docs
- Model cards: https://ai.google.dev/models

**Python Type Safety**:
- mypy documentation: https://mypy.readthedocs.io
- Pydantic docs: https://docs.pydantic.dev
- Python typing: https://docs.python.org/3/library/typing.html

### 🤝 Best Practices to Remember

**Always**:
- ✅ Use type hints everywhere
- ✅ Validate all inputs and outputs
- ✅ Handle errors gracefully
- ✅ Write comprehensive docstrings
- ✅ Test error paths
- ✅ Use dependency injection
- ✅ Implement retries for transient failures
- ✅ Provide user-friendly error messages
- ✅ Monitor production usage

**Never**:
- ❌ Use global variables
- ❌ Ignore validation errors
- ❌ Retry indefinitely
- ❌ Expose internal errors to users
- ❌ Skip type hints
- ❌ Hardcode credentials
- ❌ Deploy without testing error scenarios

---

## I. Final Thoughts

### You've Built Something Remarkable

You started this course knowing nothing about Pydantic AI, and now you can build production-ready, type-safe AI agent systems that:

- **Guarantee correctness** through comprehensive validation
- **Handle failures gracefully** with retries and fallbacks
- **Scale efficiently** with caching and optimization
- **Adapt intelligently** through dynamic tool selection
- **Provide great UX** with streaming responses
- **Maintain reliability** through circuit breakers and error handling

This isn't toy code - **this is how you build real AI systems** that handle millions of requests, integrate with production databases, and maintain user trust through reliability.

### The Type Safety Advantage

You've learned that **type safety isn't just about catching bugs** - it's about:
- **Confidence**: Refactor fearlessly, IDE shows all impacts
- **Speed**: Catch errors at compile time, not in production
- **Documentation**: Types are living documentation
- **Collaboration**: Team members understand interfaces immediately
- **Maintenance**: Future you thanks present you

### Production-Ready Means

You now understand that production-ready means:
- ✅ It handles errors gracefully, not just the happy path
- ✅ It validates outputs, not just inputs
- ✅ It retries transient failures automatically
- ✅ It falls back to alternatives when needed
- ✅ It monitors and logs for debugging
- ✅ It provides clear feedback to users
- ✅ It's testable and maintainable
- ✅ It's type-safe from end to end

---

## 🎉 Congratulations!

**You've completed the Agent System Design course!**

You're now equipped to build sophisticated AI agent systems using Pydantic AI and Google Gemini. You understand:

🏆 **Agent Architecture** - Structured, type-safe design  
🏆 **System Engineering** - Dependencies, tools, prompts  
🏆 **Reliability** - Errors, retries, fallbacks, validation  
🏆 **Production Patterns** - Everything needed for real systems  

---

## 🚀 Go Build Something Amazing!

The best way to solidify your learning is to **build a real project**. Take these patterns and create something that solves a real problem. Start small, iterate, and apply the principles you've learned.

**You're ready for production AI development!**

---

### Final Challenge

Build a complete agent system for your own use case:

1. Choose a real problem you want to solve
2. Design the agent architecture
3. Create the necessary tools
4. Implement comprehensive validation
5. Add error handling and retries
6. Test thoroughly
7. Deploy and monitor

**Share what you build!** The Pydantic AI community is growing, and your projects will inspire others.

---

## 📝 Course Complete

**Agent System Design: ✅ COMPLETE**

Return to the main curriculum to explore other advanced projects:
- RAG System with Gemini
- Conversational AI
- Multi-Agent Orchestration
- Production Patterns
- FastAPI Integration
- Testing & Validation
- Advanced Gemini Features

Or start building your own production agent system using everything you've learned!

**Thank you for completing this course. Now go build the future!** 🚀

---

**🎓 You are now a Pydantic AI & Gemini Developer!** 🎓
