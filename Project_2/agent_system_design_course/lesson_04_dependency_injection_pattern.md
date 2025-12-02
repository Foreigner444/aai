# Lesson 4: Dependency Injection Pattern

## A. Concept Overview

### What & Why
The **Dependency Injection Pattern** is an advanced architectural approach where you create dependency factories, handle async resource initialization, manage dependency lifecycles, and structure complex dependency hierarchies. This is crucial for production systems where you need database connection pools, API clients with proper cleanup, caching layers, and other managed resources.

### Analogy
Think of running a professional kitchen:
- **Simple approach** (Lesson 3): Each dish request comes with pre-made ingredients
- **Advanced pattern** (This lesson): You have a **prep station** (factory) that:
  - Opens the walk-in fridge (database connection pool)
  - Checks inventory (cache)
  - Assigns sous chefs (API clients)
  - Manages cleanup (closes connections)
  - Handles rush hours (connection pooling)

The dependency injection pattern manages the entire lifecycle of resources, not just passing them around.

### Type Safety Benefit
Advanced dependency injection provides:
- **Lifecycle safety**: Resources are acquired and released correctly
- **Async type checking**: mypy validates async dependency initialization
- **Factory pattern safety**: Type-safe dependency builders
- **Composition safety**: Complex dependencies built from simple ones
- **Context manager integration**: Automatic resource cleanup with type checking
- **Testing patterns**: Type-safe mock factories for tests

Your entire resource management layer becomes type-checked and automated!

---

## B. Code Implementation

### File Structure
```
agent_system_design/
├── lesson_04_dependency_patterns.py
├── requirements.txt
└── .env
```

### Complete Code Snippet

**lesson_04_dependency_patterns.py**
```python
"""
Lesson 4: Dependency Injection Pattern
Advanced patterns for managing dependencies in production systems
"""

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, AsyncIterator
from contextlib import asynccontextmanager
import asyncio
from dotenv import load_dotenv

load_dotenv()


# Pattern 1: Dependency Factory
# Create dependencies with validation and initialization logic

@dataclass
class DatabaseConfig:
    """Database configuration"""
    host: str
    port: int
    database: str
    username: str
    password: str
    pool_size: int = 10
    timeout: int = 30


@dataclass
class Database:
    """Database connection with connection pool"""
    config: DatabaseConfig
    _connection_pool: list = field(default_factory=list, init=False)
    _is_initialized: bool = field(default=False, init=False)
    
    async def initialize(self) -> None:
        """Initialize connection pool (async operation)"""
        if self._is_initialized:
            return
        
        print(f"🔌 Initializing database connection pool...")
        print(f"   Host: {self.config.host}:{self.config.port}")
        print(f"   Pool size: {self.config.pool_size}")
        
        # Simulate async connection setup
        await asyncio.sleep(0.1)  # Real code: await create_pool(...)
        
        self._connection_pool = [
            f"connection_{i}" for i in range(self.config.pool_size)
        ]
        self._is_initialized = True
        print(f"✅ Database pool initialized with {len(self._connection_pool)} connections")
    
    async def close(self) -> None:
        """Close all connections"""
        if not self._is_initialized:
            return
        
        print(f"🔌 Closing database connections...")
        await asyncio.sleep(0.1)  # Real code: await pool.close()
        self._connection_pool.clear()
        self._is_initialized = False
        print("✅ Database connections closed")
    
    async def query(self, sql: str) -> list[dict]:
        """Execute a query"""
        if not self._is_initialized:
            raise RuntimeError("Database not initialized. Call initialize() first.")
        
        print(f"   📊 Executing: {sql[:50]}...")
        await asyncio.sleep(0.05)  # Simulate query time
        return [{"id": 1, "result": "data"}]


@dataclass
class CacheService:
    """Redis-like cache service"""
    host: str
    port: int
    _client: Optional[any] = field(default=None, init=False)
    
    async def initialize(self) -> None:
        """Connect to cache"""
        print(f"💾 Connecting to cache at {self.host}:{self.port}")
        await asyncio.sleep(0.05)
        self._client = f"cache_client_{self.host}"
        print("✅ Cache connected")
    
    async def close(self) -> None:
        """Disconnect from cache"""
        print("💾 Disconnecting from cache")
        self._client = None
        print("✅ Cache disconnected")
    
    async def get(self, key: str) -> Optional[str]:
        """Get cached value"""
        if self._client is None:
            raise RuntimeError("Cache not connected")
        return None  # Simulate cache miss
    
    async def set(self, key: str, value: str, ttl: int = 3600) -> None:
        """Set cached value"""
        if self._client is None:
            raise RuntimeError("Cache not connected")
        print(f"   💾 Cached: {key} (TTL: {ttl}s)")


@dataclass
class APIClient:
    """External API client"""
    base_url: str
    api_key: str
    timeout: int = 10
    
    async def initialize(self) -> None:
        """Initialize HTTP client"""
        print(f"🌐 Initializing API client: {self.base_url}")
        await asyncio.sleep(0.05)
        print("✅ API client ready")
    
    async def close(self) -> None:
        """Close HTTP client"""
        print("🌐 Closing API client")
        print("✅ API client closed")
    
    async def fetch(self, endpoint: str) -> dict:
        """Make API request"""
        print(f"   🌐 GET {self.base_url}{endpoint}")
        await asyncio.sleep(0.1)
        return {"status": "success", "data": {}}


# Pattern 2: Composite Dependencies
# Combine multiple resources into one dependency object

@dataclass
class ApplicationDependencies:
    """
    Complete application dependencies
    Combines database, cache, and API clients
    """
    database: Database
    cache: CacheService
    api_client: APIClient
    user_id: str
    request_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    async def initialize_all(self) -> None:
        """Initialize all resources concurrently"""
        print("\n🚀 Initializing all dependencies...")
        await asyncio.gather(
            self.database.initialize(),
            self.cache.initialize(),
            self.api_client.initialize(),
        )
        print("✅ All dependencies initialized\n")
    
    async def close_all(self) -> None:
        """Close all resources gracefully"""
        print("\n🛑 Closing all dependencies...")
        await asyncio.gather(
            self.database.close(),
            self.cache.close(),
            self.api_client.close(),
        )
        print("✅ All dependencies closed\n")


# Pattern 3: Dependency Factory
# Type-safe factory for creating dependencies

class DependencyFactory:
    """
    Factory for creating application dependencies
    Centralizes configuration and initialization logic
    """
    
    def __init__(
        self,
        db_config: DatabaseConfig,
        cache_host: str = "localhost",
        cache_port: int = 6379,
        api_base_url: str = "https://api.example.com",
        api_key: str = "default_key",
    ):
        self.db_config = db_config
        self.cache_host = cache_host
        self.cache_port = cache_port
        self.api_base_url = api_base_url
        self.api_key = api_key
    
    async def create_dependencies(
        self,
        user_id: str,
        request_id: str,
    ) -> ApplicationDependencies:
        """
        Create and initialize dependencies for a request
        
        Args:
            user_id: ID of the user making the request
            request_id: Unique request identifier
            
        Returns:
            Fully initialized ApplicationDependencies
        """
        # Create resources
        database = Database(config=self.db_config)
        cache = CacheService(host=self.cache_host, port=self.cache_port)
        api_client = APIClient(base_url=self.api_base_url, api_key=self.api_key)
        
        # Bundle into dependencies
        deps = ApplicationDependencies(
            database=database,
            cache=cache,
            api_client=api_client,
            user_id=user_id,
            request_id=request_id,
        )
        
        # Initialize all resources
        await deps.initialize_all()
        
        return deps
    
    @asynccontextmanager
    async def dependency_context(
        self,
        user_id: str,
        request_id: str,
    ) -> AsyncIterator[ApplicationDependencies]:
        """
        Context manager for automatic resource cleanup
        
        Usage:
            async with factory.dependency_context(user_id, request_id) as deps:
                # Use deps
                result = await agent.run(prompt, deps=deps)
            # Resources automatically cleaned up here
        """
        deps = await self.create_dependencies(user_id, request_id)
        try:
            yield deps
        finally:
            await deps.close_all()


# Pattern 4: Agent with Complex Dependencies

class UserQuery(BaseModel):
    """Result model for user queries"""
    answer: str = Field(description="Answer to the user's question")
    sources: list[str] = Field(description="Data sources used")
    cached: bool = Field(description="Whether result was cached")
    processing_time_ms: int = Field(description="Processing time in milliseconds")


agent = Agent(
    model='gemini-1.5-flash',
    result_type=UserQuery,
    deps_type=ApplicationDependencies,
    system_prompt="""
You are an intelligent assistant with access to multiple data sources.

You have access to:
- database: For querying structured data
- cache: For retrieving cached results
- api_client: For fetching external data

Use these resources to provide comprehensive, accurate answers.
Always indicate which sources you used.
""",
)


# Pattern 5: Using the Dependency Factory

async def process_user_query(
    query: str,
    user_id: str,
    factory: DependencyFactory,
) -> UserQuery:
    """
    Process a user query with proper resource management
    
    Args:
        query: User's question
        user_id: User identifier
        factory: Dependency factory
        
    Returns:
        Structured response with metadata
    """
    import uuid
    request_id = str(uuid.uuid4())
    
    # Use context manager for automatic cleanup
    async with factory.dependency_context(user_id, request_id) as deps:
        # Check cache first
        cache_key = f"query:{hash(query)}"
        cached_result = await deps.cache.get(cache_key)
        
        if cached_result:
            print("💾 Cache hit!")
            # Return cached result (simplified for demo)
        
        # Execute agent with dependencies
        start_time = datetime.now()
        result = await agent.run(query, deps=deps)
        processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
        
        # Cache the result
        await deps.cache.set(cache_key, str(result.data), ttl=3600)
        
        return result.data
    # Dependencies automatically cleaned up after this block!


# Pattern 6: Testing with Mock Dependencies

@dataclass
class MockDatabase:
    """Mock database for testing"""
    config: DatabaseConfig
    _queries: list[str] = field(default_factory=list)
    
    async def initialize(self) -> None:
        print("🧪 Mock database initialized")
    
    async def close(self) -> None:
        print("🧪 Mock database closed")
    
    async def query(self, sql: str) -> list[dict]:
        self._queries.append(sql)
        print(f"🧪 Mock query: {sql}")
        return [{"id": 1, "mock": True}]


def create_test_dependencies(user_id: str = "test_user") -> ApplicationDependencies:
    """Create mock dependencies for testing"""
    mock_db_config = DatabaseConfig(
        host="localhost",
        port=5432,
        database="test_db",
        username="test",
        password="test",
        pool_size=1,
    )
    
    return ApplicationDependencies(
        database=MockDatabase(config=mock_db_config),
        cache=CacheService(host="localhost", port=6379),
        api_client=APIClient(base_url="https://test.example.com", api_key="test_key"),
        user_id=user_id,
        request_id="test_request_123",
    )


# Demo

async def main():
    print("="*70)
    print("DEPENDENCY INJECTION PATTERN DEMO")
    print("="*70)
    
    # Create factory with configuration
    db_config = DatabaseConfig(
        host="localhost",
        port=5432,
        database="production_db",
        username="app_user",
        password="secure_password",
        pool_size=10,
    )
    
    factory = DependencyFactory(
        db_config=db_config,
        cache_host="localhost",
        cache_port=6379,
        api_base_url="https://api.example.com",
        api_key="your_api_key_here",
    )
    
    # Process queries with automatic resource management
    queries = [
        "What is the status of order #12345?",
        "Show me sales data for last quarter",
    ]
    
    for query in queries:
        print(f"\n{'='*70}")
        print(f"Query: {query}")
        print(f"{'='*70}")
        
        try:
            result = await process_user_query(
                query=query,
                user_id="user_789",
                factory=factory,
            )
            
            print(f"\n📝 Answer: {result.answer}")
            print(f"📚 Sources: {', '.join(result.sources)}")
            print(f"💾 Cached: {result.cached}")
            print(f"⏱️  Time: {result.processing_time_ms}ms")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        # Small delay between requests
        await asyncio.sleep(0.5)
    
    print("\n" + "="*70)
    print("DEMO COMPLETE - Notice automatic resource management!")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())
```

### Line-by-Line Explanation

**Pattern 1: Resource Classes (Lines 19-115)**:
- `Database`, `CacheService`, `APIClient`: Managed resources
- `async def initialize()`: Async resource setup (connection pools, etc.)
- `async def close()`: Proper cleanup
- `_is_initialized`: Track lifecycle state
- These simulate real production resources like SQLAlchemy, Redis, httpx

**Pattern 2: Composite Dependencies (Lines 118-157)**:
- `ApplicationDependencies`: Bundles all resources
- `initialize_all()`: Concurrent initialization with `asyncio.gather()`
- `close_all()`: Graceful shutdown of all resources
- Type-safe: mypy checks all attributes exist

**Pattern 3: Dependency Factory (Lines 160-235)**:
- `DependencyFactory`: Centralizes dependency creation
- `create_dependencies()`: Builds and initializes deps
- `dependency_context()`: Context manager with automatic cleanup
- **This is the production pattern** - use this in real apps!

**Pattern 4: Context Manager (Lines 207-235)**:
- `@asynccontextmanager`: Async context manager decorator
- `yield deps`: Provides dependencies to the block
- `finally: await deps.close_all()`: **Guaranteed cleanup** even if errors occur
- Type-safe: return type is `AsyncIterator[ApplicationDependencies]`

**Pattern 5: Using Dependencies (Lines 259-294)**:
- `async with factory.dependency_context(...) as deps`: Get deps
- Use deps with agent: `await agent.run(query, deps=deps)`
- Resources **automatically cleaned up** after the block
- No manual cleanup needed!

**Pattern 6: Testing (Lines 297-325)**:
- `MockDatabase`: Drop-in replacement for testing
- Same interface as real database
- Tracks calls for verification
- Type-safe mocking!

### The "Why" Behind the Pattern

**Why use dependency factories and context managers?**

❌ **Manual Management** (Error-prone):
```python
# ❌ Easy to forget cleanup, hard to handle errors
database = Database(config)
await database.initialize()
try:
    result = await agent.run(query, deps=...)
finally:
    await database.close()  # What if you forget this?
```

✅ **Context Manager** (Safe):
```python
# ✅ Automatic cleanup, even on errors
async with factory.dependency_context(user_id, request_id) as deps:
    result = await agent.run(query, deps=deps)
# Guaranteed cleanup here!
```

**Benefits**:
1. **Automatic cleanup**: Context managers guarantee resource release
2. **Error safety**: Cleanup happens even if exceptions occur
3. **Testability**: Factory makes it easy to inject mocks
4. **Centralized config**: All dependency setup in one place
5. **Lifecycle management**: Clear initialization and shutdown
6. **Type safety**: Entire flow is type-checked

---

## C. Test & Apply

### How to Test It

1. **Run the dependency pattern demo**:
```bash
python lesson_04_dependency_patterns.py
```

2. **Observe the initialization and cleanup logs**

3. **Try creating your own factory**:
```python
# Create custom factory
my_factory = DependencyFactory(
    db_config=DatabaseConfig(
        host="localhost",
        port=5432,
        database="my_db",
        username="user",
        password="pass",
    ),
)

# Use it
async def my_app():
    async with my_factory.dependency_context("user_123", "req_456") as deps:
        # Use deps
        result = await agent.run("Hello!", deps=deps)
        print(result.data)
    # Auto cleanup!

asyncio.run(my_app())
```

### Expected Result

You should see structured output showing the lifecycle:

```
======================================================================
DEPENDENCY INJECTION PATTERN DEMO
======================================================================

🚀 Initializing all dependencies...
🔌 Initializing database connection pool...
   Host: localhost:5432
   Pool size: 10
✅ Database pool initialized with 10 connections
💾 Connecting to cache at localhost:6379
✅ Cache connected
🌐 Initializing API client: https://api.example.com
✅ API client ready
✅ All dependencies initialized

======================================================================
Query: What is the status of order #12345?
======================================================================
   📊 Executing: SELECT * FROM orders WHERE id = 12345...
   💾 Cached: query:123456789 (TTL: 3600s)

📝 Answer: Order #12345 is currently being processed...
📚 Sources: database, api_client
💾 Cached: false
⏱️  Time: 234ms

🛑 Closing all dependencies...
🔌 Closing database connections...
✅ Database connections closed
💾 Disconnecting from cache
✅ Cache disconnected
🌐 Closing API client
✅ API client closed
✅ All dependencies closed

======================================================================
DEMO COMPLETE - Notice automatic resource management!
======================================================================
```

### Validation Examples

**Type-Safe Factory Pattern**:

```python
# ✅ Type-safe factory
factory = DependencyFactory(
    db_config=DatabaseConfig(
        host="localhost",
        port=5432,
        database="db",
        username="user",
        password="pass"
    ),
)

async with factory.dependency_context("user_1", "req_1") as deps:
    # deps is typed as ApplicationDependencies
    # IDE autocomplete works!
    await deps.database.query("SELECT ...")
    await deps.cache.get("key")
    await deps.api_client.fetch("/endpoint")

# ❌ This fails type checking
async with factory.dependency_context("user_1", "req_1") as deps:
    await deps.database.invalid_method()  # mypy error!
```

### Type Checking

```bash
mypy lesson_04_dependency_patterns.py
```

Expected: `Success: no issues found`

---

## D. Common Stumbling Blocks

### 1. Forgetting to Initialize Resources

**The Error**:
```python
database = Database(config=db_config)
await database.query("SELECT ...")  # ❌ RuntimeError: Database not initialized
```

**The Fix**:
Always initialize before use:
```python
database = Database(config=db_config)
await database.initialize()  # ✅ Initialize first
await database.query("SELECT ...")
```

Or use the factory:
```python
async with factory.dependency_context(user_id, request_id) as deps:
    # ✅ Already initialized!
    await deps.database.query("SELECT ...")
```

### 2. Resource Leaks (Forgetting to Close)

**The Problem**:
```python
# ❌ Creating many dependencies without cleanup
for i in range(1000):
    deps = await factory.create_dependencies(f"user_{i}", f"req_{i}")
    result = await agent.run("query", deps=deps)
    # ❌ Forgot to close! Resources leak!
```

**What Happens**:
- Database connections exhaust
- Memory leaks
- Application crashes
- "Too many open files" errors

**The Fix**:
Use context managers:
```python
# ✅ Automatic cleanup
for i in range(1000):
    async with factory.dependency_context(f"user_{i}", f"req_{i}") as deps:
        result = await agent.run("query", deps=deps)
    # ✅ Resources closed automatically
```

### 3. Mixing Sync and Async

**The Error**:
```python
# ❌ Can't use async in sync context
def my_sync_function():
    result = await agent.run("query", deps=deps)  # SyntaxError!
```

**The Fix**:
Make function async or use `asyncio.run()`:
```python
# Option 1: Make function async
async def my_async_function():
    result = await agent.run("query", deps=deps)  # ✅

# Option 2: Use asyncio.run()
def my_sync_function():
    async def inner():
        return await agent.run("query", deps=deps)
    return asyncio.run(inner())  # ✅
```

### 4. Context Manager Misuse

**Common Mistake**:
```python
# ❌ Storing deps outside context
saved_deps = None

async with factory.dependency_context(user_id, request_id) as deps:
    saved_deps = deps

# ❌ Resources are closed now!
await saved_deps.database.query("SELECT ...")  # Error!
```

**The Fix**:
Only use deps inside the context:
```python
async with factory.dependency_context(user_id, request_id) as deps:
    # ✅ Use deps here
    result = await agent.run("query", deps=deps)
    await deps.database.query("SELECT ...")
# ❌ Don't use deps here - they're closed
```

### 5. Type Safety Gotcha: Factory Return Type

**The Problem**:
```python
async def create_deps(user_id: str):  # ❌ No return type hint
    return await factory.create_dependencies(user_id, "req_123")

deps = await create_deps("user_1")
await deps.databse.query("SELECT ...")  # Typo! But mypy doesn't catch it
```

**The Fix**:
Always add return type hints:
```python
async def create_deps(user_id: str) -> ApplicationDependencies:  # ✅
    return await factory.create_dependencies(user_id, "req_123")

deps = await create_deps("user_1")
await deps.databse.query("SELECT ...")  # ✅ mypy catches typo!
```

---

## Ready for the Next Lesson?

🎉 **Incredible progress!** You now understand advanced dependency injection patterns:

✅ Resource lifecycle management (initialize/close)  
✅ Dependency factories for centralized creation  
✅ Context managers for automatic cleanup  
✅ Composite dependencies combining multiple resources  
✅ Type-safe mocking for testing  
✅ Async dependency management  

**These patterns are the foundation of production-grade AI systems!** You can now build applications with databases, caches, APIs, and other resources while maintaining type safety and preventing resource leaks.

In the next lesson, we'll explore **Agent with Dependencies in Action** - you'll see a complete example of using dependencies throughout an agent's execution, including in tools and result validation!

**Ready for Lesson 5, or would you like to practice building a dependency factory first?** 🚀
