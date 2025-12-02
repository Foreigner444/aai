# Lesson 10: Tool Context and Parameters

## A. Concept Overview

### What & Why
**Tool Context and Parameters** covers advanced patterns for managing data flow between tools, using RunContext effectively to share state, and designing tool parameters that enable rich tool composition. This is crucial because complex workflows require tools to build upon each other's work, maintain consistency, and share insights across the execution chain.

### Analogy
Think of tool context like a restaurant kitchen's order ticket system:

**Without Context** = Chaos:
- Prep station makes random vegetables
- Grill cooks random proteins
- No coordination
- Results don't match the order

**With Context** (Order Ticket):
- **Order ticket** (context) travels with the dish
- Prep station reads: "Caesar salad for table 5, no croutons"
- Grill reads: "Medium-rare steak, same order"
- Expeditor reads: "Table 5 - check if allergies"
- Everyone accesses same context, works toward same goal

The context (order ticket) ensures all stations work together coherently!

### Type Safety Benefit
Tool context patterns provide:
- **Typed context access**: RunContext[YourDeps] guarantees correct types
- **Shared state safety**: All tools access same typed dependencies
- **Parameter validation**: Pydantic validates parameters between tools
- **Context immutability**: Dependencies can be immutable for safety
- **Tool composition**: Type-safe data flow between tools
- **Testing**: Mock context for isolated tool testing

Your entire workflow becomes a type-checked data pipeline!

---

## B. Code Implementation

### File Structure
```
agent_system_design/
├── lesson_10_tool_context_parameters.py
├── requirements.txt
└── .env
```

### Complete Code Snippet

**lesson_10_tool_context_parameters.py**
```python
"""
Lesson 10: Tool Context and Parameters
Advanced patterns for context sharing and parameter design
"""

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Literal
from dotenv import load_dotenv

load_dotenv()


# PATTERN 1: Rich Context with Multiple Data Sources

@dataclass
class SessionState:
    """Mutable session state that tools can update"""
    queries_executed: list[str] = field(default_factory=list)
    tools_called: list[str] = field(default_factory=list)
    cache_hits: int = 0
    cache_misses: int = 0
    
    def record_query(self, query: str):
        """Record a query execution"""
        self.queries_executed.append(query)
    
    def record_tool_call(self, tool_name: str):
        """Record a tool invocation"""
        self.tools_called.append(tool_name)


@dataclass
class DataWarehouse:
    """Simulated data warehouse"""
    
    def query_sales(self, region: str, year: int) -> dict:
        """Query sales data"""
        data = {
            "north": {"2023": 1_200_000, "2024": 1_450_000},
            "south": {"2023": 950_000, "2024": 1_100_000},
            "east": {"2023": 1_800_000, "2024": 2_100_000},
            "west": {"2023": 1_400_000, "2024": 1_600_000},
        }
        return {
            "region": region,
            "year": year,
            "sales": data.get(region, {}).get(str(year), 0)
        }
    
    def query_customers(self, region: str) -> dict:
        """Query customer data"""
        data = {
            "north": 450,
            "south": 320,
            "east": 680,
            "west": 520,
        }
        return {
            "region": region,
            "customer_count": data.get(region, 0)
        }


@dataclass
class CacheLayer:
    """Simple cache for frequently accessed data"""
    _cache: dict = field(default_factory=dict)
    
    def get(self, key: str) -> Optional[any]:
        """Get cached value"""
        return self._cache.get(key)
    
    def set(self, key: str, value: any):
        """Set cached value"""
        self._cache[key] = value


@dataclass
class ContextDeps:
    """Rich context with multiple resources"""
    data_warehouse: DataWarehouse
    cache: CacheLayer
    session: SessionState
    user_id: str
    permissions: list[str]
    request_timestamp: datetime
    
    def has_permission(self, permission: str) -> bool:
        """Check if user has permission"""
        return permission in self.permissions


# Result model
class AnalysisReport(BaseModel):
    """Analysis report with context metadata"""
    analysis: str
    metrics: dict[str, float]
    data_sources: list[str]
    tools_used: list[str]
    cache_performance: dict[str, int]
    execution_metadata: dict[str, any]


# Create agent
agent = Agent(
    model='gemini-1.5-flash',
    result_type=AnalysisReport,
    deps_type=ContextDeps,
    system_prompt="""
You are a data analyst with access to context-aware tools.

CONTEXT USAGE:
- All tools have access to shared context (ctx.deps)
- Tools can read session state to see what's been done
- Tools can update session state to record their actions
- Tools can check permissions before executing
- Tools use cache to improve performance

TOOL COORDINATION WITH CONTEXT:
1. Tools can see what other tools have been called
2. Tools can access cached results from previous calls
3. Tools maintain consistency through shared state
4. Tools respect user permissions from context

WORKFLOW:
1. Check cache before fetching data
2. Record all tool calls for auditing
3. Share insights between tools via context
4. Use session state to avoid duplicate work
""",
)


# PATTERN 2: Tools Using Context for Coordination

@agent.tool
def get_regional_sales(
    ctx: RunContext[ContextDeps],
    region: str,
    year: int
) -> dict[str, any]:
    """
    Get sales data for a region with caching.
    
    This tool demonstrates:
    - Checking cache before expensive queries
    - Recording tool calls in session state
    - Using context to access multiple resources
    
    Args:
        region: Geographic region (north, south, east, west)
        year: Year to query (2023 or 2024)
    
    Returns:
        Sales data with cache information
    """
    print(f"\n🔧 get_regional_sales(region={region}, year={year})")
    
    # Record tool call in session state
    ctx.deps.session.record_tool_call("get_regional_sales")
    
    # Check cache first
    cache_key = f"sales_{region}_{year}"
    cached_data = ctx.deps.cache.get(cache_key)
    
    if cached_data:
        print(f"   💾 Cache HIT for {cache_key}")
        ctx.deps.session.cache_hits += 1
        return {
            "region": region,
            "year": year,
            "sales": cached_data,
            "source": "cache"
        }
    
    # Cache miss - query database
    print(f"   🔍 Cache MISS - querying database")
    ctx.deps.session.cache_misses += 1
    
    # Query data warehouse
    data = ctx.deps.data_warehouse.query_sales(region, year)
    
    # Store in cache for next time
    ctx.deps.cache.set(cache_key, data["sales"])
    
    return {
        "region": region,
        "year": year,
        "sales": data["sales"],
        "source": "database"
    }


@agent.tool
def get_customer_count(
    ctx: RunContext[ContextDeps],
    region: str
) -> dict[str, any]:
    """
    Get customer count with permission check.
    
    Demonstrates:
    - Checking permissions from context
    - Recording queries in session state
    - Conditional execution based on context
    
    Args:
        region: Geographic region
    
    Returns:
        Customer count or permission error
    """
    print(f"\n🔧 get_customer_count(region={region})")
    
    # Check permissions
    if not ctx.deps.has_permission("view_customer_data"):
        print(f"   ⛔ Permission denied for user {ctx.deps.user_id}")
        return {
            "error": "Permission denied: view_customer_data required",
            "user_id": ctx.deps.user_id
        }
    
    # Record tool call
    ctx.deps.session.record_tool_call("get_customer_count")
    
    # Query database
    data = ctx.deps.data_warehouse.query_customers(region)
    
    return {
        "region": region,
        "customer_count": data["customer_count"],
        "source": "database"
    }


# PATTERN 3: Tools Building on Other Tools' Results

@agent.tool
def calculate_revenue_per_customer(
    ctx: RunContext[ContextDeps],
    region: str,
    year: int
) -> dict[str, any]:
    """
    Calculate revenue per customer by coordinating other tools.
    
    This tool demonstrates:
    - Building on results from other tools
    - Checking session state to avoid duplicate work
    - Error handling when dependencies fail
    
    Args:
        region: Geographic region
        year: Year for analysis
    
    Returns:
        Revenue per customer metric
    """
    print(f"\n🔧 calculate_revenue_per_customer(region={region}, year={year})")
    
    ctx.deps.session.record_tool_call("calculate_revenue_per_customer")
    
    # This tool needs data from TWO other tools
    # In a real scenario, Gemini would call those tools first,
    # but we can also call them directly if needed
    
    # Get sales data (will use cache if available)
    sales_result = get_regional_sales(ctx, region, year)
    
    if "error" in sales_result:
        return {"error": "Could not retrieve sales data"}
    
    # Get customer count (with permission check)
    customer_result = get_customer_count(ctx, region)
    
    if "error" in customer_result:
        return {"error": "Could not retrieve customer data"}
    
    # Calculate metric
    sales = sales_result["sales"]
    customers = customer_result["customer_count"]
    
    if customers == 0:
        return {"error": "No customers in region"}
    
    revenue_per_customer = sales / customers
    
    return {
        "region": region,
        "year": year,
        "revenue_per_customer": round(revenue_per_customer, 2),
        "sales": sales,
        "customers": customers,
        "calculation": f"{sales} / {customers}"
    }


# PATTERN 4: Tools That Use Session History

@agent.tool
def get_session_summary(
    ctx: RunContext[ContextDeps]
) -> dict[str, any]:
    """
    Get summary of current session activity.
    
    Demonstrates:
    - Reading session state accumulated by other tools
    - Providing meta-information about execution
    - No external data access needed
    
    Returns:
        Session activity summary
    """
    print(f"\n🔧 get_session_summary()")
    
    session = ctx.deps.session
    
    return {
        "user_id": ctx.deps.user_id,
        "queries_executed": len(session.queries_executed),
        "tools_called": session.tools_called,
        "unique_tools": len(set(session.tools_called)),
        "cache_performance": {
            "hits": session.cache_hits,
            "misses": session.cache_misses,
            "hit_rate": session.cache_hits / max(session.cache_hits + session.cache_misses, 1)
        },
        "request_timestamp": ctx.deps.request_timestamp.isoformat()
    }


# PATTERN 5: Tools with Rich Parameter Models

class RegionalAnalysisParams(BaseModel):
    """Structured parameters for regional analysis"""
    regions: list[str] = Field(min_length=1, max_length=4)
    years: list[int] = Field(min_length=1, max_length=2)
    include_customers: bool = True
    include_trends: bool = False


@agent.tool
def multi_regional_analysis(
    ctx: RunContext[ContextDeps],
    params: RegionalAnalysisParams
) -> dict[str, any]:
    """
    Perform analysis across multiple regions and years.
    
    Demonstrates:
    - Using Pydantic models for complex parameters
    - Automatic parameter validation
    - Batch operations with context
    
    Args:
        params: Analysis parameters (validated Pydantic model)
    
    Returns:
        Multi-dimensional analysis results
    """
    print(f"\n🔧 multi_regional_analysis(regions={params.regions}, years={params.years})")
    
    ctx.deps.session.record_tool_call("multi_regional_analysis")
    
    results = {}
    
    for region in params.regions:
        results[region] = {}
        
        for year in params.years:
            # Get sales (uses cache automatically)
            sales_data = get_regional_sales(ctx, region, year)
            results[region][year] = {
                "sales": sales_data.get("sales", 0),
                "source": sales_data.get("source", "unknown")
            }
            
            # Optionally include customer data
            if params.include_customers:
                customer_data = get_customer_count(ctx, region)
                if "error" not in customer_data:
                    results[region][year]["customers"] = customer_data["customer_count"]
    
    return {
        "analysis_params": {
            "regions": params.regions,
            "years": params.years,
            "include_customers": params.include_customers
        },
        "results": results,
        "regions_analyzed": len(params.regions),
        "years_analyzed": len(params.years)
    }


# Demonstration

def main():
    print("\n" + "="*70)
    print("TOOL CONTEXT AND PARAMETERS DEMONSTRATION")
    print("="*70)
    
    # Create rich context
    deps = ContextDeps(
        data_warehouse=DataWarehouse(),
        cache=CacheLayer(),
        session=SessionState(),
        user_id="analyst_456",
        permissions=["view_customer_data", "view_sales_data"],
        request_timestamp=datetime.now()
    )
    
    queries = [
        "What were sales in the north region in 2024?",
        "Calculate revenue per customer for the east region in 2024",
        "Compare sales across all regions for 2023 and 2024",
        "Show me a summary of this session",
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n\n{'='*70}")
        print(f"QUERY {i}: {query}")
        print(f"{'='*70}")
        
        try:
            result = agent.run_sync(query, deps=deps)
            report = result.data
            
            print(f"\n📊 ANALYSIS:")
            print(f"   {report.analysis}")
            
            print(f"\n📈 METRICS:")
            for metric, value in report.metrics.items():
                print(f"   {metric}: {value:,.2f}")
            
            print(f"\n💾 CACHE PERFORMANCE:")
            for stat, value in report.cache_performance.items():
                print(f"   {stat}: {value}")
            
            print(f"\n🔧 TOOLS USED: {', '.join(report.tools_used)}")
            print(f"📚 DATA SOURCES: {', '.join(report.data_sources)}")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
    
    # Show final session state
    print(f"\n\n{'='*70}")
    print("FINAL SESSION STATE")
    print(f"{'='*70}")
    print(f"Total tools called: {len(deps.session.tools_called)}")
    print(f"Unique tools: {set(deps.session.tools_called)}")
    print(f"Cache hits: {deps.session.cache_hits}")
    print(f"Cache misses: {deps.session.cache_misses}")
    print(f"Cache hit rate: {deps.session.cache_hits / max(deps.session.cache_hits + deps.session.cache_misses, 1):.2%}")


if __name__ == "__main__":
    main()
```

### Line-by-Line Explanation

**Pattern 1: Rich Context (Lines 17-86)**:
- `SessionState`: Mutable state tools can update
- `DataWarehouse`: Data source simulation
- `CacheLayer`: Caching for performance
- `ContextDeps`: Combines all resources
- Tools access everything through `ctx.deps`

**Pattern 2: Context Coordination (Lines 113-207)**:
- Tools check cache before expensive operations
- Tools record their execution in session state
- Tools respect permissions from context
- Tools update cache for future calls

**Pattern 3: Tool Composition (Lines 210-263)**:
- `calculate_revenue_per_customer` calls other tools
- Uses context to coordinate tool execution
- Handles errors from dependent tools
- Builds complex analysis from simple tools

**Pattern 4: Session History (Lines 266-297)**:
- Tools can inspect what's been done
- Provides meta-information about execution
- No external data needed - reads context only
- Useful for debugging and auditing

**Pattern 5: Rich Parameters (Lines 300-365)**:
- `RegionalAnalysisParams`: Pydantic model for parameters
- Automatic validation of complex inputs
- Clean interface for multi-dimensional queries
- Type-safe parameter passing

### The "Why" Behind the Pattern

**Why use context instead of passing parameters everywhere?**

❌ **Without Context** (Verbose):
```python
@agent.tool
def tool_a(db: Database, cache: Cache, session: Session, user: User, ...):
    # Every tool needs all parameters explicitly
    ...

@agent.tool
def tool_b(db: Database, cache: Cache, session: Session, user: User, ...):
    # Repeated parameters everywhere
    ...
```

✅ **With Context** (Clean):
```python
@agent.tool
def tool_a(ctx: RunContext[ContextDeps]):
    # Access everything through ctx.deps
    db = ctx.deps.data_warehouse
    cache = ctx.deps.cache
    ...

@agent.tool
def tool_b(ctx: RunContext[ContextDeps]):
    # Same clean interface
    ...
```

**Benefits**:
1. **Cleaner signatures**: Tools don't repeat parameters
2. **Shared state**: All tools access same resources
3. **Type safety**: `RunContext[ContextDeps]` is type-checked
4. **Easy updates**: Add to context, all tools get access
5. **Testing**: Mock entire context in one place

---

## C. Test & Apply

### How to Test It

1. **Run the context demo**:
```bash
python lesson_10_tool_context_parameters.py
```

2. **Observe cache hits increasing on repeated queries**

3. **Try your own context-aware tool**:
```python
@agent.tool
def my_context_tool(ctx: RunContext[ContextDeps]) -> dict:
    """Tool that uses multiple context resources"""
    # Check permission
    if not ctx.deps.has_permission("my_permission"):
        return {"error": "Access denied"}
    
    # Use cache
    cached = ctx.deps.cache.get("my_key")
    if cached:
        return {"data": cached, "source": "cache"}
    
    # Record activity
    ctx.deps.session.record_tool_call("my_context_tool")
    
    # Query database
    result = ctx.deps.data_warehouse.query_something()
    
    # Update cache
    ctx.deps.cache.set("my_key", result)
    
    return {"data": result, "source": "database"}
```

### Expected Result

You should see tools coordinating through context:

```
======================================================================
QUERY 1: What were sales in the north region in 2024?
======================================================================

🔧 get_regional_sales(region=north, year=2024)
   🔍 Cache MISS - querying database

📊 ANALYSIS:
   Sales in the north region for 2024 were $1,450,000

💾 CACHE PERFORMANCE:
   hits: 0
   misses: 1

🔧 TOOLS USED: get_regional_sales

======================================================================
QUERY 2: Calculate revenue per customer for the east region in 2024
======================================================================

🔧 calculate_revenue_per_customer(region=east, year=2024)
🔧 get_regional_sales(region=east, year=2024)
   🔍 Cache MISS - querying database
🔧 get_customer_count(region=east)

📊 ANALYSIS:
   Revenue per customer in the east region is $3,088.24

💾 CACHE PERFORMANCE:
   hits: 0
   misses: 2

🔧 TOOLS USED: calculate_revenue_per_customer, get_regional_sales, get_customer_count
```

---

## D. Common Stumbling Blocks

### 1. Modifying Context Unsafely

**The Problem**:
```python
@dataclass
class UnsafeDeps:
    data: dict  # ❌ Mutable, tools might conflict

@agent.tool
def tool_a(ctx: RunContext[UnsafeDeps]):
    ctx.deps.data["key"] = "value_a"  # ❌ Modifies shared state

@agent.tool
def tool_b(ctx: RunContext[UnsafeDeps]):
    ctx.deps.data["key"] = "value_b"  # ❌ Overwrites tool_a's change!
```

**The Fix**:
Use dedicated mutable containers:
```python
@dataclass
class SafeDeps:
    read_only_data: dict  # ✅ Don't modify
    session_state: SessionState  # ✅ Designed to be modified
    cache: CacheLayer  # ✅ Designed to be modified

@agent.tool
def tool_a(ctx: RunContext[SafeDeps]):
    # ✅ Use designated mutable containers
    ctx.deps.session_state.record("tool_a called")
    ctx.deps.cache.set("key", "value_a")
```

### 2. Forgetting RunContext Type Parameter

**The Problem**:
```python
@agent.tool
def my_tool(ctx: RunContext) -> dict:  # ❌ Missing type parameter
    # mypy can't infer ctx.deps type
    user = ctx.deps.user_id  # No autocomplete!
```

**The Fix**:
Always specify the dependency type:
```python
@agent.tool
def my_tool(ctx: RunContext[ContextDeps]) -> dict:  # ✅
    # ✅ Full type safety and autocomplete
    user = ctx.deps.user_id
```

### 3. Passing Too Much in Parameters

**The Problem**:
```python
@agent.tool
def analyze(
    ctx: RunContext[Deps],
    region: str,
    year: int,
    include_customers: bool,
    include_trends: bool,
    include_forecast: bool,
    compare_to_last_year: bool,
    ...  # ❌ Too many parameters!
):
    ...
```

**The Fix**:
Use Pydantic model for complex parameters:
```python
class AnalysisParams(BaseModel):
    region: str
    year: int
    include_customers: bool = True
    include_trends: bool = False
    include_forecast: bool = False
    compare_to_last_year: bool = False

@agent.tool
def analyze(
    ctx: RunContext[Deps],
    params: AnalysisParams  # ✅ Clean and validated!
) -> dict:
    ...
```

### 4. Not Checking Permissions

**The Problem**:
```python
@agent.tool
def get_sensitive_data(ctx: RunContext[Deps]) -> dict:
    # ❌ No permission check!
    return {"sensitive": "data"}
```

**The Fix**:
Always check permissions from context:
```python
@agent.tool
def get_sensitive_data(ctx: RunContext[Deps]) -> dict:
    # ✅ Check permission
    if not ctx.deps.has_permission("view_sensitive_data"):
        return {"error": "Permission denied"}
    
    return {"sensitive": "data"}
```

### 5. Type Safety Gotcha: Context Lifetime

**The Problem**:
```python
# ❌ Storing context outside execution
saved_context = None

@agent.tool
def tool_a(ctx: RunContext[Deps]) -> dict:
    global saved_context
    saved_context = ctx  # ❌ Don't do this!
    return {}

# Later...
saved_context.deps.something()  # ❌ Context may be invalid!
```

**The Fix**:
Only use context within tool execution:
```python
@agent.tool
def tool_a(ctx: RunContext[Deps]) -> dict:
    # ✅ Use context only within tool
    data = ctx.deps.data_warehouse.query()
    return {"data": data}
    # ✅ Context automatically cleaned up after tool returns
```

---

## Ready for the Next Lesson?

🎉 **Great work!** You now understand advanced context and parameter patterns:

✅ Rich context with multiple resources  
✅ Tools coordinating through shared state  
✅ Context-aware caching and optimization  
✅ Permission checks via context  
✅ Session state tracking  
✅ Complex parameters with Pydantic models  

**Context is the glue that makes multi-tool systems coherent!** Proper context management enables tools to work together seamlessly while maintaining type safety and clean interfaces.

In the next lesson, we'll explore **Dynamic Tool Selection** - you'll learn how to guide Gemini's tool selection through descriptions, system prompts, and tool design patterns!

**Ready for Lesson 11, or would you like to practice context-aware tools first?** 🚀
