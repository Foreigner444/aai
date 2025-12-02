# Lesson 5: Agent with Dependencies in Action

## A. Concept Overview

### What & Why
This lesson brings together everything you've learned by building a **complete agent** that uses dependencies throughout its entire execution - in the system prompt logic, during tool execution, and in result validation. This is crucial because real production agents need access to context at every stage, not just at the top level.

### Analogy
Think of a detective solving a case:
- **Detective** = Agent
- **Case file** = Dependencies (witness statements, evidence, forensic reports)
- **Investigation tools** = Agent tools (interview witnesses, analyze evidence, check alibis)
- **The twist**: The detective needs the case file **throughout** the investigation, not just at the start

Dependencies flow through the entire agent execution, available whenever needed!

### Type Safety Benefit
Using dependencies throughout execution provides:
- **Tool parameter safety**: Tools receive typed dependencies
- **Context preservation**: Type-safe context flows through execution
- **Validation with context**: Validate results against dependency constraints
- **RunContext type checking**: Access to typed context in tools
- **End-to-end type safety**: From input through tools to output validation
- **Refactoring confidence**: Change dependency structure, IDE shows all usage

Your entire agent becomes a type-safe pipeline from input to output!

---

## B. Code Implementation

### File Structure
```
agent_system_design/
├── lesson_05_agent_with_deps.py
├── requirements.txt
└── .env
```

### Complete Code Snippet

**lesson_05_agent_with_deps.py**
```python
"""
Lesson 5: Agent with Dependencies in Action
Complete example of using dependencies throughout agent execution
"""

from pydantic import BaseModel, Field, field_validator
from pydantic_ai import Agent, RunContext
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
from dotenv import load_dotenv
import json

load_dotenv()


# Step 1: Define comprehensive dependencies
# These will be accessible throughout agent execution

@dataclass
class UserAccount:
    """User account information"""
    user_id: str
    username: str
    email: str
    account_type: str  # "free", "premium", "enterprise"
    created_at: datetime
    credits_remaining: int
    max_requests_per_day: int
    
    def can_make_request(self) -> bool:
        """Check if user can make a request"""
        return self.credits_remaining > 0
    
    def deduct_credits(self, amount: int = 1) -> bool:
        """Deduct credits from account"""
        if self.credits_remaining >= amount:
            self.credits_remaining -= amount
            return True
        return False


@dataclass
class Database:
    """Simulated database"""
    _data: dict = field(default_factory=dict)
    
    def get_user_preferences(self, user_id: str) -> dict:
        """Get user preferences from database"""
        print(f"   📊 DB: Fetching preferences for user {user_id}")
        return {
            "language": "en",
            "timezone": "America/New_York",
            "notification_enabled": True,
            "theme": "dark"
        }
    
    def get_usage_history(self, user_id: str, days: int = 30) -> list[dict]:
        """Get user's recent usage"""
        print(f"   📊 DB: Fetching {days}-day history for user {user_id}")
        return [
            {"date": "2024-01-15", "requests": 42, "credits_used": 42},
            {"date": "2024-01-14", "requests": 38, "credits_used": 38},
            {"date": "2024-01-13", "requests": 51, "credits_used": 51},
        ]
    
    def log_request(self, user_id: str, request_type: str, credits_used: int) -> None:
        """Log a request"""
        print(f"   📊 DB: Logging {request_type} request for user {user_id} ({credits_used} credits)")


@dataclass
class ExternalAPI:
    """External API client"""
    api_key: str
    base_url: str = "https://api.example.com"
    
    def fetch_realtime_data(self, query: str) -> dict:
        """Fetch real-time data"""
        print(f"   🌐 API: Fetching real-time data for: {query}")
        # Simulate API response
        return {
            "query": query,
            "results": [
                {"title": "Latest news on " + query, "score": 0.95},
                {"title": "Analysis of " + query, "score": 0.87},
            ],
            "timestamp": datetime.now().isoformat()
        }


@dataclass
class RequestContext:
    """Complete request context"""
    user: UserAccount
    database: Database
    api: ExternalAPI
    request_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    def validate_access(self) -> tuple[bool, Optional[str]]:
        """Validate user can access the system"""
        if not self.user.can_make_request():
            return False, "Insufficient credits"
        
        if self.user.account_type == "free" and self.user.credits_remaining < 5:
            return True, "Warning: Low credits (upgrade recommended)"
        
        return True, None


# Step 2: Define result model with validation using dependencies

class IntelligentResponse(BaseModel):
    """Response with context-aware validation"""
    answer: str = Field(description="Main response to user's query")
    confidence: float = Field(description="Confidence score 0-1", ge=0.0, le=1.0)
    sources_used: list[str] = Field(description="Data sources consulted")
    personalization_note: str = Field(description="Note about personalization")
    credits_charged: int = Field(description="Credits charged for request", ge=1)
    suggested_followup: Optional[str] = Field(description="Suggested follow-up question")
    
    @field_validator("credits_charged")
    @classmethod
    def validate_credits_reasonable(cls, v: int) -> int:
        """Ensure credits charged is reasonable"""
        if v > 10:
            raise ValueError("Credits charged exceeds maximum (10)")
        return v


# Step 3: Create agent with tools that use dependencies
# Tools have access to RunContext[RequestContext]

agent = Agent(
    model='gemini-1.5-flash',
    result_type=IntelligentResponse,
    deps_type=RequestContext,
    system_prompt="""
You are an intelligent assistant with access to user context and external resources.

You have access to tools:
- get_user_profile: Get detailed user information
- search_knowledge_base: Search internal knowledge base
- fetch_latest_data: Get real-time external data

IMPORTANT GUIDELINES:
1. Always personalize responses based on user's account type
2. Consider user preferences when formulating answers
3. Use appropriate sources (check user history for context)
4. Charge credits fairly:
   - Simple queries: 1 credit
   - Complex queries with external data: 2-3 credits
   - Analysis with multiple sources: 3-5 credits
5. Suggest relevant follow-ups based on user's past interests

Remember: You're building a relationship with the user. Use their context!
""",
)


# Step 4: Define tools that use RunContext to access dependencies

@agent.tool
def get_user_profile(ctx: RunContext[RequestContext]) -> dict:
    """
    Get detailed user profile information
    
    This tool has access to dependencies via ctx.deps
    """
    user = ctx.deps.user
    db = ctx.deps.database
    
    print(f"\n🔧 Tool: get_user_profile")
    print(f"   User: {user.username} ({user.account_type})")
    
    # Fetch preferences from database
    preferences = db.get_user_preferences(user.user_id)
    
    # Get usage history
    history = db.get_usage_history(user.user_id, days=7)
    total_requests_week = sum(h["requests"] for h in history)
    
    profile = {
        "username": user.username,
        "account_type": user.account_type,
        "credits_remaining": user.credits_remaining,
        "preferences": preferences,
        "recent_activity": {
            "requests_this_week": total_requests_week,
            "avg_per_day": round(total_requests_week / 7, 1)
        },
        "member_since": user.created_at.strftime("%B %Y")
    }
    
    return profile


@agent.tool
def search_knowledge_base(ctx: RunContext[RequestContext], query: str) -> dict:
    """
    Search internal knowledge base
    
    Args:
        query: Search query
        
    Returns:
        Search results with relevance scores
    """
    print(f"\n🔧 Tool: search_knowledge_base")
    print(f"   Query: {query}")
    
    # Access user context
    user = ctx.deps.user
    preferences = ctx.deps.database.get_user_preferences(user.user_id)
    
    # Personalize search based on preferences
    print(f"   Personalizing for: {preferences['language']}, {preferences['theme']} theme")
    
    # Simulate knowledge base search
    results = {
        "query": query,
        "results": [
            {
                "title": f"Guide to {query}",
                "snippet": f"Comprehensive guide about {query}...",
                "relevance": 0.92
            },
            {
                "title": f"{query} Best Practices",
                "snippet": f"Learn the best practices for {query}...",
                "relevance": 0.85
            }
        ],
        "total_found": 2
    }
    
    return results


@agent.tool
def fetch_latest_data(ctx: RunContext[RequestContext], topic: str) -> dict:
    """
    Fetch real-time data from external API
    
    Args:
        topic: Topic to fetch data about
        
    Returns:
        Latest data from external source
    """
    print(f"\n🔧 Tool: fetch_latest_data")
    print(f"   Topic: {topic}")
    
    # Check if user has premium access for external data
    user = ctx.deps.user
    if user.account_type == "free":
        return {
            "error": "External data access requires premium account",
            "upgrade_url": "https://example.com/upgrade"
        }
    
    # Fetch from external API
    api_response = ctx.deps.api.fetch_realtime_data(topic)
    
    return {
        "topic": topic,
        "data": api_response["results"],
        "fetched_at": api_response["timestamp"],
        "source": "external_api"
    }


# Step 5: Use the agent with full dependency context

def process_query(
    user_query: str,
    deps: RequestContext
) -> tuple[Optional[IntelligentResponse], Optional[str]]:
    """
    Process user query with full context
    
    Args:
        user_query: The user's question
        deps: Complete request context
        
    Returns:
        Tuple of (response, error_message)
    """
    print(f"\n{'='*70}")
    print(f"Processing Query")
    print(f"{'='*70}")
    print(f"User: {deps.user.username} ({deps.user.account_type})")
    print(f"Credits: {deps.user.credits_remaining}")
    print(f"Query: {user_query}")
    
    # Validate access
    can_access, warning = deps.validate_access()
    if not can_access:
        return None, warning
    
    if warning:
        print(f"⚠️  {warning}")
    
    try:
        # Run agent with dependencies
        # Dependencies are accessible in tools via RunContext
        result = agent.run_sync(user_query, deps=deps)
        response = result.data
        
        # Deduct credits
        if deps.user.deduct_credits(response.credits_charged):
            print(f"\n💳 Charged {response.credits_charged} credits")
            print(f"   Remaining: {deps.user.credits_remaining}")
            
            # Log to database
            deps.database.log_request(
                user_id=deps.user.user_id,
                request_type="query",
                credits_used=response.credits_charged
            )
        
        return response, None
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return None, str(e)


# Step 6: Demo with different user contexts

def main():
    # Setup database and API
    database = Database()
    api = ExternalAPI(api_key="demo_key_12345")
    
    print("\n" + "="*70)
    print("AGENT WITH DEPENDENCIES IN ACTION")
    print("="*70)
    
    # Scenario 1: Free user with simple query
    print("\n\n📱 SCENARIO 1: Free User - Simple Query")
    print("="*70)
    
    free_user = UserAccount(
        user_id="u_001",
        username="Alice",
        email="alice@example.com",
        account_type="free",
        created_at=datetime.now() - timedelta(days=30),
        credits_remaining=15,
        max_requests_per_day=50
    )
    
    free_context = RequestContext(
        user=free_user,
        database=database,
        api=api,
        request_id="req_001"
    )
    
    response, error = process_query(
        "What are the benefits of async programming?",
        free_context
    )
    
    if response:
        print(f"\n✅ Response:")
        print(f"   Answer: {response.answer[:100]}...")
        print(f"   Confidence: {response.confidence:.2f}")
        print(f"   Sources: {', '.join(response.sources_used)}")
        print(f"   Personalization: {response.personalization_note}")
        if response.suggested_followup:
            print(f"   Follow-up: {response.suggested_followup}")
    
    # Scenario 2: Premium user with complex query
    print("\n\n👑 SCENARIO 2: Premium User - Complex Query with External Data")
    print("="*70)
    
    premium_user = UserAccount(
        user_id="u_002",
        username="Bob",
        email="bob@example.com",
        account_type="premium",
        created_at=datetime.now() - timedelta(days=180),
        credits_remaining=100,
        max_requests_per_day=200
    )
    
    premium_context = RequestContext(
        user=premium_user,
        database=database,
        api=api,
        request_id="req_002"
    )
    
    response, error = process_query(
        "What are the latest developments in AI safety research?",
        premium_context
    )
    
    if response:
        print(f"\n✅ Response:")
        print(f"   Answer: {response.answer[:100]}...")
        print(f"   Confidence: {response.confidence:.2f}")
        print(f"   Sources: {', '.join(response.sources_used)}")
        print(f"   Personalization: {response.personalization_note}")
        if response.suggested_followup:
            print(f"   Follow-up: {response.suggested_followup}")
    
    # Scenario 3: User with insufficient credits
    print("\n\n⚠️  SCENARIO 3: User with Insufficient Credits")
    print("="*70)
    
    low_credit_user = UserAccount(
        user_id="u_003",
        username="Charlie",
        email="charlie@example.com",
        account_type="free",
        created_at=datetime.now() - timedelta(days=5),
        credits_remaining=0,  # No credits!
        max_requests_per_day=50
    )
    
    low_credit_context = RequestContext(
        user=low_credit_user,
        database=database,
        api=api,
        request_id="req_003"
    )
    
    response, error = process_query(
        "How do I learn Python?",
        low_credit_context
    )
    
    if error:
        print(f"\n❌ Error: {error}")
        print("   💡 User needs to purchase more credits or upgrade account")
    
    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70)
    print("\nKey Takeaways:")
    print("✅ Dependencies flow through entire agent execution")
    print("✅ Tools access context via RunContext[RequestContext]")
    print("✅ Validation uses dependency constraints")
    print("✅ Different users get personalized experiences")
    print("✅ Type safety maintained throughout")


if __name__ == "__main__":
    main()
```

### Line-by-Line Explanation

**Step 1: Comprehensive Dependencies (Lines 17-91)**:
- `UserAccount`: User-specific data with business logic methods
- `Database`: Simulated data access layer
- `ExternalAPI`: External service client
- `RequestContext`: Bundles everything together
- `validate_access()`: Business logic using dependency data

**Step 2: Result Model with Validation (Lines 94-111)**:
- `IntelligentResponse`: Structured output
- `@field_validator`: Custom validation using credits
- Validation happens automatically after agent generates output

**Step 3: Agent with Tools (Lines 114-141)**:
- `deps_type=RequestContext`: Declares dependency type
- System prompt guides agent to use tools appropriately
- Agent "knows" about user context through dependencies

**Step 4: Tools with RunContext (Lines 144-250)**:
- `@agent.tool`: Decorator registers function as tool
- `ctx: RunContext[RequestContext]`: Typed context parameter
- `ctx.deps`: Access to dependencies within tool
- Tools can read user data, query database, call APIs
- **Type-safe**: IDE knows `ctx.deps.user` is `UserAccount`

**Step 5: Query Processing (Lines 253-303)**:
- `validate_access()`: Check before processing
- `agent.run_sync(user_query, deps=deps)`: Pass dependencies
- Agent and tools both have access to deps
- Deduct credits after successful response
- Log to database for analytics

**Step 6: Demo Scenarios (Lines 306-421)**:
- Same agent, three different user contexts
- Free user: limited access
- Premium user: full access with external data
- Low credit user: rejected due to insufficient credits
- Dependencies determine behavior!

### The "Why" Behind the Pattern

**Why pass dependencies throughout execution?**

This pattern enables:

1. **Context-Aware Tools**: Tools make decisions based on user state
2. **Personalization**: Same query, different users = different results
3. **Access Control**: Tools check permissions before executing
4. **Resource Management**: Track and deduct credits/quotas
5. **Audit Trail**: Log all actions with full context
6. **Type Safety**: Entire flow is type-checked

**Without this pattern**, you'd need:
- Global variables (not thread-safe)
- Passing data through every function (verbose)
- No type checking (error-prone)
- Difficult testing (hard to mock)

---

## C. Test & Apply

### How to Test It

1. **Run the complete demo**:
```bash
python lesson_05_agent_with_deps.py
```

2. **Observe how different users get different experiences**

3. **Try adding your own tool**:
```python
@agent.tool
def get_user_analytics(ctx: RunContext[RequestContext]) -> dict:
    """Get analytics for the user"""
    user = ctx.deps.user
    history = ctx.deps.database.get_usage_history(user.user_id, days=30)
    
    return {
        "total_requests": sum(h["requests"] for h in history),
        "total_credits": sum(h["credits_used"] for h in history),
        "avg_per_day": sum(h["requests"] for h in history) / len(history)
    }
```

### Expected Result

You should see three distinct scenarios:

**Scenario 1 (Free User)**:
```
📱 SCENARIO 1: Free User - Simple Query
======================================================================
Processing Query
======================================================================
User: Alice (free)
Credits: 15
Query: What are the benefits of async programming?

🔧 Tool: get_user_profile
   User: Alice (free)
   📊 DB: Fetching preferences for user u_001
   📊 DB: Fetching 7-day history for user u_001

🔧 Tool: search_knowledge_base
   Query: async programming
   Personalizing for: en, dark theme

✅ Response:
   Answer: Async programming allows your code to handle multiple tasks concurrently...
   Confidence: 0.88
   Sources: knowledge_base, user_profile
   Personalization: Customized for free account with practical examples
   Follow-up: Would you like to see code examples of async/await?

💳 Charged 1 credits
   Remaining: 14
```

**Scenario 2 (Premium User)**:
```
👑 SCENARIO 2: Premium User - Complex Query with External Data
======================================================================
Processing Query
======================================================================
User: Bob (premium)
Credits: 100
Query: What are the latest developments in AI safety research?

🔧 Tool: get_user_profile
   User: Bob (premium)
   📊 DB: Fetching preferences for user u_002

🔧 Tool: fetch_latest_data
   Topic: AI safety research
   🌐 API: Fetching real-time data for: AI safety research

✅ Response:
   Answer: Recent developments in AI safety include alignment research...
   Confidence: 0.92
   Sources: external_api, knowledge_base, user_profile
   Personalization: Premium account - includes latest research from external sources
   Follow-up: Would you like a deeper analysis of specific safety frameworks?

💳 Charged 3 credits
   Remaining: 97
```

**Scenario 3 (Insufficient Credits)**:
```
⚠️  SCENARIO 3: User with Insufficient Credits
======================================================================
Processing Query
======================================================================
User: Charlie (free)
Credits: 0
Query: How do I learn Python?

❌ Error: Insufficient credits
   💡 User needs to purchase more credits or upgrade account
```

### Validation Examples

**Type-Safe Tool Access**:

```python
@agent.tool
def my_tool(ctx: RunContext[RequestContext]) -> dict:
    # ✅ Type-safe access
    user_id = ctx.deps.user.user_id  # IDE autocomplete works!
    prefs = ctx.deps.database.get_user_preferences(user_id)
    
    # ❌ This fails type checking
    invalid = ctx.deps.user.nonexistent_field  # mypy error!
    
    return {"status": "ok"}
```

### Type Checking

```bash
mypy lesson_05_agent_with_deps.py
```

Expected: `Success: no issues found`

---

## D. Common Stumbling Blocks

### 1. Forgetting RunContext Type Parameter

**The Error**:
```python
@agent.tool
def my_tool(ctx: RunContext) -> dict:  # ❌ Missing type parameter
    user = ctx.deps.user  # mypy can't infer type!
```

**The Fix**:
Always specify the dependency type:
```python
@agent.tool
def my_tool(ctx: RunContext[RequestContext]) -> dict:  # ✅
    user = ctx.deps.user  # ✅ Type is known!
```

### 2. Accessing deps Outside Tools

**Common Confusion**:
```python
# ❌ Can't access deps in regular functions
def helper_function():
    user = ctx.deps.user  # Where is ctx?
```

**The Fix Option 1**: Pass deps explicitly:
```python
def helper_function(deps: RequestContext):  # ✅
    user = deps.user
    return user.username

@agent.tool
def my_tool(ctx: RunContext[RequestContext]) -> dict:
    name = helper_function(ctx.deps)  # ✅ Pass explicitly
    return {"name": name}
```

**The Fix Option 2**: Pass only what's needed:
```python
def helper_function(user: UserAccount) -> str:  # ✅ Even more specific
    return user.username

@agent.tool
def my_tool(ctx: RunContext[RequestContext]) -> dict:
    name = helper_function(ctx.deps.user)  # ✅
    return {"name": name}
```

### 3. Modifying Dependencies

**The Problem**:
```python
@agent.tool
def consume_credits(ctx: RunContext[RequestContext], amount: int) -> dict:
    # ⚠️ Modifying deps during execution
    ctx.deps.user.credits_remaining -= amount
    return {"remaining": ctx.deps.user.credits_remaining}
```

**What's Wrong**:
While this works, it's better to return the change and apply it after:

**Better Approach**:
```python
@agent.tool
def check_cost(ctx: RunContext[RequestContext], operation: str) -> dict:
    """Return cost without modifying state"""
    cost_map = {"simple": 1, "complex": 3, "external": 5}
    cost = cost_map.get(operation, 1)
    can_afford = ctx.deps.user.credits_remaining >= cost
    
    return {
        "cost": cost,
        "can_afford": can_afford,
        "current_balance": ctx.deps.user.credits_remaining
    }

# Apply changes after agent completes
result = agent.run_sync(query, deps=deps)
deps.user.deduct_credits(result.data.credits_charged)
```

### 4. Not Validating Access

**The Problem**:
```python
# ❌ No access validation
response = agent.run_sync(query, deps=deps)
deps.user.deduct_credits(response.data.credits_charged)
# What if user had 0 credits to start?
```

**The Fix**:
Always validate before processing:
```python
# ✅ Validate first
can_access, error = deps.validate_access()
if not can_access:
    return None, error

# Now process
response = agent.run_sync(query, deps=deps)
```

### 5. Type Safety Gotcha: Tool Return Types

**The Problem**:
```python
@agent.tool
def my_tool(ctx: RunContext[RequestContext]):  # ❌ No return type
    return {"data": "value"}  # What type is this?
```

**The Fix**:
Always specify return types:
```python
@agent.tool
def my_tool(ctx: RunContext[RequestContext]) -> dict[str, str]:  # ✅
    return {"data": "value"}

# Or use a Pydantic model for even better type safety
class ToolResult(BaseModel):
    data: str

@agent.tool
def my_tool(ctx: RunContext[RequestContext]) -> ToolResult:  # ✅✅
    return ToolResult(data="value")
```

---

## Ready for the Next Lesson?

🎉 **Outstanding work!** You've mastered using dependencies throughout agent execution:

✅ Dependencies flow from agent creation through tool execution  
✅ Tools access context via typed RunContext  
✅ Business logic methods on dependency classes  
✅ Validation with dependency constraints  
✅ Personalization based on user context  
✅ End-to-end type safety  

**This is production-ready agent design!** You can now build agents that:
- Personalize responses based on user data
- Enforce access control and quotas
- Track usage and log analytics
- Make context-aware decisions in tools
- Maintain type safety throughout

In the next lesson, we'll explore **Creating Custom Tools** - you'll learn how to build powerful tools with proper signatures, descriptions, and error handling that Gemini can intelligently select and use!

**Ready for Lesson 6, or would you like to practice building context-aware tools first?** 🚀
