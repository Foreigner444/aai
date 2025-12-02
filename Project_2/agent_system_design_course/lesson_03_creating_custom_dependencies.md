# Lesson 3: Creating Custom Dependencies

## A. Concept Overview

### What & Why
**Dependencies** in Pydantic AI are contextual data or resources that your agent needs to function - like database connections, API clients, user sessions, configuration, or any runtime state. This is crucial because it allows you to inject external context into agents in a type-safe, testable, and reusable way. Dependencies keep your agents pure and your code clean!

### Analogy
Think of a chef preparing a meal:
- The **Chef** is your Agent
- The **Recipe** is your system prompt
- The **Ingredients** are dependencies (user preferences, available ingredients, kitchen equipment)
- Without ingredients, the chef can't cook anything useful!

When a customer orders a meal:
1. You gather the ingredients (dependencies) based on the order
2. You hand them to the chef (inject into agent)
3. The chef uses them to prepare the dish (agent uses deps to generate output)

Dependencies let you separate "what the agent does" (logic) from "what context it needs" (data).

### Type Safety Benefit
Dependencies provide incredible type safety:
- **Compile-time checking**: mypy verifies dependency types before runtime
- **IDE autocomplete**: Your editor knows exactly what's available in deps
- **Impossible invalid states**: Can't pass wrong types or forget required context
- **Testability**: Easy to mock dependencies for unit tests
- **Explicit contracts**: Function signatures document exactly what context is needed
- **No global state**: Everything is passed explicitly, no hidden dependencies

With dependencies, your entire data flow is type-checked and explicit!

---

## B. Code Implementation

### File Structure
```
agent_system_design/
├── lesson_03_dependencies.py
├── requirements.txt
└── .env
```

### Complete Code Snippet

**lesson_03_dependencies.py**
```python
"""
Lesson 3: Creating Custom Dependencies
Learn to inject context and resources into agents in a type-safe way
"""

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


# Step 1: Define your dependency classes
# These hold the context/resources your agent needs

@dataclass
class UserContext:
    """User-specific context for personalized responses"""
    user_id: str
    username: str
    subscription_tier: str  # "free", "pro", "enterprise"
    joined_date: datetime
    preferences: dict[str, any]


@dataclass
class DatabaseConnection:
    """Simulated database connection (in real code, this would be sqlalchemy, etc.)"""
    connection_string: str
    is_connected: bool = True
    
    def query(self, sql: str) -> list[dict]:
        """Simulate a database query"""
        # In real code: return self.session.execute(sql).fetchall()
        print(f"🔍 Executing query: {sql}")
        return [{"id": 1, "data": "sample"}]
    
    def get_user_history(self, user_id: str) -> list[str]:
        """Get user's interaction history"""
        # Simulated data
        return [
            "Asked about Python decorators",
            "Requested code review for FastAPI app",
            "Inquired about async patterns"
        ]


@dataclass
class AppConfig:
    """Application configuration"""
    api_version: str
    max_response_length: int
    enable_analytics: bool
    feature_flags: dict[str, bool]


# Step 2: Combine dependencies into a single dependency class
# This is what gets injected into the agent

@dataclass
class AgentDependencies:
    """
    All dependencies needed by the agent
    This is the 'context' the agent operates with
    """
    user: UserContext
    database: DatabaseConnection
    config: AppConfig
    request_timestamp: datetime


# Step 3: Define your result model
class PersonalizedRecommendation(BaseModel):
    """Personalized recommendation based on user context"""
    greeting: str = Field(description="Personalized greeting using username")
    recommendation: str = Field(description="Tailored recommendation")
    reason: str = Field(description="Why this recommendation fits the user")
    tier_specific_feature: Optional[str] = Field(
        description="Feature available for user's subscription tier"
    )
    next_steps: list[str] = Field(
        description="2-3 suggested next steps",
        min_length=2,
        max_length=3
    )


# Step 4: Create agent with dependencies
# The deps_type parameter tells Pydantic AI what dependencies to expect

agent = Agent(
    model='gemini-1.5-flash',
    result_type=PersonalizedRecommendation,
    deps_type=AgentDependencies,  # ✅ Specify dependency type!
    system_prompt="""
You are a personalized AI assistant that provides tailored recommendations.

You have access to user context through dependencies:
- user.username: The user's name
- user.subscription_tier: Their subscription level (free/pro/enterprise)
- user.preferences: Their stated preferences
- database: Their interaction history
- config: Application settings

Use this context to personalize every response:
- Greet them by name
- Reference their history when relevant
- Tailor recommendations to their subscription tier
- Respect their preferences

Subscription tier features:
- free: Basic recommendations
- pro: Advanced features, priority support
- enterprise: Custom solutions, dedicated account manager
""",
)


# Step 5: Use dependencies in your agent function
def get_personalized_recommendation(
    query: str,
    deps: AgentDependencies  # Type hint ensures safety!
) -> PersonalizedRecommendation:
    """
    Get a personalized recommendation using injected dependencies
    
    Args:
        query: User's question or request
        deps: Injected dependencies with user context
        
    Returns:
        PersonalizedRecommendation: Tailored response
    """
    # The magic: pass deps to run_sync()
    # Pydantic AI will make these available to the agent's context
    result = agent.run_sync(query, deps=deps)
    return result.data


# Step 6: Demonstrate with different user contexts
def main():
    # Create different dependency contexts
    
    # User 1: Free tier user
    free_user_deps = AgentDependencies(
        user=UserContext(
            user_id="u_123",
            username="Alice",
            subscription_tier="free",
            joined_date=datetime(2024, 1, 15),
            preferences={"learning_style": "practical", "interest": "web_development"}
        ),
        database=DatabaseConnection("postgresql://localhost/app"),
        config=AppConfig(
            api_version="v1",
            max_response_length=500,
            enable_analytics=True,
            feature_flags={"beta_features": False}
        ),
        request_timestamp=datetime.now()
    )
    
    # User 2: Pro tier user
    pro_user_deps = AgentDependencies(
        user=UserContext(
            user_id="u_456",
            username="Bob",
            subscription_tier="pro",
            joined_date=datetime(2023, 6, 20),
            preferences={"learning_style": "theoretical", "interest": "machine_learning"}
        ),
        database=DatabaseConnection("postgresql://localhost/app"),
        config=AppConfig(
            api_version="v1",
            max_response_length=1000,
            enable_analytics=True,
            feature_flags={"beta_features": True}
        ),
        request_timestamp=datetime.now()
    )
    
    # Same query, different contexts!
    query = "I want to improve my Python skills"
    
    print("\n" + "="*70)
    print("SAME QUERY, DIFFERENT USER CONTEXTS")
    print("="*70)
    
    # Free tier response
    print("\n📱 FREE TIER USER (Alice)")
    print("-" * 70)
    free_rec = get_personalized_recommendation(query, free_user_deps)
    print(f"Greeting: {free_rec.greeting}")
    print(f"Recommendation: {free_rec.recommendation}")
    print(f"Reason: {free_rec.reason}")
    print(f"Tier Feature: {free_rec.tier_specific_feature or 'None (free tier)'}")
    print("Next Steps:")
    for i, step in enumerate(free_rec.next_steps, 1):
        print(f"  {i}. {step}")
    
    # Pro tier response
    print("\n👑 PRO TIER USER (Bob)")
    print("-" * 70)
    pro_rec = get_personalized_recommendation(query, pro_user_deps)
    print(f"Greeting: {pro_rec.greeting}")
    print(f"Recommendation: {pro_rec.recommendation}")
    print(f"Reason: {pro_rec.reason}")
    print(f"Tier Feature: {pro_rec.tier_specific_feature}")
    print("Next Steps:")
    for i, step in enumerate(pro_rec.next_steps, 1):
        print(f"  {i}. {step}")
    
    print("\n" + "="*70)
    print("Notice how the SAME agent produces personalized responses")
    print("based on injected dependencies!")
    print("="*70)


# Step 7: Advanced - Accessing deps inside the system prompt
# You can also use RunContext in tools (covered in later lessons)
def demonstrate_run_context():
    """Show how to access deps during agent execution"""
    
    # Create an agent that uses RunContext
    context_agent = Agent(
        model='gemini-1.5-flash',
        result_type=str,  # Simple string response for demo
        deps_type=AgentDependencies,
        system_prompt="""
You are a helpful assistant. Use the user context to personalize responses.
""",
    )
    
    # You can access deps in the execution context
    # This is useful for tools and complex agents (next lessons)
    deps = AgentDependencies(
        user=UserContext(
            user_id="u_789",
            username="Charlie",
            subscription_tier="enterprise",
            joined_date=datetime(2022, 3, 10),
            preferences={}
        ),
        database=DatabaseConnection("postgresql://localhost/app"),
        config=AppConfig(
            api_version="v1",
            max_response_length=2000,
            enable_analytics=True,
            feature_flags={"beta_features": True}
        ),
        request_timestamp=datetime.now()
    )
    
    result = context_agent.run_sync(
        "What's my subscription tier?",
        deps=deps
    )
    
    print(f"\nContext-aware response: {result.data}")


if __name__ == "__main__":
    main()
    demonstrate_run_context()
```

### Line-by-Line Explanation

**Dependency Classes (Lines 17-58)**:
- `@dataclass`: Python's clean way to create classes that hold data
- `UserContext`: User-specific information (ID, tier, preferences)
- `DatabaseConnection`: Resource that provides data access
- `AppConfig`: Application settings and feature flags
- These are **plain Python classes** - no Pydantic magic needed here

**Combined Dependencies (Lines 61-70)**:
- `AgentDependencies`: Bundles all context into one type
- This is what you'll pass to `agent.run_sync(deps=...)`
- Type hint ensures you can't forget any required context
- Think of this as your "agent's environment"

**Agent with deps_type (Lines 86-111)**:
- `deps_type=AgentDependencies`: Tells agent what dependencies to expect
- System prompt can reference dependency attributes
- Agent now has "awareness" of user context
- Still returns structured, validated output!

**Using Dependencies (Lines 114-128)**:
- `deps: AgentDependencies`: Type hint documents what's needed
- `agent.run_sync(query, deps=deps)`: Pass deps explicitly
- Pydantic AI makes deps available throughout execution
- No global state, no hidden variables - everything is explicit!

**Different Contexts = Different Results (Lines 131-222)**:
- Same agent, same query, different dependencies
- Free user gets basic features
- Pro user gets advanced features
- All type-safe and validated!

### The "Why" Behind the Pattern

**Why use dependencies instead of global variables?**

❌ **Global Variables** (Bad):
```python
current_user = None  # Global state - dangerous!

def get_recommendation(query: str):
    # What if current_user is None?
    # What if another thread changes it?
    # How do you test this?
    return agent.run_sync(query)  # Uses global current_user
```

✅ **Dependencies** (Good):
```python
def get_recommendation(query: str, deps: AgentDependencies) -> Recommendation:
    # ✅ Explicit: you can see exactly what's needed
    # ✅ Type-safe: mypy checks deps is correct type
    # ✅ Testable: easy to create mock dependencies
    # ✅ Thread-safe: each call has its own deps
    return agent.run_sync(query, deps=deps).data
```

**Benefits of Dependency Injection**:
1. **Testability**: Mock deps for unit tests
2. **Flexibility**: Different deps for different contexts
3. **Type Safety**: Compiler checks everything
4. **Explicit**: No hidden globals or magic
5. **Reusability**: Same agent, different contexts
6. **Thread Safety**: No shared mutable state

---

## C. Test & Apply

### How to Test It

1. **Run the dependency demo**:
```bash
python lesson_03_dependencies.py
```

2. **Observe how same agent produces different outputs** based on injected dependencies

3. **Try your own dependency class**:
```python
@dataclass
class MyCustomDeps:
    api_key: str
    user_role: str
    feature_enabled: bool

my_agent = Agent(
    model='gemini-1.5-flash',
    result_type=str,
    deps_type=MyCustomDeps,
    system_prompt="Use the provided context to personalize your response."
)

deps = MyCustomDeps(
    api_key="sk_test_123",
    user_role="admin",
    feature_enabled=True
)

result = my_agent.run_sync("Hello!", deps=deps)
print(result.data)
```

### Expected Result

You should see personalized outputs based on user context:

**Free Tier User (Alice)**:
```
Greeting: Hi Alice, great to see you!
Recommendation: Start with Python fundamentals through our free interactive tutorials
Reason: Based on your practical learning style and web development interest, hands-on tutorials are perfect for you
Tier Feature: None (free tier)
Next Steps:
  1. Complete the Python basics course
  2. Build a simple web app with Flask
  3. Join our community forum for support
```

**Pro Tier User (Bob)**:
```
Greeting: Welcome back, Bob!
Recommendation: Dive into advanced Python topics including async programming, decorators, and metaprogramming
Reason: Your theoretical learning style and machine learning interest pair well with advanced Python concepts
Tier Feature: Access to exclusive machine learning workshop series
Next Steps:
  1. Enroll in the Advanced Python for ML course
  2. Schedule a 1-on-1 session with a Python expert
  3. Get early access to our new async ML pipeline tutorial
```

### Validation Examples

**Type-Safe Dependency Pattern**:

```python
# ✅ Type-safe - mypy catches errors
def process_request(query: str, deps: AgentDependencies) -> Recommendation:
    result = agent.run_sync(query, deps=deps)
    return result.data

# ❌ This will fail type checking
def process_request_bad(query: str, deps: str) -> Recommendation:  # Wrong type!
    result = agent.run_sync(query, deps=deps)  # mypy error!
    return result.data

# ❌ This will fail at runtime
deps = "not a dependency object"  # Wrong type
result = agent.run_sync("hello", deps=deps)  # Error!
```

### Type Checking

Run mypy to verify type safety:

```bash
mypy lesson_03_dependencies.py
```

Expected: `Success: no issues found`

Try introducing a type error:
```python
# Wrong type for deps
deps: AgentDependencies = "wrong"  # mypy error!
```

---

## D. Common Stumbling Blocks

### 1. Forgetting deps_type Parameter

**The Error**:
```python
agent = Agent(
    model='gemini-1.5-flash',
    result_type=Recommendation,
    # ❌ Forgot deps_type!
)

result = agent.run_sync("hello", deps=my_deps)
# Error: Agent does not expect dependencies
```

**What Causes It**:
You're passing `deps` to `run_sync()` but didn't declare `deps_type` when creating the agent.

**The Fix**:
Always specify `deps_type` if you plan to use dependencies:
```python
agent = Agent(
    model='gemini-1.5-flash',
    result_type=Recommendation,
    deps_type=AgentDependencies,  # ✅ Declare dependency type
)
```

### 2. Wrong Dependency Type

**The Error**:
```python
@dataclass
class MyDeps:
    user_id: str

agent = Agent(
    model='gemini-1.5-flash',
    result_type=str,
    deps_type=MyDeps,
)

# ❌ Passing wrong type
wrong_deps = {"user_id": "123"}  # Dict, not MyDeps!
result = agent.run_sync("hello", deps=wrong_deps)
# TypeError: deps must be of type MyDeps
```

**The Fix**:
Always create an instance of your dependency class:
```python
# ✅ Correct type
correct_deps = MyDeps(user_id="123")
result = agent.run_sync("hello", deps=correct_deps)
```

### 3. Mutable Default Arguments in Dataclass

**The Problem**:
```python
@dataclass
class MyDeps:
    user_id: str
    tags: list[str] = []  # ❌ Mutable default - dangerous!

# This causes weird bugs:
deps1 = MyDeps(user_id="1")
deps1.tags.append("test")

deps2 = MyDeps(user_id="2")
print(deps2.tags)  # ["test"] - what?! They share the same list!
```

**The Fix**:
Use `field(default_factory=...)` for mutable defaults:
```python
from dataclasses import dataclass, field

@dataclass
class MyDeps:
    user_id: str
    tags: list[str] = field(default_factory=list)  # ✅ Safe!

deps1 = MyDeps(user_id="1")
deps1.tags.append("test")

deps2 = MyDeps(user_id="2")
print(deps2.tags)  # [] - correct, independent lists!
```

### 4. Accessing Dependencies in System Prompt

**Common Confusion**:
```python
system_prompt = """
You are a helpful assistant.
The user's name is {deps.user.username}.  # ❌ This doesn't work!
"""
```

**What's Wrong**:
System prompts are static strings. You can't directly interpolate dependencies.

**The Fix Option 1**: Use tools to access deps (covered in next lessons)

**The Fix Option 2**: Build dynamic prompts:
```python
def create_agent(deps: AgentDependencies) -> Agent:
    """Create agent with personalized system prompt"""
    system_prompt = f"""
You are a helpful assistant for {deps.user.username}.
Their subscription tier is {deps.user.subscription_tier}.
Tailor your responses accordingly.
"""
    return Agent(
        model='gemini-1.5-flash',
        result_type=str,
        system_prompt=system_prompt,
    )

# Create agent per user
agent = create_agent(my_deps)
```

**The Fix Option 3** (Best): Reference deps generically in prompt:
```python
system_prompt = """
You are a helpful assistant.
You have access to user context through dependencies.
Use deps.user.username to personalize greetings.
Use deps.user.subscription_tier to tailor feature recommendations.
"""
# Agent will access deps during execution via RunContext (covered in tool lessons)
```

### 5. Type Safety Gotcha: Optional Dependencies

**The Problem**:
```python
@dataclass
class MyDeps:
    user_id: str
    session: Optional[DatabaseConnection] = None

def my_function(deps: MyDeps):
    # ❌ mypy error: 'None' has no attribute 'query'
    results = deps.session.query("SELECT * FROM users")
```

**The Fix**:
Always check optional dependencies:
```python
def my_function(deps: MyDeps):
    if deps.session is None:
        raise ValueError("Database session is required")
    
    # ✅ Now mypy knows session is not None
    results = deps.session.query("SELECT * FROM users")

# Or use a type guard
def my_function(deps: MyDeps):
    assert deps.session is not None, "Session required"
    results = deps.session.query("SELECT * FROM users")
```

---

## Ready for the Next Lesson?

🎉 **Fantastic work!** You now understand how to use dependencies to:

✅ Inject context and resources into agents type-safely  
✅ Keep agents pure and testable (no global state)  
✅ Personalize agent behavior based on user context  
✅ Use dataclasses to define dependency structure  
✅ Leverage IDE autocomplete for dependency attributes  

**Dependencies are the backbone of production AI systems!** They enable testable, reusable, context-aware agents.

In the next lesson, we'll explore the **Dependency Injection Pattern** in depth - you'll learn advanced patterns like dependency factories, async dependencies, and how to structure complex dependency hierarchies!

**Ready for Lesson 4, or would you like to experiment with custom dependencies first?** 🚀
