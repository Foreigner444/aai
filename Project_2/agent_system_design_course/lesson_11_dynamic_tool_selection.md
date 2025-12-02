# Lesson 11: Dynamic Tool Selection

## A. Concept Overview

### What & Why
**Dynamic Tool Selection** is Gemini's ability to intelligently choose which tools to invoke based on the user's query, the tool descriptions, the system prompt, and the current context. Unlike hardcoded workflows, Gemini makes runtime decisions about which tools are relevant, in what order to call them, and when to stop. This is crucial because real-world queries are unpredictable - you can't anticipate every possible workflow, so you need an agent that adapts intelligently.

### Analogy
Think of dynamic tool selection like an experienced mechanic diagnosing a car problem:

**Hardcoded Workflow** = Following a rigid checklist:
1. Check battery (always)
2. Check spark plugs (always)
3. Check fuel pump (always)
4. Check transmission (always)
...even if the customer just said "my radio doesn't work"

**Dynamic Selection** = Intelligent diagnosis:
- Customer: "My car won't start"
- Mechanic thinks: *Could be battery, starter, fuel...*
- Mechanic: Checks battery first (most common) ✅
- Battery is fine, so checks starter ✅
- Starter is faulty - **stops here**, no need to check fuel, transmission, etc.

The mechanic **dynamically selects** which diagnostic tools to use based on symptoms and intermediate findings!

### Type Safety Benefit
Dynamic tool selection with type safety provides:
- **Selection validation**: Only registered, type-safe tools can be selected
- **Parameter validation**: Selected tools receive validated parameters
- **Return type safety**: Tool outputs are typed and validated
- **Error containment**: Invalid selections fail gracefully with type errors
- **Refactoring safety**: Add/remove tools, agent adapts automatically
- **Testing**: Mock tool selection for specific test scenarios

Your agent becomes an intelligent, type-safe decision engine!

---

## B. Code Implementation

### File Structure
```
agent_system_design/
├── lesson_11_dynamic_tool_selection.py
├── requirements.txt
└── .env
```

### Complete Code Snippet

**lesson_11_dynamic_tool_selection.py**
```python
"""
Lesson 11: Dynamic Tool Selection
Learn how Gemini selects tools and how to guide that process
"""

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from dataclasses import dataclass
from typing import Literal, Optional
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


# Simple dependencies
@dataclass
class ToolSelectionDeps:
    """Dependencies for tool selection examples"""
    user_id: str
    session_id: str


# Result model
class SmartResponse(BaseModel):
    """Response showing tool selection decisions"""
    answer: str
    reasoning: str = Field(description="Why certain tools were selected")
    tools_selected: list[str]
    tools_considered_but_skipped: list[str] = Field(default_factory=list)
    selection_strategy: str


# EXAMPLE 1: Poor Tool Design (Hard to Select Correctly)
# This demonstrates what NOT to do

poor_agent = Agent(
    model='gemini-1.5-flash',
    result_type=str,
    system_prompt="You have access to some tools.",  # ❌ Vague
)

@poor_agent.tool
def tool1(x: str):
    """Do something."""  # ❌ Unclear purpose
    return f"Result: {x}"

@poor_agent.tool
def tool2(y: int):
    """Helper function."""  # ❌ No context on when to use
    return y * 2

@poor_agent.tool
def process(data: str):
    """Process data."""  # ❌ What kind of processing?
    return data.upper()


# EXAMPLE 2: Good Tool Design (Easy to Select Correctly)
# This demonstrates best practices

good_agent = Agent(
    model='gemini-1.5-flash',
    result_type=SmartResponse,
    deps_type=ToolSelectionDeps,
    system_prompt="""
You are an intelligent customer service assistant with specialized tools.

TOOL SELECTION STRATEGY:

**For user account queries:**
- Use get_user_profile for general profile information
- Use get_user_orders for purchase history
- Use get_user_preferences for settings and preferences

**For product queries:**
- Use search_products for finding products
- Use get_product_details for specific product information
- Use check_inventory for stock availability

**For order queries:**
- Use track_order for shipping status
- Use get_order_details for order information
- Use cancel_order ONLY if user explicitly requests cancellation

**For support queries:**
- Use search_knowledge_base for general help articles
- Use create_support_ticket for unresolved issues

DECISION PROCESS:
1. Identify the query type (account, product, order, or support)
2. Select the most specific tool that matches the query
3. If you need information from multiple domains, use multiple tools
4. Don't use tools "just in case" - only call what's needed
5. Explain your tool selection reasoning in the response

EFFICIENCY:
- Avoid redundant tool calls
- Don't fetch data you won't use
- Stop when you have enough information to answer
""",
)


# Category 1: User Account Tools

@good_agent.tool
def get_user_profile(
    ctx: RunContext[ToolSelectionDeps]
) -> dict[str, any]:
    """
    Get user profile information including name, email, and membership status.
    
    **When to use:**
    - User asks "what's my account info?" or "my profile"
    - Need to personalize greetings
    - Need to check membership tier
    
    **When NOT to use:**
    - User is asking about orders (use get_user_orders)
    - User is asking about products (use search_products)
    - User is asking about shipping (use track_order)
    
    Returns:
        User profile with name, email, membership_tier, and join_date
    """
    print(f"\n🔧 get_user_profile()")
    return {
        "name": "Jane Smith",
        "email": "jane@example.com",
        "membership_tier": "gold",
        "join_date": "2022-03-15",
        "total_orders": 23
    }


@good_agent.tool
def get_user_orders(
    ctx: RunContext[ToolSelectionDeps],
    limit: int = 5
) -> dict[str, any]:
    """
    Get user's recent orders.
    
    **When to use:**
    - User asks about "my orders" or "purchase history"
    - User asks "what did I buy recently?"
    - Need to reference past purchases
    
    **When NOT to use:**
    - User is tracking a specific order (use track_order)
    - User is browsing products (use search_products)
    - User needs account details (use get_user_profile)
    
    Args:
        limit: Number of recent orders to return (default: 5)
    
    Returns:
        List of recent orders with id, date, status, and total
    """
    print(f"\n🔧 get_user_orders(limit={limit})")
    return {
        "orders": [
            {"id": "ORD-1001", "date": "2024-11-20", "status": "delivered", "total": 89.99},
            {"id": "ORD-0987", "date": "2024-11-05", "status": "delivered", "total": 145.50},
            {"id": "ORD-0923", "date": "2024-10-28", "status": "delivered", "total": 67.25},
        ],
        "total_orders": 23
    }


@good_agent.tool
def get_user_preferences(
    ctx: RunContext[ToolSelectionDeps]
) -> dict[str, any]:
    """
    Get user's saved preferences and settings.
    
    **When to use:**
    - User asks about "my settings" or "preferences"
    - User asks "what are my notification settings?"
    - Need to check communication preferences
    
    **When NOT to use:**
    - User is asking about orders or profile (use other tools)
    - User wants to UPDATE preferences (tell them how to do it in settings)
    
    Returns:
        User preferences including notifications, language, and saved addresses
    """
    print(f"\n🔧 get_user_preferences()")
    return {
        "notifications": {
            "email": True,
            "sms": False,
            "push": True
        },
        "language": "en-US",
        "currency": "USD",
        "saved_addresses": 2
    }


# Category 2: Product Tools

@good_agent.tool
def search_products(
    query: str,
    max_results: int = 5
) -> dict[str, any]:
    """
    Search for products in the catalog.
    
    **When to use:**
    - User asks to "find" or "search" for products
    - User describes what they're looking for
    - User asks "do you have [product type]?"
    
    **When NOT to use:**
    - User asks about a specific product ID (use get_product_details)
    - User asks about stock (use check_inventory)
    - User is asking about their past purchases (use get_user_orders)
    
    Args:
        query: Search query (product name, category, keywords)
        max_results: Maximum products to return (default: 5)
    
    Returns:
        List of matching products with id, name, price, and rating
    
    Example queries:
        "wireless headphones" → Search for headphones
        "blue running shoes" → Search for specific shoes
        "laptop under $1000" → Search with price constraint
    """
    print(f"\n🔧 search_products(query='{query}')")
    return {
        "query": query,
        "results": [
            {"id": "PROD-501", "name": f"{query.title()} Pro", "price": 149.99, "rating": 4.5},
            {"id": "PROD-502", "name": f"{query.title()} Lite", "price": 89.99, "rating": 4.2},
            {"id": "PROD-503", "name": f"{query.title()} Max", "price": 249.99, "rating": 4.8},
        ],
        "total_found": 12
    }


@good_agent.tool
def get_product_details(
    product_id: str
) -> dict[str, any]:
    """
    Get detailed information about a specific product.
    
    **When to use:**
    - User mentions a specific product ID or name
    - User asks for details about a product they found
    - User asks "tell me more about [product]"
    
    **When NOT to use:**
    - User is searching/browsing (use search_products)
    - User wants to check stock (use check_inventory after this)
    - User has general questions (use search_knowledge_base)
    
    Args:
        product_id: Product identifier (e.g., "PROD-501")
    
    Returns:
        Detailed product information including specs, reviews, and availability
    """
    print(f"\n🔧 get_product_details(product_id={product_id})")
    return {
        "id": product_id,
        "name": "Wireless Headphones Pro",
        "price": 149.99,
        "description": "Premium wireless headphones with active noise cancellation",
        "specs": {
            "battery_life": "30 hours",
            "connectivity": "Bluetooth 5.0",
            "weight": "250g"
        },
        "rating": 4.5,
        "review_count": 342,
        "in_stock": True
    }


@good_agent.tool
def check_inventory(
    product_id: str,
    location: Optional[str] = None
) -> dict[str, any]:
    """
    Check stock availability for a product.
    
    **When to use:**
    - User asks "is [product] in stock?"
    - User asks "when will it be available?"
    - Planning to mention availability in response
    
    **When NOT to use:**
    - User hasn't asked about availability
    - User is just browsing products
    - Stock info is included in get_product_details
    
    Args:
        product_id: Product identifier
        location: Optional location/warehouse to check
    
    Returns:
        Stock status, quantity available, and estimated restock date
    """
    print(f"\n🔧 check_inventory(product_id={product_id})")
    return {
        "product_id": product_id,
        "in_stock": True,
        "quantity_available": 47,
        "warehouse": "West Coast",
        "estimated_delivery": "2-3 business days"
    }


# Category 3: Order Tools

@good_agent.tool
def track_order(
    order_id: str
) -> dict[str, any]:
    """
    Track shipping status of a specific order.
    
    **When to use:**
    - User asks "where is my order?"
    - User provides an order number and wants status
    - User asks "when will my order arrive?"
    
    **When NOT to use:**
    - User is asking about past orders in general (use get_user_orders)
    - Order hasn't shipped yet
    - User wants order details, not tracking (use get_order_details)
    
    Args:
        order_id: Order identifier (e.g., "ORD-1001")
    
    Returns:
        Tracking information with status, location, and estimated delivery
    """
    print(f"\n🔧 track_order(order_id={order_id})")
    return {
        "order_id": order_id,
        "status": "in_transit",
        "current_location": "Memphis, TN",
        "last_update": "2024-12-02 14:30",
        "estimated_delivery": "2024-12-04",
        "tracking_number": "1Z999AA10123456784"
    }


@good_agent.tool
def get_order_details(
    order_id: str
) -> dict[str, any]:
    """
    Get detailed information about a specific order.
    
    **When to use:**
    - User asks "what's in my order [id]?"
    - User asks about order contents or pricing
    - User needs billing information
    
    **When NOT to use:**
    - User wants tracking info (use track_order)
    - User wants list of all orders (use get_user_orders)
    
    Args:
        order_id: Order identifier
    
    Returns:
        Complete order details including items, pricing, and shipping address
    """
    print(f"\n🔧 get_order_details(order_id={order_id})")
    return {
        "order_id": order_id,
        "date": "2024-11-20",
        "status": "delivered",
        "items": [
            {"name": "Wireless Headphones Pro", "quantity": 1, "price": 149.99},
            {"name": "USB-C Cable", "quantity": 2, "price": 12.99}
        ],
        "subtotal": 175.97,
        "shipping": 5.99,
        "tax": 14.08,
        "total": 196.04
    }


# Category 4: Support Tools

@good_agent.tool
def search_knowledge_base(
    query: str
) -> dict[str, any]:
    """
    Search help articles and documentation.
    
    **When to use:**
    - User has a general "how do I" question
    - User needs help with a process or feature
    - User asks about policies (returns, shipping, etc.)
    
    **When NOT to use:**
    - User has a specific problem that needs a ticket (use create_support_ticket)
    - User is asking about their specific account/order (use specific tools)
    
    Args:
        query: Search query for help content
    
    Returns:
        Relevant help articles with titles and summaries
    
    Example queries:
        "how to return an item" → Return policy articles
        "reset password" → Password reset instructions
        "shipping times" → Shipping information
    """
    print(f"\n🔧 search_knowledge_base(query='{query}')")
    return {
        "query": query,
        "articles": [
            {
                "title": f"How to {query}",
                "summary": f"Step-by-step guide for {query}...",
                "url": "https://help.example.com/article-1"
            },
            {
                "title": f"{query.title()} - Frequently Asked Questions",
                "summary": "Common questions and answers...",
                "url": "https://help.example.com/article-2"
            }
        ]
    }


@good_agent.tool
def create_support_ticket(
    ctx: RunContext[ToolSelectionDeps],
    issue_description: str,
    priority: Literal["low", "medium", "high"] = "medium"
) -> dict[str, any]:
    """
    Create a support ticket for issues that can't be resolved immediately.
    
    **When to use:**
    - User has a problem that requires human support
    - Issue is not covered in knowledge base
    - User explicitly asks to "speak to a person" or "create a ticket"
    
    **When NOT to use:**
    - Issue can be resolved with other tools
    - User is just asking for information (use other tools)
    - User hasn't confirmed they want to create a ticket
    
    **IMPORTANT:**
    - Always confirm with user before creating a ticket
    - Explain what will happen after ticket is created
    - This should be a last resort, not first option
    
    Args:
        issue_description: Description of the user's issue
        priority: Ticket priority (default: medium)
    
    Returns:
        Ticket information with ticket ID and estimated response time
    """
    print(f"\n🔧 create_support_ticket(priority={priority})")
    return {
        "ticket_id": "TKT-7891",
        "status": "created",
        "priority": priority,
        "estimated_response": "24 hours",
        "message": "A support specialist will contact you via email"
    }


# Demonstration

def main():
    print("\n" + "="*70)
    print("DYNAMIC TOOL SELECTION DEMONSTRATION")
    print("="*70)
    print("\nThis agent has 11 tools across 4 categories:")
    print("  👤 User Account: get_user_profile, get_user_orders, get_user_preferences")
    print("  🛍️  Products: search_products, get_product_details, check_inventory")
    print("  📦 Orders: track_order, get_order_details")
    print("  🆘 Support: search_knowledge_base, create_support_ticket")
    print("\nWatch how Gemini dynamically selects the RIGHT tools for each query!")
    
    deps = ToolSelectionDeps(user_id="user_123", session_id="sess_456")
    
    # Test queries demonstrating intelligent tool selection
    test_cases = [
        {
            "query": "What's my account information?",
            "expected_tools": ["get_user_profile"],
            "should_skip": ["get_user_orders", "search_products"]
        },
        {
            "query": "Find wireless headphones under $100",
            "expected_tools": ["search_products"],
            "should_skip": ["get_user_profile", "track_order"]
        },
        {
            "query": "Where is my order ORD-1001?",
            "expected_tools": ["track_order"],
            "should_skip": ["get_user_orders", "search_products"]
        },
        {
            "query": "Show me my recent orders and tell me if I bought headphones",
            "expected_tools": ["get_user_orders"],
            "should_skip": ["track_order", "search_products"]
        },
        {
            "query": "Tell me about product PROD-501 and check if it's in stock",
            "expected_tools": ["get_product_details", "check_inventory"],
            "should_skip": ["search_products", "get_user_orders"]
        },
        {
            "query": "How do I return an item?",
            "expected_tools": ["search_knowledge_base"],
            "should_skip": ["create_support_ticket", "track_order"]
        },
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        query = test_case["query"]
        
        print(f"\n\n{'='*70}")
        print(f"TEST {i}: {query}")
        print(f"{'='*70}")
        print(f"Expected tools: {', '.join(test_case['expected_tools'])}")
        print(f"Should skip: {', '.join(test_case['should_skip'])}")
        
        try:
            result = good_agent.run_sync(query, deps=deps)
            response = result.data
            
            print(f"\n✅ AGENT RESPONSE:")
            print(f"   {response.answer}")
            
            print(f"\n🧠 REASONING:")
            print(f"   {response.reasoning}")
            
            print(f"\n🔧 TOOLS SELECTED: {', '.join(response.tools_selected)}")
            
            if response.tools_considered_but_skipped:
                print(f"⏭️  TOOLS SKIPPED: {', '.join(response.tools_considered_but_skipped)}")
            
            print(f"\n📋 SELECTION STRATEGY: {response.selection_strategy}")
            
            # Validate selection
            expected = set(test_case["expected_tools"])
            actual = set(response.tools_selected)
            should_skip = set(test_case["should_skip"])
            
            if expected.intersection(actual):
                print(f"\n✅ CORRECT: Used expected tools")
            
            if actual.intersection(should_skip):
                print(f"\n⚠️  WARNING: Used tools that should have been skipped")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
    
    print("\n" + "="*70)
    print("KEY OBSERVATIONS")
    print("="*70)
    print("\n✅ What makes tool selection work:")
    print("   • Clear, descriptive tool names")
    print("   • Detailed docstrings with 'When to use' and 'When NOT to use'")
    print("   • System prompt with explicit selection strategy")
    print("   • Tools grouped by logical categories")
    print("   • Each tool has a single, clear purpose")
    print("\n❌ What hurts tool selection:")
    print("   • Vague tool names (tool1, tool2, helper)")
    print("   • Missing or unclear docstrings")
    print("   • Tools that do too many things")
    print("   • No guidance in system prompt")
    print("   • Overlapping tool responsibilities")


if __name__ == "__main__":
    main()
```

### Line-by-Line Explanation

**Poor Tool Design (Lines 32-52)**:
- Vague tool names: `tool1`, `tool2`, `process`
- Minimal docstrings with no context
- No guidance on when to use each tool
- **This is what NOT to do!**

**Good Tool Design (Lines 55-101)**:
- Comprehensive system prompt with selection strategy
- Clear decision process explained
- Guidance for each query type
- Efficiency tips to avoid over-using tools

**Tool Categories (Lines 104-450)**:
- **User Account Tools**: Profile, orders, preferences
- **Product Tools**: Search, details, inventory
- **Order Tools**: Tracking, order details
- **Support Tools**: Knowledge base, tickets
- Each tool has clear "When to use" and "When NOT to use" sections

**Selection Guidance in Docstrings**:
- Every tool explains its purpose
- Lists specific use cases
- Lists scenarios where other tools are better
- Provides examples of typical queries

### The "Why" Behind the Pattern

**What influences Gemini's tool selection?**

1. **Tool Name** (30% weight):
   - `get_user_profile` ✅ Clear
   - `tool1` ❌ Unclear

2. **Docstring** (40% weight):
   - Detailed description ✅
   - "When to use" / "When NOT to use" sections ✅
   - Examples ✅

3. **System Prompt** (20% weight):
   - Selection strategy explained ✅
   - Query type → tool mapping ✅
   - Efficiency guidance ✅

4. **Parameter Names** (10% weight):
   - `search_products(query="headphones")` ✅ Clear
   - `process(x="headphones")` ❌ Unclear

**Best Practice: The Tool Selection Triangle**
```
        Clear Tool Name
              /\
             /  \
            /    \
           /      \
          /        \
         /  PERFECT \
        /   SELECTION \
       /              \
      /________________\
Detailed Docstring    Strategic System Prompt
```

All three must work together!

---

## C. Test & Apply

### How to Test It

1. **Run the tool selection demo**:
```bash
python lesson_11_dynamic_tool_selection.py
```

2. **Observe which tools Gemini selects for each query**

3. **Try your own test cases**:
```python
test_cases = [
    {
        "query": "Your test query here",
        "expected_tools": ["tool_you_expect"],
        "should_skip": ["tools_that_shouldnt_be_used"]
    }
]
```

### Expected Result

You should see intelligent, efficient tool selection:

```
======================================================================
TEST 1: What's my account information?
======================================================================
Expected tools: get_user_profile
Should skip: get_user_orders, search_products

🔧 get_user_profile()

✅ AGENT RESPONSE:
   Your account information: You're Jane Smith (jane@example.com), a Gold 
   member since March 2022 with 23 total orders.

🧠 REASONING:
   I selected get_user_profile because the query specifically asks for 
   account information. I did not use get_user_orders or get_user_preferences 
   because the user didn't ask about orders or settings specifically.

🔧 TOOLS SELECTED: get_user_profile

📋 SELECTION STRATEGY: Single-tool targeted selection based on query intent

✅ CORRECT: Used expected tools

======================================================================
TEST 5: Tell me about product PROD-501 and check if it's in stock
======================================================================
Expected tools: get_product_details, check_inventory
Should skip: search_products, get_user_orders

🔧 get_product_details(product_id=PROD-501)
🔧 check_inventory(product_id=PROD-501)

✅ AGENT RESPONSE:
   The Wireless Headphones Pro (PROD-501) are premium headphones priced at 
   $149.99. They have a 4.5-star rating from 342 reviews. Great news - they're 
   in stock with 47 units available at our West Coast warehouse!

🧠 REASONING:
   I used get_product_details first to get information about PROD-501, then 
   check_inventory to confirm stock status since the user explicitly asked if 
   it's in stock. I skipped search_products because the user already knew the 
   product ID.

🔧 TOOLS SELECTED: get_product_details, check_inventory

📋 SELECTION STRATEGY: Multi-tool coordinated selection for complete answer
```

### Validation Examples

**Good Tool Selection Patterns**:

```python
# ✅ Single-tool selection for simple queries
"What's my email?" → get_user_profile (only)

# ✅ Multi-tool selection when needed
"Find laptops and check if any are in stock" 
→ search_products, then check_inventory

# ✅ Skipping unnecessary tools
"Track order ORD-1001" → track_order (skips get_user_orders, get_order_details)

# ✅ Appropriate tool chain
"My recent orders" → get_user_orders 
(not track_order for each order - inefficient!)
```

### Type Checking

```bash
mypy lesson_11_dynamic_tool_selection.py
```

Expected: `Success: no issues found`

---

## D. Common Stumbling Blocks

### 1. Overlapping Tool Responsibilities

**The Problem**:
```python
@agent.tool
def get_user_info():
    """Get user information."""  # ❌ Which information?
    return {"name": "...", "orders": [...], "preferences": {...}}

@agent.tool
def get_user_data():
    """Get user data."""  # ❌ Same as get_user_info?
    return {"name": "...", "email": "..."}
```

**What's Wrong**:
- Gemini can't decide which to use
- Unclear boundaries between tools
- Duplicate functionality

**The Fix**:
Make tools distinct and specific:
```python
@agent.tool
def get_user_profile():
    """Get user profile: name, email, membership. Use for account info."""
    return {"name": "...", "email": "...", "membership": "..."}

@agent.tool
def get_user_orders():
    """Get user order history. Use when user asks about purchases."""
    return {"orders": [...]}

@agent.tool
def get_user_preferences():
    """Get user settings and preferences. Use for notification/language settings."""
    return {"preferences": {...}}
```

### 2. Missing "When NOT to Use"

**The Problem**:
```python
@agent.tool
def search_database(query: str):
    """
    Search the database.
    
    When to use:
    - User wants to find something
    """  # ❌ No guidance on when NOT to use
```

**What Happens**:
Gemini might use this for EVERY query that involves finding something, even when more specific tools exist.

**The Fix**:
Always include "When NOT to use":
```python
@agent.tool
def search_database(query: str):
    """
    Search the database for records.
    
    When to use:
    - User wants to find database records
    - General data search needed
    
    When NOT to use:
    - Searching for products (use search_products)
    - Searching for users (use search_users)
    - Searching help articles (use search_knowledge_base)
    
    This is a general fallback when specific search tools don't apply.
    """
```

### 3. System Prompt Conflicts with Tool Descriptions

**The Problem**:
```python
system_prompt = """
Always check inventory before showing product details.
"""

@agent.tool
def get_product_details(product_id: str):
    """
    Get product details.
    
    When to use: User asks about a product
    When NOT to use: User only wants stock info (use check_inventory)
    """  # ❌ Conflicts with system prompt!
```

**What Happens**:
Confusion - system prompt says always check inventory first, but tool says don't use it unless needed.

**The Fix**:
Ensure system prompt and tool descriptions align:
```python
system_prompt = """
For product queries:
1. If user asks about specific product, use get_product_details
2. If user asks about stock specifically, use check_inventory
3. Product details already include basic stock status
"""

@agent.tool
def get_product_details(product_id: str):
    """
    Get product details including basic availability.
    
    When to use: User asks about a product
    
    Note: Includes basic stock status. For detailed inventory 
    info (quantities, warehouse location), use check_inventory.
    """  # ✅ Aligned with system prompt
```

### 4. Too Many Similar Tools

**The Problem**:
```python
@agent.tool
def get_user_profile(): ...

@agent.tool
def fetch_user_profile(): ...

@agent.tool
def retrieve_user_profile(): ...

@agent.tool
def load_user_profile(): ...
# ❌ All do the same thing with different names!
```

**What Happens**:
- Gemini randomly picks one
- Inconsistent behavior
- Wasted tool slots

**The Fix**:
One tool, one responsibility:
```python
@agent.tool
def get_user_profile():
    """
    Get user profile information.
    
    This is THE tool for user profile data.
    """  # ✅ One clear tool
```

### 5. Type Safety Gotcha: Tool Selection in Tests

**The Problem**:
```python
# ❌ Hard to test which tools were selected
def test_agent():
    result = agent.run_sync("query")
    # How do I know which tools were used?
```

**The Fix**:
Include tool tracking in your result model:
```python
class TrackedResult(BaseModel):
    answer: str
    tools_used: list[str]  # ✅ Track tool usage

# In your agent's result, list tools used
# Then test can verify:
def test_agent():
    result = agent.run_sync("find products")
    assert "search_products" in result.tools_used  # ✅ Testable
    assert "get_user_profile" not in result.tools_used  # ✅ Verify skipped
```

---

## Ready for the Next Lesson?

🎉 **Excellent work!** You now understand how to guide Gemini's tool selection:

✅ Clear, descriptive tool names  
✅ Detailed docstrings with "When to use" / "When NOT to use"  
✅ Strategic system prompts that explain selection logic  
✅ Distinct tool responsibilities  
✅ Tool categories that make sense  
✅ Examples in tool descriptions  

**Dynamic tool selection is what makes agents intelligent!** With proper guidance, Gemini becomes an expert at choosing the right tool at the right time, making your agent efficient and reliable.

In the next lesson, we'll explore **Streaming Responses** - you'll learn how to return results incrementally for better user experience, handle streaming tool calls, and implement server-sent events!

**Ready for Lesson 12, or would you like to practice tool selection patterns first?** 🚀
