# Lesson 8: Tool Descriptions for Gemini

## A. Concept Overview

### What & Why
**Tool Descriptions** are the docstrings and metadata that explain to Gemini what your tools do, when to use them, and how to use them correctly. This is crucial because Gemini uses these descriptions to decide which tools to invoke - poor descriptions lead to wrong tool selections and failed executions, while great descriptions enable perfect tool usage.

### Analogy
Think of tool descriptions like instruction manuals in a toolbox:

**Bad Manual**:
- "Hammer" (that's it)

**Good Manual**:
- **Tool**: Hammer
- **Purpose**: Drive nails into wood or remove them
- **When to use**: Constructing or deconstructing wooden structures
- **How to use**: Hold handle firmly, strike nail head squarely
- **Don't use for**: Screws (use screwdriver), delicate materials (use rubber mallet)
- **Safety**: Watch your fingers, wear eye protection

Clear instructions = correct tool selection and usage!

### Type Safety Benefit
Well-written tool descriptions provide:
- **Validation guidance**: Descriptions explain constraints and valid inputs
- **Type expectations**: Clear documentation of parameter and return types
- **Error prevention**: Gemini understands limitations and edge cases
- **IDE integration**: Docstrings appear in editor tooltips
- **Self-documentation**: Code documents itself for future developers
- **Test generation**: Descriptions guide test case creation

Great descriptions make your entire agent system more reliable!

---

## B. Code Implementation

### File Structure
```
agent_system_design/
├── lesson_08_tool_descriptions.py
├── requirements.txt
└── .env
```

### Complete Code Snippet

**lesson_08_tool_descriptions.py**
```python
"""
Lesson 8: Tool Descriptions for Gemini
Master the art of writing tool descriptions that guide Gemini effectively
"""

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from dataclasses import dataclass
from typing import Literal, Optional
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ToolDeps:
    """Simple dependencies"""
    user_id: str
    session_id: str


class ActionResult(BaseModel):
    """Result of agent's action"""
    action_taken: str
    tool_used: str
    result_summary: str
    success: bool


agent = Agent(
    model='gemini-1.5-flash',
    result_type=ActionResult,
    deps_type=ToolDeps,
    system_prompt="""
You are an intelligent assistant with access to specialized tools.

Read each tool's description carefully to understand:
- What the tool does
- When to use it vs other tools
- What parameters it needs
- What results it returns
- Any limitations or constraints

Choose tools wisely based on user requests.
""",
)


# ❌ ANTIPATTERN: Poor Description

@agent.tool
def bad_tool(x: int, y: int) -> int:
    """Does something."""  # ❌ Vague, no details!
    return x + y


# ✅ PATTERN 1: Clear, Comprehensive Description

@agent.tool
def calculate_distance(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float
) -> dict[str, float]:
    """
    Calculate the distance between two geographic coordinates.
    
    This tool computes the great-circle distance between two points
    on Earth's surface using the Haversine formula. Use this when you
    need to determine how far apart two locations are.
    
    Args:
        lat1: Latitude of first point in decimal degrees (-90 to 90)
        lon1: Longitude of first point in decimal degrees (-180 to 180)
        lat2: Latitude of second point in decimal degrees (-90 to 90)
        lon2: Longitude of second point in decimal degrees (-180 to 180)
    
    Returns:
        Dictionary containing:
        - distance_km: Distance in kilometers
        - distance_mi: Distance in miles
        - straight_line: Whether this is straight-line (true) or driving distance (false)
    
    Example:
        calculate_distance(40.7128, -74.0060, 34.0522, -118.2437)
        # Returns distance from New York to Los Angeles
    
    Note:
        This calculates straight-line distance, not driving/walking routes.
        For actual travel distances, use get_driving_directions instead.
    """
    from math import radians, sin, cos, sqrt, atan2
    
    print(f"\n🔧 calculate_distance({lat1}, {lon1} → {lat2}, {lon2})")
    
    # Haversine formula
    R = 6371  # Earth's radius in km
    
    lat1_rad, lon1_rad = radians(lat1), radians(lon1)
    lat2_rad, lon2_rad = radians(lat2), radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = sin(dlat/2)**2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    
    distance_km = R * c
    distance_mi = distance_km * 0.621371
    
    return {
        "distance_km": round(distance_km, 2),
        "distance_mi": round(distance_mi, 2),
        "straight_line": True
    }


# ✅ PATTERN 2: When to Use vs When Not to Use

@agent.tool
def send_email(
    to_address: str,
    subject: str,
    body: str,
    priority: Literal["low", "normal", "high"] = "normal"
) -> dict[str, any]:
    """
    Send an email to a specified recipient.
    
    **When to use this tool:**
    - User explicitly asks to send an email
    - User wants to notify someone of information
    - User requests to "email", "message via email", or "send to"
    
    **When NOT to use this tool:**
    - For instant messaging (use send_slack_message instead)
    - For SMS (use send_sms instead)
    - For in-app notifications (use create_notification instead)
    - User only wants a draft (use create_email_draft instead)
    
    **Important constraints:**
    - to_address must be a valid email format
    - Email is sent immediately (cannot be undone)
    - User will NOT see the email before sending
    - Always confirm with user before sending important emails
    
    Args:
        to_address: Recipient's email address (validated format)
        subject: Email subject line (required, max 200 characters)
        body: Email body content (plain text, max 10000 characters)
        priority: Email priority flag (default: "normal")
    
    Returns:
        Dictionary with:
        - message_id: Unique identifier for sent email
        - status: "sent", "queued", or "failed"
        - sent_at: ISO timestamp of when email was sent
        - error: Error message if status is "failed", otherwise null
    
    Example:
        send_email(
            to_address="user@example.com",
            subject="Meeting Reminder",
            body="Don't forget our 2pm meeting today!",
            priority="high"
        )
    """
    print(f"\n🔧 send_email(to={to_address}, subject='{subject}')")
    
    # Validate email format (simplified)
    if "@" not in to_address or "." not in to_address:
        return {
            "message_id": None,
            "status": "failed",
            "sent_at": None,
            "error": "Invalid email address format"
        }
    
    # Simulate sending
    message_id = f"msg_{datetime.now().timestamp()}"
    
    return {
        "message_id": message_id,
        "status": "sent",
        "sent_at": datetime.now().isoformat(),
        "error": None
    }


# ✅ PATTERN 3: Detailed Parameter Explanations

@agent.tool
def search_products(
    query: str,
    category: Optional[str] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    sort_by: Literal["relevance", "price_asc", "price_desc", "rating"] = "relevance",
    limit: int = 10
) -> dict[str, any]:
    """
    Search for products in the catalog.
    
    This tool searches the product database using text matching and filters.
    Returns a ranked list of products matching the criteria.
    
    Args:
        query: 
            Search query text. Can include product names, descriptions, or keywords.
            Examples: "wireless headphones", "red dress size M", "laptop under $1000"
            
        category:
            Optional category filter. Only returns products from this category.
            Valid categories: "electronics", "clothing", "home", "sports", "books"
            Use None to search all categories.
            
        min_price:
            Minimum price filter in USD. Products below this price are excluded.
            Use None for no minimum. Can be combined with max_price for a range.
            
        max_price:
            Maximum price filter in USD. Products above this price are excluded.
            Use None for no maximum. Can be combined with min_price for a range.
            
        sort_by:
            How to sort the results:
            - "relevance": Best match to query first (default, recommended)
            - "price_asc": Lowest price first
            - "price_desc": Highest price first
            - "rating": Highest customer rating first
            
        limit:
            Maximum number of results to return (1-100).
            Default is 10, which is optimal for most use cases.
            Higher limits take longer to process.
    
    Returns:
        Dictionary with:
        - products: List of product objects (see structure below)
        - total_found: Total number of matching products (may be more than limit)
        - query: The search query used
        - filters_applied: Which filters were active
        
        Product object structure:
        {
            "id": "prod_123",
            "name": "Product Name",
            "price": 29.99,
            "category": "electronics",
            "rating": 4.5,
            "in_stock": true
        }
    
    Example searches:
        # Simple search
        search_products("laptop")
        
        # Search with category filter
        search_products("shoes", category="sports")
        
        # Search with price range
        search_products("coffee maker", min_price=20, max_price=100)
        
        # Search with sorting
        search_products("tablet", sort_by="price_asc", limit=5)
    """
    print(f"\n🔧 search_products(query='{query}', category={category}, price={min_price}-{max_price})")
    
    # Simulate search
    products = [
        {
            "id": f"prod_{i}",
            "name": f"{query.title()} Model {i}",
            "price": 29.99 + (i * 10),
            "category": category or "electronics",
            "rating": 4.0 + (i * 0.1),
            "in_stock": True
        }
        for i in range(min(limit, 3))  # Return up to 3 sample products
    ]
    
    return {
        "products": products,
        "total_found": 42,
        "query": query,
        "filters_applied": {
            "category": category,
            "min_price": min_price,
            "max_price": max_price,
            "sort_by": sort_by
        }
    }


# ✅ PATTERN 4: Examples in Description

@agent.tool
def parse_date(
    date_string: str,
    format_hint: Optional[str] = None
) -> dict[str, any]:
    """
    Parse a date string into a structured date object.
    
    This tool can handle various date formats and natural language.
    
    Args:
        date_string: The date string to parse
        format_hint: Optional hint about the format (e.g., "US", "EU", "ISO")
    
    Returns:
        Parsed date information or error
    
    Supported formats and examples:
    
    ISO format (recommended):
        "2024-01-15" → January 15, 2024
        "2024-01-15T14:30:00" → January 15, 2024 at 2:30 PM
    
    US format (with format_hint="US"):
        "01/15/2024" → January 15, 2024
        "1/15/24" → January 15, 2024
        "Jan 15, 2024" → January 15, 2024
    
    European format (with format_hint="EU"):
        "15/01/2024" → January 15, 2024
        "15.01.2024" → January 15, 2024
    
    Natural language:
        "today" → Current date
        "tomorrow" → Next day
        "next Monday" → Upcoming Monday
        "in 3 days" → Date 3 days from now
    
    Ambiguous dates:
        "12/01/2024" - Could be Dec 1 or Jan 12
        Use format_hint to disambiguate:
        - format_hint="US" → December 1, 2024
        - format_hint="EU" → January 12, 2024
    
    Edge cases:
        - "February 30" → Returns error (invalid date)
        - "13/01/2024" → Assumes EU format (13th month doesn't exist)
        - "" (empty string) → Returns error
    """
    from dateutil import parser
    
    print(f"\n🔧 parse_date('{date_string}', format_hint={format_hint})")
    
    try:
        # Simple parsing (in real code, use dateutil.parser)
        if date_string.lower() == "today":
            parsed = datetime.now()
        elif date_string.lower() == "tomorrow":
            parsed = datetime.now() + timedelta(days=1)
        else:
            # Simplified ISO parsing
            parsed = datetime.fromisoformat(date_string.replace("Z", "+00:00"))
        
        return {
            "success": True,
            "year": parsed.year,
            "month": parsed.month,
            "day": parsed.day,
            "iso_format": parsed.date().isoformat(),
            "human_readable": parsed.strftime("%B %d, %Y")
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": f"Could not parse date: {str(e)}",
            "input": date_string
        }


# ✅ PATTERN 5: Clear Limitations and Constraints

@agent.tool
def get_weather_forecast(
    city: str,
    days: int = 3
) -> dict[str, any]:
    """
    Get weather forecast for a city.
    
    **Capabilities:**
    - Provides temperature, conditions, and precipitation
    - Supports forecasts up to 7 days ahead
    - Covers major cities worldwide
    
    **Limitations:**
    - Only works for cities, not specific addresses or coordinates
    - Forecasts are less accurate beyond 3 days
    - Historical weather (past dates) NOT supported - use get_historical_weather
    - Hyperlocal weather (specific neighborhoods) NOT available
    - No minute-by-minute or hourly data - only daily forecasts
    
    **Data freshness:**
    - Forecast data updated every 6 hours
    - May not reflect very recent weather changes
    - For current conditions, use get_current_weather instead
    
    Args:
        city: City name (e.g., "New York", "London", "Tokyo")
               Can include country for disambiguation: "Paris, France" vs "Paris, Texas"
        days: Number of days to forecast (1-7, default: 3)
              Days 1-3: High accuracy
              Days 4-7: Moderate accuracy
    
    Returns:
        Dictionary with:
        - city: Confirmed city name
        - forecasts: List of daily forecasts
        - updated_at: When forecast data was last updated
        - accuracy_note: Accuracy information based on forecast length
    
    Error conditions:
        - City not found → Returns error with suggested alternatives
        - days > 7 → Returns error with explanation
        - days < 1 → Returns error with explanation
    """
    print(f"\n🔧 get_weather_forecast(city='{city}', days={days})")
    
    # Validate inputs
    if days < 1 or days > 7:
        return {
            "error": "Days must be between 1 and 7",
            "valid_range": "1-7",
            "requested": days
        }
    
    # Simulate forecast
    forecasts = [
        {
            "date": (datetime.now() + timedelta(days=i)).date().isoformat(),
            "temp_high": 72 + i,
            "temp_low": 58 + i,
            "conditions": "Partly cloudy",
            "precipitation_chance": 20 + (i * 10)
        }
        for i in range(days)
    ]
    
    accuracy = "high" if days <= 3 else "moderate"
    
    return {
        "city": city,
        "forecasts": forecasts,
        "updated_at": datetime.now().isoformat(),
        "accuracy_note": f"Forecast accuracy: {accuracy} ({days} days ahead)"
    }


# ✅ PATTERN 6: Context-Aware Description

@agent.tool
def approve_expense(
    ctx: RunContext[ToolDeps],
    expense_id: str,
    amount: float,
    reason: Optional[str] = None
) -> dict[str, any]:
    """
    Approve an expense report.
    
    **USER CONTEXT MATTERS:**
    This tool uses the current user's permissions to approve expenses.
    Different users have different approval limits:
    - Junior employees: Cannot approve (will return error)
    - Managers: Can approve up to $5,000
    - Directors: Can approve up to $50,000
    - Executives: Can approve any amount
    
    **When to use:**
    - User explicitly says "approve expense [id]"
    - User confirms approval after reviewing expense details
    
    **When NOT to use:**
    - User is just viewing expense information (use get_expense_details)
    - User wants to edit expense (use update_expense)
    - User wants to reject expense (use reject_expense)
    - Expense is already approved or rejected
    
    **Side effects:**
    - ⚠️ THIS ACTION IS PERMANENT (cannot be undone)
    - Triggers email notification to expense submitter
    - Updates accounting system immediately
    - May trigger payment processing
    
    **Security:**
    - Requires valid user session (validated via ctx.deps)
    - Logs approval action with timestamp and user ID
    - Enforces approval limits based on user role
    
    Args:
        expense_id: Unique expense report identifier (format: "EXP-12345")
        amount: Total amount being approved in USD
        reason: Optional reason for approval (recommended for large amounts)
    
    Returns:
        Dictionary with:
        - approved: Whether approval succeeded
        - expense_id: The expense ID that was processed
        - approved_by: User who approved (from context)
        - approved_at: Timestamp of approval
        - error: Error message if approval failed, otherwise null
    
    Common errors:
        - "Insufficient permissions" → User cannot approve this amount
        - "Expense not found" → Invalid expense_id
        - "Already processed" → Expense was already approved or rejected
    """
    print(f"\n🔧 approve_expense(expense_id={expense_id}, amount=${amount})")
    
    user_id = ctx.deps.user_id
    
    # Simulate approval logic
    approved = amount < 5000  # Simplified: assume limit is $5k
    
    if not approved:
        return {
            "approved": False,
            "expense_id": expense_id,
            "approved_by": None,
            "approved_at": None,
            "error": f"Amount ${amount} exceeds your approval limit of $5,000"
        }
    
    return {
        "approved": True,
        "expense_id": expense_id,
        "approved_by": user_id,
        "approved_at": datetime.now().isoformat(),
        "error": None
    }


# Demo

def main():
    print("\n" + "="*70)
    print("TOOL DESCRIPTIONS FOR GEMINI")
    print("="*70)
    
    deps = ToolDeps(user_id="manager_123", session_id="sess_456")
    
    queries = [
        "How far is it from New York (40.7128, -74.0060) to Boston (42.3601, -71.0589)?",
        "Send an email to john@example.com about tomorrow's meeting",
        "Search for wireless headphones under $100",
        "What's the weather forecast for Seattle for the next 5 days?",
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n\n{'='*70}")
        print(f"QUERY {i}: {query}")
        print(f"{'='*70}")
        
        try:
            result = agent.run_sync(query, deps=deps)
            action = result.data
            
            print(f"\n✅ RESULT:")
            print(f"   Action: {action.action_taken}")
            print(f"   Tool: {action.tool_used}")
            print(f"   Summary: {action.result_summary}")
            print(f"   Success: {action.success}")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
    
    print("\n" + "="*70)
    print("DESCRIPTION PATTERNS SUMMARY")
    print("="*70)
    print("\n✅ Pattern 1: Clear, comprehensive description")
    print("✅ Pattern 2: When to use vs when NOT to use")
    print("✅ Pattern 3: Detailed parameter explanations")
    print("✅ Pattern 4: Examples in description")
    print("✅ Pattern 5: Clear limitations and constraints")
    print("✅ Pattern 6: Context-aware descriptions")
    print("\n💡 Great descriptions guide Gemini to perfect tool usage!")


if __name__ == "__main__":
    from datetime import timedelta
    main()
```

### Line-by-Line Explanation

**Antipattern: Poor Description (Lines 46-49)**:
- Minimal docstring
- No parameter explanations
- No return value description
- Gemini has no guidance
- **Don't do this!**

**Pattern 1: Comprehensive (Lines 52-113)**:
- Clear purpose statement
- Detailed parameter descriptions with ranges
- Return value structure explained
- Example usage included
- Notes about limitations
- Comparison to related tools

**Pattern 2: When to Use (Lines 116-190)**:
- **When to use** section with specific triggers
- **When NOT to use** with alternatives
- **Important constraints** highlighted
- **Side effects** called out
- Makes tool selection clear for Gemini

**Pattern 3: Parameter Details (Lines 193-304)**:
- Each parameter gets its own detailed explanation
- Examples for each parameter
- Valid values clearly stated
- Relationships between parameters explained
- Return value structure fully documented

**Pattern 4: Examples (Lines 307-408)**:
- Multiple example formats shown
- Edge cases documented
- Common mistakes addressed
- Expected outputs illustrated
- Helps Gemini understand usage patterns

**Pattern 5: Limitations (Lines 411-497)**:
- **Capabilities** section: what it CAN do
- **Limitations** section: what it CANNOT do
- **Data freshness** information
- **Error conditions** listed
- Sets realistic expectations

**Pattern 6: Context-Aware (Lines 500-597)**:
- **User context matters** section
- Different behaviors for different users
- **Side effects** prominently displayed
- **Security** considerations
- Makes dependencies explicit

### The "Why" Behind the Pattern

**Why detailed descriptions matter:**

Think about these tool calls:

❌ **With Poor Description**:
```python
@agent.tool
def process_data(data: dict) -> dict:
    """Process data."""
    ...

# Gemini has no idea:
# - What kind of data?
# - What processing happens?
# - What comes back?
# Result: Tool rarely used, or used incorrectly
```

✅ **With Great Description**:
```python
@agent.tool
def process_sales_data(data: dict[str, list[float]]) -> dict[str, float]:
    """
    Calculate sales metrics from daily sales data.
    
    Takes daily sales figures and computes:
    - Total revenue
    - Average daily sales
    - Highest sales day
    
    Use this when user asks about sales performance or metrics.
    
    Args:
        data: Dict mapping product IDs to lists of daily sales amounts
              Example: {"prod_1": [100.0, 150.0, 200.0]}
    
    Returns:
        Dict with metrics:
        - total_revenue: Sum of all sales
        - average_daily: Mean sales per day
        - highest_day_amount: Peak single-day sales
    """
    ...

# Gemini knows exactly when and how to use this!
```

**Benefits**:
1. **Correct tool selection**: Gemini picks the right tool
2. **Correct parameters**: Gemini provides valid inputs
3. **Error prevention**: Gemini avoids invalid usage
4. **User confidence**: Users trust the agent's actions
5. **Maintainability**: Future developers understand the tool

---

## C. Test & Apply

### How to Test It

1. **Run the descriptions demo**:
```bash
python lesson_08_tool_descriptions.py
```

2. **Observe how Gemini selects tools based on descriptions**

3. **Write a great description for your own tool**:
```python
@agent.tool
def my_awesome_tool(
    param1: str,
    param2: int = 10
) -> dict[str, any]:
    """
    [Start with a clear one-line summary]
    
    [Follow with a detailed explanation of what it does]
    
    **When to use:**
    - [Specific trigger 1]
    - [Specific trigger 2]
    
    **When NOT to use:**
    - [Alternative tool 1]
    - [Alternative tool 2]
    
    Args:
        param1: [Detailed explanation with examples]
        param2: [Detailed explanation with default reasoning]
    
    Returns:
        [Detailed structure with all fields explained]
    
    Example:
        my_awesome_tool("example input", param2=20)
        # [Expected output]
    
    Limitations:
        - [Important limitation 1]
        - [Important limitation 2]
    """
    # Implementation
    ...
```

### Expected Result

You should see Gemini selecting tools confidently:

```
======================================================================
QUERY 1: How far is it from New York to Boston?
======================================================================

🔧 calculate_distance(40.7128, -74.0060 → 42.3601, -71.0589)

✅ RESULT:
   Action: Calculated geographic distance
   Tool: calculate_distance
   Summary: The straight-line distance from New York to Boston is approximately 306.2 km (190.3 miles)
   Success: True

======================================================================
QUERY 2: Send an email to john@example.com about tomorrow's meeting
======================================================================

🔧 send_email(to=john@example.com, subject='Meeting Reminder')

✅ RESULT:
   Action: Sent email notification
   Tool: send_email
   Summary: Email sent successfully to john@example.com with meeting reminder
   Success: True
```

### Validation Examples

**Description Checklist**:

✅ Clear one-line summary  
✅ Detailed purpose explanation  
✅ "When to use" section  
✅ "When NOT to use" section (with alternatives)  
✅ All parameters explained with examples  
✅ Return value structure documented  
✅ Example usage included  
✅ Limitations clearly stated  
✅ Edge cases addressed  
✅ Error conditions listed  

### Type Checking

Great descriptions don't affect type checking, but they improve reliability:

```bash
mypy lesson_08_tool_descriptions.py
```

Expected: `Success: no issues found`

---

## D. Common Stumbling Blocks

### 1. Vague Descriptions

**The Problem**:
```python
@agent.tool
def process(data: str) -> str:
    """Process the data."""  # ❌ What does "process" mean?
    ...
```

**The Fix**:
```python
@agent.tool
def sanitize_user_input(data: str) -> str:
    """
    Remove potentially harmful characters from user input.
    
    Strips HTML tags, SQL injection patterns, and XSS attempts.
    Use this before storing user-provided text in the database.
    
    Args:
        data: Raw user input string
    
    Returns:
        Sanitized string safe for database storage
    """
    ...
```

### 2. Missing "When NOT to Use"

**The Problem**:
```python
@agent.tool
def search_users(query: str) -> list[dict]:
    """
    Search for users matching the query.
    """  # ❌ When should I use search_products instead?
```

**The Fix**:
```python
@agent.tool
def search_users(query: str) -> list[dict]:
    """
    Search for users matching the query.
    
    **When to use:**
    - User asks about people, accounts, or profiles
    - Need to find user by name, email, or username
    
    **When NOT to use:**
    - Searching for products (use search_products)
    - Searching for documents (use search_documents)
    - Searching for orders (use search_orders)
    """
    ...
```

### 3. Undocumented Limitations

**The Problem**:
```python
@agent.tool
def get_file_contents(path: str) -> str:
    """Get the contents of a file."""  # ❌ What about large files? Binary files?
    ...
```

**The Fix**:
```python
@agent.tool
def get_file_contents(path: str) -> str:
    """
    Get the contents of a text file.
    
    **Limitations:**
    - Only works for text files (not binary)
    - Maximum file size: 1 MB
    - Encoding: UTF-8 only
    - Does NOT support: images, PDFs, executables
    
    For binary files, use read_binary_file instead.
    For large files, use stream_file_contents instead.
    """
    ...
```

### 4. No Examples

**The Problem**:
```python
@agent.tool
def parse_timeframe(spec: str) -> dict:
    """Parse a timeframe specification."""  # ❌ What format is spec?
    ...
```

**The Fix**:
```python
@agent.tool
def parse_timeframe(spec: str) -> dict:
    """
    Parse a timeframe specification into start and end dates.
    
    Supported formats:
    
    Relative:
        "last 7 days" → 7 days ago to now
        "last month" → Previous calendar month
        "this week" → Monday to Sunday of current week
    
    Absolute:
        "2024-01-01 to 2024-01-31" → January 2024
        "Jan 2024" → Entire month of January 2024
    
    Args:
        spec: Timeframe specification string (see examples above)
    
    Returns:
        Dict with:
        - start_date: ISO date string (YYYY-MM-DD)
        - end_date: ISO date string (YYYY-MM-DD)
        - human_readable: Description of the timeframe
    """
    ...
```

### 5. Type Safety Gotcha: Description Doesn't Match Implementation

**The Problem**:
```python
@agent.tool
def calculate_tax(income: float, rate: float) -> float:
    """
    Calculate tax owed.
    
    Returns:
        Tax amount in dollars  # ❌ Says float, but...
    """
    return {
        "tax_amount": income * rate,  # ❌ Returns dict!
        "income": income,
        "rate": rate
    }
```

**The Fix**:
Make description match reality:
```python
@agent.tool
def calculate_tax(income: float, rate: float) -> dict[str, float]:
    """
    Calculate tax owed.
    
    Returns:
        Dictionary with:  # ✅ Matches return type
        - tax_amount: Tax owed in dollars
        - income: Income amount used
        - rate: Tax rate applied
    """
    return {
        "tax_amount": income * rate,
        "income": income,
        "rate": rate
    }
```

---

## Ready for the Next Lesson?

🎉 **Excellent work!** You now know how to write tool descriptions that guide Gemini perfectly:

✅ Clear, comprehensive descriptions  
✅ "When to use" vs "When NOT to use" sections  
✅ Detailed parameter explanations  
✅ Example usage included  
✅ Limitations and constraints clearly stated  
✅ Context-aware documentation  

**Great descriptions are the difference between tools that work and tools that sit unused!** Well-documented tools enable Gemini to make intelligent decisions about when and how to use them.

In the next lesson, we'll explore **Multi-Tool Agents** - you'll learn how to build agents with multiple tools that work together, how Gemini selects between them, and patterns for tool composition!

**Ready for Lesson 9, or would you like to practice writing tool descriptions first?** 🚀
