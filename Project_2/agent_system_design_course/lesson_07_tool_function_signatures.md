# Lesson 7: Tool Function Signatures

## A. Concept Overview

### What & Why
**Tool Function Signatures** define the interface between your tools and the AI model. A well-designed signature tells Gemini exactly what parameters are required, which are optional, what types are expected, and what defaults to use. This is crucial because the signature directly impacts how reliably Gemini can call your tools - poor signatures lead to errors, while great signatures enable flawless tool usage.

### Analogy
Think of a tool signature like a restaurant order form:

**Bad Order Form**:
- "Tell us what you want" (vague, no structure)

**Good Order Form**:
- **Main dish** (required, choose one): Burger | Pizza | Salad
- **Size** (required): Small | Medium | Large
- **Toppings** (optional, multiple): Cheese, Onions, Tomatoes
- **Spice level** (optional, default: Medium): Mild | Medium | Hot
- **Special instructions** (optional): Text field

The form structure guides customers to provide exactly the information the kitchen needs!

### Type Safety Benefit
Well-designed tool signatures provide:
- **Compile-time validation**: mypy catches signature errors before runtime
- **Parameter validation**: Pydantic validates all inputs automatically
- **Default value safety**: Type-checked default values prevent errors
- **Optional parameter clarity**: Explicit about what's required vs optional
- **Return type guarantees**: Callers know exactly what to expect
- **IDE autocomplete**: Full editor support for tool parameters
- **Self-documentation**: Signatures serve as API documentation

Your tools become robust, self-documenting APIs!

---

## B. Code Implementation

### File Structure
```
agent_system_design/
├── lesson_07_tool_signatures.py
├── requirements.txt
└── .env
```

### Complete Code Snippet

**lesson_07_tool_signatures.py**
```python
"""
Lesson 7: Tool Function Signatures
Master the art of designing clear, robust tool signatures
"""

from pydantic import BaseModel, Field, field_validator
from pydantic_ai import Agent, RunContext
from dataclasses import dataclass
from datetime import datetime, date, time
from typing import Literal, Optional, Union
from enum import Enum
from dotenv import load_dotenv

load_dotenv()


# Simple dependencies
@dataclass
class SimpleDeps:
    """Minimal dependencies for demonstration"""
    user_id: str


# Result model
class ToolUsageReport(BaseModel):
    """Report on tool usage"""
    query: str
    tools_used: list[str]
    result_summary: str
    parameter_examples: dict[str, any]


# Create agent
agent = Agent(
    model='gemini-1.5-flash',
    result_type=ToolUsageReport,
    deps_type=SimpleDeps,
    system_prompt="""
You are a demonstration agent showcasing various tool signature patterns.

Use the available tools to answer user queries, demonstrating different
parameter types, defaults, and validation patterns.

In your response, explain which tools you used and show example parameters.
""",
)


# Pattern 1: Required Parameters Only (Simplest)

@agent.tool
def calculate_area(length: float, width: float) -> dict[str, float]:
    """
    Calculate the area of a rectangle.
    
    Both parameters are required - the tool cannot function without them.
    
    Args:
        length: Length of the rectangle in meters
        width: Width of the rectangle in meters
    
    Returns:
        Dictionary with area and perimeter
    """
    print(f"\n🔧 calculate_area(length={length}, width={width})")
    
    area = length * width
    perimeter = 2 * (length + width)
    
    return {
        "area": area,
        "perimeter": perimeter,
        "unit": "square meters"
    }


# Pattern 2: Optional Parameters with Defaults

@agent.tool
def format_currency(
    amount: float,
    currency: str = "USD",
    show_symbol: bool = True,
    decimal_places: int = 2
) -> dict[str, any]:
    """
    Format a number as currency.
    
    Demonstrates optional parameters with sensible defaults.
    
    Args:
        amount: The monetary amount (required)
        currency: Currency code (optional, default: "USD")
        show_symbol: Whether to show currency symbol (optional, default: True)
        decimal_places: Number of decimal places (optional, default: 2)
    
    Returns:
        Formatted currency string and metadata
    """
    print(f"\n🔧 format_currency(amount={amount}, currency={currency}, show_symbol={show_symbol})")
    
    # Currency symbols
    symbols = {
        "USD": "$",
        "EUR": "€",
        "GBP": "£",
        "JPY": "¥"
    }
    
    # Format number
    formatted_amount = f"{amount:,.{decimal_places}f}"
    
    # Add symbol if requested
    if show_symbol:
        symbol = symbols.get(currency, currency + " ")
        formatted = f"{symbol}{formatted_amount}"
    else:
        formatted = formatted_amount
    
    return {
        "formatted": formatted,
        "amount": amount,
        "currency": currency,
        "decimal_places": decimal_places
    }


# Pattern 3: Literal Types (Constrained Choices)

@agent.tool
def send_notification(
    message: str,
    priority: Literal["low", "medium", "high", "urgent"],
    channel: Literal["email", "sms", "push", "slack"] = "email"
) -> dict[str, any]:
    """
    Send a notification through specified channel.
    
    Uses Literal types to constrain inputs to valid choices.
    
    Args:
        message: Notification message (required)
        priority: Priority level (required, must be: low, medium, high, or urgent)
        channel: Delivery channel (optional, default: email)
    
    Returns:
        Notification status and delivery details
    """
    print(f"\n🔧 send_notification(priority={priority}, channel={channel})")
    print(f"   Message: {message[:50]}...")
    
    return {
        "status": "sent",
        "message_id": f"msg_{datetime.now().timestamp()}",
        "priority": priority,
        "channel": channel,
        "estimated_delivery": "within 5 minutes"
    }


# Pattern 4: Optional Parameters with None Default

@agent.tool
def search_database(
    query: str,
    limit: int = 10,
    offset: int = 0,
    sort_by: Optional[str] = None,
    filter_category: Optional[str] = None
) -> dict[str, any]:
    """
    Search database with optional filtering and sorting.
    
    Demonstrates None defaults for truly optional parameters.
    
    Args:
        query: Search query string (required)
        limit: Maximum results to return (optional, default: 10)
        offset: Number of results to skip (optional, default: 0)
        sort_by: Field to sort by (optional, default: None = relevance)
        filter_category: Category filter (optional, default: None = all categories)
    
    Returns:
        Search results with metadata
    """
    print(f"\n🔧 search_database(query='{query}', limit={limit}, offset={offset})")
    if sort_by:
        print(f"   Sorting by: {sort_by}")
    if filter_category:
        print(f"   Filtering category: {filter_category}")
    
    # Simulate search results
    results = [
        {"id": i, "title": f"Result {i} for {query}", "score": 0.9 - (i * 0.1)}
        for i in range(offset, offset + min(limit, 5))
    ]
    
    return {
        "query": query,
        "results": results,
        "total_found": 42,
        "limit": limit,
        "offset": offset,
        "sort_by": sort_by or "relevance",
        "filter_category": filter_category or "all"
    }


# Pattern 5: Union Types (Multiple Acceptable Types)

@agent.tool
def schedule_event(
    title: str,
    start_time: Union[datetime, str],
    duration_minutes: int = 60,
    attendees: Optional[list[str]] = None
) -> dict[str, any]:
    """
    Schedule an event with flexible time input.
    
    Demonstrates Union types for parameters that accept multiple types.
    
    Args:
        title: Event title (required)
        start_time: Start time as datetime object or ISO string (required)
        duration_minutes: Event duration (optional, default: 60)
        attendees: List of attendee emails (optional, default: None)
    
    Returns:
        Event details and scheduling confirmation
    """
    print(f"\n🔧 schedule_event(title='{title}', duration={duration_minutes}min)")
    
    # Handle Union type - convert string to datetime if needed
    if isinstance(start_time, str):
        try:
            start_time = datetime.fromisoformat(start_time)
        except ValueError:
            return {"error": "Invalid datetime format. Use ISO format: YYYY-MM-DDTHH:MM:SS"}
    
    end_time = start_time + timedelta(minutes=duration_minutes)
    
    return {
        "event_id": f"evt_{datetime.now().timestamp()}",
        "title": title,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_minutes": duration_minutes,
        "attendees": attendees or [],
        "status": "scheduled"
    }


# Pattern 6: Complex Nested Parameters (Use with Caution)

@agent.tool
def generate_report(
    report_type: Literal["sales", "analytics", "financial"],
    date_range: dict[str, str],
    options: Optional[dict[str, any]] = None
) -> dict[str, any]:
    """
    Generate a report with complex configuration.
    
    Demonstrates nested dict parameters (use sparingly).
    
    Args:
        report_type: Type of report to generate (required)
        date_range: Date range as dict with 'start' and 'end' keys (required)
        options: Additional report options as dict (optional)
    
    Returns:
        Generated report metadata
    
    Example date_range:
        {"start": "2024-01-01", "end": "2024-01-31"}
    
    Example options:
        {"include_charts": true, "format": "pdf", "email_recipients": ["user@example.com"]}
    """
    print(f"\n🔧 generate_report(type={report_type})")
    print(f"   Date range: {date_range.get('start')} to {date_range.get('end')}")
    
    if options:
        print(f"   Options: {options}")
    
    return {
        "report_id": f"rpt_{datetime.now().timestamp()}",
        "report_type": report_type,
        "date_range": date_range,
        "options": options or {},
        "status": "generated",
        "download_url": f"https://example.com/reports/rpt_{report_type}.pdf"
    }


# Pattern 7: Using Pydantic Models for Complex Parameters (Best Practice)

class ReportConfig(BaseModel):
    """Configuration for report generation"""
    report_type: Literal["sales", "analytics", "financial"]
    start_date: date
    end_date: date
    include_charts: bool = True
    format: Literal["pdf", "excel", "html"] = "pdf"
    email_recipients: list[str] = Field(default_factory=list)
    
    @field_validator("end_date")
    @classmethod
    def validate_date_range(cls, v: date, info) -> date:
        """Ensure end_date is after start_date"""
        if hasattr(info, 'data') and 'start_date' in info.data:
            if v < info.data['start_date']:
                raise ValueError("end_date must be after start_date")
        return v


@agent.tool
def generate_report_typed(config: ReportConfig) -> dict[str, any]:
    """
    Generate a report with type-safe configuration.
    
    This is the BEST PRACTICE - use Pydantic models for complex parameters.
    Provides automatic validation, better error messages, and type safety.
    
    Args:
        config: Report configuration (type-safe Pydantic model)
    
    Returns:
        Generated report metadata
    """
    print(f"\n🔧 generate_report_typed(type={config.report_type})")
    print(f"   Date range: {config.start_date} to {config.end_date}")
    print(f"   Format: {config.format}")
    
    return {
        "report_id": f"rpt_{datetime.now().timestamp()}",
        "report_type": config.report_type,
        "date_range": {
            "start": config.start_date.isoformat(),
            "end": config.end_date.isoformat()
        },
        "format": config.format,
        "include_charts": config.include_charts,
        "email_recipients": config.email_recipients,
        "status": "generated",
        "download_url": f"https://example.com/reports/{config.report_type}.{config.format}"
    }


# Pattern 8: Variable-Length Arguments (Advanced, Use Sparingly)

@agent.tool
def calculate_statistics(*values: float, include_mode: bool = False) -> dict[str, float]:
    """
    Calculate statistics for a list of values.
    
    Demonstrates *args for variable-length parameters.
    
    Args:
        *values: Variable number of numeric values (at least 1 required)
        include_mode: Whether to calculate mode (optional, default: False)
    
    Returns:
        Statistical measures
    """
    print(f"\n🔧 calculate_statistics(values={values})")
    
    if not values:
        return {"error": "At least one value required"}
    
    # Calculate statistics
    mean = sum(values) / len(values)
    sorted_values = sorted(values)
    
    # Median
    n = len(sorted_values)
    if n % 2 == 0:
        median = (sorted_values[n//2 - 1] + sorted_values[n//2]) / 2
    else:
        median = sorted_values[n//2]
    
    result = {
        "count": len(values),
        "mean": round(mean, 2),
        "median": round(median, 2),
        "min": min(values),
        "max": max(values),
        "range": max(values) - min(values)
    }
    
    # Mode (if requested)
    if include_mode:
        from collections import Counter
        counts = Counter(values)
        mode_value, mode_count = counts.most_common(1)[0]
        result["mode"] = mode_value
        result["mode_frequency"] = mode_count
    
    return result


# Demo

def main():
    print("\n" + "="*70)
    print("TOOL FUNCTION SIGNATURES DEMONSTRATION")
    print("="*70)
    
    deps = SimpleDeps(user_id="demo_user")
    
    queries = [
        "Calculate the area of a 5 by 3 meter rectangle",
        "Format $1234.56 in EUR with the currency symbol",
        "Send a high priority notification via SMS: 'Server is down!'",
        "Search for 'Python tutorials' and sort by date",
        "Schedule a 90-minute meeting called 'Team Sync' for tomorrow at 2pm",
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n\n{'='*70}")
        print(f"QUERY {i}: {query}")
        print(f"{'='*70}")
        
        try:
            result = agent.run_sync(query, deps=deps)
            report = result.data
            
            print(f"\n✅ RESULT:")
            print(f"   Tools Used: {', '.join(report.tools_used)}")
            print(f"   Summary: {report.result_summary}")
            if report.parameter_examples:
                print(f"   Example Parameters: {report.parameter_examples}")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
    
    print("\n" + "="*70)
    print("SIGNATURE PATTERNS SUMMARY")
    print("="*70)
    print("\n✅ Pattern 1: Required only - simplest, when all params needed")
    print("✅ Pattern 2: Optional with defaults - common, user-friendly")
    print("✅ Pattern 3: Literal types - constrain to valid choices")
    print("✅ Pattern 4: Optional[T] = None - truly optional parameters")
    print("✅ Pattern 5: Union types - accept multiple types")
    print("✅ Pattern 6: Dict params - use sparingly, hard to validate")
    print("✅ Pattern 7: Pydantic models - BEST for complex params")
    print("✅ Pattern 8: *args - variable length, use rarely")


if __name__ == "__main__":
    from datetime import timedelta
    main()
```

### Line-by-Line Explanation

**Pattern 1: Required Only (Lines 49-73)**:
- Simplest pattern: all parameters required
- No defaults, no optionals
- Use when tool cannot function without all inputs
- Clear and explicit

**Pattern 2: Optional with Defaults (Lines 76-118)**:
- Most common pattern
- Sensible defaults make tool easy to use
- Required parameters first, optional after
- Defaults should be the most common use case

**Pattern 3: Literal Types (Lines 121-155)**:
- Constrains input to specific values
- Gemini knows valid choices
- Type-safe: invalid values rejected at runtime
- Better than string validation in tool body

**Pattern 4: Optional[T] = None (Lines 158-200)**:
- Use when parameter is truly optional
- `None` signals "not provided"
- Tool must handle `None` case explicitly
- Different from default value pattern

**Pattern 5: Union Types (Lines 203-247)**:
- Accept multiple types for same parameter
- Tool must handle each type
- Use sparingly - can complicate tool logic
- Check type with `isinstance()`

**Pattern 6: Dict Parameters (Lines 250-292)**:
- Flexible but hard to validate
- No IDE autocomplete
- Prone to errors
- Use only when structure is truly dynamic

**Pattern 7: Pydantic Models (Lines 295-348)**:
- **BEST PRACTICE** for complex parameters
- Full validation automatic
- Type-safe and self-documenting
- Can include field validators
- IDE autocomplete works perfectly

**Pattern 8: *args (Lines 351-399)**:
- Variable-length arguments
- Use rarely - harder for agent to use correctly
- Good for mathematical functions
- Document minimum number of args

### The "Why" Behind the Pattern

**Why signature design matters:**

❌ **Poor Signature** (Unreliable):
```python
@agent.tool
def do_stuff(params: dict) -> any:  # What goes in params? What comes out?
    ...
```

✅ **Great Signature** (Reliable):
```python
@agent.tool
def calculate_discount(
    price: float,
    discount_percent: float,
    tax_rate: float = 0.08
) -> dict[str, float]:
    """Calculate final price with discount and tax."""
    ...
```

**Signature Design Principles**:
1. **Required first, optional last**
2. **Use Literal for constrained choices**
3. **Pydantic models for complex params**
4. **Avoid `any` and `dict` types**
5. **Clear parameter names**
6. **Comprehensive docstrings**

---

## C. Test & Apply

### How to Test It

1. **Run the signature demo**:
```bash
python lesson_07_tool_signatures.py
```

2. **Observe how different signatures behave**

3. **Design your own tool signature**:
```python
@agent.tool
def book_flight(
    origin: str,
    destination: str,
    departure_date: date,
    return_date: Optional[date] = None,  # One-way if None
    cabin_class: Literal["economy", "business", "first"] = "economy",
    passengers: int = 1
) -> dict[str, any]:
    """
    Book a flight with specified parameters.
    
    Args:
        origin: Departure airport code (required)
        destination: Arrival airport code (required)
        departure_date: Departure date (required)
        return_date: Return date for round-trip (optional, None = one-way)
        cabin_class: Seating class (optional, default: economy)
        passengers: Number of passengers (optional, default: 1)
    
    Returns:
        Booking confirmation details
    """
    # Implementation...
```

### Expected Result

You should see tools being called with correct parameter types:

```
======================================================================
QUERY 1: Calculate the area of a 5 by 3 meter rectangle
======================================================================

🔧 calculate_area(length=5.0, width=3.0)

✅ RESULT:
   Tools Used: calculate_area
   Summary: The area is 15.0 square meters with a perimeter of 16.0 meters
   Example Parameters: {'length': 5.0, 'width': 3.0}

======================================================================
QUERY 2: Format $1234.56 in EUR with the currency symbol
======================================================================

🔧 format_currency(amount=1234.56, currency=EUR, show_symbol=True)

✅ RESULT:
   Tools Used: format_currency
   Summary: €1,234.56
   Example Parameters: {'amount': 1234.56, 'currency': 'EUR', 'show_symbol': True}
```

### Validation Examples

**Good Signature Checklist**:

```python
@agent.tool
def my_tool(
    # ✅ Required parameters first
    required_param: str,
    
    # ✅ Optional parameters after
    optional_with_default: int = 10,
    
    # ✅ Literal for choices
    choice: Literal["a", "b", "c"] = "a",
    
    # ✅ Optional[T] for truly optional
    maybe_value: Optional[float] = None
    
# ✅ Clear return type
) -> dict[str, any]:
    """
    ✅ Comprehensive docstring explaining:
    - What the tool does
    - Each parameter's purpose
    - Return value structure
    - Example usage if complex
    """
    pass
```

### Type Checking

```bash
mypy lesson_07_tool_signatures.py
```

Expected: `Success: no issues found`

Try adding an error:
```python
@agent.tool
def bad_tool(x: int) -> str:
    return 123  # ❌ mypy error: expected str, got int
```

---

## D. Common Stumbling Blocks

### 1. Optional Parameters Before Required

**The Error**:
```python
@agent.tool
def bad_signature(
    optional: str = "default",  # ❌ Optional first
    required: int  # Required after optional - syntax error!
) -> dict:
    ...
```

**The Fix**:
Required parameters always come first:
```python
@agent.tool
def good_signature(
    required: int,  # ✅ Required first
    optional: str = "default"  # ✅ Optional after
) -> dict:
    ...
```

### 2. Mutable Default Arguments

**The Problem**:
```python
@agent.tool
def append_item(items: list[str] = []) -> list[str]:  # ❌ Mutable default
    items.append("new")
    return items

# Weird behavior:
result1 = append_item()  # ["new"]
result2 = append_item()  # ["new", "new"] - same list!
```

**The Fix**:
```python
@agent.tool
def append_item(items: Optional[list[str]] = None) -> list[str]:  # ✅
    if items is None:
        items = []
    items.append("new")
    return items
```

### 3. Vague Parameter Types

**The Problem**:
```python
@agent.tool
def process_data(data: any) -> any:  # ❌ No type information
    ...
```

**What's Wrong**:
- No validation
- No IDE support
- Agent doesn't know what to pass
- Runtime errors likely

**The Fix**:
Be specific:
```python
@agent.tool
def process_data(data: dict[str, list[float]]) -> dict[str, float]:  # ✅
    """Process data dictionary mapping categories to value lists."""
    ...
```

### 4. Missing Return Type

**The Problem**:
```python
@agent.tool
def calculate(x: float, y: float):  # ❌ No return type
    return x + y  # What type is returned?
```

**The Fix**:
Always specify return type:
```python
@agent.tool
def calculate(x: float, y: float) -> float:  # ✅
    """Add two numbers."""
    return x + y
```

### 5. Type Safety Gotcha: Dict vs Pydantic Model

**Antipattern**:
```python
@agent.tool
def create_user(user_data: dict) -> dict:  # ❌ No validation
    # What keys should user_data have?
    # What are their types?
    ...
```

**Best Practice**:
```python
class UserData(BaseModel):
    name: str
    email: str
    age: int = Field(ge=0, le=150)

@agent.tool
def create_user(user_data: UserData) -> dict:  # ✅ Validated!
    # All fields guaranteed to exist and have correct types
    ...
```

---

## Ready for the Next Lesson?

🎉 **Excellent work!** You now understand how to design robust tool signatures:

✅ Required vs optional parameters  
✅ Default values and their trade-offs  
✅ Literal types for constrained choices  
✅ Optional[T] for truly optional params  
✅ Union types for multiple acceptable types  
✅ Pydantic models for complex parameters (best practice)  
✅ Proper ordering and type hints  

**Well-designed signatures are the foundation of reliable tool usage!** Great signatures make it nearly impossible for Gemini to call your tools incorrectly.

In the next lesson, we'll explore **Tool Descriptions for Gemini** - you'll learn how to write docstrings and descriptions that help Gemini understand when and how to use your tools effectively!

**Ready for Lesson 8, or would you like to practice designing tool signatures first?** 🚀
