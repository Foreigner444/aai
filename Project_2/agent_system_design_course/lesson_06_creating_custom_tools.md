# Lesson 6: Creating Custom Tools

## A. Concept Overview

### What & Why
**Custom Tools** are functions that you register with your agent to extend its capabilities beyond pure text generation. Tools allow agents to perform actions like database queries, API calls, calculations, file operations, or any other programmatic task. This is crucial because it transforms agents from text generators into **intelligent automation systems** that can interact with the real world.

### Analogy
Think of an agent as a highly intelligent intern:
- **Without tools**: They can only give you advice and written reports (text generation)
- **With tools**: They can actually DO things:
  - `search_database` = Look up information in your filing cabinet
  - `send_email` = Draft and send emails on your behalf
  - `calculate_metrics` = Crunch numbers in Excel
  - `check_calendar` = View your schedule
  - `create_ticket` = File support tickets

Tools transform agents from "advisors" to "executors"!

### Type Safety Benefit
Custom tools provide exceptional type safety:
- **Parameter validation**: Pydantic validates all tool inputs automatically
- **Return type checking**: mypy verifies tool return types
- **Description enforcement**: Tools must document their purpose
- **Signature validation**: Function signatures are type-checked at compile time
- **IDE support**: Autocomplete for tool parameters and return values
- **Runtime safety**: Invalid tool calls are caught before execution

Every tool becomes a type-safe, self-documenting function!

---

## B. Code Implementation

### File Structure
```
agent_system_design/
├── lesson_06_custom_tools.py
├── requirements.txt
└── .env
```

### Complete Code Snippet

**lesson_06_custom_tools.py**
```python
"""
Lesson 6: Creating Custom Tools
Learn to build powerful, type-safe tools for your agents
"""

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal, Optional
import math
import json
from dotenv import load_dotenv

load_dotenv()


# Dependencies for our tools
@dataclass
class ToolDependencies:
    """Dependencies that tools can access"""
    user_id: str
    database: dict = None  # Simulated database
    
    def __post_init__(self):
        if self.database is None:
            self.database = {
                "users": {
                    "user_1": {"name": "Alice", "balance": 1500.00},
                    "user_2": {"name": "Bob", "balance": 3200.00},
                },
                "transactions": [
                    {"user": "user_1", "amount": -45.00, "date": "2024-01-15", "category": "food"},
                    {"user": "user_1", "amount": 1500.00, "date": "2024-01-01", "category": "salary"},
                ]
            }


# Result model
class FinancialAnalysis(BaseModel):
    """Financial analysis result"""
    summary: str = Field(description="Summary of the analysis")
    key_metrics: dict[str, float] = Field(description="Important metrics")
    recommendations: list[str] = Field(description="Actionable recommendations")
    data_sources: list[str] = Field(description="Tools/data sources used")


# Create agent
agent = Agent(
    model='gemini-1.5-flash',
    result_type=FinancialAnalysis,
    deps_type=ToolDependencies,
    system_prompt="""
You are a financial analysis assistant with access to powerful tools.

Available tools:
- calculate_budget: Perform financial calculations
- get_account_balance: Retrieve user's account balance
- get_transaction_history: Fetch recent transactions
- categorize_spending: Analyze spending by category
- predict_savings: Project future savings

Use tools strategically to provide accurate, data-driven analysis.
Always cite which tools you used in the data_sources field.
""",
)


# Tool Pattern 1: Simple Calculation Tool (No Dependencies)

@agent.tool
def calculate_budget(
    monthly_income: float,
    fixed_expenses: float,
    savings_rate: float
) -> dict[str, float]:
    """
    Calculate budget breakdown based on income and expenses.
    
    Args:
        monthly_income: Total monthly income in dollars
        fixed_expenses: Total fixed monthly expenses in dollars
        savings_rate: Desired savings rate as decimal (e.g., 0.2 for 20%)
    
    Returns:
        Dictionary with budget breakdown including savings, discretionary, etc.
    """
    print(f"\n🔧 Tool: calculate_budget")
    print(f"   Income: ${monthly_income:,.2f}")
    print(f"   Fixed Expenses: ${fixed_expenses:,.2f}")
    print(f"   Savings Rate: {savings_rate:.1%}")
    
    # Calculate budget components
    savings_amount = monthly_income * savings_rate
    after_expenses = monthly_income - fixed_expenses
    after_savings = after_expenses - savings_amount
    
    budget = {
        "monthly_income": monthly_income,
        "fixed_expenses": fixed_expenses,
        "savings_amount": savings_amount,
        "discretionary_spending": after_savings,
        "savings_rate_actual": savings_amount / monthly_income if monthly_income > 0 else 0,
    }
    
    print(f"   💰 Discretionary: ${budget['discretionary_spending']:,.2f}")
    print(f"   🏦 Savings: ${savings_amount:,.2f}")
    
    return budget


# Tool Pattern 2: Tool with Dependencies (Database Access)

@agent.tool
def get_account_balance(ctx: RunContext[ToolDependencies]) -> dict[str, any]:
    """
    Get the current account balance for the user.
    
    This tool uses dependencies to access user-specific data.
    
    Returns:
        Dictionary with balance information and account status
    """
    print(f"\n🔧 Tool: get_account_balance")
    
    user_id = ctx.deps.user_id
    database = ctx.deps.database
    
    user_data = database["users"].get(user_id)
    
    if not user_data:
        return {"error": f"User {user_id} not found"}
    
    balance = user_data["balance"]
    name = user_data["name"]
    
    print(f"   User: {name}")
    print(f"   Balance: ${balance:,.2f}")
    
    # Determine account status
    status = "healthy" if balance > 1000 else "low" if balance > 100 else "critical"
    
    return {
        "name": name,
        "balance": balance,
        "status": status,
        "currency": "USD"
    }


# Tool Pattern 3: Tool with Complex Return Type

@agent.tool
def get_transaction_history(
    ctx: RunContext[ToolDependencies],
    days: int = 30,
    category: Optional[str] = None
) -> dict[str, any]:
    """
    Retrieve recent transaction history for the user.
    
    Args:
        days: Number of days to look back (default: 30)
        category: Optional category filter (e.g., "food", "salary")
    
    Returns:
        Dictionary with transactions list and summary statistics
    """
    print(f"\n🔧 Tool: get_transaction_history")
    print(f"   Days: {days}")
    if category:
        print(f"   Category filter: {category}")
    
    user_id = ctx.deps.user_id
    database = ctx.deps.database
    
    # Get all user transactions
    all_transactions = [
        t for t in database["transactions"]
        if t["user"] == user_id
    ]
    
    # Filter by category if specified
    if category:
        all_transactions = [
            t for t in all_transactions
            if t["category"] == category
        ]
    
    # Calculate statistics
    total_spent = sum(t["amount"] for t in all_transactions if t["amount"] < 0)
    total_earned = sum(t["amount"] for t in all_transactions if t["amount"] > 0)
    net_change = total_earned + total_spent
    
    print(f"   Transactions found: {len(all_transactions)}")
    print(f"   Net change: ${net_change:,.2f}")
    
    return {
        "transactions": all_transactions,
        "count": len(all_transactions),
        "summary": {
            "total_spent": abs(total_spent),
            "total_earned": total_earned,
            "net_change": net_change
        },
        "period_days": days,
        "category_filter": category
    }


# Tool Pattern 4: Analytical Tool with Multiple Inputs

@agent.tool
def categorize_spending(
    ctx: RunContext[ToolDependencies],
    days: int = 30
) -> dict[str, any]:
    """
    Analyze spending broken down by category.
    
    Args:
        days: Number of days to analyze
    
    Returns:
        Spending breakdown by category with percentages
    """
    print(f"\n🔧 Tool: categorize_spending")
    print(f"   Analysis period: {days} days")
    
    user_id = ctx.deps.user_id
    database = ctx.deps.database
    
    # Get user's spending transactions (negative amounts)
    transactions = [
        t for t in database["transactions"]
        if t["user"] == user_id and t["amount"] < 0
    ]
    
    # Group by category
    by_category = {}
    total_spent = 0
    
    for t in transactions:
        category = t["category"]
        amount = abs(t["amount"])
        by_category[category] = by_category.get(category, 0) + amount
        total_spent += amount
    
    # Calculate percentages
    category_analysis = {
        cat: {
            "amount": amount,
            "percentage": (amount / total_spent * 100) if total_spent > 0 else 0
        }
        for cat, amount in by_category.items()
    }
    
    print(f"   Total spent: ${total_spent:,.2f}")
    print(f"   Categories: {', '.join(by_category.keys())}")
    
    return {
        "categories": category_analysis,
        "total_spent": total_spent,
        "period_days": days,
        "largest_category": max(by_category, key=by_category.get) if by_category else None
    }


# Tool Pattern 5: Predictive Tool

@agent.tool
def predict_savings(
    current_balance: float,
    monthly_income: float,
    monthly_expenses: float,
    months: int = 12
) -> dict[str, any]:
    """
    Project savings growth over time.
    
    Args:
        current_balance: Starting balance
        monthly_income: Expected monthly income
        monthly_expenses: Expected monthly expenses
        months: Number of months to project (default: 12)
    
    Returns:
        Projection with monthly breakdown and final balance
    """
    print(f"\n🔧 Tool: predict_savings")
    print(f"   Starting balance: ${current_balance:,.2f}")
    print(f"   Projection period: {months} months")
    
    monthly_savings = monthly_income - monthly_expenses
    
    # Project month by month
    projections = []
    balance = current_balance
    
    for month in range(1, months + 1):
        balance += monthly_savings
        projections.append({
            "month": month,
            "balance": round(balance, 2),
            "cumulative_savings": round(monthly_savings * month, 2)
        })
    
    final_balance = balance
    total_saved = monthly_savings * months
    
    print(f"   Final projected balance: ${final_balance:,.2f}")
    print(f"   Total savings: ${total_saved:,.2f}")
    
    return {
        "starting_balance": current_balance,
        "monthly_savings": monthly_savings,
        "final_balance": final_balance,
        "total_saved": total_saved,
        "months": months,
        "projections": projections[:6],  # Include first 6 months detail
        "savings_rate": (monthly_savings / monthly_income) if monthly_income > 0 else 0
    }


# Tool Pattern 6: Tool with Literal Type (Constrained Choices)

@agent.tool
def get_financial_advice(
    topic: Literal["budgeting", "investing", "debt", "emergency_fund", "retirement"]
) -> dict[str, any]:
    """
    Get expert financial advice on a specific topic.
    
    Args:
        topic: The financial topic to get advice on.
               Must be one of: budgeting, investing, debt, emergency_fund, retirement
    
    Returns:
        Structured advice with key principles and action steps
    """
    print(f"\n🔧 Tool: get_financial_advice")
    print(f"   Topic: {topic}")
    
    advice_database = {
        "budgeting": {
            "principle": "Track every dollar and allocate intentionally",
            "steps": [
                "Calculate total monthly income",
                "List all fixed expenses",
                "Set savings goals (aim for 20% of income)",
                "Allocate remaining for discretionary spending"
            ],
            "rule_of_thumb": "50/30/20 rule: 50% needs, 30% wants, 20% savings"
        },
        "investing": {
            "principle": "Start early, diversify, think long-term",
            "steps": [
                "Maximize employer 401(k) match",
                "Open a Roth IRA",
                "Invest in low-cost index funds",
                "Rebalance annually"
            ],
            "rule_of_thumb": "Invest 10-15% of income, diversify across asset classes"
        },
        "debt": {
            "principle": "High-interest debt is an emergency",
            "steps": [
                "List all debts with interest rates",
                "Pay minimums on all debts",
                "Attack highest interest debt aggressively",
                "Consider debt consolidation if rates are high"
            ],
            "rule_of_thumb": "Avalanche method: highest interest first"
        },
        "emergency_fund": {
            "principle": "Financial security starts with cash reserves",
            "steps": [
                "Start with $1,000 mini emergency fund",
                "Build to 3-6 months of expenses",
                "Keep in high-yield savings account",
                "Don't touch except for true emergencies"
            ],
            "rule_of_thumb": "6 months expenses for dual income, 9-12 for single income"
        },
        "retirement": {
            "principle": "Time is your greatest asset",
            "steps": [
                "Start contributing to retirement accounts now",
                "Take full advantage of employer match",
                "Increase contributions with raises",
                "Estimate needs: 25x annual expenses"
            ],
            "rule_of_thumb": "Save 15% of income; retire when you have 25x annual expenses"
        }
    }
    
    advice = advice_database[topic]
    
    return {
        "topic": topic,
        "core_principle": advice["principle"],
        "action_steps": advice["steps"],
        "rule_of_thumb": advice["rule_of_thumb"]
    }


# Demonstration

def main():
    print("\n" + "="*70)
    print("CUSTOM TOOLS DEMONSTRATION")
    print("="*70)
    
    # Create dependencies
    deps = ToolDependencies(user_id="user_1")
    
    # Test queries that will trigger different tools
    queries = [
        "What's my current financial situation?",
        "I make $5000/month and spend $3000. How should I budget?",
        "Analyze my spending patterns",
        "If I save $500/month, what will I have in a year?",
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n\n{'='*70}")
        print(f"QUERY {i}: {query}")
        print(f"{'='*70}")
        
        try:
            result = agent.run_sync(query, deps=deps)
            analysis = result.data
            
            print(f"\n✅ ANALYSIS:")
            print(f"   {analysis.summary}")
            print(f"\n📊 KEY METRICS:")
            for metric, value in analysis.key_metrics.items():
                print(f"      {metric}: {value}")
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in analysis.recommendations:
                print(f"      • {rec}")
            print(f"\n📚 Data Sources: {', '.join(analysis.data_sources)}")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
    
    print("\n" + "="*70)
    print("TOOLS DEMONSTRATION COMPLETE")
    print("="*70)
    print("\n Key Takeaways:")
    print(" ✅ Tools extend agent capabilities beyond text generation")
    print(" ✅ Type hints ensure tool parameters and returns are validated")
    print(" ✅ Tools can access dependencies via RunContext")
    print(" ✅ Gemini intelligently selects appropriate tools")
    print(" ✅ Tools are composable - agents can use multiple tools per query")


if __name__ == "__main__":
    main()
```

### Line-by-Line Explanation

**Tool Pattern 1: Simple Tool (Lines 62-93)**:
- `@agent.tool`: Registers function as agent tool
- No `ctx` parameter: Doesn't need dependencies
- Type hints on parameters: Pydantic validates inputs
- Docstring: Describes tool for both Gemini and developers
- Return type hint: `dict[str, float]` is validated

**Tool Pattern 2: Tool with Dependencies (Lines 96-129)**:
- `ctx: RunContext[ToolDependencies]`: Access to dependencies
- `ctx.deps.user_id`: Type-safe dependency access
- Returns error dict if user not found
- Uses dependencies to fetch user-specific data

**Tool Pattern 3: Complex Return (Lines 132-186)**:
- Multiple parameters with defaults
- Optional parameter: `Optional[str]`
- Complex return type with nested data
- Performs filtering and aggregation
- Returns structured statistics

**Tool Pattern 4: Analytical Tool (Lines 189-240)**:
- Performs analysis and calculations
- Groups data by category
- Calculates percentages and totals
- Returns insights, not just raw data

**Tool Pattern 5: Predictive Tool (Lines 243-298)**:
- Takes multiple inputs
- Performs projections over time
- Returns structured forecast
- Includes intermediate values

**Tool Pattern 6: Literal Type Tool (Lines 301-369)**:
- `Literal[...]`: Constrains input to specific values
- Gemini knows the valid options
- Returns advice from knowledge base
- Type-safe: can't pass invalid topic

### The "Why" Behind the Pattern

**Why use tools instead of just prompting?**

❌ **Without Tools** (Unreliable):
```
User: What's my account balance?
Agent: I don't have access to your account information, but typically...
```

✅ **With Tools** (Accurate):
```
User: What's my account balance?
Agent: *calls get_account_balance tool*
Agent: Your current balance is $1,500.00, and your account status is healthy.
```

**Benefits of Tools**:
1. **Accuracy**: Real data, not hallucinations
2. **Actions**: Can actually DO things
3. **Integration**: Connect to databases, APIs, services
4. **Composability**: Chain multiple tools
5. **Type Safety**: All inputs/outputs validated
6. **Auditability**: Log tool usage

---

## C. Test & Apply

### How to Test It

1. **Run the tools demo**:
```bash
python lesson_06_custom_tools.py
```

2. **Observe which tools Gemini selects for each query**

3. **Create your own tool**:
```python
@agent.tool
def convert_currency(
    amount: float,
    from_currency: str,
    to_currency: str
) -> dict[str, any]:
    """
    Convert amount from one currency to another.
    
    Args:
        amount: Amount to convert
        from_currency: Source currency code (e.g., "USD")
        to_currency: Target currency code (e.g., "EUR")
    
    Returns:
        Conversion result with rate and converted amount
    """
    # Simulated exchange rates
    rates = {"USD": 1.0, "EUR": 0.85, "GBP": 0.73, "JPY": 110.0}
    
    from_rate = rates.get(from_currency, 1.0)
    to_rate = rates.get(to_currency, 1.0)
    
    converted = amount * (to_rate / from_rate)
    
    return {
        "original_amount": amount,
        "original_currency": from_currency,
        "converted_amount": round(converted, 2),
        "converted_currency": to_currency,
        "exchange_rate": to_rate / from_rate
    }
```

### Expected Result

You should see Gemini intelligently selecting and using tools:

```
======================================================================
QUERY 1: What's my current financial situation?
======================================================================

🔧 Tool: get_account_balance
   User: Alice
   Balance: $1,500.00

🔧 Tool: get_transaction_history
   Days: 30
   Transactions found: 2
   Net change: $1,455.00

🔧 Tool: categorize_spending
   Analysis period: 30 days
   Total spent: $45.00
   Categories: food

✅ ANALYSIS:
   Alice, your current financial situation is healthy. You have a balance of $1,500.00...

📊 KEY METRICS:
      current_balance: 1500.0
      monthly_spending: 45.0
      net_worth_change: 1455.0

💡 RECOMMENDATIONS:
      • Continue maintaining your healthy balance above $1,000
      • Consider increasing your savings rate given your positive cash flow
      • Review spending in the food category to identify optimization opportunities

📚 Data Sources: get_account_balance, get_transaction_history, categorize_spending
```

### Validation Examples

**Type-Safe Tool Definition**:

```python
# ✅ Correct: Type hints and docstring
@agent.tool
def calculate_tax(income: float, rate: float) -> dict[str, float]:
    """Calculate tax owed on income."""
    return {"tax_owed": income * rate}

# ❌ Missing type hints
@agent.tool
def calculate_tax(income, rate):  # mypy warning!
    return {"tax_owed": income * rate}

# ❌ Missing docstring
@agent.tool
def calculate_tax(income: float, rate: float) -> dict[str, float]:
    # No docstring - agent won't know what this does!
    return {"tax_owed": income * rate}
```

### Type Checking

```bash
mypy lesson_06_custom_tools.py
```

Expected: `Success: no issues found`

---

## D. Common Stumbling Blocks

### 1. Missing Docstring

**The Problem**:
```python
@agent.tool
def my_tool(value: int) -> int:
    # ❌ No docstring!
    return value * 2
```

**What Happens**:
Gemini doesn't know what the tool does or when to use it.

**The Fix**:
Always include a detailed docstring:
```python
@agent.tool
def my_tool(value: int) -> int:
    """
    Double the input value.
    
    Args:
        value: The number to double
    
    Returns:
        The doubled value
    """
    return value * 2
```

### 2. Missing Type Hints

**The Problem**:
```python
@agent.tool
def calculate(x, y):  # ❌ No type hints
    return x + y
```

**What Happens**:
- No parameter validation
- mypy can't check correctness
- Gemini doesn't know expected types
- Runtime errors possible

**The Fix**:
```python
@agent.tool
def calculate(x: float, y: float) -> float:  # ✅ Type hints
    """Add two numbers."""
    return x + y
```

### 3. Tool Raises Unhandled Exceptions

**The Problem**:
```python
@agent.tool
def divide(a: float, b: float) -> float:
    """Divide a by b."""
    return a / b  # ❌ Crashes if b == 0
```

**The Fix**:
Handle errors gracefully:
```python
@agent.tool
def divide(a: float, b: float) -> dict[str, any]:
    """
    Divide a by b safely.
    
    Returns error if division by zero.
    """
    if b == 0:
        return {"error": "Cannot divide by zero", "result": None}
    
    return {"result": a / b, "error": None}
```

### 4. Tool is Too Complex

**The Problem**:
```python
@agent.tool
def do_everything(
    user_id: str,
    action: str,
    params: dict,
    options: Optional[dict] = None,
    flags: list[str] = None
) -> dict:
    """Does many different things based on action."""
    # 100 lines of if/elif/else...
    # ❌ Too complex! Hard for agent to use correctly
```

**The Fix**:
Create multiple focused tools:
```python
@agent.tool
def get_user_data(user_id: str) -> dict:
    """Get user data."""
    ...

@agent.tool
def update_user_preferences(user_id: str, preferences: dict) -> dict:
    """Update user preferences."""
    ...

@agent.tool
def delete_user_account(user_id: str) -> dict:
    """Delete user account."""
    ...
```

### 5. Type Safety Gotcha: Mutable Defaults

**The Problem**:
```python
@agent.tool
def add_items(items: list[str] = []) -> list[str]:  # ❌ Mutable default!
    """Add items to list."""
    items.append("new_item")
    return items

# Calling twice causes weird behavior:
result1 = add_items()  # ["new_item"]
result2 = add_items()  # ["new_item", "new_item"] - shared list!
```

**The Fix**:
Use `None` and create new instance:
```python
@agent.tool
def add_items(items: Optional[list[str]] = None) -> list[str]:
    """Add items to list."""
    if items is None:
        items = []
    items.append("new_item")
    return items
```

---

## Ready for the Next Lesson?

🎉 **Fantastic work!** You can now create powerful custom tools:

✅ Simple calculation tools without dependencies  
✅ Tools that access dependencies via RunContext  
✅ Complex analytical tools with multiple parameters  
✅ Predictive tools that generate forecasts  
✅ Tools with Literal types for constrained inputs  
✅ Type-safe, self-documenting tool functions  

**Custom tools are what make agents truly useful!** You've transformed your agent from a text generator into an automation system that can interact with real data and services.

In the next lesson, we'll explore **Tool Function Signatures** in depth - you'll learn advanced patterns for parameter validation, optional parameters, default values, and how to design tool signatures that guide Gemini to use them correctly!

**Ready for Lesson 7, or would you like to practice creating more custom tools first?** 🚀
