# Lesson 9: Multi-Tool Agents

## A. Concept Overview

### What & Why
**Multi-Tool Agents** are agents equipped with multiple specialized tools that Gemini can intelligently select and coordinate to solve complex tasks. Instead of a single tool doing everything, you create focused tools that do one thing well, and Gemini orchestrates them based on the user's needs. This is crucial because real-world problems require diverse capabilities - data retrieval, calculations, API calls, transformations - all working together seamlessly.

### Analogy
Think of a multi-tool agent like a professional kitchen with specialized stations:

**Single-Tool Agent** = Fast food restaurant:
- One person, one station, limited menu
- "I can make burgers, that's it"

**Multi-Tool Agent** = Professional kitchen:
- **Prep station**: Chops vegetables, preps ingredients
- **Grill station**: Cooks proteins
- **Sauté station**: Makes sauces and sautéed dishes
- **Pastry station**: Handles desserts
- **Expeditor (Gemini)**: Coordinates all stations to complete the order

When a customer orders a meal, the expeditor (Gemini) intelligently coordinates:
1. Prep station prepares ingredients
2. Grill station cooks the protein
3. Sauté station makes the sauce
4. Everything comes together as a complete dish

Each station is specialized, and the expeditor knows which stations to use and in what order!

### Type Safety Benefit
Multi-tool agents provide comprehensive type safety:
- **Tool registry validation**: All tools are type-checked at registration
- **Parameter type safety**: Each tool's parameters are independently validated
- **Return type composition**: Combine typed outputs from multiple tools
- **Tool selection safety**: Gemini can only select registered tools
- **Dependency sharing**: All tools access the same typed dependencies
- **Error isolation**: One tool's failure doesn't corrupt others

Your entire agent becomes a type-safe workflow orchestration system!

---

## B. Code Implementation

### File Structure
```
agent_system_design/
├── lesson_09_multi_tool_agents.py
├── requirements.txt
└── .env
```

### Complete Code Snippet

**lesson_09_multi_tool_agents.py**
```python
"""
Lesson 9: Multi-Tool Agents
Build agents with multiple coordinated tools
"""

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from typing import Literal, Optional
import json
from dotenv import load_dotenv

load_dotenv()


# Dependencies that all tools can access
@dataclass
class CompanyData:
    """Simulated company database"""
    
    def get_employee_count(self) -> int:
        """Get total employee count"""
        return 247
    
    def get_department_count(self, department: str) -> int:
        """Get employee count by department"""
        dept_counts = {
            "engineering": 85,
            "sales": 62,
            "marketing": 34,
            "operations": 41,
            "hr": 25
        }
        return dept_counts.get(department.lower(), 0)
    
    def get_revenue_data(self, year: int) -> dict:
        """Get revenue data for a year"""
        return {
            "year": year,
            "total_revenue": 15_750_000,
            "q1": 3_500_000,
            "q2": 4_200_000,
            "q3": 3_900_000,
            "q4": 4_150_000
        }
    
    def get_department_budget(self, department: str) -> float:
        """Get department budget"""
        budgets = {
            "engineering": 4_500_000,
            "sales": 3_200_000,
            "marketing": 1_800_000,
            "operations": 2_100_000,
            "hr": 900_000
        }
        return budgets.get(department.lower(), 0)


@dataclass
class MultiToolDeps:
    """Dependencies for multi-tool agent"""
    company_data: CompanyData
    user_id: str
    request_timestamp: datetime


# Result model
class AnalysisResult(BaseModel):
    """Comprehensive analysis result"""
    summary: str = Field(description="Overall summary of analysis")
    key_findings: list[str] = Field(
        description="3-5 key insights",
        min_length=1,
        max_length=5
    )
    metrics: dict[str, float] = Field(description="Calculated metrics")
    data_sources: list[str] = Field(description="Data sources used")
    tools_invoked: list[str] = Field(description="Tools that were called")
    confidence: float = Field(description="Confidence in analysis", ge=0.0, le=1.0)
    recommendations: list[str] = Field(description="Actionable recommendations")


# Create multi-tool agent
agent = Agent(
    model='gemini-1.5-flash',
    result_type=AnalysisResult,
    deps_type=MultiToolDeps,
    system_prompt="""
You are a business intelligence analyst with access to multiple specialized tools.

AVAILABLE TOOLS:
1. get_headcount_data - Employee counts (total and by department)
2. get_financial_data - Revenue and financial metrics
3. calculate_ratio - Calculate ratios and percentages
4. calculate_growth_rate - Calculate growth between periods
5. get_budget_data - Department budget information
6. compare_departments - Compare metrics across departments

TOOL COORDINATION STRATEGY:
- Use multiple tools when needed to build comprehensive analysis
- Combine data from different tools for deeper insights
- Use calculation tools to derive metrics from raw data
- Compare data across dimensions (departments, time periods, etc.)

ANALYSIS APPROACH:
1. Gather relevant data from data retrieval tools
2. Use calculation tools to derive insights
3. Use comparison tools to identify patterns
4. Synthesize findings into actionable recommendations

OUTPUT REQUIREMENTS:
- Provide 3-5 key findings based on tool outputs
- Include calculated metrics with clear labels
- List all tools you invoked
- List all data sources accessed
- Give confidence score based on data completeness
- Provide specific, actionable recommendations

Remember: More tools = more comprehensive analysis!
""",
)


# TOOL CATEGORY 1: Data Retrieval Tools
# These tools fetch raw data from various sources

@agent.tool
def get_headcount_data(
    ctx: RunContext[MultiToolDeps],
    department: Optional[str] = None
) -> dict[str, any]:
    """
    Get employee headcount data.
    
    Retrieves total company headcount or department-specific counts.
    Use this when analyzing workforce, staffing levels, or organizational structure.
    
    Args:
        department: Optional department name (engineering, sales, marketing, operations, hr)
                   If None, returns total company headcount
    
    Returns:
        Dictionary with headcount data:
        - total: Total employee count
        - department: Department name (if specified)
        - department_count: Count for specific department (if specified)
        - data_source: Where the data came from
    
    Example:
        get_headcount_data() → Total company headcount
        get_headcount_data(department="engineering") → Engineering team size
    """
    print(f"\n🔧 get_headcount_data(department={department})")
    
    company = ctx.deps.company_data
    total = company.get_employee_count()
    
    result = {
        "total": total,
        "data_source": "hr_database"
    }
    
    if department:
        dept_count = company.get_department_count(department)
        result["department"] = department
        result["department_count"] = dept_count
        result["percentage_of_total"] = round((dept_count / total) * 100, 1) if total > 0 else 0
    
    return result


@agent.tool
def get_financial_data(
    ctx: RunContext[MultiToolDeps],
    year: int,
    breakdown: bool = False
) -> dict[str, any]:
    """
    Get financial/revenue data for analysis.
    
    Retrieves revenue data for a specific year with optional quarterly breakdown.
    Use this for financial analysis, revenue trends, or performance evaluation.
    
    Args:
        year: Year to retrieve data for (e.g., 2024)
        breakdown: If True, includes quarterly breakdown (default: False)
    
    Returns:
        Dictionary with financial data:
        - year: Year of data
        - total_revenue: Annual revenue
        - quarterly_breakdown: Q1-Q4 revenue (if breakdown=True)
        - data_source: Where the data came from
    
    Example:
        get_financial_data(2024) → Annual revenue
        get_financial_data(2024, breakdown=True) → Revenue with quarterly detail
    """
    print(f"\n🔧 get_financial_data(year={year}, breakdown={breakdown})")
    
    company = ctx.deps.company_data
    revenue_data = company.get_revenue_data(year)
    
    result = {
        "year": year,
        "total_revenue": revenue_data["total_revenue"],
        "data_source": "financial_system"
    }
    
    if breakdown:
        result["quarterly_breakdown"] = {
            "Q1": revenue_data["q1"],
            "Q2": revenue_data["q2"],
            "Q3": revenue_data["q3"],
            "Q4": revenue_data["q4"]
        }
    
    return result


@agent.tool
def get_budget_data(
    ctx: RunContext[MultiToolDeps],
    department: str
) -> dict[str, any]:
    """
    Get department budget information.
    
    Retrieves allocated budget for a specific department.
    Use for budget analysis, resource allocation, or cost planning.
    
    Args:
        department: Department name (engineering, sales, marketing, operations, hr)
    
    Returns:
        Dictionary with budget data:
        - department: Department name
        - budget: Allocated budget in dollars
        - data_source: Where the data came from
    
    Example:
        get_budget_data("engineering") → Engineering department budget
    """
    print(f"\n🔧 get_budget_data(department={department})")
    
    company = ctx.deps.company_data
    budget = company.get_department_budget(department)
    
    return {
        "department": department,
        "budget": budget,
        "data_source": "budget_system"
    }


# TOOL CATEGORY 2: Calculation Tools
# These tools perform calculations on data

@agent.tool
def calculate_ratio(
    numerator: float,
    denominator: float,
    ratio_name: str
) -> dict[str, any]:
    """
    Calculate a ratio between two values.
    
    General-purpose ratio calculator. Use for per-capita metrics,
    efficiency ratios, or any comparative calculation.
    
    Args:
        numerator: Top value in ratio
        denominator: Bottom value in ratio
        ratio_name: Descriptive name for the ratio (e.g., "revenue_per_employee")
    
    Returns:
        Dictionary with:
        - ratio_name: Name of the calculated ratio
        - result: Calculated ratio value
        - numerator: Input numerator
        - denominator: Input denominator
        - formatted: Human-readable formatted result
    
    Example:
        calculate_ratio(15750000, 247, "revenue_per_employee")
        → Revenue per employee calculation
    """
    print(f"\n🔧 calculate_ratio({numerator} / {denominator} = {ratio_name})")
    
    if denominator == 0:
        return {
            "ratio_name": ratio_name,
            "result": None,
            "error": "Division by zero"
        }
    
    result = numerator / denominator
    
    return {
        "ratio_name": ratio_name,
        "result": round(result, 2),
        "numerator": numerator,
        "denominator": denominator,
        "formatted": f"{result:,.2f}"
    }


@agent.tool
def calculate_growth_rate(
    previous_value: float,
    current_value: float,
    period_description: str
) -> dict[str, any]:
    """
    Calculate growth rate between two values.
    
    Calculates percentage change between two periods.
    Use for year-over-year growth, quarter-over-quarter changes, etc.
    
    Args:
        previous_value: Earlier period value
        current_value: Later period value
        period_description: Description of the period (e.g., "Q1 to Q2 2024")
    
    Returns:
        Dictionary with:
        - period: Period description
        - growth_rate: Percentage growth (positive or negative)
        - previous_value: Starting value
        - current_value: Ending value
        - absolute_change: Dollar/unit change
        - direction: "growth" or "decline"
    
    Example:
        calculate_growth_rate(3500000, 4200000, "Q1 to Q2")
        → 20% growth from Q1 to Q2
    """
    print(f"\n🔧 calculate_growth_rate({previous_value} → {current_value})")
    
    if previous_value == 0:
        return {
            "period": period_description,
            "growth_rate": None,
            "error": "Cannot calculate growth from zero"
        }
    
    absolute_change = current_value - previous_value
    growth_rate = (absolute_change / previous_value) * 100
    
    return {
        "period": period_description,
        "growth_rate": round(growth_rate, 2),
        "previous_value": previous_value,
        "current_value": current_value,
        "absolute_change": absolute_change,
        "direction": "growth" if growth_rate > 0 else "decline"
    }


# TOOL CATEGORY 3: Comparison Tools
# These tools compare data across dimensions

@agent.tool
def compare_departments(
    ctx: RunContext[MultiToolDeps],
    departments: list[str],
    metric: Literal["headcount", "budget"]
) -> dict[str, any]:
    """
    Compare metrics across multiple departments.
    
    Performs side-by-side comparison of departments on a specific metric.
    Use for benchmarking, resource allocation analysis, or identifying outliers.
    
    Args:
        departments: List of department names to compare
        metric: Which metric to compare ("headcount" or "budget")
    
    Returns:
        Dictionary with:
        - metric: Metric being compared
        - departments: List of department names
        - values: Dict mapping department to metric value
        - highest: Department with highest value
        - lowest: Department with lowest value
        - average: Average value across departments
        - total: Sum of all values
    
    Example:
        compare_departments(["engineering", "sales"], "headcount")
        → Compare team sizes
    """
    print(f"\n🔧 compare_departments(departments={departments}, metric={metric})")
    
    company = ctx.deps.company_data
    
    values = {}
    for dept in departments:
        if metric == "headcount":
            values[dept] = company.get_department_count(dept)
        elif metric == "budget":
            values[dept] = company.get_department_budget(dept)
    
    if not values:
        return {"error": "No valid departments provided"}
    
    total = sum(values.values())
    average = total / len(values)
    
    highest_dept = max(values, key=values.get)
    lowest_dept = min(values, key=values.get)
    
    return {
        "metric": metric,
        "departments": departments,
        "values": values,
        "highest": {"department": highest_dept, "value": values[highest_dept]},
        "lowest": {"department": lowest_dept, "value": values[lowest_dept]},
        "average": round(average, 2),
        "total": total
    }


# Demonstration

def main():
    print("\n" + "="*70)
    print("MULTI-TOOL AGENT DEMONSTRATION")
    print("="*70)
    print("\nThis agent has 6 specialized tools working together:")
    print("  📊 Data Retrieval: get_headcount_data, get_financial_data, get_budget_data")
    print("  🧮 Calculations: calculate_ratio, calculate_growth_rate")
    print("  📈 Comparisons: compare_departments")
    
    # Create dependencies
    deps = MultiToolDeps(
        company_data=CompanyData(),
        user_id="analyst_001",
        request_timestamp=datetime.now()
    )
    
    # Test queries that require multiple tools
    queries = [
        "What is our revenue per employee?",
        "Compare engineering and sales departments by headcount and budget",
        "Analyze our quarterly revenue growth for 2024",
        "Which department has the best revenue-to-budget ratio?",
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n\n{'='*70}")
        print(f"QUERY {i}: {query}")
        print(f"{'='*70}")
        
        try:
            result = agent.run_sync(query, deps=deps)
            analysis = result.data
            
            print(f"\n📋 ANALYSIS SUMMARY:")
            print(f"   {analysis.summary}")
            
            print(f"\n🔍 KEY FINDINGS:")
            for j, finding in enumerate(analysis.key_findings, 1):
                print(f"   {j}. {finding}")
            
            print(f"\n📊 METRICS:")
            for metric_name, value in analysis.metrics.items():
                print(f"   {metric_name}: {value:,.2f}")
            
            print(f"\n💡 RECOMMENDATIONS:")
            for j, rec in enumerate(analysis.recommendations, 1):
                print(f"   {j}. {rec}")
            
            print(f"\n🔧 TOOLS USED: {', '.join(analysis.tools_invoked)}")
            print(f"📚 DATA SOURCES: {', '.join(analysis.data_sources)}")
            print(f"✅ CONFIDENCE: {analysis.confidence:.2f}")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
    
    print("\n" + "="*70)
    print("MULTI-TOOL COORDINATION DEMONSTRATED")
    print("="*70)
    print("\n✨ Key Observations:")
    print("   • Gemini intelligently selected relevant tools for each query")
    print("   • Tools were coordinated to build comprehensive analysis")
    print("   • Data from multiple tools was synthesized into insights")
    print("   • Each tool remained focused and single-purpose")
    print("   • Type safety maintained across all tool interactions")


if __name__ == "__main__":
    main()
```

### Line-by-Line Explanation

**Dependencies (Lines 17-75)**:
- `CompanyData`: Simulates a company database
- Multiple methods for different data types
- All tools can access this shared data source
- Type-safe data access

**Multi-Tool Agent (Lines 85-145)**:
- Agent with 6 specialized tools
- System prompt explains tool coordination strategy
- Guides Gemini on when and how to use multiple tools
- Emphasizes comprehensive analysis through tool combination

**Category 1: Data Retrieval Tools (Lines 148-254)**:
- `get_headcount_data`: Employee counts
- `get_financial_data`: Revenue information
- `get_budget_data`: Department budgets
- Each tool focused on one data domain
- All return structured, consistent formats

**Category 2: Calculation Tools (Lines 257-361)**:
- `calculate_ratio`: General ratio calculations
- `calculate_growth_rate`: Percentage change calculations
- Pure computational tools (no dependencies needed)
- Reusable across different data types

**Category 3: Comparison Tools (Lines 364-424)**:
- `compare_departments`: Side-by-side comparisons
- Combines data from multiple sources
- Identifies patterns (highest, lowest, average)
- Useful for benchmarking and analysis

### The "Why" Behind the Pattern

**Why multiple specialized tools instead of one general tool?**

❌ **Single General Tool** (Antipattern):
```python
@agent.tool
def do_analysis(query: str, data_type: str, operation: str, params: dict):
    """Do any kind of analysis."""  # Too vague!
    if data_type == "headcount":
        if operation == "get":
            # 100 lines of if/else...
        elif operation == "compare":
            # More branching...
    elif data_type == "financial":
        # Even more branching...
    # This becomes unmaintainable!
```

✅ **Multiple Specialized Tools** (Best Practice):
```python
# Each tool has ONE clear purpose
@agent.tool
def get_headcount_data(...):
    """Get employee headcount."""
    # Simple, focused implementation

@agent.tool
def calculate_ratio(...):
    """Calculate a ratio."""
    # Simple, focused implementation

# Gemini combines them intelligently!
```

**Benefits of Multi-Tool Pattern**:
1. **Clear responsibilities**: Each tool does one thing well
2. **Easy to test**: Test each tool independently
3. **Easy to maintain**: Changes to one tool don't affect others
4. **Reusable**: Tools can be combined in many ways
5. **Understandable**: Gemini knows exactly what each tool does
6. **Type-safe**: Each tool has its own validated signature

---

## C. Test & Apply

### How to Test It

1. **Run the multi-tool demo**:
```bash
python lesson_09_multi_tool_agents.py
```

2. **Observe how Gemini coordinates multiple tools**

3. **Try your own multi-tool query**:
```python
queries = [
    "What's our cost per employee by department?",
    "Which quarter had the strongest revenue growth?",
    "How does marketing's budget compare to their headcount?",
]
```

### Expected Result

You should see Gemini intelligently using multiple tools:

```
======================================================================
QUERY 1: What is our revenue per employee?
======================================================================

🔧 get_financial_data(year=2024, breakdown=False)
🔧 get_headcount_data(department=None)
🔧 calculate_ratio(15750000 / 247 = revenue_per_employee)

📋 ANALYSIS SUMMARY:
   The company generates approximately $63,765 in revenue per employee,
   indicating strong productivity and efficiency across the organization.

🔍 KEY FINDINGS:
   1. Total annual revenue is $15,750,000
   2. Total employee count is 247
   3. Revenue per employee ratio is $63,765
   4. This is above industry average for similar companies

📊 METRICS:
   revenue_per_employee: 63,765.00
   total_revenue: 15,750,000.00
   total_employees: 247.00

💡 RECOMMENDATIONS:
   1. Benchmark against industry standards to identify improvement areas
   2. Analyze productivity trends over time
   3. Consider this metric in hiring decisions

🔧 TOOLS USED: get_financial_data, get_headcount_data, calculate_ratio
📚 DATA SOURCES: financial_system, hr_database
✅ CONFIDENCE: 0.95
```

### Validation Examples

**Multi-Tool Query Pattern**:

```python
# ✅ Query that requires multiple tools
"Compare engineering and sales by budget and headcount"
# Triggers:
# 1. get_budget_data("engineering")
# 2. get_budget_data("sales")
# 3. get_headcount_data(department="engineering")
# 4. get_headcount_data(department="sales")
# 5. calculate_ratio (multiple times)
# 6. compare_departments

# ✅ Query that chains tool outputs
"What's the quarterly growth rate?"
# Triggers:
# 1. get_financial_data(breakdown=True)
# 2. calculate_growth_rate(Q1, Q2, ...)
# 3. calculate_growth_rate(Q2, Q3, ...)
# 4. calculate_growth_rate(Q3, Q4, ...)
```

### Type Checking

```bash
mypy lesson_09_multi_tool_agents.py
```

Expected: `Success: no issues found`

---

## D. Common Stumbling Blocks

### 1. Tools That Do Too Much

**The Problem**:
```python
@agent.tool
def analyze_everything(
    data_type: str,
    operation: str,
    departments: list[str],
    years: list[int],
    options: dict
):
    """Analyze any data in any way."""  # ❌ Too generic!
    # 500 lines of if/else statements...
```

**What's Wrong**:
- Gemini doesn't know when to use it
- Hard to describe clearly
- Difficult to test
- Hard to maintain

**The Fix**:
Break into specialized tools:
```python
@agent.tool
def get_department_metrics(department: str, year: int) -> dict:
    """Get metrics for one department in one year."""
    ...

@agent.tool
def compare_years(department: str, year1: int, year2: int) -> dict:
    """Compare one department across two years."""
    ...

# Each tool is clear and focused!
```

### 2. Tools with Hidden Dependencies

**The Problem**:
```python
# Tool A modifies global state
@agent.tool
def tool_a():
    global cache
    cache["data"] = "something"  # ❌ Hidden side effect

# Tool B depends on Tool A having run first
@agent.tool
def tool_b():
    return cache["data"]  # ❌ Assumes Tool A ran first
```

**What's Wrong**:
- Order-dependent execution
- Hidden dependencies
- Not thread-safe
- Hard to test

**The Fix**:
Use explicit dependencies:
```python
@agent.tool
def tool_a(ctx: RunContext[Deps]) -> dict:
    """Get data and return it."""
    data = fetch_data()
    return {"data": data}

@agent.tool
def tool_b(ctx: RunContext[Deps], data: dict) -> dict:
    """Process data (pass data explicitly)."""
    return process(data)

# Dependencies are explicit!
```

### 3. Unclear Tool Names

**The Problem**:
```python
@agent.tool
def tool_1():  # ❌ What does this do?
    ...

@agent.tool
def helper():  # ❌ Vague
    ...

@agent.tool
def process():  # ❌ Process what?
    ...
```

**The Fix**:
Use descriptive, action-oriented names:
```python
@agent.tool
def get_user_profile():  # ✅ Clear action and target
    ...

@agent.tool
def calculate_monthly_revenue():  # ✅ Specific and descriptive
    ...

@agent.tool
def send_email_notification():  # ✅ Clear what it does
    ...
```

### 4. Missing Tool Coordination Guidance

**The Problem**:
```python
system_prompt = """
You have access to tools.
"""  # ❌ No guidance on how to use them together
```

**The Fix**:
Provide coordination strategy:
```python
system_prompt = """
You have access to multiple tools that work together:

DATA TOOLS:
- get_user_data: User information
- get_order_data: Order history

CALCULATION TOOLS:
- calculate_total: Sum values
- calculate_average: Average values

WORKFLOW:
1. Fetch raw data using data tools
2. Process with calculation tools
3. Combine results for comprehensive analysis

EXAMPLE:
To analyze user spending:
1. get_user_data(user_id) → user info
2. get_order_data(user_id) → orders
3. calculate_total(order amounts) → total spent
4. Combine for final analysis
"""  # ✅ Clear coordination strategy
```

### 5. Type Safety Gotcha: Tool Output Assumptions

**The Problem**:
```python
@agent.tool
def get_data() -> dict:
    """Get some data."""
    return {"value": 42}

@agent.tool
def process_data(data: dict) -> dict:
    """Process data."""
    # ❌ Assumes "value" key exists
    return {"result": data["value"] * 2}  # KeyError if missing!
```

**The Fix**:
Use Pydantic models for tool outputs:
```python
class DataOutput(BaseModel):
    value: int

class ProcessedOutput(BaseModel):
    result: int

@agent.tool
def get_data() -> DataOutput:  # ✅ Structured output
    """Get some data."""
    return DataOutput(value=42)

@agent.tool
def process_data(data: DataOutput) -> ProcessedOutput:  # ✅ Validated input
    """Process data."""
    return ProcessedOutput(result=data.value * 2)  # ✅ Type-safe!
```

---

## Ready for the Next Lesson?

🎉 **Excellent work!** You now understand how to build multi-tool agents:

✅ Multiple specialized tools working together  
✅ Tool categories (data, calculation, comparison)  
✅ Gemini's intelligent tool coordination  
✅ Clear tool naming and descriptions  
✅ Tool independence and reusability  
✅ Type-safe tool composition  

**Multi-tool agents unlock the true power of AI systems!** Instead of trying to do everything in one tool, you build a toolkit of capabilities that Gemini orchestrates intelligently.

In the next lesson, we'll explore **Tool Context and Parameters** - you'll learn advanced patterns for sharing context between tools, passing tool outputs to other tools, and managing state across multi-step workflows!

**Ready for Lesson 10, or would you like to practice building multi-tool agents first?** 🚀
