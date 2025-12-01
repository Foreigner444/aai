# PydanticAI Gemini Mentor - Agent System Design Project
## Lesson 6: Creating Custom Tools

Welcome back, agent architect! 🎯 You've built agents, given them personalities, connected them to dependencies, and orchestrated complex workflows. Now it's time to supercharge your agents with **custom tools** - specialized functions they can call during execution to perform specific tasks!

---

## What Are Custom Tools?

**Custom tools** are specialized functions that agents can invoke during their reasoning process. Think of them as the "actions" your agent can take - like making API calls, performing calculations, querying databases, or executing business logic.

**Analogy Time**: Imagine you're a detective (your agent) with a toolkit:
- **Magnifying glass** (search tool) - to examine evidence
- **Phone** (communication tool) - to call witnesses  
- **Database access** (research tool) - to lookup records
- **Calculator** (math tool) - to analyze data

The detective doesn't need to memorize how to use each tool perfectly - they just need to know when to reach for which one!

---

## Why Use Custom Tools?

1. **Separation of Concerns**: Keep agent reasoning separate from execution logic
2. **Reusability**: Same tools can be used by multiple agents
3. **Error Handling**: Tools can handle errors gracefully without breaking agent flow
4. **Testing**: Tools can be unit tested independently
5. **Flexibility**: Easy to swap implementations (real API ↔ mock API)

---

## Basic Tool Structure

Let's start with a simple tool that performs a calculation:

```python
from pydantic import BaseModel, Field
from typing import Any, Dict
import math

class CalculatorTool:
    """A simple calculator tool for basic math operations."""
    
    def calculate(self, operation: str, a: float, b: float) -> Dict[str, Any]:
        """
        Perform basic mathematical operations.
        
        Args:
            operation: Type of operation ('add', 'subtract', 'multiply', 'divide', 'power', 'sqrt')
            a: First number
            b: Second number (not needed for sqrt)
            
        Returns:
            Dictionary with result and operation details
        """
        try:
            if operation == 'add':
                result = a + b
                operation_symbol = '+'
            elif operation == 'subtract':
                result = a - b
                operation_symbol = '-'
            elif operation == 'multiply':
                result = a * b
                operation_symbol = '×'
            elif operation == 'divide':
                if b == 0:
                    return {
                        'error': 'Cannot divide by zero',
                        'operation': operation,
                        'inputs': [a, b]
                    }
                result = a / b
                operation_symbol = '÷'
            elif operation == 'power':
                result = a ** b
                operation_symbol = '^'
            elif operation == 'sqrt':
                result = math.sqrt(a)
                operation_symbol = '√'
                b = None  # Not used for square root
            else:
                return {
                    'error': f'Unknown operation: {operation}',
                    'available_operations': ['add', 'subtract', 'multiply', 'divide', 'power', 'sqrt']
                }
            
            return {
                'result': result,
                'operation': operation,
                'operation_symbol': operation_symbol,
                'inputs': [a, b] if b is not None else [a],
                'calculation': f"{a} {operation_symbol} {b if b is not None else ''} = {result}"
            }
            
        except Exception as e:
            return {
                'error': f'Calculation failed: {str(e)}',
                'operation': operation,
                'inputs': [a, b]
            }

# Example usage
calculator = CalculatorTool()

# Test different operations
print("=== Calculator Tool Examples ===")
print(f"Addition: {calculator.calculate('add', 10, 5)}")
print(f"Subtraction: {calculator.calculate('subtract', 10, 5)}")
print(f"Multiplication: {calculator.calculate('multiply', 10, 5)}")
print(f"Division: {calculator.calculate('divide', 10, 5)}")
print(f"Power: {calculator.calculate('power', 2, 8)}")
print(f"Square Root: {calculator.calculate('sqrt', 16, None)}")
print(f"Error Case: {calculator.calculate('divide', 10, 0)}")
```

**Output:**
```
=== Calculator Tool Examples ===
Addition: {'result': 15, 'operation': 'add', 'operation_symbol': '+', 'inputs': [10, 5], 'calculation': '10 + 5 = 15'}
Subtraction: {'result': 5, 'operation': 'subtract', 'operation_symbol': '-', 'inputs': [10, 5], 'calculation': '10 - 5 = 5'}
Multiplication: {'result': 50, 'operation': 'multiply', 'operation_symbol': '×', 'inputs': [10, 5], 'calculation': '10 × 5 = 50'}
Division: {'result': 2.0, 'operation': 'divide', 'operation_symbol': '÷', 'inputs': [10, 5], 'calculation': '10 ÷ 5 = 2.0'}
Power: {'result': 256, 'operation': 'power', 'operation_symbol': '^', 'inputs': [2, 8], 'calculation': '2 ^ 8 = 256'}
Square Root: {'result': 4.0, 'operation': 'sqrt', 'operation_symbol': '√', 'inputs': [16], 'calculation': '√16 = 4.0'}
Error Case: {'error': 'Cannot divide by zero', 'operation': 'divide', 'inputs': [10, 0]}
```

---

## Integrating Tools with Agents

Now let's create an agent that can use our calculator tool:

```python
from pydantic import BaseModel
from pydantic_ai import Agent
import asyncio

class MathQuery(BaseModel):
    """Input for math-related queries."""
    question: str = Field(description="A math question or calculation request")

class MathResult(BaseModel):
    """Result of a math calculation."""
    original_question: str
    operation_used: str
    inputs: list
    result: float
    calculation_steps: str
    explanation: str

# Create the calculator tool
calculator_tool = CalculatorTool()

# Create agent with the tool
math_agent = Agent(
    'gemini-1.5-flash',
    result_type=MathResult,
    system_prompt='''
    You are a helpful math assistant. When users ask math questions:
    1. Identify the operation needed (add, subtract, multiply, divide, power, sqrt)
    2. Extract the numbers from the question
    3. Use the calculator tool to get the result
    4. Provide a clear explanation
    
    Always show your work and explain the steps clearly.
    ''',
    tools=[calculator_tool]  # This is where we add our custom tool!
)

# Test the agent
async def test_math_agent():
    result = await math_agent.run("What's 25 times 16?")
    print(f"Question: {result.data.original_question}")
    print(f"Operation: {result.data.operation_used}")
    print(f"Inputs: {result.data.inputs}")
    print(f"Result: {result.data.result}")
    print(f"Steps: {result.data.calculation_steps}")
    print(f"Explanation: {result.data.explanation}")

# Run the test
asyncio.run(test_math_agent())
```

**Expected Output:**
```
Question: What's 25 times 16?
Operation: multiply
Inputs: [25, 16]
Result: 400.0
Steps: 25 × 16 = 400
Explanation: To multiply 25 by 16, I calculated 25 × 16 = 400. This is basic multiplication of two numbers.
```

---

## Advanced Tool: Data Processing Tool

Let's build a more sophisticated tool that processes data:

```python
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

class DataProcessorTool:
    """Tool for processing and analyzing data."""
    
    def __init__(self):
        # Simulated data store
        self.sample_data = [
            {"id": 1, "name": "Alice", "sales": 1500, "region": "North", "date": "2024-01-15"},
            {"id": 2, "name": "Bob", "sales": 2300, "region": "South", "date": "2024-01-16"},
            {"id": 3, "name": "Carol", "sales": 1800, "region": "North", "date": "2024-01-17"},
            {"id": 4, "name": "David", "sales": 3200, "region": "West", "date": "2024-01-18"},
            {"id": 5, "name": "Eve", "sales": 2100, "region": "South", "date": "2024-01-19"},
        ]
    
    def filter_data(self, data: List[Dict], criteria: Dict[str, Any]) -> List[Dict]:
        """Filter data based on criteria."""
        filtered = data
        for key, value in criteria.items():
            if key == "min_sales":
                filtered = [item for item in filtered if item.get("sales", 0) >= value]
            elif key == "max_sales":
                filtered = [item for item in filtered if item.get("sales", 0) <= value]
            elif key == "region":
                filtered = [item for item in filtered if item.get("region") == value]
        return filtered
    
    def calculate_stats(self, data: List[Dict], field: str) -> Dict[str, Any]:
        """Calculate statistics for a numeric field."""
        values = [item.get(field, 0) for item in data if field in item]
        
        if not values:
            return {"error": f"No data found for field: {field}"}
        
        total = sum(values)
        count = len(values)
        average = total / count
        minimum = min(values)
        maximum = max(values)
        
        return {
            "field": field,
            "count": count,
            "total": total,
            "average": round(average, 2),
            "minimum": minimum,
            "maximum": maximum,
            "items_analyzed": values
        }
    
    def group_by_field(self, data: List[Dict], group_field: str, value_field: str) -> Dict[str, Any]:
        """Group data by a field and calculate sum of another field."""
        grouped = {}
        
        for item in data:
            group_key = item.get(group_field, "Unknown")
            value = item.get(value_field, 0)
            
            if group_key not in grouped:
                grouped[group_key] = []
            grouped[group_key].append(value)
        
        # Calculate statistics for each group
        result = {}
        for group, values in grouped.items():
            result[group] = {
                "count": len(values),
                "total": sum(values),
                "average": round(sum(values) / len(values), 2),
                "items": values
            }
        
        return result
    
    def get_top_performers(self, data: List[Dict], field: str, limit: int = 3) -> List[Dict]:
        """Get top performers based on a field."""
        sorted_data = sorted(data, key=lambda x: x.get(field, 0), reverse=True)
        return sorted_data[:limit]

# Create the data processor tool
data_processor = DataProcessorTool()

# Example usage
print("=== Data Processing Tool Examples ===")

# 1. Filter data
print("\n1. Filter by region 'North':")
north_data = data_processor.filter_data(data_processor.sample_data, {"region": "North"})
for item in north_data:
    print(f"  {item}")

print("\n2. Filter by minimum sales 2000:")
high_sales = data_processor.filter_data(data_processor.sample_data, {"min_sales": 2000})
for item in high_sales:
    print(f"  {item}")

# 2. Calculate statistics
print("\n3. Sales statistics:")
stats = data_processor.calculate_stats(data_processor.sample_data, "sales")
print(f"  {json.dumps(stats, indent=2)}")

# 3. Group by region
print("\n4. Sales grouped by region:")
grouped = data_processor.group_by_field(data_processor.sample_data, "region", "sales")
print(f"  {json.dumps(grouped, indent=2)}")

# 4. Top performers
print("\n5. Top 3 performers:")
top_performers = data_processor.get_top_performers(data_processor.sample_data, "sales", 3)
for i, performer in enumerate(top_performers, 1):
    print(f"  {i}. {performer['name']}: ${performer['sales']}")
```

**Output:**
```
=== Data Processing Tool Examples ===

1. Filter by region 'North':
  {'id': 1, 'name': 'Alice', 'sales': 1500, 'region': 'North', 'date': '2024-01-15'}
  {'id': 3, 'name': 'Carol', 'sales': 1800, 'region': 'North', 'date': '2024-01-17'}

2. Filter by minimum sales 2000:
  {'id': 2, 'name': 'Bob', 'sales': 2300, 'region': 'South', 'date': '2024-01-16'}
  {'id': 4, 'name': 'David', 'sales': 3200, 'region': 'West', 'date': '2024-01-18'}
  {'id': 5, 'name': 'Eve', 'sales': 2100, 'region': 'South', 'date': '2024-01-19'}

3. Sales statistics:
{
  "field": "sales",
  "count": 5,
  "total": 10900,
  "average": 2180.0,
  "minimum": 1500,
  "maximum": 3200,
  "items_analyzed": [1500, 2300, 1800, 3200, 2100]
}

4. Sales grouped by region:
{
  "North": {
    "count": 2,
    "total": 3300,
    "average": 1650.0,
    "items": [1500, 1800]
  },
  "South": {
    "count": 2,
    "total": 4400,
    "average": 2200.0,
    "items": [2300, 2100]
  },
  "West": {
    "count": 1,
    "total": 3200,
    "average": 3200.0,
    "items": [3200]
  }
}

5. Top 3 performers:
  1. David: $3200
  2. Bob: $2300
  3. Eve: $2100
```

---

## Data Analysis Agent with Tools

Now let's create an agent that combines multiple tools:

```python
from pydantic import BaseModel, Field
from typing import List

class DataAnalysisQuery(BaseModel):
    """Input for data analysis requests."""
    question: str = Field(description="A data analysis question about sales data")

class DataAnalysisResult(BaseModel):
    """Result of data analysis."""
    original_question: str
    analysis_type: str
    findings: List[str]
    data_summary: Dict[str, Any]
    recommendations: List[str]
    confidence_level: str = Field(description="How confident we are in the analysis (high/medium/low)")

# Create agents with multiple tools
data_analysis_agent = Agent(
    'gemini-1.5-flash',
    result_type=DataAnalysisResult,
    system_prompt='''
    You are a data analyst assistant. You have access to:
    1. Calculator tool - for mathematical operations
    2. Data processor tool - for filtering, statistics, grouping data
    
    When users ask questions about data:
    1. Use the data processor tool to analyze the sample sales data
    2. Use the calculator tool if you need mathematical operations
    3. Provide insights and recommendations based on the results
    4. Always explain your methodology
    
    Be thorough and provide actionable insights.
    ''',
    tools=[calculator_tool, data_processor]
)

# Alternative: Tool-rich agent for complex analysis
complex_analysis_agent = Agent(
    'gemini-1.5-flash',
    result_type=DataAnalysisResult,
    system_prompt='''
    You are a senior data analyst. You have access to powerful data processing tools:
    - Calculator: for mathematical operations
    - Data Processor: for filtering, statistics, grouping, and ranking data
    
    Your approach:
    1. First understand what the user wants to know
    2. Plan your analysis approach
    3. Use appropriate tools to get the data
    4. Perform calculations if needed
    5. Provide insights with confidence levels
    
    Always show your work and explain your reasoning.
    ''',
    tools=[calculator_tool, data_processor]
)

async def test_data_analysis():
    print("=== Testing Data Analysis Agent ===")
    
    # Test question
    questions = [
        "Show me the top 3 sales performers and calculate their average performance",
        "What's the total sales for each region, and which region is performing best?",
        "Find all employees with sales above average and calculate how much above average they are"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n--- Analysis {i} ---")
        print(f"Question: {question}")
        
        try:
            result = await data_analysis_agent.run(question)
            print(f"Analysis Type: {result.data.analysis_type}")
            print(f"Findings:")
            for finding in result.data.findings:
                print(f"  • {finding}")
            print(f"Data Summary: {result.data.data_summary}")
            print(f"Recommendations:")
            for rec in result.data.recommendations:
                print(f"  • {rec}")
            print(f"Confidence: {result.data.confidence_level}")
        except Exception as e:
            print(f"Error: {e}")

# Run the test
asyncio.run(test_data_analysis())
```

---

## Tool Chaining: Building Complex Workflows

Let's create a tool that coordinates multiple other tools:

```python
class WorkflowOrchestrator:
    """Orchestrates multiple tools to perform complex workflows."""
    
    def __init__(self, calculator, data_processor):
        self.calculator = calculator
        self.data_processor = data_processor
    
    def sales_performance_analysis(self) -> Dict[str, Any]:
        """Complete sales performance analysis using multiple tools."""
        try:
            # Step 1: Get basic statistics
            stats = self.data_processor.calculate_stats(self.data_processor.sample_data, "sales")
            
            # Step 2: Get top performers
            top_performers = self.data_processor.get_top_performers(
                self.data_processor.sample_data, "sales", 3
            )
            
            # Step 3: Group by region for comparison
            regional_performance = self.data_processor.group_by_field(
                self.data_processor.sample_data, "region", "sales"
            )
            
            # Step 4: Calculate growth metrics using calculator
            average_sales = stats["average"]
            top_sales = top_performers[0]["sales"]
            performance_multiplier = self.calculator.calculate(
                "divide", top_sales, average_sales
            )
            
            # Step 5: Find underperformers
            underperformers = self.data_processor.filter_data(
                self.data_processor.sample_data, {"max_sales": average_sales}
            )
            
            return {
                "overall_stats": stats,
                "top_performers": top_performers,
                "regional_breakdown": regional_performance,
                "performance_multiplier": performance_multiplier,
                "underperformers": underperformers,
                "key_insights": [
                    f"Top performer is {performance_multiplier['result']:.2f}x above average",
                    f"Best performing region: {max(regional_performance.keys(), key=lambda k: regional_performance[k]['total'])}",
                    f"Number of underperformers: {len(underperformers)}"
                ]
            }
            
        except Exception as e:
            return {
                "error": f"Analysis failed: {str(e)}",
                "step_completed": "partial"
            }

# Create workflow orchestrator
orchestrator = WorkflowOrchestrator(calculator_tool, data_processor)

# Test the complete workflow
print("=== Complete Sales Performance Analysis ===")
analysis = orchestrator.sales_performance_analysis()

if "error" not in analysis:
    print(f"Top Performer: {analysis['top_performers'][0]['name']} - ${analysis['top_performers'][0]['sales']}")
    print(f"Average Performance: {analysis['overall_stats']['average']}")
    print(f"Performance Multiplier: {analysis['performance_multiplier']['result']:.2f}x")
    
    print("\nRegional Performance:")
    for region, data in analysis['regional_breakdown'].items():
        print(f"  {region}: ${data['total']} (avg: ${data['average']})")
    
    print("\nKey Insights:")
    for insight in analysis['key_insights']:
        print(f"  • {insight}")
else:
    print(f"Error: {analysis['error']}")
```

**Output:**
```
=== Complete Sales Performance Analysis ===
Top Performer: David - $3200
Average Performance: 2180.0
Performance Multiplier: 1.47x

Regional Performance:
  North: $3300 (avg: $1650)
  South: $4400 (avg: $2200)
  West: $3200 (avg: $3200)

Key Insights:
  • Top performer is 1.47x above average
  • Best performing region: South
  • Number of underperformers: 2
```

---

## Tool Result Models

Create proper Pydantic models for tool results to ensure type safety:

```python
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Literal

class CalculationResult(BaseModel):
    """Result of a calculation operation."""
    operation: str
    inputs: List[float]
    result: float
    calculation: str
    error: Optional[str] = None
    
    @validator('result')
    def result_must_be_finite(cls, v):
        if not isinstance(v, (int, float)) or v != v:  # Check for NaN
            raise ValueError('Result must be a valid number')
        return v

class DataFilterResult(BaseModel):
    """Result of data filtering operation."""
    original_count: int
    filtered_count: int
    criteria: Dict[str, Any]
    filtered_data: List[Dict[str, Any]]
    filter_description: str

class StatisticsResult(BaseModel):
    """Result of statistical analysis."""
    field: str
    count: int
    total: float
    average: float
    minimum: float
    maximum: float
    confidence_level: Literal['high', 'medium', 'low'] = 'medium'

class ToolExecutionResult(BaseModel):
    """General result from tool execution."""
    tool_name: str
    operation: str
    success: bool
    result: Dict[str, Any]
    execution_time_ms: Optional[int] = None
    error_message: Optional[str] = None

# Enhanced Calculator Tool with proper result models
class EnhancedCalculatorTool:
    """Enhanced calculator with proper result models."""
    
    def calculate(self, operation: str, a: float, b: Optional[float] = None) -> CalculationResult:
        """Perform calculation with proper result modeling."""
        try:
            if operation == 'sqrt':
                result = math.sqrt(a)
                calculation = f"√{a} = {result}"
                inputs = [a]
            else:
                result = a ** b if operation == 'power' else getattr(operator, operation)(a, b)
                calculation = f"{a} {operation} {b} = {result}"
                inputs = [a, b]
            
            return CalculationResult(
                operation=operation,
                inputs=inputs,
                result=result,
                calculation=calculation
            )
            
        except Exception as e:
            return CalculationResult(
                operation=operation,
                inputs=[a, b] if b else [a],
                result=0.0,  # Default value for failed calculations
                calculation="",
                error=str(e)
            )

# Example usage with proper modeling
enhanced_calc = EnhancedCalculatorTool()
result = enhanced_calc.calculate('multiply', 15, 8)

print(f"Enhanced Calculator Result:")
print(f"  Operation: {result.operation}")
print(f"  Calculation: {result.calculation}")
print(f"  Result: {result.result}")
print(f"  Has Error: {result.error is not None}")
```

---

## Tool Error Handling and Recovery

Implement robust error handling in your tools:

```python
import logging
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def tool_error_handler(func):
    """Decorator for consistent error handling in tools."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            logger.info(f"Successfully executed {func.__name__}")
            return result
        except ValueError as e:
            logger.warning(f"Invalid input for {func.__name__}: {e}")
            return {
                'error': f'Invalid input: {str(e)}',
                'error_type': 'ValueError',
                'operation': func.__name__
            }
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}")
            return {
                'error': f'Unexpected error: {str(e)}',
                'error_type': type(e).__name__,
                'operation': func.__name__
            }
    return wrapper

class RobustCalculatorTool:
    """Calculator with robust error handling."""
    
    @tool_error_handler
    def calculate(self, operation: str, a: float, b: Optional[float] = None) -> Dict[str, Any]:
        """Perform calculation with comprehensive error handling."""
        # Input validation
        if not isinstance(a, (int, float)):
            raise ValueError(f"First argument must be a number, got {type(a)}")
        
        if b is not None and not isinstance(b, (int, float)):
            raise ValueError(f"Second argument must be a number, got {type(b)}")
        
        if operation == 'divide' and b == 0:
            raise ValueError("Cannot divide by zero")
        
        if operation == 'sqrt' and a < 0:
            raise ValueError("Cannot calculate square root of negative number")
        
        # Perform operation
        if operation == 'sqrt':
            result = math.sqrt(a)
            calculation = f"√{a} = {result}"
        elif operation == 'power':
            result = a ** b
            calculation = f"{a}^{b} = {result}"
        else:
            operation_map = {
                'add': lambda x, y: x + y,
                'subtract': lambda x, y: x - y,
                'multiply': lambda x, y: x * y,
                'divide': lambda x, y: x / y
            }
            
            if operation not in operation_map:
                raise ValueError(f"Unsupported operation: {operation}")
            
            result = operation_map[operation](a, b)
            calculation = f"{a} {operation} {b} = {result}"
        
        return {
            'operation': operation,
            'inputs': [a, b] if b is not None else [a],
            'result': result,
            'calculation': calculation,
            'success': True
        }
    
    @tool_error_handler
    def batch_calculate(self, calculations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Perform multiple calculations in batch."""
        results = []
        for calc in calculations:
            operation = calc.get('operation')
            a = calc.get('a')
            b = calc.get('b')
            
            result = self.calculate(operation, a, b)
            result['batch_item'] = calc.get('description', f"{operation} calculation")
            results.append(result)
        
        return results

# Test error handling
robust_calc = RobustCalculatorTool()

print("=== Testing Robust Error Handling ===")

# Test cases
test_cases = [
    {'operation': 'multiply', 'a': 5, 'b': 3, 'description': 'Normal multiplication'},
    {'operation': 'divide', 'a': 10, 'b': 0, 'description': 'Division by zero'},
    {'operation': 'sqrt', 'a': -4, 'description': 'Square root of negative'},
    {'operation': 'unknown', 'a': 5, 'b': 3, 'description': 'Unknown operation'},
]

results = robust_calc.batch_calculate(test_cases)

for i, result in enumerate(results, 1):
    print(f"\nTest {i}: {test_cases[i-1]['description']}")
    if result.get('success'):
        print(f"  ✅ Result: {result['calculation']}")
    else:
        print(f"  ❌ Error: {result['error']}")
        print(f"  📋 Error Type: {result['error_type']}")
```

**Expected Output:**
```
=== Testing Robust Error Handling ===

Test 1: Normal multiplication
  ✅ Result: 5 multiply 3 = 15

Test 2: Division by zero
  ❌ Error: Invalid input: Cannot divide by zero
  📋 Error Type: ValueError

Test 3: Square root of negative
  ❌ Error: Invalid input: Cannot calculate square root of negative number
  📋 Error Type: ValueError

Test 4: Unknown operation
  ❌ Error: Invalid input: Unsupported operation: unknown
  📋 Error Type: ValueError
```

---

## Real-World Tool Example: API Integration Tool

Here's a realistic example of a tool that integrates with external services:

```python
import aiohttp
import asyncio
from typing import Dict, Any, Optional

class WeatherAPITool:
    """Simulated weather API tool for demonstration."""
    
    def __init__(self):
        # Simulated weather data (in real implementation, this would be actual API calls)
        self.weather_data = {
            "London": {"temp": 15, "condition": "Cloudy", "humidity": 75},
            "New York": {"temp": 22, "condition": "Sunny", "humidity": 45},
            "Tokyo": {"temp": 18, "condition": "Rainy", "humidity": 80},
            "Paris": {"temp": 19, "condition": "Partly Cloudy", "humidity": 60},
        }
    
    async def get_weather(self, city: str, units: str = "celsius") -> Dict[str, Any]:
        """Get current weather for a city."""
        try:
            # Simulate API delay
            await asyncio.sleep(0.1)
            
            city_key = city.title()
            if city_key not in self.weather_data:
                return {
                    'success': False,
                    'error': f'Weather data not available for {city}',
                    'available_cities': list(self.weather_data.keys())
                }
            
            weather = self.weather_data[city_key].copy()
            
            # Convert temperature if needed
            if units.lower() == 'fahrenheit':
                weather['temp_fahrenheit'] = round(weather['temp'] * 9/5 + 32, 1)
                weather['temp'] = weather['temp_fahrenheit']
            elif units.lower() == 'kelvin':
                weather['temp_kelvin'] = round(weather['temp'] + 273.15, 1)
                weather['temp'] = weather['temp_kelvin']
            
            return {
                'success': True,
                'city': city,
                'units': units,
                'weather': weather,
                'timestamp': '2024-01-20T12:00:00Z'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Failed to fetch weather for {city}: {str(e)}'
            }
    
    async def get_multiple_weather(self, cities: List[str], units: str = "celsius") -> Dict[str, Any]:
        """Get weather for multiple cities concurrently."""
        tasks = [self.get_weather(city, units) for city in cities]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        successful = []
        failed = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed.append({'city': cities[i], 'error': str(result)})
            elif result.get('success'):
                successful.append(result)
            else:
                failed.append(result)
        
        return {
            'successful_count': len(successful),
            'failed_count': len(failed),
            'successful': successful,
            'failed': failed,
            'summary': f"Retrieved weather for {len(successful)} cities, {len(failed)} failed"
        }

# Create weather tool
weather_tool = WeatherAPITool()

# Test weather tool
async def test_weather_tool():
    print("=== Weather API Tool Testing ===")
    
    # Single city test
    print("\n1. Single city weather:")
    weather = await weather_tool.get_weather("London", "fahrenheit")
    if weather['success']:
        w = weather['weather']
        print(f"  {weather['city']}: {w['temp']}°F, {w['condition']}, {w['humidity']}% humidity")
    else:
        print(f"  Error: {weather['error']}")
    
    # Multiple cities test
    print("\n2. Multiple cities weather:")
    cities = ["New York", "Tokyo", "Paris", "Berlin"]
    multi_weather = await weather_tool.get_multiple_weather(cities, "celsius")
    
    print(f"  Summary: {multi_weather['summary']}")
    print("  Successful cities:")
    for result in multi_weather['successful']:
        w = result['weather']
        print(f"    {result['city']}: {w['temp']}°C, {w['condition']}")
    
    if multi_weather['failed']:
        print("  Failed cities:")
        for fail in multi_weather['failed']:
            if 'error' in fail:
                print(f"    {fail['city']}: {fail['error']}")

# Run the test
asyncio.run(test_weather_tool())
```

---

## Agent That Uses External Tools

Create an agent that leverages our weather tool:

```python
from pydantic import BaseModel
from pydantic_ai import Agent
import asyncio

class TravelQuery(BaseModel):
    """Input for travel-related queries."""
    question: str = Field(description="A travel planning question")

class TravelRecommendation(BaseModel):
    """Travel recommendations based on weather."""
    destination: str
    weather_conditions: Dict[str, Any]
    recommendations: List[str]
    best_time_to_visit: str
    activities: List[str]
    packing_suggestions: List[str]

# Create travel agent with weather tool
travel_agent = Agent(
    'gemini-1.5-flash',
    result_type=TravelRecommendation,
    system_prompt='''
    You are a travel planning assistant. You have access to a weather API tool.
    
    When users ask about travel:
    1. Identify the destination city they want to visit
    2. Use the weather tool to get current conditions
    3. Based on weather, provide travel recommendations
    4. Suggest appropriate activities and packing items
    5. Give general travel advice
    
    Be practical and weather-aware in your suggestions.
    ''',
    tools=[weather_tool]
)

async def test_travel_agent():
    print("=== Travel Agent with Weather Tool ===")
    
    questions = [
        "I want to visit New York next week, what should I pack?",
        "Planning a trip to Tokyo, what activities are good for the current weather?",
        "Is London a good destination right now?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n--- Travel Query {i} ---")
        print(f"Question: {question}")
        
        try:
            result = await travel_agent.run(question)
            rec = result.data
            
            print(f"Destination: {rec.destination}")
            print(f"Weather: {rec.weather_conditions}")
            print(f"Recommendations:")
            for rec_item in rec.recommendations:
                print(f"  • {rec_item}")
            print(f"Activities: {', '.join(rec.activities)}")
            print(f"Packing: {', '.join(rec.packing_suggestions)}")
            
        except Exception as e:
            print(f"Error: {e}")

# Test the travel agent
asyncio.run(test_travel_agent())
```

---

## Tool Discovery and Documentation

Create a tool registry to help agents discover available tools:

```python
from typing import Dict, Any, List, Callable
import inspect

class ToolRegistry:
    """Registry for managing and discovering tools."""
    
    def __init__(self):
        self.tools: Dict[str, Callable] = {}
        self.descriptions: Dict[str, Dict[str, Any]] = {}
    
    def register_tool(self, name: str, func: Callable, description: str, parameters: Dict[str, Any]):
        """Register a tool with metadata."""
        self.tools[name] = func
        self.descriptions[name] = {
            'description': description,
            'parameters': parameters,
            'function_signature': str(inspect.signature(func))
        }
    
    def get_tool_info(self, name: str) -> Dict[str, Any]:
        """Get information about a specific tool."""
        if name not in self.tools:
            return {'error': f'Tool "{name}" not found'}
        
        return {
            'name': name,
            **self.descriptions[name]
        }
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List all available tools."""
        return [
            {'name': name, **self.descriptions[name]}
            for name in self.tools.keys()
        ]
    
    def find_tools_by_category(self, category: str) -> List[str]:
        """Find tools by category (simulated)."""
        # This would be more sophisticated in a real implementation
        math_tools = ['calculator', 'data_processor']
        api_tools = ['weather_api', 'sales_api']
        
        return math_tools if category == 'math' else api_tools

# Create and populate tool registry
registry = ToolRegistry()

# Register our tools
registry.register_tool(
    'calculator',
    calculator_tool.calculate,
    'Performs basic mathematical operations (add, subtract, multiply, divide, power, sqrt)',
    {
        'operation': 'str - operation type',
        'a': 'float - first number',
        'b': 'float - second number (optional for sqrt)'
    }
)

registry.register_tool(
    'weather_api',
    weather_tool.get_weather,
    'Get current weather conditions for cities worldwide',
    {
        'city': 'str - city name',
        'units': 'str - temperature units (celsius, fahrenheit, kelvin)'
    }
)

registry.register_tool(
    'data_processor',
    data_processor.calculate_stats,
    'Calculate statistics for numerical data',
    {
        'data': 'list - list of dictionaries',
        'field': 'str - field name to analyze'
    }
)

# Display tool registry
print("=== Tool Registry ===")
print("\nAll available tools:")
tools = registry.list_tools()
for tool in tools:
    print(f"\n🔧 {tool['name']}")
    print(f"   Description: {tool['description']}")
    print(f"   Signature: {tool['function_signature']}")

print(f"\n📊 Total tools registered: {len(tools)}")
```

---

## Best Practices for Custom Tools

### 1. **Tool Design Principles**

```python
class BestPracticesTool:
    """Example of tool best practices."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize with configuration."""
        self.config = config or {}
        self.timeout = self.config.get('timeout', 30)
        self.retry_count = self.config.get('retry_count', 3)
    
    def validate_inputs(self, **kwargs) -> bool:
        """Validate all inputs before processing."""
        for key, value in kwargs.items():
            if value is None:
                raise ValueError(f"Required parameter '{key}' cannot be None")
        return True
    
    def _safe_execute(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """Execute function with error handling and logging."""
        try:
            result = func(*args, **kwargs)
            return {
                'success': True,
                'result': result,
                'execution_time': 'N/A'  # Would measure in real implementation
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__
            }
    
    def cache_result(self, cache_key: str, result: Any):
        """Simple caching mechanism."""
        # In real implementation, use proper cache (Redis, etc.)
        pass
    
    def get_cache(self, cache_key: str) -> Any:
        """Retrieve cached result."""
        # In real implementation, use proper cache
        return None

# Example of well-structured tool
def create_analysis_tool() -> BestPracticesTool:
    """Create a tool following best practices."""
    config = {
        'timeout': 60,
        'retry_count': 3,
        'enable_caching': True
    }
    return BestPracticesTool(config)
```

### 2. **Tool Testing Strategy**

```python
def test_tool_thoroughly(tool: Any, test_cases: List[Dict[str, Any]]):
    """Comprehensive tool testing framework."""
    print(f"Testing {tool.__class__.__name__}")
    
    for i, test_case in enumerate(test_cases, 1):
        try:
            # Execute the tool
            result = tool.calculate(**test_case['inputs'])
            
            # Verify expected outcome
            if test_case.get('should_succeed', True):
                if result.get('success'):
                    print(f"  ✅ Test {i}: PASSED")
                else:
                    print(f"  ❌ Test {i}: FAILED - Expected success, got error: {result.get('error')}")
            else:
                if not result.get('success'):
                    print(f"  ✅ Test {i}: PASSED - Expected failure, got error")
                else:
                    print(f"  ❌ Test {i}: FAILED - Expected failure, got success")
                    
        except Exception as e:
            print(f"  ⚠️  Test {i}: EXCEPTION - {str(e)}")

# Test cases for calculator
test_cases = [
    {'inputs': {'operation': 'add', 'a': 5, 'b': 3}, 'should_succeed': True},
    {'inputs': {'operation': 'divide', 'a': 10, 'b': 0}, 'should_succeed': False},
    {'inputs': {'operation': 'unknown', 'a': 5, 'b': 3}, 'should_succeed': False},
]

print("=== Tool Testing ===")
test_tool_thoroughly(calculator_tool, test_cases)
```

---

## Summary: Mastering Custom Tools

**Key Concepts Covered:**

1. **Tool Structure**: Well-defined functions with clear inputs/outputs
2. **Agent Integration**: Tools are passed to agents via the `tools` parameter
3. **Error Handling**: Robust error handling with meaningful error messages
4. **Tool Chaining**: Combining multiple tools for complex workflows
5. **Result Modeling**: Using Pydantic models for type safety
6. **Real-world Integration**: Simulated API tools for practical examples
7. **Tool Registry**: Managing and discovering available tools
8. **Best Practices**: Testing, validation, caching, and configuration

**Real-World Applications:**
- **Financial Tools**: Currency conversion, portfolio analysis, risk calculation
- **Data Processing Tools**: Database queries, data cleaning, statistical analysis
- **Communication Tools**: Email sending, SMS, API calls
- **Business Logic Tools**: Workflow automation, decision trees, rule engines
- **Integration Tools**: CRM, ERP, external service connectors

**Next Level:** Tools that can call other agents as functions, creating meta-systems where agents can delegate work to specialized sub-agents!

---

## 🎯 Practice Exercise

Create a **File Management Tool** that can:
1. Read text files and extract specific information
2. Count words, lines, and characters
3. Search for patterns in files
4. Create summary reports

Then integrate it with an agent that answers questions about file contents!

Ready to move to **Lesson 7: Advanced Agent Patterns** where we'll explore complex agent behaviors like streaming, streaming, and agent composition? Let's build some truly sophisticated AI systems! 🚀