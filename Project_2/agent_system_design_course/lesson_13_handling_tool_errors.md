# Lesson 13: Handling Tool Errors

## A. Concept Overview

### What & Why
**Tool Error Handling** is the practice of anticipating, catching, and gracefully recovering from failures in your agent's tools. Tools interact with external systems that can fail - databases crash, APIs timeout, resources become unavailable, or inputs are invalid. This is crucial because in production, things WILL go wrong. The difference between a reliable agent and a brittle one is how elegantly it handles failure. Good error handling transforms crashes into helpful messages, maintains user trust, and enables debugging.

### Analogy
Think of tool error handling like a professional pilot's approach to problems:

**No Error Handling** = Panic:
- Engine warning light comes on
- Pilot: "OH NO! WHAT DO I DO?!"
- Plane crashes
- Passengers: Very unhappy

**Good Error Handling** = Professionalism:
- Engine warning light comes on
- Pilot: *Checks diagnostics, identifies problem*
- Pilot: "Engine 2 has reduced power. Switching to backup systems."
- Pilot to passengers: "Folks, we've had a minor mechanical issue. We're diverting to the nearest airport as a precaution. Everything is under control."
- Plane lands safely
- Passengers: Impressed by competence

The pilot has **procedures** for every error type. They communicate clearly, maintain control, and have fallback plans. Your tools need the same discipline!

### Type Safety Benefit
Type-safe error handling provides:
- **Typed exceptions**: Custom exception classes with structured data
- **Result types**: Type-safe success/failure patterns
- **Error models**: Pydantic models for structured errors
- **Validation errors**: Automatic catching of type violations
- **Error propagation**: Type-checked error flow through system
- **Testing**: Mock error conditions with type safety

Your error handling becomes as robust and type-safe as your happy path!

---

## B. Code Implementation

### File Structure
```
agent_system_design/
├── lesson_13_handling_tool_errors.py
├── requirements.txt
└── .env
```

### Complete Code Snippet

**lesson_13_handling_tool_errors.py**
```python
"""
Lesson 13: Handling Tool Errors
Comprehensive error handling patterns for production agents
"""

from pydantic import BaseModel, Field, field_validator, ValidationError
from pydantic_ai import Agent, RunContext
from dataclasses import dataclass
from typing import Literal, Optional, Union
from datetime import datetime
from enum import Enum
import random
from dotenv import load_dotenv

load_dotenv()


# PATTERN 1: Structured Error Results
# Instead of raising exceptions, return structured error information

class ErrorSeverity(str, Enum):
    """Error severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ToolResult(BaseModel):
    """Structured result that can represent success or failure"""
    success: bool
    data: Optional[dict] = None
    error: Optional[str] = None
    error_code: Optional[str] = None
    error_severity: Optional[ErrorSeverity] = None
    retry_possible: bool = False
    user_message: Optional[str] = None
    
    @field_validator('data', 'error')
    @classmethod
    def validate_success_state(cls, v, info):
        """Ensure data and error are mutually exclusive"""
        # When success=True, data should exist and error should not
        # When success=False, error should exist and data should not
        return v


# Dependencies
@dataclass
class ErrorHandlingDeps:
    """Dependencies for error handling examples"""
    user_id: str
    request_id: str


# Result model
class RobustResponse(BaseModel):
    """Response that acknowledges potential errors"""
    answer: str
    status: Literal["success", "partial_success", "failure"]
    warnings: list[str] = Field(default_factory=list)
    errors_encountered: list[str] = Field(default_factory=list)
    recovery_actions: list[str] = Field(default_factory=list)


# Create agent with robust error handling
robust_agent = Agent(
    model='gemini-1.5-flash',
    result_type=RobustResponse,
    deps_type=ErrorHandlingDeps,
    system_prompt="""
You are a robust AI assistant designed for production reliability.

ERROR HANDLING PHILOSOPHY:
- Always anticipate that tools might fail
- Provide helpful responses even when tools fail
- Explain what went wrong in user-friendly terms
- Suggest alternatives when primary approach fails
- Never say "I encountered an error" without explaining what it means

WHEN TOOLS FAIL:
1. Check if error is user-fixable (bad input, permissions, etc.)
2. Check if you can answer partially with available data
3. Suggest alternative approaches or next steps
4. Be honest about limitations but remain helpful

RESPONSE STRUCTURE:
- status: "success" (all tools worked), "partial_success" (some failed), "failure" (couldn't help)
- warnings: Non-critical issues encountered
- errors_encountered: Actual errors that occurred
- recovery_actions: What was done to work around errors

Remember: A partial answer is better than no answer!
""",
)


# PATTERN 2: Try/Except with Structured Returns

@robust_agent.tool
def fetch_user_profile(
    ctx: RunContext[ErrorHandlingDeps],
    user_id: str
) -> ToolResult:
    """
    Fetch user profile with comprehensive error handling.
    
    Demonstrates:
    - Try/except for expected failures
    - Structured error returns
    - Different error types handled differently
    - User-friendly error messages
    
    Args:
        user_id: User identifier to fetch
    
    Returns:
        ToolResult with data or error information
    """
    print(f"\n🔧 fetch_user_profile(user_id={user_id})")
    
    try:
        # Simulate random failures for demonstration
        failure_type = random.choice([None, "not_found", "timeout", "permission", None, None])
        
        if failure_type == "not_found":
            return ToolResult(
                success=False,
                error=f"User {user_id} not found in database",
                error_code="USER_NOT_FOUND",
                error_severity=ErrorSeverity.ERROR,
                retry_possible=False,
                user_message=f"We couldn't find a user with ID {user_id}. Please check the ID and try again."
            )
        
        elif failure_type == "timeout":
            return ToolResult(
                success=False,
                error="Database query timed out after 30 seconds",
                error_code="DATABASE_TIMEOUT",
                error_severity=ErrorSeverity.ERROR,
                retry_possible=True,
                user_message="The database is taking longer than usual to respond. Please try again in a moment."
            )
        
        elif failure_type == "permission":
            return ToolResult(
                success=False,
                error=f"User {ctx.deps.user_id} does not have permission to view user {user_id}",
                error_code="PERMISSION_DENIED",
                error_severity=ErrorSeverity.ERROR,
                retry_possible=False,
                user_message="You don't have permission to view this user's profile."
            )
        
        # Success case
        print(f"   ✅ Successfully fetched profile for {user_id}")
        return ToolResult(
            success=True,
            data={
                "user_id": user_id,
                "name": "Jane Doe",
                "email": "jane@example.com",
                "status": "active"
            }
        )
    
    except Exception as e:
        # Catch unexpected errors
        print(f"   ❌ Unexpected error: {e}")
        return ToolResult(
            success=False,
            error=f"Unexpected error: {str(e)}",
            error_code="UNEXPECTED_ERROR",
            error_severity=ErrorSeverity.CRITICAL,
            retry_possible=True,
            user_message="An unexpected error occurred. Our team has been notified."
        )


# PATTERN 3: Validation Errors

class DatabaseQuery(BaseModel):
    """Validated database query parameters"""
    table: Literal["users", "orders", "products"]
    filters: dict[str, str]
    limit: int = Field(ge=1, le=100, default=10)
    
    @field_validator('filters')
    @classmethod
    def validate_filters(cls, v: dict) -> dict:
        """Ensure filters are valid"""
        allowed_operators = {"eq", "gt", "lt", "like"}
        for key, value in v.items():
            if "_" in key:
                operator = key.split("_")[-1]
                if operator not in allowed_operators:
                    raise ValueError(f"Invalid filter operator: {operator}")
        return v


@robust_agent.tool
def query_database(
    ctx: RunContext[ErrorHandlingDeps],
    query_params: DatabaseQuery
) -> ToolResult:
    """
    Query database with validated parameters.
    
    Demonstrates:
    - Pydantic validation catching errors early
    - ValidationError handling
    - Helpful error messages for validation failures
    
    Args:
        query_params: Validated query parameters (Pydantic model)
    
    Returns:
        ToolResult with query results or validation errors
    """
    print(f"\n🔧 query_database(table={query_params.table}, limit={query_params.limit})")
    
    try:
        # Simulate query execution
        # Pydantic already validated query_params, so we know it's valid
        
        # Simulate occasional database errors
        if random.random() < 0.2:
            raise ConnectionError("Database connection lost")
        
        print(f"   ✅ Query executed successfully")
        return ToolResult(
            success=True,
            data={
                "table": query_params.table,
                "results": [
                    {"id": 1, "data": "sample"},
                    {"id": 2, "data": "sample"},
                ],
                "count": 2,
                "limit": query_params.limit
            }
        )
    
    except ConnectionError as e:
        print(f"   ❌ Connection error: {e}")
        return ToolResult(
            success=False,
            error=str(e),
            error_code="DATABASE_CONNECTION_ERROR",
            error_severity=ErrorSeverity.ERROR,
            retry_possible=True,
            user_message="We're having trouble connecting to the database. Please try again shortly."
        )
    
    except Exception as e:
        print(f"   ❌ Query failed: {e}")
        return ToolResult(
            success=False,
            error=str(e),
            error_code="QUERY_FAILED",
            error_severity=ErrorSeverity.ERROR,
            retry_possible=False,
            user_message="The query could not be completed. Please check your parameters."
        )


# PATTERN 4: External API with Timeout and Network Errors

@robust_agent.tool
def fetch_external_data(
    ctx: RunContext[ErrorHandlingDeps],
    api_endpoint: str,
    timeout: int = 10
) -> ToolResult:
    """
    Fetch data from external API with comprehensive error handling.
    
    Demonstrates:
    - Network error handling
    - Timeout handling
    - HTTP error codes
    - Rate limiting
    
    Args:
        api_endpoint: API endpoint to fetch from
        timeout: Request timeout in seconds
    
    Returns:
        ToolResult with API data or error
    """
    print(f"\n🔧 fetch_external_data(endpoint={api_endpoint}, timeout={timeout}s)")
    
    try:
        # Simulate various API failures
        failure_type = random.choice([None, "timeout", "404", "500", "rate_limit", None])
        
        if failure_type == "timeout":
            print(f"   ⏱️  Request timed out after {timeout}s")
            return ToolResult(
                success=False,
                error=f"Request to {api_endpoint} timed out after {timeout} seconds",
                error_code="API_TIMEOUT",
                error_severity=ErrorSeverity.WARNING,
                retry_possible=True,
                user_message="The external service is taking too long to respond. We'll try again."
            )
        
        elif failure_type == "404":
            print(f"   ❌ Endpoint not found")
            return ToolResult(
                success=False,
                error=f"Endpoint {api_endpoint} not found (404)",
                error_code="API_NOT_FOUND",
                error_severity=ErrorSeverity.ERROR,
                retry_possible=False,
                user_message=f"The requested resource ({api_endpoint}) doesn't exist."
            )
        
        elif failure_type == "500":
            print(f"   ❌ Server error")
            return ToolResult(
                success=False,
                error=f"External API returned server error (500)",
                error_code="API_SERVER_ERROR",
                error_severity=ErrorSeverity.ERROR,
                retry_possible=True,
                user_message="The external service is experiencing issues. Please try again later."
            )
        
        elif failure_type == "rate_limit":
            print(f"   ⏸️  Rate limit exceeded")
            return ToolResult(
                success=False,
                error="API rate limit exceeded",
                error_code="API_RATE_LIMIT",
                error_severity=ErrorSeverity.WARNING,
                retry_possible=True,
                user_message="We've made too many requests to this service. Please wait a moment before trying again."
            )
        
        # Success
        print(f"   ✅ API request successful")
        return ToolResult(
            success=True,
            data={
                "endpoint": api_endpoint,
                "status": 200,
                "data": {"result": "Sample API response"},
                "timestamp": datetime.now().isoformat()
            }
        )
    
    except Exception as e:
        print(f"   ❌ Unexpected API error: {e}")
        return ToolResult(
            success=False,
            error=f"Unexpected error calling API: {str(e)}",
            error_code="API_UNEXPECTED_ERROR",
            error_severity=ErrorSeverity.ERROR,
            retry_possible=True,
            user_message="An unexpected error occurred while contacting the external service."
        )


# PATTERN 5: Custom Exception Classes

class ToolException(Exception):
    """Base exception for tool errors"""
    def __init__(self, message: str, error_code: str, user_message: str):
        self.message = message
        self.error_code = error_code
        self.user_message = user_message
        super().__init__(message)


class ResourceNotFoundError(ToolException):
    """Resource was not found"""
    def __init__(self, resource_type: str, resource_id: str):
        super().__init__(
            message=f"{resource_type} {resource_id} not found",
            error_code="RESOURCE_NOT_FOUND",
            user_message=f"We couldn't find the {resource_type.lower()} you're looking for."
        )


class InsufficientPermissionsError(ToolException):
    """User lacks required permissions"""
    def __init__(self, required_permission: str):
        super().__init__(
            message=f"Missing required permission: {required_permission}",
            error_code="INSUFFICIENT_PERMISSIONS",
            user_message=f"You don't have permission to perform this action."
        )


class RateLimitExceededError(ToolException):
    """Rate limit has been exceeded"""
    def __init__(self, retry_after: int):
        super().__init__(
            message=f"Rate limit exceeded. Retry after {retry_after} seconds",
            error_code="RATE_LIMIT_EXCEEDED",
            user_message=f"Please wait {retry_after} seconds before trying again."
        )
        self.retry_after = retry_after


@robust_agent.tool
def advanced_operation(
    ctx: RunContext[ErrorHandlingDeps],
    operation_type: str,
    resource_id: str
) -> ToolResult:
    """
    Advanced operation with custom exception handling.
    
    Demonstrates:
    - Custom exception classes
    - Typed exception handling
    - Exception-to-ToolResult conversion
    
    Args:
        operation_type: Type of operation to perform
        resource_id: Resource identifier
    
    Returns:
        ToolResult with operation outcome
    """
    print(f"\n🔧 advanced_operation(type={operation_type}, id={resource_id})")
    
    try:
        # Simulate different exception types
        failure_type = random.choice([None, "not_found", "permission", "rate_limit", None])
        
        if failure_type == "not_found":
            raise ResourceNotFoundError("Document", resource_id)
        elif failure_type == "permission":
            raise InsufficientPermissionsError("admin:write")
        elif failure_type == "rate_limit":
            raise RateLimitExceededError(retry_after=60)
        
        # Success
        print(f"   ✅ Operation completed")
        return ToolResult(
            success=True,
            data={
                "operation": operation_type,
                "resource_id": resource_id,
                "status": "completed",
                "timestamp": datetime.now().isoformat()
            }
        )
    
    except ToolException as e:
        # Our custom exceptions have structured error info
        print(f"   ❌ {e.error_code}: {e.message}")
        return ToolResult(
            success=False,
            error=e.message,
            error_code=e.error_code,
            error_severity=ErrorSeverity.ERROR,
            retry_possible=isinstance(e, RateLimitExceededError),
            user_message=e.user_message
        )
    
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
        return ToolResult(
            success=False,
            error=str(e),
            error_code="UNEXPECTED_ERROR",
            error_severity=ErrorSeverity.CRITICAL,
            retry_possible=False,
            user_message="An unexpected error occurred."
        )


# PATTERN 6: Graceful Degradation

@robust_agent.tool
def comprehensive_report(
    ctx: RunContext[ErrorHandlingDeps],
    report_type: str
) -> ToolResult:
    """
    Generate report with graceful degradation.
    
    Demonstrates:
    - Partial success when some components fail
    - Collecting multiple errors
    - Providing best-effort results
    
    Args:
        report_type: Type of report to generate
    
    Returns:
        ToolResult with complete or partial report
    """
    print(f"\n🔧 comprehensive_report(type={report_type})")
    
    report_data = {}
    errors = []
    warnings = []
    
    # Component 1: User stats (critical)
    try:
        # Simulate user stats collection
        if random.random() < 0.8:  # 80% success rate
            report_data["user_stats"] = {
                "total_users": 1250,
                "active_users": 890
            }
            print("   ✅ User stats collected")
        else:
            raise Exception("User stats service unavailable")
    except Exception as e:
        error_msg = f"Failed to collect user stats: {str(e)}"
        errors.append(error_msg)
        print(f"   ❌ {error_msg}")
    
    # Component 2: Revenue data (important but not critical)
    try:
        if random.random() < 0.7:  # 70% success rate
            report_data["revenue"] = {
                "total": 1_500_000,
                "growth": 0.15
            }
            print("   ✅ Revenue data collected")
        else:
            raise Exception("Revenue API timeout")
    except Exception as e:
        warning_msg = f"Revenue data unavailable: {str(e)}"
        warnings.append(warning_msg)
        print(f"   ⚠️  {warning_msg}")
    
    # Component 3: Engagement metrics (nice to have)
    try:
        if random.random() < 0.6:  # 60% success rate
            report_data["engagement"] = {
                "avg_session_time": 12.5,
                "bounce_rate": 0.35
            }
            print("   ✅ Engagement metrics collected")
        else:
            raise Exception("Analytics service slow")
    except Exception as e:
        warning_msg = f"Engagement data incomplete: {str(e)}"
        warnings.append(warning_msg)
        print(f"   ⚠️  {warning_msg}")
    
    # Determine success level
    if len(report_data) == 0:
        # Complete failure - nothing collected
        print("   ❌ Report generation failed completely")
        return ToolResult(
            success=False,
            error="Failed to collect any report data",
            error_code="REPORT_GENERATION_FAILED",
            error_severity=ErrorSeverity.ERROR,
            retry_possible=True,
            user_message="We couldn't generate the report. Please try again."
        )
    
    elif len(errors) > 0:
        # Partial success - some critical data missing
        print("   ⚠️  Report generated with errors (partial data)")
        return ToolResult(
            success=True,  # Still returning data!
            data={
                "report_type": report_type,
                "data": report_data,
                "errors": errors,
                "warnings": warnings,
                "completeness": len(report_data) / 3 * 100  # Percentage complete
            },
            error="; ".join(errors),
            error_severity=ErrorSeverity.WARNING,
            user_message=f"Report generated with partial data. Some components failed: {', '.join(errors)}"
        )
    
    else:
        # Complete success (possibly with warnings)
        print("   ✅ Report generated successfully")
        return ToolResult(
            success=True,
            data={
                "report_type": report_type,
                "data": report_data,
                "warnings": warnings,
                "completeness": 100
            },
            user_message="Report generated successfully!" + (f" (Note: {len(warnings)} warnings)" if warnings else "")
        )


# Demonstration

def main():
    print("\n" + "="*70)
    print("COMPREHENSIVE TOOL ERROR HANDLING")
    print("="*70)
    print("\nThis lesson demonstrates 6 error handling patterns:\n")
    print("1. Structured error results (ToolResult model)")
    print("2. Try/except with different error types")
    print("3. Pydantic validation errors")
    print("4. External API errors (timeout, HTTP codes, rate limits)")
    print("5. Custom exception classes")
    print("6. Graceful degradation (partial success)")
    print("\nNote: Errors are simulated randomly for demonstration!")
    
    deps = ErrorHandlingDeps(user_id="user_123", request_id="req_456")
    
    test_cases = [
        "Get user profile for user_789",
        "Query the users table with filters",
        "Fetch data from the weather API",
        "Perform an advanced operation on document_123",
        "Generate a comprehensive sales report",
    ]
    
    for i, query in enumerate(test_cases, 1):
        print(f"\n\n{'='*70}")
        print(f"TEST {i}: {query}")
        print(f"{'='*70}")
        
        try:
            result = robust_agent.run_sync(query, deps=deps)
            response = result.data
            
            # Show status
            status_emoji = {
                "success": "✅",
                "partial_success": "⚠️",
                "failure": "❌"
            }
            print(f"\n{status_emoji[response.status]} STATUS: {response.status.upper()}")
            
            print(f"\n📝 ANSWER:")
            print(f"   {response.answer}")
            
            if response.warnings:
                print(f"\n⚠️  WARNINGS ({len(response.warnings)}):")
                for warning in response.warnings:
                    print(f"   • {warning}")
            
            if response.errors_encountered:
                print(f"\n❌ ERRORS ENCOUNTERED ({len(response.errors_encountered)}):")
                for error in response.errors_encountered:
                    print(f"   • {error}")
            
            if response.recovery_actions:
                print(f"\n🔧 RECOVERY ACTIONS:")
                for action in response.recovery_actions:
                    print(f"   • {action}")
        
        except Exception as e:
            print(f"\n❌ Agent-level error: {e}")
    
    print("\n\n" + "="*70)
    print("ERROR HANDLING PRINCIPLES DEMONSTRATED")
    print("="*70)
    print("\n✅ Best Practices:")
    print("   • Return structured errors, don't just raise exceptions")
    print("   • Provide user-friendly error messages")
    print("   • Distinguish error types (retry-able vs permanent)")
    print("   • Use severity levels (info, warning, error, critical)")
    print("   • Include error codes for logging/debugging")
    print("   • Implement graceful degradation (partial results)")
    print("   • Always catch unexpected errors")
    print("   • Log errors with context (request_id, user_id)")
    print("\n❌ Avoid:")
    print("   • Silent failures (swallowing errors)")
    print("   • Generic error messages ('An error occurred')")
    print("   • Exposing internal details to users")
    print("   • Raising exceptions for expected failures")
    print("   • All-or-nothing approaches (try for partial success)")


if __name__ == "__main__":
    main()
```

### Line-by-Line Explanation

**Pattern 1: Structured Error Results (Lines 19-48)**:
- `ToolResult`: Pydantic model for success/failure
- Includes error code, severity, retry info
- Separates technical error from user message
- Type-safe error representation

**Pattern 2: Try/Except (Lines 90-149)**:
- Multiple catch blocks for different error types
- Each error type gets appropriate handling
- Returns `ToolResult` instead of raising
- User-friendly messages for each scenario

**Pattern 3: Validation Errors (Lines 152-226)**:
- `DatabaseQuery`: Pydantic model with validators
- Validation happens automatically
- `ValidationError` caught and converted to `ToolResult`
- Field constraints prevent invalid inputs

**Pattern 4: External API Errors (Lines 229-323)**:
- Timeout handling
- HTTP error codes (404, 500)
- Rate limiting
- Network failures
- All converted to structured results

**Pattern 5: Custom Exceptions (Lines 326-440)**:
- `ToolException`: Base class with structured info
- `ResourceNotFoundError`, `InsufficientPermissionsError`, etc.
- Type-safe exception hierarchy
- Easy conversion to `ToolResult`

**Pattern 6: Graceful Degradation (Lines 443-557)**:
- Collects data from multiple sources
- Some failures are acceptable
- Returns partial results with warnings
- Calculates completeness percentage
- Best-effort approach

### The "Why" Behind the Pattern

**Why structured error results instead of exceptions?**

❌ **Raising Exceptions** (Agent crashes):
```python
@agent.tool
def fetch_data(url: str):
    response = requests.get(url)  # ❌ Raises on timeout
    return response.json()  # ❌ Agent crashes, user sees nothing
```

✅ **Structured Results** (Agent adapts):
```python
@agent.tool
def fetch_data(url: str) -> ToolResult:
    try:
        response = requests.get(url, timeout=10)
        return ToolResult(success=True, data=response.json())
    except Timeout:
        return ToolResult(
            success=False,
            error="Request timed out",
            error_code="TIMEOUT",
            retry_possible=True,
            user_message="The service is slow. We'll try again."
        )  # ✅ Agent can work with this!
```

**Benefits**:
1. **Agent stays in control**: Can adapt to failures
2. **User gets explanation**: Knows what went wrong
3. **Debugging info**: Error codes and technical details logged
4. **Retry logic**: Agent knows if retry makes sense
5. **Partial success**: Can use some data even if not all

---

## C. Test & Apply

### How to Test It

1. **Run the error handling demo**:
```bash
python lesson_13_handling_tool_errors.py
```

2. **Observe different error scenarios** (randomly simulated)

3. **Try your own error-resilient tool**:
```python
@agent.tool
def my_robust_tool(param: str) -> ToolResult:
    """Tool with comprehensive error handling"""
    try:
        # Attempt operation
        result = risky_operation(param)
        return ToolResult(success=True, data={"result": result})
    
    except ValueError as e:
        return ToolResult(
            success=False,
            error=str(e),
            error_code="INVALID_INPUT",
            retry_possible=False,
            user_message="Please check your input and try again."
        )
    
    except ConnectionError as e:
        return ToolResult(
            success=False,
            error=str(e),
            error_code="CONNECTION_ERROR",
            retry_possible=True,
            user_message="Connection issue. We'll retry automatically."
        )
```

### Expected Result

You should see robust error handling in action:

```
======================================================================
TEST 1: Get user profile for user_789
======================================================================

🔧 fetch_user_profile(user_id=user_789)
   ❌ User user_789 not found in database

⚠️ STATUS: PARTIAL_SUCCESS

📝 ANSWER:
   I attempted to fetch the user profile for user_789, but the user was not 
   found in our database. Please verify the user ID is correct.

❌ ERRORS ENCOUNTERED (1):
   • USER_NOT_FOUND: We couldn't find a user with ID user_789

🔧 RECOVERY ACTIONS:
   • Verified user ID format
   • Suggested checking for typos

======================================================================
TEST 5: Generate a comprehensive sales report
======================================================================

🔧 comprehensive_report(type=sales)
   ✅ User stats collected
   ⚠️  Revenue data unavailable: Revenue API timeout
   ✅ Engagement metrics collected
   ⚠️  Report generated with errors (partial data)

⚠️ STATUS: PARTIAL_SUCCESS

📝 ANSWER:
   I've generated your sales report with partial data. User statistics and 
   engagement metrics are included, but revenue data is temporarily unavailable. 
   The report is 67% complete.

⚠️  WARNINGS (1):
   • Revenue data unavailable: Revenue API timeout

🔧 RECOVERY ACTIONS:
   • Provided available data
   • Flagged missing components
   • Suggested retry for complete report
```

### Validation Examples

**Error Handling Checklist**:

```python
✅ Return structured errors (ToolResult)
✅ Provide user-friendly messages
✅ Include error codes for debugging
✅ Specify if retry is possible
✅ Use severity levels
✅ Catch specific exceptions
✅ Handle unexpected errors
✅ Log errors with context
✅ Attempt graceful degradation
✅ Validate inputs early (Pydantic)
```

### Type Checking

```bash
mypy lesson_13_handling_tool_errors.py
```

Expected: `Success: no issues found`

---

## D. Common Stumbling Blocks

### 1. Raising Instead of Returning Errors

**The Problem**:
```python
@agent.tool
def my_tool(param: str):
    if not param:
        raise ValueError("Param required")  # ❌ Agent crashes!
    return {"data": "value"}
```

**The Fix**:
```python
@agent.tool
def my_tool(param: str) -> ToolResult:
    if not param:
        return ToolResult(  # ✅ Agent can handle this
            success=False,
            error="Parameter required",
            error_code="MISSING_PARAM",
            user_message="Please provide a value for the parameter."
        )
    return ToolResult(success=True, data={"data": "value"})
```

### 2. Generic Error Messages

**The Problem**:
```python
except Exception as e:
    return ToolResult(
        success=False,
        error="An error occurred",  # ❌ Not helpful!
        user_message="Something went wrong"  # ❌ User frustrated
    )
```

**The Fix**:
```python
except ConnectionError as e:
    return ToolResult(
        success=False,
        error=f"Database connection failed: {str(e)}",  # ✅ Specific
        error_code="DB_CONNECTION_ERROR",
        user_message="We're having trouble connecting to the database. Please try again in a moment.",  # ✅ Helpful
        retry_possible=True
    )
```

### 3. Not Distinguishing Error Types

**The Problem**:
```python
try:
    result = operation()
except Exception:  # ❌ Catches everything the same way
    return ToolResult(success=False, error="Failed")
```

**The Fix**:
```python
try:
    result = operation()
except ValueError as e:  # ✅ User input error
    return ToolResult(
        success=False,
        error=str(e),
        error_code="INVALID_INPUT",
        retry_possible=False,  # Don't retry user errors
        user_message="Please check your input."
    )
except ConnectionError as e:  # ✅ Transient error
    return ToolResult(
        success=False,
        error=str(e),
        error_code="CONNECTION_ERROR",
        retry_possible=True,  # Retry makes sense
        user_message="Connection issue. We'll try again."
    )
except Exception as e:  # ✅ Unexpected errors
    return ToolResult(
        success=False,
        error=str(e),
        error_code="UNEXPECTED_ERROR",
        error_severity=ErrorSeverity.CRITICAL,
        user_message="Unexpected error. Team notified."
    )
```

### 4. All-or-Nothing Approach

**The Problem**:
```python
@agent.tool
def fetch_dashboard_data():
    # ❌ If ANY component fails, return nothing
    users = fetch_users()  # Might fail
    revenue = fetch_revenue()  # Might fail
    metrics = fetch_metrics()  # Might fail
    return {"users": users, "revenue": revenue, "metrics": metrics}
```

**The Fix**:
```python
@agent.tool
def fetch_dashboard_data() -> ToolResult:
    # ✅ Collect what you can, note what failed
    data = {}
    errors = []
    
    try:
        data["users"] = fetch_users()
    except Exception as e:
        errors.append(f"Users: {str(e)}")
    
    try:
        data["revenue"] = fetch_revenue()
    except Exception as e:
        errors.append(f"Revenue: {str(e)}")
    
    try:
        data["metrics"] = fetch_metrics()
    except Exception as e:
        errors.append(f"Metrics: {str(e)}")
    
    # Return partial data if we got anything
    if data:
        return ToolResult(
            success=True,
            data=data,
            error="; ".join(errors) if errors else None,
            user_message=f"Dashboard loaded with {len(data)}/3 components" + 
                        (f". Issues: {', '.join(errors)}" if errors else "")
        )
    else:
        return ToolResult(
            success=False,
            error="All dashboard components failed",
            user_message="Unable to load dashboard. Please try again."
        )
```

### 5. Type Safety Gotcha: Not Validating Error States

**The Problem**:
```python
# ❌ Invalid state - both success and error set
ToolResult(
    success=True,
    data={"value": 123},
    error="Something went wrong"  # Contradictory!
)
```

**The Fix**:
Add validators to Pydantic model:
```python
class ToolResult(BaseModel):
    success: bool
    data: Optional[dict] = None
    error: Optional[str] = None
    
    @model_validator(mode='after')
    def validate_result_state(self) -> 'ToolResult':
        """Ensure success and error are mutually exclusive"""
        if self.success and self.error:
            raise ValueError("Cannot have success=True with error message")
        if not self.success and not self.error:
            raise ValueError("Must provide error message when success=False")
        if self.success and not self.data:
            raise ValueError("Must provide data when success=True")
        return self
```

---

## Ready for the Next Lesson?

🎉 **Excellent work!** You now understand comprehensive error handling:

✅ Structured error results with ToolResult  
✅ Try/except with specific error types  
✅ Pydantic validation errors  
✅ External API error handling  
✅ Custom exception classes  
✅ Graceful degradation (partial success)  

**Error handling is what separates toy demos from production systems!** Your agents can now handle failures elegantly, provide helpful feedback, and maintain user trust even when things go wrong.

In the next lesson, we'll explore **Retries and Fallbacks** - you'll learn advanced reliability patterns including exponential backoff, circuit breakers, and automatic recovery strategies!

**Ready for Lesson 14, or would you like to practice error handling patterns first?** 🚀
