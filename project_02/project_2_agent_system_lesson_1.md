# Project 2: Agent System Design - Lesson 1: Understanding Agent Architecture

## A. Concept Overview

**What & Why:** An Agent in Pydantic AI is like a specialized AI worker that follows strict rules. Think of it as a highly trained expert who can only work within specific guidelines, ensuring everything it produces meets your exact requirements.

**Analogy:** Imagine a restaurant with specialized chefs:
- **Agent** = A sushi chef who can ONLY make sushi (not pasta, not dessert)
- **System Prompt** = The restaurant's menu and cooking instructions
- **Tools** = The chef's equipment (knives, rice cooker, fish slicer)
- **Dependencies** = The pantry, inventory system, and other support staff
- **Result Model** = The exact plating requirements for each dish

**Type Safety Benefit:** Just like the sushi chef can't accidentally serve you pizza (they only know sushi recipes), your Agent can't produce outputs that don't match your Pydantic model. The AI is "type-safe" - it can only give you exactly what you defined.

## B. Code Implementation

### File Structure
```
agent_design_basics/
├── main.py                 # Main agent setup and execution
├── models.py              # Pydantic models for inputs/outputs
├── agent.py              # Agent definition
└── requirements.txt       # Dependencies
```

### Complete Code Implementation

**File: models.py**
```python
from pydantic import BaseModel, Field
from typing import Optional, Literal
from enum import Enum

class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

class TaskInput(BaseModel):
    """Input model for our agent"""
    task_description: str = Field(..., description="What task needs to be done")
    priority: Literal["low", "medium", "high"] = "medium"
    deadline_days: Optional[int] = Field(None, ge=0, le=365)

class TaskOutput(BaseModel):
    """Output model for our agent"""
    task_id: str = Field(..., description="Unique identifier for the task")
    status: TaskStatus = Field(..., description="Current task status")
    estimated_duration_hours: int = Field(..., ge=1, le=40)
    required_resources: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)
```

**File: agent.py**
```python
from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
from models import TaskInput, TaskOutput

# Create the Gemini model (you'll need your API key)
model = GeminiModel('gemini-flash')

# Create the Agent with our structure
task_planning_agent = Agent(
    model=model,
    result_type=TaskOutput,  # This enforces type safety!
    system_prompt='''
    You are a professional project management assistant. Your job is to analyze 
    task descriptions and create detailed planning with realistic estimates.
    
    Always provide:
    - A unique task ID
    - Realistic duration estimates (1-40 hours)
    - Practical next steps
    - Required resources as a list
    
    Be practical and professional in your responses.
    '''
)

# Add dependency for context (optional but powerful)
def get_current_date():
    from datetime import datetime
    return datetime.now().isoformat()

# Configure agent with dependencies
task_planning_agent.deps = {
    'current_date': get_current_date
}
```

**File: main.py**
```python
from agent import task_planning_agent
from models import TaskInput

def main():
    # Create a task to analyze
    task_input = TaskInput(
        task_description="Build a user login system with email verification",
        priority="high",
        deadline_days=30
    )
    
    print("🤖 Planning your task with our AI agent...")
    print(f"📝 Task: {task_input.task_description}")
    print(f"⏰ Priority: {task_input.priority}")
    print(f"🗓️  Deadline: {task_input.deadline_days} days")
    print("\n" + "="*50)
    
    # Run the agent (synchronous version for simplicity)
    result = task_planning_agent.run_sync(task_input.model_dump_json())
    
    print(f"✅ Task ID: {result.data.task_id}")
    print(f"📊 Status: {result.data.status}")
    print(f"⏱️  Estimated Duration: {result.data.estimated_duration_hours} hours")
    print(f"🎯 Required Resources: {', '.join(result.data.required_resources)}")
    print(f"📋 Next Steps:")
    for i, step in enumerate(result.data.next_steps, 1):
        print(f"   {i}. {step}")

if __name__ == "__main__":
    main()
```

**File: requirements.txt**
```
pydantic-ai>=0.0.8
google-generativeai>=0.3.0
pydantic>=2.0.0
python-dateutil>=2.8.0
```

### Line-by-Line Explanation

1. **Pydantic Models (models.py):** Define the exact structure of what goes in and comes out
2. **Agent Creation (agent.py):** The core AI worker that follows strict rules
3. **System Prompt:** Instructions that guide how the AI thinks and responds
4. **Dependencies:** Additional context or tools the agent can use
5. **Result Type:** Guarantees the output matches our Pydantic model

### The "Why" Behind the Pattern

This architecture ensures **type safety** because:
- The Agent **can only** return TaskOutput objects
- If Gemini tries to return something else, Pydantic raises a ValidationError
- Your code never has to deal with unexpected formats
- You get IDE autocomplete and type checking everywhere

## C. Test & Apply

### How to Test It
```bash
# Set your Gemini API key
export GEMINI_API_KEY="your-api-key-here"

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

### Expected Result
You should see structured output like:
```
🤖 Planning your task with our AI agent...
📝 Task: Build a user login system with email verification
⏰ Priority: high
🗓️  Deadline: 30 days

==================================================
✅ Task ID: LOGIN-VERIFICATION-001
📊 Status: pending
⏱️  Estimated Duration: 16 hours
🎯 Required Resources: Database, Email service, Frontend framework
📋 Next Steps:
   1. Design database schema for user accounts
   2. Set up email service integration
   3. Create frontend login components
   4. Implement email verification workflow
```

### Validation Examples
- ✅ **Valid Input:** "Create a dashboard with user analytics" → Returns proper TaskOutput
- ❌ **Invalid Output:** If Gemini tries to return `{"task_name": "..."}` instead of our model → ValidationError
- ❌ **Type Error:** If Gemini returns `"16"` instead of `16` for duration → ValidationError

### Type Checking
```bash
# Install mypy for static type checking
pip install mypy

# Run type checking
mypy main.py agent.py models.py
```

## D. Common Stumbling Blocks

### Proactive Debugging
**Common Mistake #1: API Key Issues**
```
❌ Error: "API key not found" or "Authentication failed"
✅ Fix: Set GEMINI_API_KEY environment variable before running

# In your terminal:
export GEMINI_API_KEY="your-real-api-key-here"
python main.py
```

**Common Mistake #2: ValidationError from Pydantic**
```
❌ Error: ValidationError: 1 validation error for TaskOutput
     estimated_duration_hours
     Input should be a valid integer [type=int_type]
✅ Fix: Check your system prompt - be more specific about output format
```

### Type Safety Gotchas
- **Never trust the AI to guess your model structure** - always be explicit
- **Use Field() descriptions** - these help Gemini understand requirements
- **Test edge cases** - try invalid inputs to see if validation catches them
- **Use Literal types** for exact values (like our priority field)

## Ready for the Next Step?
You now understand the core architecture! The Agent is your type-safe AI worker that can only produce outputs matching your Pydantic models.

**Next:** We'll learn about System Prompts and Instructions - the "brain" of your agent that controls how it thinks and behaves.

Ready to dive into system prompts, or would you like to practice with this agent first? 🤔
