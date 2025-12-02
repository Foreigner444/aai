# Lesson 1: Understanding Agent Architecture

## A. Concept Overview

### What & Why
**Agent Architecture** is the foundation of Pydantic AI. An agent is a reusable, type-safe container that wraps an AI model (like Gemini), defines what structured outputs it should return, and can be equipped with tools and dependencies. This is crucial because it separates your AI logic from your application code, making it testable, reusable, and type-safe.

### Analogy
Think of a Pydantic AI Agent like a **specialized employee in a company**:
- The **Agent** is the employee with a specific job role
- The **Model** (Gemini) is their brain/intelligence
- The **Result Type** (Pydantic model) is the exact format of reports they must submit
- **Tools** are the resources/systems they can access to do their job
- **Dependencies** are the context/information they need to start working (customer ID, database connection, etc.)
- **System Prompts** are their job description and training manual

Just like you wouldn't hire someone without defining their role, you don't create an agent without defining its structure!

### Type Safety Benefit
Agent architecture provides type safety at every level:
- **Input validation**: Your prompts and parameters are type-checked
- **Output validation**: Gemini's responses are automatically validated against your Pydantic models
- **Tool safety**: Tool functions have typed parameters and returns
- **Dependency injection**: Context is passed in a type-safe way
- **IDE support**: Full autocomplete for agent methods and results

Before your code even runs, Python's type system (with mypy) catches 90% of potential bugs!

---

## B. Code Implementation

### File Structure
```
agent_system_design/
├── lesson_01_basic_agent.py
└── requirements.txt
```

### Complete Code Snippet

**requirements.txt**
```txt
pydantic-ai==0.0.14
pydantic==2.9.2
google-generativeai==0.8.3
python-dotenv==1.0.0
```

**lesson_01_basic_agent.py**
```python
"""
Lesson 1: Understanding Agent Architecture
A simple agent that returns structured, validated outputs
"""

from pydantic import BaseModel, Field
from pydantic_ai import Agent
import os
from dotenv import load_dotenv

# Load environment variables (for API keys)
load_dotenv()

# Step 1: Define the structure of what the agent will return
# This is a Pydantic model - it defines the exact shape and types
class MovieRecommendation(BaseModel):
    """The structured output our agent will return"""
    title: str = Field(description="The movie title")
    year: int = Field(description="Release year", ge=1900, le=2025)
    genre: str = Field(description="Primary genre")
    rating: float = Field(description="Rating out of 10", ge=0.0, le=10.0)
    reason: str = Field(description="Why this movie is recommended")


# Step 2: Create the Agent
# This is the core of Pydantic AI - an Agent wraps a model and defines its output
agent = Agent(
    model='gemini-1.5-flash',  # Using Gemini Flash - fast and cost-effective
    result_type=MovieRecommendation,  # Agent MUST return this type
    system_prompt=(
        "You are a movie recommendation expert. "
        "Recommend movies based on user preferences. "
        "Always provide accurate release years and ratings."
    ),
)


# Step 3: Run the agent
def get_movie_recommendation(user_preference: str) -> MovieRecommendation:
    """
    Get a type-safe movie recommendation from the agent
    
    Args:
        user_preference: What the user is looking for in a movie
        
    Returns:
        MovieRecommendation: A validated movie recommendation
    """
    # run_sync() executes the agent and returns validated result
    result = agent.run_sync(user_preference)
    
    # result.data is guaranteed to be a MovieRecommendation
    # If Gemini returns invalid data, Pydantic raises ValidationError
    return result.data


# Step 4: Test the agent
if __name__ == "__main__":
    # Test with different preferences
    preferences = [
        "I want a mind-bending sci-fi movie",
        "Recommend a feel-good comedy from the 90s",
        "I love psychological thrillers"
    ]
    
    for preference in preferences:
        print(f"\n{'='*60}")
        print(f"Request: {preference}")
        print(f"{'='*60}")
        
        try:
            recommendation = get_movie_recommendation(preference)
            
            # Because it's a Pydantic model, we get:
            # 1. Type-safe attribute access (IDE autocomplete works!)
            # 2. Guaranteed valid data (year between 1900-2025, rating 0-10)
            # 3. Clean, structured output
            
            print(f"🎬 {recommendation.title} ({recommendation.year})")
            print(f"📁 Genre: {recommendation.genre}")
            print(f"⭐ Rating: {recommendation.rating}/10")
            print(f"💡 Why: {recommendation.reason}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
```

### Line-by-Line Explanation

**The Pydantic Model (Lines 15-21)**:
- `MovieRecommendation` defines the exact structure of outputs
- `Field()` adds constraints: `ge=1900` means "greater than or equal to 1900"
- These constraints are automatically validated - Gemini can't return invalid data
- Descriptions help both developers (documentation) and the model (context)

**The Agent (Lines 24-32)**:
- `Agent()` is the core Pydantic AI class
- `model='gemini-1.5-flash'` specifies which Gemini model to use
- `result_type=MovieRecommendation` tells Pydantic AI to validate outputs against this model
- `system_prompt` is the agent's instructions - like a job description

**Running the Agent (Lines 35-47)**:
- `agent.run_sync()` sends the prompt to Gemini and returns a validated result
- The result is wrapped in a `RunResult` object
- `result.data` contains the validated Pydantic model
- If validation fails, `ValidationError` is raised automatically

**Type Safety in Action (Lines 63-67)**:
- Your IDE knows `recommendation.title` is a `str`
- `recommendation.year` is guaranteed to be between 1900-2025
- `recommendation.rating` is guaranteed to be between 0.0-10.0
- No manual validation needed - Pydantic does it all!

### The "Why" Behind the Pattern

This architecture provides several critical benefits:

1. **Separation of Concerns**: Agent definition is separate from usage
2. **Reusability**: Create the agent once, use it everywhere
3. **Type Safety**: Catch errors at validation time, not in production
4. **Testability**: Easy to mock and test (we'll cover this in later lessons)
5. **Model Agnostic**: Change from Gemini to another model with one line
6. **Automatic Validation**: No manual parsing or validation code needed

---

## C. Test & Apply

### How to Test It

1. **Set up your environment**:
```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

2. **Set up your Gemini API key**:
```bash
# Create a .env file
echo "GOOGLE_API_KEY=your_api_key_here" > .env
```

Get your free API key from: https://ai.google.dev/

3. **Run the agent**:
```bash
python lesson_01_basic_agent.py
```

### Expected Result

You should see output like this:

```
============================================================
Request: I want a mind-bending sci-fi movie
============================================================
🎬 Inception (2010)
📁 Genre: Science Fiction
⭐ Rating: 8.8/10
💡 Why: Inception features complex dream-within-a-dream layers that will keep you guessing and discussing the ending long after the credits roll.

============================================================
Request: Recommend a feel-good comedy from the 90s
============================================================
🎬 The Princess Bride (1987)
📁 Genre: Comedy
⭐ Rating: 8.0/10
💡 Why: A timeless fairy tale adventure with quotable dialogue, charming characters, and the perfect blend of humor, romance, and adventure.
```

### Validation Examples

**Valid Output** (passes validation):
```python
{
    "title": "Inception",
    "year": 2010,
    "genre": "Science Fiction",
    "rating": 8.8,
    "reason": "Complex dream layers..."
}
```

**Invalid Output** (would raise ValidationError):
```python
{
    "title": "Inception",
    "year": 2050,  # ❌ Future year - exceeds le=2025 constraint
    "genre": "Science Fiction",
    "rating": 15.0,  # ❌ Rating too high - exceeds le=10.0 constraint
    "reason": "Complex dream layers..."
}
```

If Gemini somehow returns this, you'd see:
```
ValidationError: 2 validation errors for MovieRecommendation
year
  Input should be less than or equal to 2025 [type=less_than_equal]
rating
  Input should be less than or equal to 10 [type=less_than_equal]
```

### Type Checking

Add a `mypy` check to verify type safety:

```bash
pip install mypy
mypy lesson_01_basic_agent.py
```

Expected output: `Success: no issues found in 1 source file`

Try adding an error to see type checking in action:

```python
# This will fail type checking:
recommendation = get_movie_recommendation("sci-fi")
print(recommendation.title.upper())  # ✅ OK - title is str
print(recommendation.year + "test")  # ❌ mypy error - can't add str to int
```

---

## D. Common Stumbling Blocks

### 1. Missing API Key

**The Error**:
```
google.api_core.exceptions.Unauthenticated: 401 API key not valid
```

**What Causes It**:
You haven't set up your Google Gemini API key or it's incorrect.

**The Fix**:
1. Get your API key from https://ai.google.dev/
2. Create a `.env` file in your project root:
```bash
GOOGLE_API_KEY=your_actual_api_key_here
```
3. Make sure you're calling `load_dotenv()` before creating the agent
4. Or set it as an environment variable:
```bash
export GOOGLE_API_KEY=your_api_key_here
```

### 2. Validation Error on Agent Output

**The Error**:
```
pydantic_core._pydantic_core.ValidationError: 1 validation error for MovieRecommendation
year
  Input should be a valid integer [type=int_type]
```

**What Causes It**:
Gemini returned data that doesn't match your Pydantic model. This could happen if:
- Your system prompt isn't clear enough
- The model returned a string instead of an integer
- Field constraints are too strict

**The Fix**:
1. **Check your system prompt** - be explicit about output requirements
2. **Inspect the actual output**:
```python
result = agent.run_sync(user_preference)
print(result.data)  # See what Gemini actually returned
```
3. **Adjust your model constraints** if they're too restrictive
4. **Add better field descriptions** to guide the model:
```python
year: int = Field(
    description="Release year as a 4-digit integer (e.g., 2010)"
)
```

### 3. Type Safety Gotcha: result vs result.data

**Common Mistake**:
```python
result = agent.run_sync("recommend a movie")
print(result.title)  # ❌ AttributeError: 'RunResult' has no attribute 'title'
```

**What's Happening**:
`agent.run_sync()` returns a `RunResult` wrapper object, not the model directly.

**The Fix**:
Always use `.data` to access the validated model:
```python
result = agent.run_sync("recommend a movie")
print(result.data.title)  # ✅ Correct

# Or extract it immediately:
recommendation: MovieRecommendation = agent.run_sync("recommend a movie").data
print(recommendation.title)  # ✅ Also correct
```

### 4. Type Hint Confusion

**Common Mistake**:
```python
def get_movie_recommendation(user_preference: str):  # ❌ No return type
    return agent.run_sync(user_preference).data
```

**Type Safety Issue**:
Without a return type hint, your IDE and mypy can't help you catch errors.

**The Fix**:
Always add return type hints:
```python
def get_movie_recommendation(user_preference: str) -> MovieRecommendation:  # ✅
    return agent.run_sync(user_preference).data
```

Now your IDE will:
- Autocomplete `recommendation.title`, `recommendation.year`, etc.
- Warn you if you try to use the result incorrectly
- Show you the model's documentation on hover

---

## Ready for the Next Lesson?

🎉 **Congratulations!** You've just learned the foundation of Pydantic AI agent architecture! You now understand:

✅ How agents wrap models and define structured outputs  
✅ Why Pydantic models provide automatic validation  
✅ How to create a basic agent with Gemini  
✅ The importance of type hints for safety and IDE support  

In the next lesson, we'll dive into **System Prompts and Instructions** - you'll learn how to craft effective prompts that guide Gemini to produce exactly the outputs you need, every time.

**Does this make sense? Let me know if you'd like me to explain anything in a different way, or if you're ready for Lesson 2!** 🚀
