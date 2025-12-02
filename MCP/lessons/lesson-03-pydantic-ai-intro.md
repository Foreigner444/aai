# Lesson 3: Pydantic AI Introduction

## What is Pydantic AI?

**Pydantic AI** is a Python framework for building production-ready AI applications. It's built on top of Pydantic (the popular data validation library) and provides a clean, type-safe way to work with language models.

### Why Pydantic AI?

**Traditional AI code**:
```python
response = model.generate("What's the weather?")
data = json.loads(response)
temp = data.get("temperature")
if temp is None:
    print("Error!")
```

Problems:
- No type safety
- Manual JSON parsing
- No validation
- Hard to test

**Pydantic AI code**:
```python
from pydantic import BaseModel
from pydantic_ai import Agent

class WeatherResponse(BaseModel):
    temperature: float
    condition: str
    humidity: int

agent = Agent('gemini-1.5-flash', result_type=WeatherResponse)
result = await agent.run("What's the weather in London?")
result.data.temperature
```

Benefits:
- ✅ Type-safe
- ✅ Automatic validation
- ✅ IDE autocomplete
- ✅ Easy to test

## Installation

```bash
pip install pydantic-ai google-generativeai
```

You'll need:
- `pydantic-ai`: The framework
- `google-generativeai`: Gemini model support

## Your First Agent

Let's build a simple agent step by step.

### Step 1: Basic Agent

```python
from pydantic_ai import Agent

agent = Agent('gemini-1.5-flash')

result = await agent.run("Tell me a joke about Python programming")
print(result.data)
```

**Output**: A string with the joke.

### Step 2: Structured Output

```python
from pydantic import BaseModel
from pydantic_ai import Agent

class Joke(BaseModel):
    setup: str
    punchline: str
    rating: int

agent = Agent('gemini-1.5-flash', result_type=Joke)

result = await agent.run("Tell me a joke about Python")
print(f"Setup: {result.data.setup}")
print(f"Punchline: {result.data.punchline}")
print(f"Rating: {result.data.rating}/10")
```

**Output**: A validated `Joke` object with proper fields.

### Step 3: System Prompt

```python
agent = Agent(
    'gemini-1.5-flash',
    result_type=Joke,
    system_prompt=(
        "You are a comedian specializing in programming jokes. "
        "Rate your own jokes honestly from 1-10."
    )
)

result = await agent.run("Tell me a joke about async/await")
```

The system prompt sets the agent's behavior and personality.

## Core Concepts

### 1. Agents

Agents are the main building blocks. They:
- Connect to a language model
- Process user messages
- Return typed responses
- Can use tools and dependencies

```python
agent = Agent(
    model='gemini-1.5-flash',
    result_type=MyResponse,
    system_prompt="You are a helpful assistant",
    deps_type=MyDependencies
)
```

### 2. Result Types

Result types define the structure of agent responses using Pydantic models:

```python
from pydantic import BaseModel, Field
from typing import Literal

class WeatherReport(BaseModel):
    location: str
    temperature: float = Field(description="Temperature in Celsius")
    condition: Literal["sunny", "cloudy", "rainy", "snowy"]
    humidity: int = Field(ge=0, le=100)
    wind_speed: float
    forecast: str = Field(description="Brief forecast description")

agent = Agent('gemini-1.5-flash', result_type=WeatherReport)
```

The model will automatically return data matching this structure!

### 3. Dependencies

Dependencies provide context to your agent:

```python
from dataclasses import dataclass

@dataclass
class UserContext:
    user_id: str
    preferences: dict
    location: str

agent = Agent(
    'gemini-1.5-flash',
    deps_type=UserContext
)

deps = UserContext(
    user_id="user123",
    preferences={"units": "celsius"},
    location="London"
)

result = await agent.run("What's the weather?", deps=deps)
```

The agent can access `deps` when processing requests.

### 4. Tools

Tools let your agent perform actions:

```python
from pydantic_ai import Agent, RunContext

agent = Agent('gemini-1.5-flash')

@agent.tool
def get_current_time(ctx: RunContext[str]) -> str:
    from datetime import datetime
    return datetime.now().strftime("%H:%M:%S")

@agent.tool
def calculate(ctx: RunContext[str], expression: str) -> float:
    return eval(expression)

result = await agent.run("What time is it and what's 15 * 7?")
```

The agent automatically decides when to use these tools!

## Working with Gemini Models

### Available Models

```python
"gemini-1.5-flash"
"gemini-1.5-pro"
"gemini-2.0-flash-exp"
```

**gemini-1.5-flash**: Fast, cost-effective, great for most tasks
**gemini-1.5-pro**: More capable, better for complex reasoning
**gemini-2.0-flash-exp**: Experimental, cutting-edge features

### Model Configuration

```python
from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel

model = GeminiModel(
    'gemini-1.5-flash',
    api_key='your-api-key-here'
)

agent = Agent(model)
```

### Environment Variables

Better approach for API keys:

```bash
export GEMINI_API_KEY='your-api-key-here'
```

Then in code:
```python
import os

agent = Agent(
    'gemini-1.5-flash',
)
```

Pydantic AI automatically uses `GEMINI_API_KEY` environment variable.

## Async vs Sync

Pydantic AI is async-first, but supports sync too:

### Async (Recommended)

```python
import asyncio
from pydantic_ai import Agent

agent = Agent('gemini-1.5-flash')

async def main():
    result = await agent.run("Hello!")
    print(result.data)

asyncio.run(main())
```

### Sync

```python
result = agent.run_sync("Hello!")
print(result.data)
```

Use async when possible for better performance with multiple requests.

## Error Handling

```python
from pydantic_ai import Agent, ModelRetry, UnexpectedModelBehavior

agent = Agent('gemini-1.5-flash', result_type=WeatherReport)

try:
    result = await agent.run("What's the weather in London?")
    print(result.data)
except ModelRetry as e:
    print(f"Model needs retry: {e}")
except UnexpectedModelBehavior as e:
    print(f"Unexpected behavior: {e}")
except Exception as e:
    print(f"Error: {e}")
```

## Validation

Pydantic automatically validates responses:

```python
from pydantic import BaseModel, Field, validator

class Temperature(BaseModel):
    value: float
    unit: Literal["C", "F"]
    
    @validator('value')
    def check_reasonable(cls, v):
        if v < -100 or v > 100:
            raise ValueError("Temperature seems unrealistic")
        return v

agent = Agent('gemini-1.5-flash', result_type=Temperature)
```

If the model returns invalid data, Pydantic AI will:
1. Validate the response
2. If invalid, ask the model to correct it
3. Retry a few times
4. Raise an exception if it keeps failing

## Streaming

For long responses, use streaming:

```python
agent = Agent('gemini-1.5-flash')

async with agent.run_stream("Write a long story") as response:
    async for message in response.stream():
        print(message, end='', flush=True)
```

## Complete Example: Weather Agent

```python
import asyncio
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from typing import Literal

class WeatherData(BaseModel):
    location: str
    temperature: float
    condition: Literal["sunny", "cloudy", "rainy", "snowy"]
    humidity: int
    summary: str

agent = Agent(
    'gemini-1.5-flash',
    result_type=WeatherData,
    system_prompt="You are a weather assistant. Provide accurate weather information."
)

@agent.tool
def fetch_weather(ctx: RunContext[None], city: str) -> dict:
    weather_db = {
        "London": {"temp": 15, "condition": "cloudy", "humidity": 70},
        "Tokyo": {"temp": 22, "condition": "sunny", "humidity": 50},
        "NYC": {"temp": 10, "condition": "rainy", "humidity": 80},
    }
    return weather_db.get(city, {"temp": 20, "condition": "sunny", "humidity": 60})

async def main():
    result = await agent.run("What's the weather like in London?")
    
    weather = result.data
    print(f"Location: {weather.location}")
    print(f"Temperature: {weather.temperature}°C")
    print(f"Condition: {weather.condition}")
    print(f"Humidity: {weather.humidity}%")
    print(f"Summary: {weather.summary}")

if __name__ == "__main__":
    asyncio.run(main())
```

**How it works**:
1. User asks about weather
2. Agent decides to call `fetch_weather` tool
3. Tool returns mock data
4. Agent formats into `WeatherData` structure
5. Returns validated, typed response

## Practice Exercise

Build a **Calculator Agent** with the following:

1. **Result Type**: Create a `CalculationResult` model with:
   - `expression: str` - The math expression
   - `result: float` - The calculated result
   - `explanation: str` - How the calculation was done

2. **Tool**: Create a `calculate` tool that takes an expression and returns the result

3. **System Prompt**: Tell the agent to break down complex math step-by-step

4. **Test it** with:
   - Simple: "What's 15 + 27?"
   - Complex: "Calculate compound interest: $1000 at 5% for 3 years"

Save your code as `calculator_agent.py`.

## Summary

- Pydantic AI provides type-safe AI development
- Agents connect to models and process requests
- Result types define structured outputs
- Tools let agents perform actions
- Dependencies provide context
- Built-in validation and error handling
- Works seamlessly with Gemini models

---

**Next**: [Lesson 4: Google Gemini Setup](lesson-04-gemini-setup.md)
