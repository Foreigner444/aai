# Lesson 7: Pydantic AI Agents

## Building Production Agents

In this lesson, we'll build sophisticated agents that use Gemini models and can integrate with MCP servers.

## Agent Architecture

A well-designed agent has:

1. **Model**: The LLM (Gemini)
2. **System Prompt**: Personality and instructions
3. **Tools**: Actions the agent can perform
4. **Dependencies**: Context and state
5. **Result Type**: Structured output format

## Basic Agent Pattern

```python
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext

class MyResult(BaseModel):
    answer: str
    confidence: float

agent = Agent(
    model='gemini-1.5-flash',
    result_type=MyResult,
    system_prompt="You are a helpful assistant."
)

result = await agent.run("What's 2+2?")
print(result.data.answer)
```

## Advanced Agent with Dependencies

Dependencies provide runtime context to your agent:

```python
from dataclasses import dataclass
from datetime import datetime
from pydantic_ai import Agent, RunContext

@dataclass
class UserContext:
    user_id: str
    username: str
    preferences: dict
    session_start: datetime

class ChatResponse(BaseModel):
    message: str
    personalized: bool

agent = Agent(
    'gemini-1.5-flash',
    result_type=ChatResponse,
    deps_type=UserContext,
    system_prompt="""
    You are a personalized assistant. Use the user's context to provide
    tailored responses. Reference their name and preferences when appropriate.
    """
)

@agent.tool
def get_user_history(ctx: RunContext[UserContext]) -> list[str]:
    return database.get_history(ctx.deps.user_id)

@agent.tool
def save_preference(ctx: RunContext[UserContext], key: str, value: str) -> str:
    ctx.deps.preferences[key] = value
    database.save_preferences(ctx.deps.user_id, ctx.deps.preferences)
    return f"Saved {key}={value}"

async def chat(user_id: str, message: str):
    user = database.get_user(user_id)
    
    deps = UserContext(
        user_id=user.id,
        username=user.username,
        preferences=user.preferences,
        session_start=datetime.now()
    )
    
    result = await agent.run(message, deps=deps)
    return result.data.message
```

**How it works**:
1. Create `UserContext` with user data
2. Pass to agent via `deps` parameter
3. Agent and tools access via `ctx.deps`
4. Personalized responses based on context

## Agent Tools

Tools are Python functions decorated with `@agent.tool`:

### Basic Tool

```python
@agent.tool
def get_current_time(ctx: RunContext[None]) -> str:
    from datetime import datetime
    return datetime.now().strftime("%H:%M:%S")
```

### Tool with Parameters

```python
@agent.tool
def search_web(ctx: RunContext[None], query: str, max_results: int = 5) -> list[dict]:
    results = web_api.search(query, limit=max_results)
    return [
        {"title": r.title, "url": r.url, "snippet": r.snippet}
        for r in results
    ]
```

### Tool with Dependencies

```python
@agent.tool
def get_user_tasks(ctx: RunContext[UserContext]) -> list[dict]:
    user_id = ctx.deps.user_id
    tasks = database.get_tasks(user_id)
    return [
        {"id": t.id, "title": t.title, "status": t.status}
        for t in tasks
    ]
```

### Async Tools

```python
@agent.tool
async def fetch_weather(ctx: RunContext[None], city: str) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.get(f"https://api.weather.com/{city}")
        return response.json()
```

## Structured Output with Pydantic

Define complex response structures:

```python
from pydantic import BaseModel, Field
from typing import List, Literal

class ResearchPaper(BaseModel):
    title: str
    authors: List[str]
    year: int
    abstract: str = Field(max_length=500)
    relevance_score: float = Field(ge=0, le=1)

class ResearchResponse(BaseModel):
    query: str
    papers: List[ResearchPaper]
    total_found: int
    summary: str

agent = Agent(
    'gemini-1.5-flash',
    result_type=ResearchResponse
)

result = await agent.run("Find papers about transformers in NLP")
for paper in result.data.papers:
    print(f"{paper.title} ({paper.year}) - Score: {paper.relevance_score}")
```

**Benefits**:
- Type safety
- Automatic validation
- IDE autocomplete
- Self-documenting

## System Prompts

System prompts define agent behavior:

### Static System Prompt

```python
agent = Agent(
    'gemini-1.5-flash',
    system_prompt="""
    You are an expert Python developer with 10 years of experience.
    Always provide code examples with your explanations.
    Use type hints and follow PEP 8 style guidelines.
    Explain your reasoning step-by-step.
    """
)
```

### Dynamic System Prompt

```python
def get_system_prompt(ctx: RunContext[UserContext]) -> str:
    user = ctx.deps
    return f"""
    You are assisting {user.username}.
    Their preferences: {user.preferences}
    Current session started: {user.session_start}
    
    Tailor your responses to their preferences and history.
    Be friendly and professional.
    """

agent = Agent(
    'gemini-1.5-flash',
    deps_type=UserContext,
    system_prompt=get_system_prompt
)
```

### Multi-Part System Prompts

```python
from pydantic_ai import SystemPrompt

@agent.system_prompt
def get_prompt(ctx: RunContext[UserContext]) -> list[SystemPrompt]:
    prompts = [
        SystemPrompt("You are a coding assistant."),
        SystemPrompt(f"User: {ctx.deps.username}"),
    ]
    
    if ctx.deps.preferences.get("expert_mode"):
        prompts.append(SystemPrompt("Provide advanced, detailed explanations."))
    else:
        prompts.append(SystemPrompt("Keep explanations simple and beginner-friendly."))
    
    return prompts
```

## Conversation History

Maintain context across messages:

```python
from typing import List

class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class ConversationAgent:
    def __init__(self):
        self.agent = Agent('gemini-1.5-flash')
        self.history: List[Message] = []
    
    async def chat(self, user_message: str) -> str:
        self.history.append(Message(role="user", content=user_message))
        
        context = "\n".join([
            f"{msg.role}: {msg.content}"
            for msg in self.history[-10:]
        ])
        
        full_prompt = f"Conversation history:\n{context}\n\nRespond to the latest message."
        
        result = await self.agent.run(full_prompt)
        response = result.data
        
        self.history.append(Message(role="assistant", content=response))
        
        return response

conversation = ConversationAgent()
print(await conversation.chat("Hi, I'm learning Python"))
print(await conversation.chat("What are decorators?"))
print(await conversation.chat("Can you show an example?"))
```

## Error Handling

Handle errors gracefully:

```python
from pydantic_ai import Agent, ModelRetry, UnexpectedModelBehavior
import logging

logger = logging.getLogger(__name__)

class SafeAgent:
    def __init__(self):
        self.agent = Agent('gemini-1.5-flash')
    
    async def run(self, prompt: str, max_retries: int = 3) -> str | None:
        for attempt in range(max_retries):
            try:
                result = await self.agent.run(prompt)
                return result.data
            
            except ModelRetry as e:
                logger.warning(f"Model retry needed (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    logger.error("Max retries exceeded")
                    return None
            
            except UnexpectedModelBehavior as e:
                logger.error(f"Unexpected model behavior: {e}")
                return None
            
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                if attempt == max_retries - 1:
                    return None
        
        return None

agent = SafeAgent()
response = await agent.run("Hello!")
if response:
    print(response)
else:
    print("Failed to get response")
```

## Streaming Responses

For long outputs:

```python
agent = Agent('gemini-1.5-flash')

async def stream_response(prompt: str):
    async with agent.run_stream(prompt) as response:
        async for chunk in response.stream():
            print(chunk, end='', flush=True)
        print()

await stream_response("Write a long story about AI")
```

## Model Selection

Choose the right model for your use case:

```python
class SmartAgent:
    def __init__(self):
        self.fast_agent = Agent('gemini-1.5-flash')
        self.smart_agent = Agent('gemini-1.5-pro')
    
    async def run(self, prompt: str, use_smart: bool = False):
        agent = self.smart_agent if use_smart else self.fast_agent
        result = await agent.run(prompt)
        return result.data

agent = SmartAgent()

quick_response = await agent.run("What's 2+2?")

detailed_response = await agent.run(
    "Explain quantum computing in detail",
    use_smart=True
)
```

## Agent Composition

Combine multiple agents:

```python
class MultiAgentSystem:
    def __init__(self):
        self.classifier = Agent(
            'gemini-1.5-flash',
            result_type=QueryType,
            system_prompt="Classify user queries into types: technical, general, creative"
        )
        
        self.technical_agent = Agent(
            'gemini-1.5-pro',
            system_prompt="You are a technical expert. Provide detailed, accurate answers."
        )
        
        self.creative_agent = Agent(
            'gemini-1.5-flash',
            system_prompt="You are creative and imaginative. Provide engaging, creative responses."
        )
        
        self.general_agent = Agent(
            'gemini-1.5-flash',
            system_prompt="You are a friendly general assistant."
        )
    
    async def process(self, query: str) -> str:
        classification = await self.classifier.run(query)
        
        if classification.data.type == "technical":
            agent = self.technical_agent
        elif classification.data.type == "creative":
            agent = self.creative_agent
        else:
            agent = self.general_agent
        
        result = await agent.run(query)
        return result.data

system = MultiAgentSystem()
response = await system.process("Explain neural networks")
```

## Real-World Example: Customer Support Agent

```python
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from typing import List, Literal
from dataclasses import dataclass
import asyncio

@dataclass
class SupportContext:
    customer_id: str
    tier: Literal["free", "premium", "enterprise"]
    ticket_history: List[dict]

class SupportResponse(BaseModel):
    response: str
    escalate: bool
    suggested_articles: List[str]
    followup_required: bool

support_agent = Agent(
    'gemini-1.5-flash',
    result_type=SupportResponse,
    deps_type=SupportContext,
    system_prompt="""
    You are a customer support agent. Be helpful, professional, and empathetic.
    
    Based on customer tier:
    - Free: Offer self-service resources
    - Premium: Provide detailed help
    - Enterprise: Prioritize and offer direct assistance
    
    Escalate if: technical bug, billing issue, or customer is frustrated.
    """
)

@support_agent.tool
def search_knowledge_base(ctx: RunContext[SupportContext], query: str) -> List[dict]:
    articles = knowledge_base.search(query)
    return [
        {"title": a.title, "url": a.url, "summary": a.summary}
        for a in articles[:3]
    ]

@support_agent.tool
def get_customer_history(ctx: RunContext[SupportContext]) -> List[dict]:
    return ctx.deps.ticket_history

@support_agent.tool
def check_account_status(ctx: RunContext[SupportContext]) -> dict:
    account = database.get_account(ctx.deps.customer_id)
    return {
        "tier": ctx.deps.tier,
        "active": account.active,
        "last_payment": account.last_payment.isoformat()
    }

async def handle_support_ticket(customer_id: str, message: str):
    customer = database.get_customer(customer_id)
    
    deps = SupportContext(
        customer_id=customer_id,
        tier=customer.tier,
        ticket_history=database.get_ticket_history(customer_id, limit=5)
    )
    
    result = await support_agent.run(message, deps=deps)
    
    response_data = result.data
    
    if response_data.escalate:
        await notify_human_agent(customer_id, message)
    
    return {
        "response": response_data.response,
        "articles": response_data.suggested_articles,
        "escalated": response_data.escalate
    }
```

## Practice Exercise

Build a **Code Review Agent** with:

### Requirements

1. **Result Type**:
```python
class CodeReview(BaseModel):
    overall_score: int  # 1-10
    issues: List[Issue]
    suggestions: List[str]
    approved: bool
```

2. **Tools**:
   - `check_syntax(code: str)` - Validate syntax
   - `run_linter(code: str)` - Run linting
   - `check_security(code: str)` - Security scan
   - `get_best_practices(language: str)` - Get style guide

3. **Dependencies**:
```python
@dataclass
class ReviewContext:
    language: str
    strictness: Literal["lenient", "normal", "strict"]
    focus_areas: List[str]
```

4. **System Prompt**: Customize based on strictness level

5. **Test** with different code samples and strictness levels

Save as `code_review_agent.py`.

## Summary

- Agents combine models, prompts, tools, and dependencies
- Dependencies provide runtime context
- Tools let agents perform actions
- Structured outputs ensure type safety
- System prompts define behavior
- Compose multiple agents for complex systems
- Handle errors gracefully
- Choose appropriate models for tasks

---

**Next**: [Lesson 8: Integrating Everything](lesson-08-integrating-everything.md)
