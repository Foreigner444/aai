# 📘 Lesson 8: Understanding Gemini Model Variants

Google offers several Gemini models, each with different strengths. Let's learn which to use and when! 🧠

---

## A. Concept Overview

### What & Why

Google's Gemini family includes multiple models optimized for different use cases:

| Model | Best For | Speed | Cost |
|-------|----------|-------|------|
| **Gemini 1.5 Flash** | Fast, high-volume tasks | ⚡ Fastest | 💰 Cheapest |
| **Gemini 1.5 Pro** | Complex reasoning | 🐢 Slower | 💰💰 More expensive |
| **Gemini 1.0 Pro** | Legacy support | 🐢 Medium | 💰 Medium |

For most Pydantic AI applications, **Gemini 1.5 Flash** is the best starting point!

### The Analogy 🚗

Think of Gemini models like vehicles:

- **Gemini 1.5 Flash** = Sports car - Fast, efficient, great for most trips
- **Gemini 1.5 Pro** = Luxury SUV - More capable, handles rough terrain, but uses more fuel
- **Gemini 1.0 Pro** = Reliable sedan - Gets the job done, but older model

For daily commutes (typical AI tasks), the sports car is perfect. For moving furniture (complex reasoning), you might need the SUV.

### Type Safety Benefit

Understanding models helps you:
- Choose the right speed/capability tradeoff
- Optimize costs for production
- Handle context window limits correctly
- Make informed decisions about model selection

---

## B. Code Implementation

### File Structure
```
pydantic-ai-project/
├── models/
├── config.py
├── model_comparison.py    # Compare different models
└── ...
```

### Gemini Model Overview

```python
"""
Gemini Model Reference Guide

This file documents the available Gemini models and their capabilities.
"""

# ============================================================
# GEMINI 1.5 FLASH
# ============================================================
# Model ID: gemini-1.5-flash
# 
# Best for:
# - High-volume applications
# - Simple to moderate complexity tasks
# - Cost-sensitive deployments
# - Real-time/streaming applications
# - Structured data extraction
#
# Specifications:
# - Context Window: 1,048,576 tokens (1M!)
# - Output Tokens: 8,192
# - Speed: Fastest
# - Cost: Lowest
#
# Use cases:
# - Chat applications
# - Content classification
# - Data extraction
# - Summarization
# - Translation
# - Simple Q&A

# ============================================================
# GEMINI 1.5 PRO
# ============================================================
# Model ID: gemini-1.5-pro
#
# Best for:
# - Complex reasoning tasks
# - Long document analysis
# - Code generation and review
# - Multi-step problem solving
# - Tasks requiring high accuracy
#
# Specifications:
# - Context Window: 2,097,152 tokens (2M!)
# - Output Tokens: 8,192
# - Speed: Slower than Flash
# - Cost: Higher than Flash
#
# Use cases:
# - Document analysis
# - Complex code tasks
# - Research synthesis
# - Legal/medical analysis
# - Multi-document comparison

# ============================================================
# GEMINI 1.0 PRO
# ============================================================
# Model ID: gemini-pro (or gemini-1.0-pro)
#
# Best for:
# - Legacy compatibility
# - When 1.5 isn't needed
#
# Specifications:
# - Context Window: 32,768 tokens (32K)
# - Output Tokens: 8,192
# - Speed: Medium
# - Cost: Medium
#
# Note: Consider using 1.5 Flash instead for most tasks
```

### Using Different Models with Pydantic AI

```python
"""
Demonstrate using different Gemini models with Pydantic AI.
"""
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from typing import Literal


class SentimentResult(BaseModel):
    """Result of sentiment analysis."""
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str


# Create agents with different models
flash_agent = Agent(
    'gemini-1.5-flash',  # Fast and cheap
    result_type=SentimentResult,
    system_prompt="You are a sentiment analyzer. Analyze the sentiment of the given text."
)

pro_agent = Agent(
    'gemini-1.5-pro',  # More capable, slower
    result_type=SentimentResult,
    system_prompt="You are a sentiment analyzer. Analyze the sentiment of the given text."
)


# Example usage (requires API key)
if __name__ == "__main__":
    text = "I absolutely love this product! It exceeded all my expectations."
    
    # Using Flash (recommended for most cases)
    result = flash_agent.run_sync(text)
    print(f"Flash result: {result.data}")
    
    # Using Pro (for complex analysis)
    # result = pro_agent.run_sync(text)
    # print(f"Pro result: {result.data}")
```

### Model Selection Helper

Create `model_selector.py`:

```python
"""
Helper for selecting the right Gemini model based on task requirements.
"""
from enum import Enum
from dataclasses import dataclass


class TaskComplexity(str, Enum):
    """Complexity level of the task."""
    SIMPLE = "simple"          # Classification, extraction, simple Q&A
    MODERATE = "moderate"      # Summarization, translation, content generation
    COMPLEX = "complex"        # Multi-step reasoning, analysis, code review


class VolumeLevel(str, Enum):
    """Expected request volume."""
    LOW = "low"          # < 100 requests/day
    MEDIUM = "medium"    # 100-1000 requests/day
    HIGH = "high"        # > 1000 requests/day


@dataclass
class ModelRecommendation:
    """Recommended model and reasoning."""
    model_id: str
    model_name: str
    reasoning: str
    estimated_cost: str


def recommend_model(
    complexity: TaskComplexity,
    volume: VolumeLevel,
    needs_long_context: bool = False,
    latency_critical: bool = False
) -> ModelRecommendation:
    """
    Recommend the best Gemini model based on requirements.
    
    Args:
        complexity: How complex is the task?
        volume: How many requests do you expect?
        needs_long_context: Do you need >100K token context?
        latency_critical: Is response time critical?
    
    Returns:
        ModelRecommendation with model ID and reasoning
    """
    
    # Decision logic
    if latency_critical:
        return ModelRecommendation(
            model_id="gemini-1.5-flash",
            model_name="Gemini 1.5 Flash",
            reasoning="Flash is fastest; essential for latency-critical apps",
            estimated_cost="$0.075/1M tokens"
        )
    
    if volume == VolumeLevel.HIGH:
        return ModelRecommendation(
            model_id="gemini-1.5-flash",
            model_name="Gemini 1.5 Flash",
            reasoning="High volume = cost sensitive; Flash is most economical",
            estimated_cost="$0.075/1M tokens"
        )
    
    if complexity == TaskComplexity.COMPLEX and needs_long_context:
        return ModelRecommendation(
            model_id="gemini-1.5-pro",
            model_name="Gemini 1.5 Pro",
            reasoning="Complex task + long context = Pro's sweet spot (2M context)",
            estimated_cost="$3.50/1M tokens"
        )
    
    if complexity == TaskComplexity.COMPLEX:
        return ModelRecommendation(
            model_id="gemini-1.5-pro",
            model_name="Gemini 1.5 Pro",
            reasoning="Complex reasoning tasks benefit from Pro's capabilities",
            estimated_cost="$3.50/1M tokens"
        )
    
    # Default: Flash handles most tasks well
    return ModelRecommendation(
        model_id="gemini-1.5-flash",
        model_name="Gemini 1.5 Flash",
        reasoning="Flash handles simple/moderate tasks efficiently",
        estimated_cost="$0.075/1M tokens"
    )


# Interactive recommendation
if __name__ == "__main__":
    print("=" * 60)
    print("Gemini Model Recommendation Tool")
    print("=" * 60)
    
    # Example scenarios
    scenarios = [
        {
            "name": "Chat Application",
            "complexity": TaskComplexity.SIMPLE,
            "volume": VolumeLevel.HIGH,
            "needs_long_context": False,
            "latency_critical": True
        },
        {
            "name": "Document Analysis",
            "complexity": TaskComplexity.COMPLEX,
            "volume": VolumeLevel.LOW,
            "needs_long_context": True,
            "latency_critical": False
        },
        {
            "name": "Data Extraction API",
            "complexity": TaskComplexity.MODERATE,
            "volume": VolumeLevel.MEDIUM,
            "needs_long_context": False,
            "latency_critical": False
        }
    ]
    
    for scenario in scenarios:
        rec = recommend_model(
            complexity=scenario["complexity"],
            volume=scenario["volume"],
            needs_long_context=scenario["needs_long_context"],
            latency_critical=scenario["latency_critical"]
        )
        
        print(f"\n📋 Scenario: {scenario['name']}")
        print(f"   Complexity: {scenario['complexity'].value}")
        print(f"   Volume: {scenario['volume'].value}")
        print(f"   Long context: {scenario['needs_long_context']}")
        print(f"   Latency critical: {scenario['latency_critical']}")
        print(f"\n   ✅ Recommended: {rec.model_name}")
        print(f"   📝 Reasoning: {rec.reasoning}")
        print(f"   💰 Est. cost: {rec.estimated_cost}")
```

### Context Window Comparison

```python
"""
Understanding context windows across Gemini models.

Context window = how much text the model can "see" at once
"""

# Context window sizes (in tokens)
CONTEXT_WINDOWS = {
    "gemini-1.5-flash": 1_048_576,   # 1M tokens
    "gemini-1.5-pro": 2_097_152,      # 2M tokens  
    "gemini-1.0-pro": 32_768,         # 32K tokens
}

# Approximate token-to-word ratio
TOKENS_PER_WORD = 1.3  # English average


def estimate_word_capacity(model: str) -> int:
    """Estimate how many words fit in the context window."""
    tokens = CONTEXT_WINDOWS.get(model, 0)
    return int(tokens / TOKENS_PER_WORD)


def estimate_pages(model: str, words_per_page: int = 300) -> int:
    """Estimate how many pages of text fit in context."""
    words = estimate_word_capacity(model)
    return words // words_per_page


if __name__ == "__main__":
    print("Gemini Context Window Comparison")
    print("=" * 60)
    
    for model, tokens in CONTEXT_WINDOWS.items():
        words = estimate_word_capacity(model)
        pages = estimate_pages(model)
        
        print(f"\n{model}:")
        print(f"  Tokens: {tokens:,}")
        print(f"  ~Words: {words:,}")
        print(f"  ~Pages: {pages:,} (at 300 words/page)")
    
    print("\n" + "=" * 60)
    print("Reference: 'War and Peace' is about 580,000 words")
    print("Gemini 1.5 Flash can handle ~800,000 words!")
    print("Gemini 1.5 Pro can handle ~1,600,000 words!")
```

---

## C. Test & Apply

### Quick Reference Table

| Task | Recommended Model | Why |
|------|-------------------|-----|
| **Chat/Conversation** | gemini-1.5-flash | Speed matters for chat |
| **Data Extraction** | gemini-1.5-flash | Simple task, high efficiency |
| **Sentiment Analysis** | gemini-1.5-flash | Classification is simple |
| **Summarization** | gemini-1.5-flash | Usually sufficient |
| **Code Review** | gemini-1.5-pro | Benefits from deeper reasoning |
| **Long Document Q&A** | gemini-1.5-pro | Needs large context + reasoning |
| **Legal Analysis** | gemini-1.5-pro | Complex, accuracy-critical |
| **Research Synthesis** | gemini-1.5-pro | Multi-step reasoning |

### Model String Reference

For Pydantic AI, use these exact model strings:

```python
from pydantic_ai import Agent

# Recommended for most tasks
agent = Agent('gemini-1.5-flash', result_type=MyModel)

# For complex reasoning
agent = Agent('gemini-1.5-pro', result_type=MyModel)

# Legacy (use 1.5-flash instead)
agent = Agent('gemini-1.0-pro', result_type=MyModel)
```

### Cost Estimation

Here's a rough cost calculator:

```python
"""
Cost estimation for Gemini API usage.
Prices as of 2024 - always check current pricing.
"""

# Prices per million tokens (approximate)
PRICING = {
    "gemini-1.5-flash": {
        "input": 0.075,   # $0.075 per 1M input tokens
        "output": 0.30,   # $0.30 per 1M output tokens
    },
    "gemini-1.5-pro": {
        "input": 3.50,    # $3.50 per 1M input tokens
        "output": 10.50,  # $10.50 per 1M output tokens
    },
}


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int
) -> float:
    """Estimate cost for a request."""
    prices = PRICING.get(model, PRICING["gemini-1.5-flash"])
    
    input_cost = (input_tokens / 1_000_000) * prices["input"]
    output_cost = (output_tokens / 1_000_000) * prices["output"]
    
    return input_cost + output_cost


def estimate_monthly_cost(
    model: str,
    requests_per_day: int,
    avg_input_tokens: int = 500,
    avg_output_tokens: int = 200
) -> float:
    """Estimate monthly cost for regular usage."""
    cost_per_request = estimate_cost(model, avg_input_tokens, avg_output_tokens)
    daily_cost = cost_per_request * requests_per_day
    monthly_cost = daily_cost * 30
    return monthly_cost


if __name__ == "__main__":
    # Example: 1000 requests/day with typical token counts
    flash_cost = estimate_monthly_cost(
        "gemini-1.5-flash", 
        requests_per_day=1000
    )
    pro_cost = estimate_monthly_cost(
        "gemini-1.5-pro",
        requests_per_day=1000
    )
    
    print(f"Monthly cost estimate (1000 req/day):")
    print(f"  Flash: ${flash_cost:.2f}")
    print(f"  Pro: ${pro_cost:.2f}")
    print(f"  Savings with Flash: ${pro_cost - flash_cost:.2f} ({(1-flash_cost/pro_cost)*100:.0f}%)")
```

---

## D. Common Stumbling Blocks

### "Model not found" error

Check your model string is exactly correct:
```python
# ✅ Correct
agent = Agent('gemini-1.5-flash', ...)

# ❌ Wrong - various typos
agent = Agent('gemini-flash', ...)      # Missing version
agent = Agent('gemini-1.5-Flash', ...)  # Capital F
agent = Agent('gemini1.5-flash', ...)   # Missing hyphen
```

### "Context length exceeded"

Your input is too long for the model:
```python
# Check your input length
input_text = "..." # your text
tokens_estimate = len(input_text.split()) * 1.3  # rough estimate

if tokens_estimate > 1_000_000:  # Flash limit
    print("Consider using gemini-1.5-pro or splitting input")
```

### "Which model should I start with?"

**Always start with `gemini-1.5-flash`!**

- It's fast, cheap, and handles most tasks well
- Only switch to Pro if Flash isn't giving good results
- For learning and development, Flash is perfect

### "My task is complex but Flash works fine"

Great! Stick with Flash! The model recommendations are guidelines, not rules. If Flash handles your task well:
- Lower cost
- Faster responses
- Same reliability

Only upgrade to Pro when you hit actual limitations.

---

## ✅ Lesson 8 Complete!

### Key Takeaways

1. **Start with `gemini-1.5-flash`** - best for most tasks
2. **Use `gemini-1.5-pro`** for complex reasoning and long context
3. **Context windows are huge** - 1M+ tokens handles most use cases
4. **Cost matters at scale** - Flash is 47x cheaper than Pro
5. **Test before deciding** - try Flash first, upgrade if needed

### Model Selection Checklist

When choosing a model, ask:
- [ ] Is latency critical? → Flash
- [ ] High volume? → Flash
- [ ] Simple/moderate task? → Flash
- [ ] Complex reasoning needed? → Pro
- [ ] Very long documents? → Pro
- [ ] Budget constrained? → Flash

### What's Next?

In Lesson 9, you'll create your first Pydantic AI Agent - combining everything you've learned so far!

---

*Model selected? Let's build your first agent in Lesson 9!* 🚀
