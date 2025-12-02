# Lesson 2: System Prompts and Instructions

## A. Concept Overview

### What & Why
**System Prompts** are the foundational instructions you give to your agent that define its behavior, personality, expertise, and output requirements. This is crucial because a well-crafted system prompt ensures consistent, high-quality outputs that match your Pydantic models perfectly. Think of it as the agent's permanent memory and job description.

### Analogy
Imagine hiring a customer service representative. You don't just say "help customers" - you provide:
- **Role definition**: "You are a senior technical support specialist"
- **Expertise**: "You have deep knowledge of our SaaS product"
- **Behavior guidelines**: "Always be empathetic and solution-focused"
- **Output requirements**: "Structure your responses with: problem summary, solution steps, and follow-up actions"
- **Constraints**: "Never promise features that don't exist"

The system prompt is that comprehensive training manual. Without it, responses are inconsistent and unreliable.

### Type Safety Benefit
While system prompts are strings (not typed), they directly impact type safety by:
- **Guiding output structure**: Telling the model exactly what fields to populate
- **Reducing validation errors**: Clear instructions = fewer malformed outputs
- **Constraining values**: Specifying valid ranges, formats, and enum values
- **Improving reliability**: Consistent outputs mean fewer runtime exceptions
- **Enabling complex models**: Detailed prompts help models fill nested structures correctly

A great system prompt is your first line of defense against validation errors!

---

## B. Code Implementation

### File Structure
```
agent_system_design/
├── lesson_02_system_prompts.py
├── requirements.txt  (same as lesson 1)
└── .env  (your GOOGLE_API_KEY)
```

### Complete Code Snippet

**lesson_02_system_prompts.py**
```python
"""
Lesson 2: System Prompts and Instructions
Learn to craft effective system prompts for consistent, structured outputs
"""

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

# Define a more complex structured output
class ProductReview(BaseModel):
    """Structured product review analysis"""
    product_name: str = Field(description="Name of the product")
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="Overall sentiment of the review"
    )
    rating: int = Field(description="Star rating", ge=1, le=5)
    key_points: list[str] = Field(
        description="3-5 key points from the review",
        min_length=3,
        max_length=5
    )
    is_verified_purchase: bool = Field(
        description="Whether the reviewer purchased the product"
    )
    recommendation: str = Field(
        description="Brief recommendation based on the review"
    )


# Example 1: Poor system prompt (vague and minimal)
poor_agent = Agent(
    model='gemini-1.5-flash',
    result_type=ProductReview,
    system_prompt="Analyze product reviews.",  # ❌ Too vague!
)


# Example 2: Good system prompt (detailed and structured)
good_agent = Agent(
    model='gemini-1.5-flash',
    result_type=ProductReview,
    system_prompt="""
You are an expert product review analyzer with years of experience in e-commerce.

Your task is to analyze customer reviews and extract structured insights.

ROLE & EXPERTISE:
- You understand customer sentiment, product quality indicators, and purchasing patterns
- You can identify authentic reviews vs potentially fake ones
- You focus on factual information and objective analysis

OUTPUT REQUIREMENTS:
- sentiment: Determine if the review is positive, negative, or neutral based on overall tone
- rating: Extract or infer the star rating (1-5 stars)
- key_points: Identify 3-5 most important points mentioned (pros, cons, or notable features)
- is_verified_purchase: Look for indicators that the person actually bought the product
- recommendation: Provide a one-sentence actionable insight

GUIDELINES:
- Be objective and fact-based
- If sentiment is mixed, focus on the dominant tone
- Key points should be concise (one short sentence each)
- Only mark is_verified_purchase as true if there's clear evidence
- Recommendation should be specific and actionable for potential buyers
""".strip(),
)


# Example 3: Excellent system prompt with examples (best practice)
excellent_agent = Agent(
    model='gemini-1.5-flash',
    result_type=ProductReview,
    system_prompt="""
You are an expert product review analyzer specializing in extracting structured insights from customer feedback.

ROLE & EXPERTISE:
You have analyzed over 100,000 product reviews across multiple categories. You understand:
- Sentiment analysis and tone detection
- Product quality indicators and common issues
- Verified vs unverified purchase patterns
- Consumer decision-making factors

YOUR TASK:
Analyze customer reviews and extract structured, actionable insights.

OUTPUT FIELD GUIDELINES:

1. product_name:
   - Extract the exact product name mentioned or infer from context
   - Use the full product name, not abbreviations

2. sentiment:
   - "positive": Overall satisfied, would recommend, 4-5 star tone
   - "negative": Dissatisfied, frustrated, 1-2 star tone
   - "neutral": Mixed feelings, 3 star tone, or factual without emotion
   
3. rating:
   - Extract explicit star rating if mentioned
   - Infer from sentiment: very positive=5, positive=4, neutral=3, negative=2, very negative=1
   
4. key_points:
   - Extract 3-5 most important points (pros, cons, features)
   - Format: Short, clear sentences (max 15 words each)
   - Cover both strengths and weaknesses if present
   - Example: "Battery lasts 2 full days with heavy use"
   
5. is_verified_purchase:
   - True: Review mentions "purchased", "bought", "received", or shows detailed usage
   - False: Vague, generic, or no purchase indicators
   
6. recommendation:
   - One sentence summarizing who should buy this (or avoid it)
   - Be specific: mention use cases, user types, or conditions
   - Example: "Great for professionals needing long battery life, but skip if you need top-tier camera quality"

EXAMPLES:

Input: "I bought this laptop 2 months ago and I'm blown away! The battery easily lasts 10 hours, the screen is gorgeous, and it handles my video editing smoothly. The only downside is the webcam quality is mediocre. Totally worth the $1200."

Expected Output:
- product_name: "Laptop"
- sentiment: "positive"
- rating: 5
- key_points: ["Battery lasts 10+ hours", "Excellent screen quality", "Handles video editing smoothly", "Webcam quality is mediocre"]
- is_verified_purchase: true
- recommendation: "Excellent choice for content creators who need performance and battery life; consider external webcam for video calls"

IMPORTANT:
- Always fill all required fields
- Be objective and evidence-based
- If information is unclear, make reasonable inferences
- Keep language professional and concise
""".strip(),
)


# Test function to compare agents
def analyze_review(agent: Agent, review_text: str, agent_name: str) -> None:
    """Analyze a review and display results"""
    print(f"\n{'='*70}")
    print(f"AGENT: {agent_name}")
    print(f"{'='*70}")
    print(f"Review: {review_text[:100]}...\n")
    
    try:
        result = agent.run_sync(review_text)
        review = result.data
        
        print(f"📦 Product: {review.product_name}")
        print(f"😊 Sentiment: {review.sentiment}")
        print(f"⭐ Rating: {review.rating}/5")
        print(f"✅ Verified: {review.is_verified_purchase}")
        print(f"\n🔑 Key Points:")
        for i, point in enumerate(review.key_points, 1):
            print(f"   {i}. {point}")
        print(f"\n💡 Recommendation: {review.recommendation}")
        
    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {e}")


if __name__ == "__main__":
    # Test review
    sample_review = """
    I purchased these wireless headphones 3 weeks ago for my daily commute. 
    The noise cancellation is absolutely phenomenal - I can't hear the subway at all! 
    Sound quality is crisp and bass is deep without being overwhelming. 
    Battery life is solid, about 20 hours per charge. 
    However, they're a bit tight on my head after wearing for 2+ hours, and 
    the touch controls are overly sensitive. For $150, I'd say it's a good deal 
    despite the minor comfort issues. Would definitely recommend for commuters!
    """
    
    # Compare different system prompts
    print("\n" + "="*70)
    print("COMPARING SYSTEM PROMPT QUALITY")
    print("="*70)
    
    # Test with poor prompt
    analyze_review(poor_agent, sample_review, "POOR PROMPT (minimal instructions)")
    
    # Test with good prompt
    analyze_review(good_agent, sample_review, "GOOD PROMPT (detailed instructions)")
    
    # Test with excellent prompt
    analyze_review(excellent_agent, sample_review, "EXCELLENT PROMPT (with examples)")
    
    print("\n" + "="*70)
    print("SYSTEM PROMPT COMPARISON COMPLETE")
    print("="*70)
    print("\nNotice how the excellent prompt produces the most consistent,")
    print("detailed, and accurate analysis!")
```

### Line-by-Line Explanation

**Complex Pydantic Model (Lines 12-30)**:
- `Literal["positive", "negative", "neutral"]`: Constrains sentiment to exactly 3 values
- `list[str]`: A list of strings for multiple key points
- `min_length=3, max_length=5`: Enforces 3-5 items in the list
- These constraints guide the model AND validate outputs

**Poor System Prompt (Lines 33-37)**:
- Only 3 words: "Analyze product reviews"
- No context, no examples, no field guidance
- Will produce inconsistent results and validation errors
- Don't do this! ❌

**Good System Prompt (Lines 40-68)**:
- Defines role and expertise clearly
- Explains each output field
- Provides behavioral guidelines
- Much better, but still room for improvement

**Excellent System Prompt (Lines 71-140)**:
- **Role definition**: Establishes expertise and credibility
- **Field-by-field guidance**: Explains exactly how to populate each field
- **Examples**: Shows concrete input/output pairs
- **Edge case handling**: Addresses ambiguous situations
- This produces the most consistent, accurate results ✅

**System Prompt Best Practices**:
1. **Start with role**: "You are an expert..."
2. **Define the task**: "Your job is to..."
3. **Explain each field**: Map Pydantic fields to instructions
4. **Provide examples**: Show ideal input/output pairs
5. **Handle edge cases**: Address ambiguity explicitly
6. **Use clear structure**: Headers, bullet points, numbered lists

### The "Why" Behind the Pattern

**Why invest time in system prompts?**

1. **Reduces validation errors**: Clear instructions = outputs that match your Pydantic models
2. **Improves consistency**: Same input produces similar outputs across runs
3. **Better inference quality**: Models perform better with context and examples
4. **Faster debugging**: When errors occur, you know the prompt is comprehensive
5. **Documentation**: Your prompt documents what the agent should do
6. **Cost efficiency**: Fewer retries = fewer API calls = lower costs

**System Prompt = Your Contract with the Model**

---

## C. Test & Apply

### How to Test It

1. **Run the comparison script**:
```bash
python lesson_02_system_prompts.py
```

2. **Observe the differences** between poor, good, and excellent prompts

3. **Experiment with your own prompts**:
```python
# Create a custom agent with your prompt
custom_agent = Agent(
    model='gemini-1.5-flash',
    result_type=ProductReview,
    system_prompt="YOUR CUSTOM PROMPT HERE",
)

analyze_review(custom_agent, sample_review, "MY CUSTOM PROMPT")
```

### Expected Result

You should see three different analyses of the same review:

**POOR PROMPT Output** (likely inconsistent):
```
📦 Product: Headphones
😊 Sentiment: positive
⭐ Rating: 4/5
✅ Verified: true
🔑 Key Points:
   1. Good noise cancellation
   2. Good battery
   3. Comfort issues
💡 Recommendation: Good for commuters
```

**EXCELLENT PROMPT Output** (detailed and consistent):
```
📦 Product: Wireless Headphones
😊 Sentiment: positive
⭐ Rating: 4/5
✅ Verified: true
🔑 Key Points:
   1. Phenomenal noise cancellation blocks subway noise completely
   2. Crisp sound quality with well-balanced bass
   3. Solid 20-hour battery life
   4. Becomes uncomfortable after 2+ hours of wear
   5. Touch controls are overly sensitive
💡 Recommendation: Excellent for commuters and travelers who prioritize noise cancellation and sound quality; consider for shorter sessions if sensitive to headphone pressure
```

### Validation Examples

**System Prompt Structure Template**:

```python
system_prompt = """
You are a [ROLE] with expertise in [DOMAIN].

BACKGROUND & EXPERTISE:
- [Expertise point 1]
- [Expertise point 2]

YOUR TASK:
[Clear, specific description of what the agent should do]

OUTPUT FIELD GUIDELINES:
1. field_name_1:
   - [How to determine this value]
   - [Valid options or format]
   - [Example]

2. field_name_2:
   - [How to determine this value]
   - [Valid options or format]
   - [Example]

EXAMPLES:
Input: [Example input]
Expected Output:
- field_name_1: [value]
- field_name_2: [value]

IMPORTANT NOTES:
- [Critical guideline 1]
- [Critical guideline 2]
""".strip()
```

### Type Checking

System prompts are strings, but you can validate their effectiveness:

```python
from pydantic import ValidationError

def test_system_prompt(agent: Agent, test_cases: list[str]) -> float:
    """
    Test a system prompt's reliability
    Returns: Success rate (0.0 to 1.0)
    """
    successes = 0
    for test_case in test_cases:
        try:
            result = agent.run_sync(test_case)
            successes += 1
        except ValidationError:
            pass  # Validation failure
    
    return successes / len(test_cases)

# Compare prompt quality
test_cases = [sample_review, another_review, third_review]
poor_success_rate = test_system_prompt(poor_agent, test_cases)
excellent_success_rate = test_system_prompt(excellent_agent, test_cases)

print(f"Poor prompt success rate: {poor_success_rate:.1%}")
print(f"Excellent prompt success rate: {excellent_success_rate:.1%}")
```

---

## D. Common Stumbling Blocks

### 1. System Prompt Too Vague

**The Problem**:
```python
system_prompt = "You help with tasks."  # ❌ Way too vague!
```

**What Happens**:
- Inconsistent outputs
- Frequent validation errors
- Model "guesses" what you want
- Different results for similar inputs

**The Fix**:
Be **extremely specific**:
```python
system_prompt = """
You are a task categorization specialist.

Your job is to analyze task descriptions and categorize them by:
- priority (high/medium/low)
- estimated_hours (1-40)
- category (one of: development, design, testing, documentation)

Guidelines:
- "urgent" or "asap" in description = high priority
- "bug" or "fix" = development category
- default estimated_hours to 4 if unclear
"""
```

### 2. Not Aligning Prompt with Pydantic Model

**The Problem**:
```python
class MovieInfo(BaseModel):
    title: str
    release_year: int
    directors: list[str]  # Note: plural!
    box_office_millions: float

agent = Agent(
    result_type=MovieInfo,
    system_prompt="Tell me about movies."  # ❌ Doesn't mention the fields!
)
```

**What Happens**:
Model doesn't know it needs to provide multiple directors, or format box office correctly.

**The Fix**:
**Map each Pydantic field to prompt instructions**:
```python
system_prompt = """
You are a movie database expert.

Provide information in this exact structure:
- title: The official movie title
- release_year: 4-digit year (e.g., 2010)
- directors: List ALL directors (even if just one, return as a list)
- box_office_millions: Total worldwide box office in millions (as a number, e.g., 2787.9)

Example:
For "Avatar": 
- directors should be ["James Cameron"] (a list!)
- box_office_millions should be 2787.9 (not "$2.79B" or "2.79 billion")
"""
```

### 3. Forgetting to Handle Edge Cases

**The Problem**:
```python
system_prompt = "Extract the date from the text."
# What if there's no date? Multiple dates? Ambiguous format?
```

**What Happens**:
- Model makes inconsistent assumptions
- Validation errors on edge cases
- Unpredictable behavior

**The Fix**:
**Explicitly handle edge cases in your prompt**:
```python
system_prompt = """
Extract dates from text.

Rules:
- If multiple dates present, return the earliest one
- If no date present, set date to None
- If date is ambiguous (e.g., "next Friday"), use context to infer
- Always return ISO format: YYYY-MM-DD
- If only month/year given, use first day of month (e.g., "Jan 2024" → "2024-01-01")
"""
```

### 4. System Prompt Too Long (Context Waste)

**The Problem**:
```python
system_prompt = """
You are the world's most advanced AI agent, built by a team of researchers 
over 10 years. Your capabilities include... [3000 words of fluff]
"""  # ❌ Wastes tokens, adds no value
```

**What Happens**:
- Wastes context window space
- Increases costs (you pay per token)
- Doesn't improve output quality
- May confuse the model with irrelevant info

**The Fix**:
**Be comprehensive but concise**:
```python
system_prompt = """
You are a data extraction specialist.

Task: Extract contact information from text.

Fields:
- name: Full name (First Last)
- email: Valid email or None
- phone: Format as +1-XXX-XXX-XXXX or None

Rules:
- If multiple contacts, return the first one
- Validate email format strictly
- Phone should include country code if present
"""  # ✅ Complete but concise
```

### 5. Type Safety Gotcha: Dynamic Prompts

**Common Pattern**:
```python
# ❌ Easy to make mistakes
system_prompt = f"You are a {role} who {task}"  # What if role is None?
```

**Type-Safe Approach**:
```python
def create_system_prompt(role: str, task: str, fields: list[str]) -> str:
    """
    Type-safe system prompt builder
    
    Args:
        role: The agent's role (e.g., "data analyst")
        task: The agent's task (e.g., "analyze sales data")
        fields: List of Pydantic field names to include
    
    Returns:
        Complete system prompt string
    """
    # Validate inputs
    assert role.strip(), "Role cannot be empty"
    assert task.strip(), "Task cannot be empty"
    assert len(fields) > 0, "Must specify at least one field"
    
    field_descriptions = "\n".join([f"- {field}" for field in fields])
    
    return f"""
You are a {role}.

Your task: {task}

Required fields:
{field_descriptions}
""".strip()

# Usage - type-safe and validated!
prompt = create_system_prompt(
    role="product analyst",
    task="analyze customer reviews",
    fields=["sentiment", "rating", "key_points"]
)
```

---

## Ready for the Next Lesson?

🎉 **Excellent work!** You now understand how to craft powerful system prompts that:

✅ Define agent roles and expertise clearly  
✅ Map Pydantic fields to specific instructions  
✅ Provide examples for consistency  
✅ Handle edge cases explicitly  
✅ Balance comprehensiveness with conciseness  

**System prompts are the foundation of reliable AI systems!** A great prompt reduces validation errors by 80%+ and ensures consistent, predictable outputs.

In the next lesson, we'll explore **Creating Custom Dependencies** - you'll learn how to inject context, database connections, API clients, and other resources into your agents in a type-safe way!

**Ready for Lesson 3, or would you like to practice writing system prompts first?** 🚀
