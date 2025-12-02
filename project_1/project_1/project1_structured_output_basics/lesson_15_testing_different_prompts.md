# 📘 Lesson 15: Testing Different Prompts

The quality of your prompts directly affects the quality of AI outputs. Let's learn prompt engineering! 🎯

---

## A. Concept Overview

### What & Why

**Prompt engineering** is the art of crafting instructions that get the AI to return exactly what you want. Even with Pydantic AI's structured outputs, the *quality* of the data depends on how well you guide the AI.

Good prompts:
- Get accurate, relevant extractions
- Reduce validation errors
- Produce consistent results
- Handle edge cases gracefully

### The Analogy 📝

Think of prompting like giving directions:

**Bad directions:** "Go to the store"
- Which store? What should they buy? When should they go?

**Good directions:** "Go to Whole Foods on Main Street, buy 2 gallons of milk and a dozen eggs, and be back by 6 PM"
- Clear destination, specific items, time constraint

The more specific you are, the better the results!

### Type Safety Benefit

Good prompts combined with Pydantic models:
- Reduce validation errors significantly
- Produce more accurate data
- Make your system more reliable
- Decrease the need for retries

---

## B. Code Implementation

### File Structure
```
pydantic-ai-project/
├── experiments/
│   └── prompt_testing.py    # Prompt experiments
└── ...
```

### Prompt Testing Framework

```python
"""
A framework for testing and comparing different prompts.
"""
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from dataclasses import dataclass
from typing import Any
import time


@dataclass
class PromptTestResult:
    """Result of testing a prompt."""
    prompt_name: str
    success: bool
    response: Any
    duration_ms: float
    error: str | None = None


class PromptTester:
    """Test different prompts and compare results."""
    
    def __init__(self, model_class: type[BaseModel], model: str = 'gemini-1.5-flash'):
        self.model_class = model_class
        self.model = model
    
    def test_prompt(self, prompt_name: str, system_prompt: str, test_input: str) -> PromptTestResult:
        """Test a single prompt configuration."""
        agent = Agent(self.model, result_type=self.model_class, system_prompt=system_prompt)
        
        start_time = time.time()
        try:
            result = agent.run_sync(test_input)
            duration_ms = (time.time() - start_time) * 1000
            return PromptTestResult(
                prompt_name=prompt_name,
                success=True,
                response=result.data,
                duration_ms=duration_ms
            )
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return PromptTestResult(
                prompt_name=prompt_name,
                success=False,
                response=None,
                duration_ms=duration_ms,
                error=str(e)
            )
    
    def compare_prompts(
        self,
        prompts: dict[str, str],
        test_inputs: list[str]
    ) -> dict[str, list[PromptTestResult]]:
        """Compare multiple prompts across multiple inputs."""
        results = {name: [] for name in prompts}
        
        for input_text in test_inputs:
            print(f"\nTesting input: {input_text[:50]}...")
            for name, system_prompt in prompts.items():
                result = self.test_prompt(name, system_prompt, input_text)
                results[name].append(result)
                status = "✅" if result.success else "❌"
                print(f"  {status} {name}: {result.duration_ms:.0f}ms")
        
        return results
    
    def summary(self, results: dict[str, list[PromptTestResult]]) -> None:
        """Print a summary of prompt comparison results."""
        print("\n" + "=" * 60)
        print("PROMPT COMPARISON SUMMARY")
        print("=" * 60)
        
        for name, prompt_results in results.items():
            total = len(prompt_results)
            successes = sum(1 for r in prompt_results if r.success)
            avg_time = sum(r.duration_ms for r in prompt_results) / total
            
            print(f"\n{name}:")
            print(f"  Success rate: {successes}/{total} ({100*successes/total:.0f}%)")
            print(f"  Avg response time: {avg_time:.0f}ms")


# Example usage
class ContactInfo(BaseModel):
    """Extracted contact information."""
    name: str = Field(description="Full name")
    email: str | None = Field(default=None, description="Email address")
    phone: str | None = Field(default=None, description="Phone number")


# Define different prompts to test
prompts_to_test = {
    "minimal": "Extract contact info.",
    
    "detailed": (
        "Extract contact information from the text. "
        "Find the person's name, email address, and phone number. "
        "If any field is not present, set it to null."
    ),
    
    "structured": (
        "You are a contact information extractor. "
        "Your task is to identify and extract:\n"
        "1. The person's full name\n"
        "2. Their email address (if mentioned)\n"
        "3. Their phone number (if mentioned)\n"
        "Be precise and only extract information that is explicitly stated."
    ),
    
    "with_examples": (
        "Extract contact information. Examples:\n"
        "Input: 'Call John at 555-1234' → name='John', phone='555-1234'\n"
        "Input: 'Email sarah@test.com for help' → name='Sarah', email='sarah@test.com'\n"
        "Now extract from the given text."
    ),
}

test_inputs = [
    "Please contact John Smith at john@example.com or call 555-123-4567.",
    "For more info, reach out to Sarah Johnson: sarah.j@company.com",
    "Call our support team at 1-800-555-0199",
    "Email Dr. Michael Chen (m.chen@university.edu) with questions.",
]

# Run the comparison
tester = PromptTester(ContactInfo)
results = tester.compare_prompts(prompts_to_test, test_inputs)
tester.summary(results)
```

### Prompt Engineering Patterns

```python
"""
Collection of effective prompt patterns for Pydantic AI.
"""
from pydantic import BaseModel, Field
from pydantic_ai import Agent


class Product(BaseModel):
    name: str
    category: str
    price: float
    features: list[str]


# Pattern 1: Role + Task + Format
ROLE_TASK_FORMAT = """
Role: You are a product information specialist.
Task: Extract detailed product information from the given text.
Format: Provide the product name, category, price, and key features.
"""

# Pattern 2: Context + Instructions + Constraints
CONTEXT_INSTRUCTIONS = """
Context: You are processing e-commerce product descriptions.
Instructions: Extract the product details accurately.
Constraints:
- Price must be a number (no currency symbols)
- Features should be a list of distinct characteristics
- Category should be broad (Electronics, Clothing, etc.)
"""

# Pattern 3: Step-by-Step
STEP_BY_STEP = """
Follow these steps to extract product information:
1. Identify the product name (the main item being described)
2. Determine the category (Electronics, Home, Fashion, etc.)
3. Find the price (convert to a number, e.g., "$99.99" → 99.99)
4. List the key features (distinct characteristics mentioned)
"""

# Pattern 4: Few-Shot Examples
FEW_SHOT = """
Extract product information from text.

Example 1:
Text: "iPhone 15 Pro - $999, features A17 chip, titanium design"
Output: name="iPhone 15 Pro", category="Electronics", price=999.0, features=["A17 chip", "titanium design"]

Example 2:
Text: "Cozy fleece blanket, only $29.99! Super soft and machine washable."
Output: name="Cozy fleece blanket", category="Home", price=29.99, features=["Super soft", "Machine washable"]

Now extract from the given text:
"""

# Pattern 5: Negative Examples (what NOT to do)
WITH_NEGATIVES = """
Extract product information from the text.

DO:
- Extract the exact product name
- Convert prices to numbers
- List specific features

DON'T:
- Include promotional language in features
- Make up information not in the text
- Include prices in the product name
"""


# Test all patterns
def test_prompt_patterns():
    test_text = """
    New Sony WH-1000XM5 Headphones! Now just $349.99!
    Industry-leading noise cancellation, 30-hour battery life,
    and crystal-clear call quality. Premium comfort for all-day wear.
    """
    
    patterns = {
        "role_task_format": ROLE_TASK_FORMAT,
        "context_instructions": CONTEXT_INSTRUCTIONS,
        "step_by_step": STEP_BY_STEP,
        "few_shot": FEW_SHOT,
        "with_negatives": WITH_NEGATIVES,
    }
    
    print("Testing different prompt patterns...")
    print("=" * 60)
    
    for name, prompt in patterns.items():
        agent = Agent('gemini-1.5-flash', result_type=Product, system_prompt=prompt)
        try:
            result = agent.run_sync(test_text)
            print(f"\n✅ {name}:")
            print(f"   Name: {result.data.name}")
            print(f"   Category: {result.data.category}")
            print(f"   Price: ${result.data.price}")
            print(f"   Features: {result.data.features}")
        except Exception as e:
            print(f"\n❌ {name}: {e}")


if __name__ == "__main__":
    test_prompt_patterns()
```

### Dynamic Prompt Generation

```python
"""
Generate prompts dynamically based on the Pydantic model.
"""
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from typing import get_type_hints


def generate_prompt_from_model(model_class: type[BaseModel]) -> str:
    """Generate a system prompt from a Pydantic model's structure."""
    lines = [
        f"Extract information to fill a {model_class.__name__} record.",
        "",
        "Required fields:"
    ]
    
    for field_name, field_info in model_class.model_fields.items():
        # Get the type
        field_type = field_info.annotation
        type_name = getattr(field_type, '__name__', str(field_type))
        
        # Check if required
        required = field_info.is_required()
        req_str = "required" if required else "optional"
        
        # Get description if available
        description = field_info.description or "no description"
        
        lines.append(f"  - {field_name} ({type_name}, {req_str}): {description}")
    
    lines.extend([
        "",
        "Instructions:",
        "- Only extract information that is explicitly stated",
        "- Use null for optional fields if not found",
        "- Be precise and accurate"
    ])
    
    return "\n".join(lines)


# Example
class JobPosting(BaseModel):
    """A job posting."""
    title: str = Field(description="The job title")
    company: str = Field(description="Company name")
    location: str = Field(description="Job location")
    salary_min: float | None = Field(default=None, description="Minimum salary")
    salary_max: float | None = Field(default=None, description="Maximum salary")
    requirements: list[str] = Field(description="Job requirements")


# Generate prompt automatically
auto_prompt = generate_prompt_from_model(JobPosting)
print("Generated prompt:")
print(auto_prompt)

# Use it
agent = Agent('gemini-1.5-flash', result_type=JobPosting, system_prompt=auto_prompt)
```

### A/B Testing Prompts

```python
"""
A/B test different prompts in production.
"""
import random
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from dataclasses import dataclass, field
from typing import Callable
from datetime import datetime


@dataclass
class ABTestConfig:
    """Configuration for A/B testing prompts."""
    name: str
    prompt_a: str
    prompt_b: str
    split_ratio: float = 0.5  # 50/50 by default


@dataclass
class ABTestMetrics:
    """Metrics collected during A/B testing."""
    prompt_a_successes: int = 0
    prompt_a_failures: int = 0
    prompt_b_successes: int = 0
    prompt_b_failures: int = 0
    
    def success_rate_a(self) -> float:
        total = self.prompt_a_successes + self.prompt_a_failures
        return self.prompt_a_successes / total if total > 0 else 0
    
    def success_rate_b(self) -> float:
        total = self.prompt_b_successes + self.prompt_b_failures
        return self.prompt_b_successes / total if total > 0 else 0
    
    def winner(self) -> str:
        if self.success_rate_a() > self.success_rate_b():
            return "A"
        elif self.success_rate_b() > self.success_rate_a():
            return "B"
        return "Tie"


class ABTester:
    """Run A/B tests on prompts."""
    
    def __init__(self, model_class: type[BaseModel], config: ABTestConfig):
        self.model_class = model_class
        self.config = config
        self.metrics = ABTestMetrics()
        
        self.agent_a = Agent(
            'gemini-1.5-flash',
            result_type=model_class,
            system_prompt=config.prompt_a
        )
        self.agent_b = Agent(
            'gemini-1.5-flash',
            result_type=model_class,
            system_prompt=config.prompt_b
        )
    
    def run(self, input_text: str):
        """Run the A/B test for a single input."""
        # Randomly select A or B
        use_a = random.random() < self.config.split_ratio
        agent = self.agent_a if use_a else self.agent_b
        variant = "A" if use_a else "B"
        
        try:
            result = agent.run_sync(input_text)
            if use_a:
                self.metrics.prompt_a_successes += 1
            else:
                self.metrics.prompt_b_successes += 1
            return result.data, variant
        except Exception as e:
            if use_a:
                self.metrics.prompt_a_failures += 1
            else:
                self.metrics.prompt_b_failures += 1
            raise
    
    def report(self) -> str:
        """Generate A/B test report."""
        return f"""
A/B Test Report: {self.config.name}
{'=' * 50}
Prompt A:
  Successes: {self.metrics.prompt_a_successes}
  Failures: {self.metrics.prompt_a_failures}
  Success Rate: {self.metrics.success_rate_a():.1%}

Prompt B:
  Successes: {self.metrics.prompt_b_successes}
  Failures: {self.metrics.prompt_b_failures}
  Success Rate: {self.metrics.success_rate_b():.1%}

Winner: Prompt {self.metrics.winner()}
"""


# Example usage
class Sentiment(BaseModel):
    sentiment: str
    confidence: float


config = ABTestConfig(
    name="Sentiment Analysis Prompt Test",
    prompt_a="Analyze sentiment. Be concise.",
    prompt_b=(
        "Analyze the sentiment of the text. "
        "Determine if it's positive, negative, or neutral. "
        "Rate your confidence from 0.0 to 1.0."
    )
)

tester = ABTester(Sentiment, config)

# Run tests
test_texts = [
    "I love this product!",
    "Terrible experience.",
    "It's okay I guess.",
    "Best purchase ever!",
    "Would not recommend.",
]

for text in test_texts:
    try:
        result, variant = tester.run(text)
        print(f"[{variant}] {text[:30]}... → {result.sentiment}")
    except:
        print(f"[{variant}] {text[:30]}... → FAILED")

print(tester.report())
```

---

## C. Test & Apply

### Prompt Engineering Best Practices

| Practice | Example |
|----------|---------|
| Be specific | "Extract the price as a number" vs "Get price" |
| Use examples | "e.g., $99.99 → 99.99" |
| Define format | "Return as a list of strings" |
| Set constraints | "Only include explicitly stated info" |
| Handle edge cases | "If not found, use null" |

### Prompt Templates

```python
# Template 1: Simple extraction
SIMPLE_TEMPLATE = """
Extract {entity_type} from the text.
Required fields: {fields}
"""

# Template 2: Detailed extraction
DETAILED_TEMPLATE = """
You are a {role} specialist.
Your task is to extract {entity_type} information.

Requirements:
{requirements}

Constraints:
{constraints}
"""

# Template 3: With validation hints
VALIDATION_TEMPLATE = """
Extract {entity_type}. Ensure:
- All required fields are filled
- Numbers are actual numbers (not strings)
- Lists contain at least one item
- Optional fields are null if not found
"""
```

---

## D. Common Stumbling Blocks

### "My prompts work sometimes but not always"

Add more specific instructions:
```python
# ❌ Vague
"Extract the name"

# ✅ Specific
"Extract the person's full name (first and last). If only one name is given, use it as-is."
```

### "The AI adds extra commentary"

Be explicit about format:
```python
"Only return the extracted data. Do not add explanations or commentary."
```

### "Results are inconsistent across runs"

Use examples to anchor behavior:
```python
"""
Example: "John Smith, CEO" → name="John Smith", title="CEO"
Example: "Contact sarah@test.com" → name="Sarah", email="sarah@test.com"

Now extract from: {input}
"""
```

### "How do I know if my prompt is good?"

Test systematically:
1. Create test cases with known correct answers
2. Run multiple times to check consistency
3. Track success rates over time
4. Compare different prompt versions

---

## ✅ Lesson 15 Complete!

### Key Takeaways

1. **Prompt quality** directly affects output quality
2. **Be specific** - vague prompts get vague results
3. **Use examples** - they anchor AI behavior
4. **Test systematically** - don't guess, measure
5. **A/B test** - compare prompt versions
6. **Iterate** - prompts can always be improved

### Prompt Engineering Checklist

- [ ] Clear role/task definition
- [ ] Specific extraction instructions
- [ ] Examples of expected output
- [ ] Constraints and edge case handling
- [ ] Systematic testing in place

### What's Next?

In Lesson 16, we'll build the complete structured output application, bringing everything together!

---

*Prompts perfected! Let's build the complete app in Lesson 16!* 🚀
