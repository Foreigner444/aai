# Project 2: Agent System Design - Lesson 2: System Prompts and Instructions

## A. Concept Overview

**What & Why:** System prompts are the "brain training" for your AI agent. They define how the AI thinks, what tone to use, what knowledge to apply, and what behaviors to follow. Think of it as the agent's instruction manual written in its preferred language.

**Analogy:** System prompts are like job descriptions for specialized AI workers:
- **Prompt = Job Description** - Defines role, responsibilities, and working style
- **Task = Daily Task** - Specific work to do on a given day
- **Response = Work Output** - Professional deliverable following company standards

For example, a "Customer Support" prompt teaches the agent to be empathetic and solution-focused, while a "Technical Analyst" prompt makes it more formal and data-driven.

**Type Safety Benefit:** System prompts work alongside Pydantic models to ensure your agent not only produces the right structure but also thinks and communicates in the exact style you need for your application.

## B. Code Implementation

### File Structure
```
agent_prompts_advanced/
├── main.py                 # Advanced agent with multiple prompts
├── models.py              # Pydantic models
├── agent_configs.py       # Different agent configurations
├── prompt_templates.py    # Reusable prompt components
└── requirements.txt       # Dependencies
```

### Complete Code Implementation

**File: models.py**
```python
from pydantic import BaseModel, Field
from typing import List, Literal, Optional
from enum import Enum

class CommunicationStyle(str, Enum):
    PROFESSIONAL = "professional"
    CASUAL = "casual" 
    TECHNICAL = "technical"
    EMPATHETIC = "empathetic"

class AnalysisOutput(BaseModel):
    """Output model for analysis tasks"""
    summary: str = Field(..., description="Brief summary of findings")
    key_points: List[str] = Field(..., description="Main points identified")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence in analysis")
    recommendations: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)

class SupportResponse(BaseModel):
    """Output model for customer support tasks"""
    response_text: str = Field(..., description="Customer-facing response")
    tone: CommunicationStyle = Field(..., description="Communication tone used")
    resolution_steps: List[str] = Field(..., description="Steps to resolve issue")
    escalation_needed: bool = Field(False, description="Requires human escalation")
    follow_up_required: bool = Field(False, description="Needs follow-up")
```

**File: prompt_templates.py**
```python
from typing import Literal

class PromptTemplates:
    """Reusable prompt components"""
    
    BASE_SYSTEM_PROMPT = '''
    You are a specialized AI assistant with expertise in {domain}.
    Your responses must always be:
    - Accurate and based on factual information
    - Structured according to the required output format
    - Appropriate for the specified communication style
    
    Always think step by step before responding.
    '''
    
    TECHNICAL_ANALYST_PROMPT = '''
    {base_prompt}
    
    You are a technical analyst with deep expertise in software development and system architecture.
    
    Communication Guidelines:
    - Use precise technical terminology
    - Include specific implementation details
    - Reference best practices and industry standards
    - Be concise but comprehensive
    - Focus on practical, actionable insights
    
    Analysis Framework:
    1. Break down complex problems into components
    2. Identify potential technical challenges
    3. Suggest proven solution patterns
    4. Consider scalability and maintainability
    5. Highlight security and performance implications
    '''
    
    CUSTOMER_SUPPORT_PROMPT = '''
    {base_prompt}
    
    You are a customer support representative who genuinely cares about helping users.
    
    Communication Guidelines:
    - Use warm, empathetic language
    - Acknowledge the user's frustration or concern
    - Provide clear, step-by-step solutions
    - Offer alternative approaches when possible
    - End with positive reassurance
    
    Problem-Solving Approach:
    1. Express empathy and understanding
    2. Clarify the specific issue
    3. Provide immediate actionable steps
    4. Offer additional resources if helpful
    5. Confirm the solution works for them
    '''
```

**File: agent_configs.py**
```python
from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
from prompt_templates import PromptTemplates

def create_technical_analyst_agent():
    """Create an agent specialized in technical analysis"""
    model = GeminiModel('gemini-pro')
    
    system_prompt = PromptTemplates.TECHNICAL_ANALYST_PROMPT.format(
        base_prompt=PromptTemplates.BASE_SYSTEM_PROMPT.format(domain="software development"),
        style="professional"
    )
    
    return Agent(
        model=model,
        result_type=AnalysisOutput,
        system_prompt=system_prompt,
        temperature=0.3,  # Lower temperature for more consistent, focused responses
        max_tokens=1500
    )

def create_support_agent():
    """Create an agent specialized in customer support"""
    model = GeminiModel('gemini-flash')  # Faster for conversational responses
    
    system_prompt = PromptTemplates.CUSTOMER_SUPPORT_PROMPT.format(
        base_prompt=PromptTemplates.BASE_SYSTEM_PROMPT.format(domain="customer service"),
        style="empathetic"
    )
    
    return Agent(
        model=model,
        result_type=SupportResponse,
        system_prompt=system_prompt,
        temperature=0.7,  # Higher temperature for more natural conversation
        max_tokens=800
    )

def create_adaptive_agent(output_type, domain, style):
    """Create an agent with dynamic prompting"""
    model = GeminiModel('gemini-pro')
    
    system_prompt = f'''
    You are a specialized AI assistant in {domain} with a {style} communication style.
    
    Your role:
    - Understand the user's needs accurately
    - Provide relevant, actionable responses
    - Maintain consistent tone throughout the conversation
    - Follow the structured output format exactly
    
    Always remember: You are representing a professional service that values quality and user satisfaction.
    '''
    
    return Agent(
        model=model,
        result_type=output_type,
        system_prompt=system_prompt,
        temperature=0.5  # Balanced creativity and consistency
    )
```

**File: main.py**
```python
from agent_configs import create_technical_analyst_agent, create_support_agent, create_adaptive_agent
from models import AnalysisOutput, SupportResponse, CommunicationStyle

def demonstrate_technical_analysis():
    """Show technical analysis agent in action"""
    print("🔬 TECHNICAL ANALYSIS AGENT DEMO")
    print("=" * 50)
    
    # Create and configure technical analyst
    analyst = create_technical_analyst_agent()
    
    # Example technical analysis task
    task = """
    Analyze this system architecture for a microservices application:
    
    - Frontend: React SPA served via CDN
    - API Gateway: Kong with rate limiting
    - Microservices: 12 Node.js services in Kubernetes
    - Database: PostgreSQL with read replicas
    - Cache: Redis cluster
    - Message Queue: RabbitMQ
    
    Consider scalability, reliability, and security implications.
    """
    
    print(f"📋 Analyzing: Microservices Architecture")
    result = analyst.run_sync(task)
    
    print(f"✅ Summary: {result.data.summary}")
    print(f"🎯 Key Points:")
    for point in result.data.key_points:
        print(f"   • {point}")
    print(f"📊 Confidence: {result.data.confidence_score:.1%}")
    print(f"🔧 Recommendations:")
    for rec in result.data.recommendations:
        print(f"   1. {rec}")
    print(f"⚠️ Risks:")
    for risk in result.data.risks:
        print(f"   • {risk}")

def demonstrate_customer_support():
    """Show customer support agent in action"""
    print("\n💬 CUSTOMER SUPPORT AGENT DEMO")
    print("=" * 50)
    
    # Create and configure support agent
    support_agent = create_support_agent()
    
    # Example customer support scenario
    customer_issue = """
    A customer contacts us saying:
    "I've been trying to update my payment information for the past hour 
    but the system keeps showing an error. I'm supposed to be charged today 
    and I'm really worried about losing access to my account. This is 
    frustrating!"
    """
    
    print(f"👤 Customer Issue: Payment Update Problem")
    result = support_agent.run_sync(customer_issue)
    
    print(f"💬 Response: {result.data.response_text}")
    print(f"🎭 Tone: {result.data.tone}")
    print(f"🛠️ Resolution Steps:")
    for step in result.data.resolution_steps:
        print(f"   1. {step}")
    print(f"🚨 Escalation Needed: {'Yes' if result.data.escalation_needed else 'No'}")
    print(f"📞 Follow-up Required: {'Yes' if result.data.follow_up_required else 'No'}")

def demonstrate_adaptive_behavior():
    """Show how the same task gets different responses based on prompt design"""
    print("\n🔄 ADAPTIVE BEHAVIOR DEMO")
    print("=" * 50)
    
    # Same input, different agent personalities
    input_text = "How do I optimize my database queries?"
    
    # Technical analyst approach
    tech_agent = create_adaptive_agent(
        output_type=AnalysisOutput,
        domain="database optimization",
        style="technical"
    )
    
    tech_result = tech_agent.run_sync(input_text)
    print(f"🔬 Technical Response: {tech_result.data.summary}")
    
    # Support representative approach  
    support_agent = create_adaptive_agent(
        output_type=SupportResponse,
        domain="customer education",
        style="empathetic"
    )
    
    support_result = support_agent.run_sync(input_text)
    print(f"💬 Support Response: {support_result.data.response_text}")

def main():
    """Run all demonstrations"""
    print("🎯 ADVANCED AGENT PROMPTING DEMONSTRATIONS")
    print("=" * 60)
    print("Watch how different system prompts create different AI personalities!")
    print()
    
    demonstrate_technical_analysis()
    demonstrate_customer_support()
    demonstrate_adaptive_behavior()
    
    print("\n🎉 Key Takeaway: System prompts shape how your agent thinks and communicates!")

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

1. **Prompt Templates:** Reusable components that make your prompts consistent and maintainable
2. **Multiple Agent Configurations:** Different system prompts create different "personalities"
3. **Adaptive Agents:** Dynamic prompting based on context and requirements
4. **Temperature Settings:** Control response creativity vs. consistency
5. **Structured Communication:** Each agent type follows specific communication guidelines

### The "Why" Behind the Pattern

This approach ensures **behavior control** because:
- **Consistency:** Same type of request always gets same quality response
- **Customization:** Different domains get specialized expertise
- **Maintainability:** Prompts are modular and reusable
- **Safety:** System prompts provide guardrails for appropriate responses
- **Type Safety Integration:** Prompts work with Pydantic models for complete control

## C. Test & Apply

### How to Test It
```bash
# Set your Gemini API key
export GEMINI_API_KEY="your-api-key-here"

# Install dependencies
pip install -r requirements.txt

# Run the demonstrations
python main.py
```

### Expected Result
You'll see three different AI "personalities" in action:

```
🎯 ADVANCED AGENT PROMPTING DEMONSTRATIONS
============================================================
Watch how different system prompts create different AI personalities!

🔬 TECHNICAL ANALYSIS AGENT DEMO
==================================================
📋 Analyzing: Microservices Architecture
✅ Summary: The microservices architecture shows good separation of concerns...
🎯 Key Points:
   • Clear service boundaries with API gateway
   • Database scaling through read replicas
   • Message queue for async communication
📊 Confidence: 0.85
🔧 Recommendations:
   1. Implement circuit breakers for resilience
   2. Add distributed tracing for monitoring
⚠️ Risks:
   • Single point of failure at API gateway
   • Potential data consistency issues

💬 CUSTOMER SUPPORT AGENT DEMO
==================================================
👤 Customer Issue: Payment Update Problem
💬 Response: I completely understand your frustration with the payment update issue...
🎭 Tone: empathetic
🛠️ Resolution Steps:
   1. Acknowledge the customer's concern
   2. Guide through alternative payment update method
🚨 Escalation Needed: No
📞 Follow-up Required: No

🔄 ADAPTIVE BEHAVIOR DEMO
==================================================
🔬 Technical Response: Database query optimization involves several strategies...
💬 Support Response: I'd be happy to help you optimize your database queries!...
```

### Validation Examples
- ✅ **Technical Input:** "Analyze API performance bottlenecks" → Detailed technical analysis
- ❌ **Wrong Tone:** If support agent responds too formally → Check prompt communication guidelines
- ❌ **Inconsistent Style:** If technical analysis lacks specifics → Verify technical prompt includes detail requirements

### Type Checking
```bash
# Run type checking to ensure models are properly structured
mypy main.py agent_configs.py models.py
```

## D. Common Stumbling Blocks

### Proactive Debugging
**Common Mistake #1: Vague System Prompts**
```
❌ Error: Agent provides generic, unhelpful responses
✅ Fix: Be specific about communication style, expertise level, and response structure

# Instead of:
"You are a helpful assistant"

# Use:
"You are a technical analyst specializing in database optimization. 
Provide specific, actionable recommendations with implementation details."
```

**Common Mistake #2: Conflicting Instructions**
```
❌ Error: Agent doesn't know which instruction to follow
✅ Fix: Prioritize instructions and avoid contradictory requirements

# Good: Clear hierarchy
"Your primary goal is technical accuracy. Secondary: be concise. 
Never compromise accuracy for brevity."

# Bad: Conflicting requirements
"Be very detailed but also be very brief"
```

**Common Mistake #3: Temperature Mismatch**
```
❌ Error: Technical analysis too creative, support responses too rigid
✅ Fix: Match temperature to use case

# Technical analysis: Lower temperature (0.2-0.4) for consistency
# Creative writing: Higher temperature (0.7-0.9) for variety  
# Customer support: Medium temperature (0.6-0.8) for natural conversation
```

### Type Safety Gotchas
- **Prompt injection:** Don't include your Pydantic model structure in the system prompt
- **Role conflicts:** System prompt and result_type should complement, not conflict
- **Temperature effects:** Higher temperature can cause more validation errors
- **Token limits:** Very long system prompts can exceed model context limits

### Prompt Engineering Best Practices
1. **Be specific about output format** in the prompt AND in your Pydantic model
2. **Test edge cases** by providing intentionally difficult inputs
3. **Use examples** in your prompts for complex output formats
4. **Iterate based on results** - system prompts need refinement
5. **Document your choices** - explain why you chose specific prompting strategies

## Ready for the Next Step?
You now understand how system prompts shape your agent's "personality" and behavior! 

**Next:** We'll explore **Custom Dependencies** - how to give your agents access to external context, tools, and data sources.

System prompts are like giving your AI a specific education and personality. The right prompt can transform a generic AI into a specialized expert for your exact use case! 🧠

**Ready for dependencies, or want to experiment with different prompting styles first?** 🤔
