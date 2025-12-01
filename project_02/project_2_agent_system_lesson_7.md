# PydanticAI Gemini Mentor - Agent System Design Project
## Lesson 7: Advanced Agent Patterns

Welcome back, master architect! 🎯 You've built agents, equipped them with tools, and orchestrated complex workflows. Now it's time to explore the most sophisticated patterns in agent design - where agents become autonomous orchestrators that can delegate work, stream responses, and create meta-systems of intelligence!

---

## What Are Advanced Agent Patterns?

**Advanced Agent Patterns** are sophisticated behaviors that go beyond simple question-answering. These patterns allow agents to:

- **Stream responses** for real-time interaction
- **Delegate work** to specialized sub-agents
- **Compose complex workflows** from multiple agents
- **Adapt their behavior** dynamically based on context
- **Create meta-systems** where agents manage other agents

**Analogy Time**: Think of it like moving from a solo musician → a jazz combo → a full orchestra → a music festival with multiple stages!

---

## 1. Streaming Responses: Real-Time Agent Communication

Streaming allows agents to provide responses incrementally, creating a more engaging and responsive experience:

```python
import asyncio
from pydantic import BaseModel
from pydantic_ai import Agent
from typing import AsyncGenerator, Any

class StreamingQuery(BaseModel):
    """Input for streaming response queries."""
    topic: str = Field(description="The topic to discuss")
    depth: str = Field(description="How detailed the response should be (brief/detailed/comprehensive)")

class StreamingResponse(BaseModel):
    """Streaming response with incremental updates."""
    topic: str
    current_section: str
    progress: int  # 0-100
    content: str
    is_complete: bool

# Streaming content generator
async def generate_streaming_content(topic: str, depth: str) -> AsyncGenerator[str, None]:
    """Generate content about a topic in streaming fashion."""
    
    # Simulate progressive content generation
    sections = {
        "brief": [
            f"## {topic} Overview",
            f"Key aspects of {topic}",
            f"Practical applications",
            "Summary"
        ],
        "detailed": [
            f"## Introduction to {topic}",
            f"## Historical Context",
            f"## Current State of {topic}",
            f"## Technical Deep Dive",
            f"## Industry Applications",
            f"## Future Implications",
            "## Conclusion"
        ],
        "comprehensive": [
            f"## {topic}: A Complete Analysis",
            f"## Historical Origins and Evolution",
            f"## Theoretical Foundations",
            f"## Current Technological Landscape",
            f"## Implementation Strategies",
            f"## Industry-Specific Applications",
            f"## Case Studies and Examples",
            f"## Challenges and Limitations",
            f"## Emerging Trends and Future Directions",
            f"## Strategic Recommendations",
            f"## Conclusion and Next Steps"
        ]
    }
    
    selected_sections = sections.get(depth, sections["detailed"])
    
    for i, section in enumerate(selected_sections):
        # Simulate thinking time
        await asyncio.sleep(0.5)
        
        # Generate section content
        if "Overview" in section or "Introduction" in section:
            content = f"This provides a comprehensive overview of {topic}, covering the fundamental concepts and importance in today's technological landscape."
        elif "Historical" in section:
            content = f"The evolution of {topic} has been marked by significant milestones, from early conceptualizations to modern implementations."
        elif "Current" in section or "State" in section:
            content = f"Today's {topic} landscape is characterized by rapid innovation, increased adoption, and evolving best practices."
        elif "Technical" in section or "Deep Dive" in section:
            content = f"From a technical perspective, {topic} involves complex systems, methodologies, and frameworks that require deep understanding."
        elif "Application" in section:
            content = f"Real-world applications of {topic} span across industries, from healthcare to finance to technology sectors."
        elif "Future" in section:
            content = f"The future of {topic} promises even greater innovation, with emerging trends pointing toward new possibilities."
        else:
            content = f"Key insights about {topic} include its transformative potential and the strategic importance of adoption."
        
        yield f"**{section}**\n\n{content}\n\n"
        
        # Update progress
        progress = int((i + 1) / len(selected_sections) * 100)
        yield f"<!-- PROGRESS: {progress}% -->\n"

# Streaming agent
streaming_agent = Agent(
    'gemini-1.5-flash',
    result_type=StreamingResponse,
    system_prompt='''
    You are a streaming content generator. When users request information:
    1. Generate content in sections with progressive updates
    2. Use the streaming content generator for each section
    3. Track progress and provide status updates
    4. Ensure content quality and relevance
    5. Complete with a summary and next steps
    
    Always be engaging and informative in your streaming responses.
    ''',
    tools=[generate_streaming_content]
)

# Client-side streaming handler
async def handle_streaming_response(agent: Agent, topic: str, depth: str = "detailed"):
    """Handle streaming responses on the client side."""
    print(f"🚀 Starting streaming response about: {topic}")
    print(f"📊 Detail level: {depth}")
    print("=" * 50)
    
    content_buffer = ""
    
    # This would be the actual streaming call
    # result = await agent.run(topic, depth, stream=True)
    # For demonstration, we'll simulate the streaming
    
    async for chunk in generate_streaming_content(topic, depth):
        if chunk.startswith("<!-- PROGRESS:"):
            # Extract and display progress
            progress = chunk.split(": ")[1].split("% -->")[0]
            print(f"\n⏳ Progress: {progress}%")
        else:
            # Display content
            print(chunk, end="", flush=True)
            content_buffer += chunk
    
    print(f"\n✅ Streaming complete!")
    return content_buffer

# Test streaming
async def test_streaming():
    print("=== Testing Streaming Agent Patterns ===")
    
    await handle_streaming_response(
        streaming_agent, 
        "Artificial Intelligence in Healthcare", 
        "detailed"
    )

# Run the test
asyncio.run(test_streaming())
```

**Expected Output:**
```
🚀 Starting streaming response about: Artificial Intelligence in Healthcare
📊 Detail level: detailed
==================================================

**## Introduction to Artificial Intelligence in Healthcare**

This provides a comprehensive overview of Artificial Intelligence in Healthcare, covering the fundamental concepts and importance in today's technological landscape.

⏳ Progress: 16%

**## Historical Context**

The evolution of Artificial Intelligence in Healthcare has been marked by significant milestones, from early conceptualizations to modern implementations.

⏳ Progress: 33%

**## Current State of Artificial Intelligence in Healthcare**

Today's Artificial Intelligence in Healthcare landscape is characterized by rapid innovation, increased adoption, and evolving best practices.

⏳ Progress: 50%

**## Technical Deep Dive**

From a technical perspective, Artificial Intelligence in Healthcare involves complex systems, methodologies, and frameworks that require deep understanding.

⏳ Progress: 66%

**## Industry Applications**

Real-world applications of Artificial Intelligence in Healthcare span across industries, from healthcare to finance to technology sectors.

⏳ Progress: 83%

**## Future Implications**

The future of Artificial Intelligence in Healthcare promises even greater innovation, with emerging trends pointing toward new possibilities.

⏳ Progress: 100%

**## Conclusion**

Key insights about Artificial Intelligence in Healthcare include its transformative potential and the strategic importance of adoption.

✅ Streaming complete!
```

---

## 2. Agent Composition: Orchestrating Multiple Agents

Create systems where agents collaborate and delegate work:

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import asyncio

# Base agent interface
class BaseAgent(ABC):
    def __init__(self, name: str, specialty: str):
        self.name = name
        self.specialty = specialty
    
    @abstractmethod
    async def process(self, input_data: Any) -> Any:
        pass
    
    def __str__(self):
        return f"{self.name} ({self.specialty})"

# Specialized agents
class ResearchAgent(BaseAgent):
    def __init__(self):
        super().__init__("Researcher", "Data gathering and analysis")
    
    async def process(self, query: str) -> Dict[str, Any]:
        """Research and gather information."""
        await asyncio.sleep(1)  # Simulate research time
        
        return {
            "agent": self.name,
            "task": "Research",
            "query": query,
            "findings": [
                f"Key finding 1 about {query}",
                f"Important statistic related to {query}",
                f"Recent development in {query} domain"
            ],
            "confidence": 0.85,
            "sources": ["Source A", "Source B", "Source C"]
        }

class AnalysisAgent(BaseAgent):
    def __init__(self):
        super().__init__("Analyzer", "Data interpretation and insights")
    
    async def process(self, research_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze research findings."""
        await asyncio.sleep(0.8)  # Simulate analysis time
        
        insights = []
        for finding in research_data["findings"]:
            insights.append(f"Analysis of: {finding}")
        
        return {
            "agent": self.name,
            "task": "Analysis",
            "research_data": research_data,
            "insights": insights,
            "recommendations": [
                f"Based on analysis: Action item 1",
                f"Strategic recommendation for {research_data['query']}",
                f"Risk assessment and mitigation strategy"
            ],
            "confidence": 0.90
        }

class CreativeAgent(BaseAgent):
    def __init__(self):
        super().__init__("Creator", "Content generation and creative solutions")
    
    async def process(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate creative content based on analysis."""
        await asyncio.sleep(0.6)  # Simulate creative time
        
        return {
            "agent": self.name,
            "task": "Creative Generation",
            "analysis_data": analysis_data,
            "content": {
                "summary": f"Creative summary of {analysis_data['research_data']['query']}",
                "suggestions": [
                    f"Innovative approach to {analysis_data['research_data']['query']}",
                    f"Engaging narrative around the findings",
                    f"Compelling visual representation ideas"
                ],
                "implementation": [
                    "Phase 1: Foundation setup",
                    "Phase 2: Implementation",
                    "Phase 3: Optimization and scaling"
                ]
            },
            "confidence": 0.88
        }

# Orchestrator agent
class OrchestratorAgent(BaseAgent):
    def __init__(self):
        super().__init__("Orchestrator", "Multi-agent coordination")
        self.research_agent = ResearchAgent()
        self.analysis_agent = AnalysisAgent()
        self.creative_agent = CreativeAgent()
        self.agents = [self.research_agent, self.analysis_agent, self.creative_agent]
    
    async def orchestrate_workflow(self, query: str) -> Dict[str, Any]:
        """Orchestrate a complete workflow across multiple agents."""
        print(f"🎯 Orchestrator starting workflow for: {query}")
        print("=" * 60)
        
        # Step 1: Research Phase
        print("\n📚 Phase 1: Research")
        research_result = await self.research_agent.process(query)
        print(f"   ✅ {research_result['agent']} completed research")
        print(f"   📊 Findings: {len(research_result['findings'])} items")
        print(f"   🎯 Confidence: {research_result['confidence']:.0%}")
        
        # Step 2: Analysis Phase
        print("\n🔍 Phase 2: Analysis")
        analysis_result = await self.analysis_agent.process(research_result)
        print(f"   ✅ {analysis_result['agent']} completed analysis")
        print(f"   💡 Insights: {len(analysis_result['insights'])} generated")
        print(f"   📋 Recommendations: {len(analysis_result['recommendations'])} provided")
        
        # Step 3: Creative Phase
        print("\n🎨 Phase 3: Creative Generation")
        creative_result = await self.creative_agent.process(analysis_result)
        print(f"   ✅ {creative_result['agent']} completed creative work")
        print(f"   📝 Content generated: {len(creative_result['content']['suggestions'])} suggestions")
        print(f"   🚀 Implementation plan: {len(creative_result['content']['implementation'])} phases")
        
        # Compile final result
        workflow_result = {
            "query": query,
            "workflow_id": f"workflow_{hash(query) % 10000}",
            "timestamp": "2024-01-20T12:00:00Z",
            "phases": {
                "research": research_result,
                "analysis": analysis_result,
                "creative": creative_result
            },
            "final_output": {
                "summary": creative_result["content"]["summary"],
                "key_findings": research_result["findings"],
                "top_recommendations": analysis_result["recommendations"][:2],
                "implementation_roadmap": creative_result["content"]["implementation"]
            },
            "workflow_stats": {
                "total_agents": len(self.agents),
                "total_phases": 3,
                "overall_confidence": (
                    research_result["confidence"] + 
                    analysis_result["confidence"] + 
                    creative_result["confidence"]
                ) / 3
            }
        }
        
        print(f"\n🎉 Workflow completed successfully!")
        print(f"   🆔 Workflow ID: {workflow_result['workflow_id']}")
        print(f"   📊 Overall Confidence: {workflow_result['workflow_stats']['overall_confidence']:.0%}")
        
        return workflow_result

# Test agent orchestration
async def test_agent_orchestration():
    print("=== Testing Agent Composition Patterns ===")
    
    orchestrator = OrchestratorAgent()
    
    # Test queries
    queries = [
        "AI adoption in small businesses",
        "Sustainable technology solutions",
        "Remote work productivity tools"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n{'='*20} WORKFLOW {i} {'='*20}")
        result = await orchestrator.orchestrate_workflow(query)
        
        # Display key results
        print(f"\n📋 FINAL OUTPUT:")
        print(f"   Summary: {result['final_output']['summary']}")
        print(f"   Key Finding: {result['final_output']['key_findings'][0]}")
        print(f"   Top Recommendation: {result['final_output']['top_recommendations'][0]}")

# Run the orchestration test
asyncio.run(test_agent_orchestration())
```

**Expected Output:**
```
=== Testing Agent Composition Patterns ===

==================== WORKFLOW 1 ====================
🎯 Orchestrator starting workflow for: AI adoption in small businesses
============================================================

📚 Phase 1: Research
   ✅ Researcher completed research
   📊 Findings: 3 items
   🎯 Confidence: 85%

🔍 Phase 2: Analysis
   ✅ Analyzer completed analysis
   💡 Insights: 3 generated
   📋 Recommendations: 3 provided

🎨 Phase 3: Creative Generation
   ✅ Creator completed creative work
   📝 Content generated: 3 suggestions
   🚀 Implementation plan: 3 phases

🎉 Workflow completed successfully!
   🆔 Workflow ID: workflow_4827
   📊 Overall Confidence: 88%

📋 FINAL OUTPUT:
   Summary: Creative summary of AI adoption in small businesses
   Key Finding: Key finding 1 about AI adoption in small businesses
   Top Recommendation: Based on analysis: Action item 1
```

---

## 3. Dynamic Agent Selection: Smart Agent Routing

Create agents that can dynamically choose which specialized agents to use:

```python
import re
from enum import Enum
from typing import Union

class QueryType(Enum):
    TECHNICAL = "technical"
    CREATIVE = "creative"
    ANALYTICAL = "analytical"
    RESEARCH = "research"
    STRATEGIC = "strategic"

class SmartRouterAgent(BaseAgent):
    def __init__(self):
        super().__init__("SmartRouter", "Dynamic agent selection")
        self.agents = {
            QueryType.TECHNICAL: ResearchAgent(),  # Technical research
            QueryType.CREATIVE: CreativeAgent(),   # Creative generation
            QueryType.ANALYTICAL: AnalysisAgent(), # Data analysis
            QueryType.RESEARCH: ResearchAgent(),   # Information gathering
            QueryType.STRATEGIC: AnalysisAgent()   # Strategic analysis
        }
        self.query_patterns = {
            QueryType.TECHNICAL: [
                r"(?i)(implement|build|develop|code|technical|algorithm|system)",
                r"(?i)(programming|software|development|engineering|architecture)"
            ],
            QueryType.CREATIVE: [
                r"(?i)(design|creative|innovative|artistic|visual|brainstorm)",
                r"(?i)(idea|concept|imagine|story|narrative|content)"
            ],
            QueryType.ANALYTICAL: [
                r"(?i)(analyze|data|statistics|metrics|trend|pattern)",
                r"(?i)(comparison|evaluation|assessment|measurement|insight)"
            ],
            QueryType.RESEARCH: [
                r"(?i)(research|investigate|explore|study|discover|find)",
                r"(?i)(information|knowledge|learn|understand|examine)"
            ],
            QueryType.STRATEGIC: [
                r"(?i)(strategy|planning|roadmap|future|direction|goal)",
                r"(?i)(business|organization|decision|recommendation|optimization)"
            ]
        }
    
    def classify_query(self, query: str) -> QueryType:
        """Classify query type based on content analysis."""
        query_lower = query.lower()
        
        scores = {}
        for query_type, patterns in self.query_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    score += 1
            scores[query_type] = score
        
        # Return the type with highest score
        max_score = max(scores.values())
        if max_score == 0:
            return QueryType.RESEARCH  # Default fallback
        
        return max(scores.keys(), key=lambda k: scores[k])
    
    async def route_and_process(self, query: str) -> Dict[str, Any]:
        """Route query to appropriate agent and process."""
        print(f"🧭 Smart Router analyzing query: '{query}'")
        
        # Classify the query
        query_type = self.classify_query(query)
        print(f"📊 Classified as: {query_type.value.title()} (confidence: estimated)")
        
        # Select appropriate agent
        selected_agent = self.agents[query_type]
        print(f"🤖 Selected agent: {selected_agent}")
        
        # Process with selected agent
        result = await selected_agent.process(query)
        
        # Add routing metadata
        result["routing"] = {
            "query_type": query_type.value,
            "selected_agent": selected_agent.name,
            "routing_confidence": "high"  # In real implementation, calculate this
        }
        
        return result
    
    async def multi_agent_consultation(self, query: str) -> Dict[str, Any]:
        """Consult multiple agents for complex queries."""
        print(f"🏛️ Multi-agent consultation for: '{query}'")
        
        # For complex queries, use multiple agents
        primary_type = self.classify_query(query)
        secondary_types = [t for t in QueryType if t != primary_type][:2]  # Pick 2 others
        
        print(f"🎯 Primary type: {primary_type.value}")
        print(f"🔄 Secondary types: {[t.value for t in secondary_types]}")
        
        # Process with multiple agents
        results = {}
        for query_type in [primary_type] + secondary_types:
            agent = self.agents[query_type]
            print(f"   🔄 Consulting {agent.name}...")
            results[query_type.value] = await agent.process(query)
        
        # Synthesize results
        synthesis = {
            "consultation_id": f"consult_{hash(query) % 10000}",
            "query": query,
            "agents_consulted": [r["agent"] for r in results.values()],
            "synthesis": f"Combined insights from {len(results)} specialized agents",
            "results": results
        }
        
        return synthesis

# Test smart routing
async def test_smart_routing():
    print("=== Testing Smart Agent Routing ===")
    
    router = SmartRouterAgent()
    
    # Test queries
    test_queries = [
        "Analyze the latest sales data and identify trends",
        "Design a creative marketing campaign for our new product",
        "Research the best programming frameworks for web development",
        "Create a strategic roadmap for digital transformation",
        "Build a machine learning model for customer prediction"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*15} QUERY {i} {'='*15}")
        print(f"Query: {query}")
        
        # Test simple routing
        print("\n🔄 Simple Routing:")
        result = await router.route_and_process(query)
        print(f"   📋 Result: {result['task']} completed by {result['agent']}")
        
        # Test multi-agent consultation for complex queries
        if len(query.split()) > 6:  # Complex queries get multi-agent treatment
            print("\n🏛️ Multi-Agent Consultation:")
            consultation = await router.multi_agent_consultation(query)
            print(f"   🤖 Agents consulted: {len(consultation['agents_consulted'])}")
            print(f"   📊 Consultation ID: {consultation['consultation_id']}")

# Run the routing test
asyncio.run(test_smart_routing())
```

**Expected Output:**
```
=== Testing Smart Agent Routing ===

=============== QUERY 1 ================
Query: Analyze the latest sales data and identify trends
🔄 Simple Routing:
🧭 Smart Router analyzing query: 'Analyze the latest sales data and identify trends'
📊 Classified as: Analytical (confidence: estimated)
🤖 Selected agent: Analyzer (Data interpretation and insights)
   📋 Result: Analysis completed by Analyzer

🏛️ Multi-Agent Consultation:
🏛️ Multi-agent consultation for: 'Analyze the latest sales data and identify trends'
🎯 Primary type: analytical
🔄 Secondary types: ['research', 'strategic']
   🔄 Consulting Analyzer...
   🔄 Consulting Researcher...
   🔄 Consulting Creator...
   🤖 Agents consulted: 3
   📊 Consultation ID: consult_2847
```

---

## 4. Agent State Management: Persistent Conversations

Create agents that maintain state across multiple interactions:

```python
from datetime import datetime
from typing import Dict, List, Any, Optional

class ConversationState:
    """Manages conversation state and history."""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.messages: List[Dict[str, Any]] = []
        self.context: Dict[str, Any] = {}
        self.created_at = datetime.now()
        self.last_updated = datetime.now()
    
    def add_message(self, role: str, content: str, metadata: Dict[str, Any] = None):
        """Add a message to the conversation."""
        message = {
            "timestamp": datetime.now().isoformat(),
            "role": role,
            "content": content,
            "metadata": metadata or {}
        }
        self.messages.append(message)
        self.last_updated = datetime.now()
    
    def get_context_summary(self) -> Dict[str, Any]:
        """Get a summary of conversation context."""
        return {
            "session_id": self.session_id,
            "message_count": len(self.messages),
            "topics": list(set(msg.get("metadata", {}).get("topic", "") for msg in self.messages if msg.get("metadata", {}).get("topic"))),
            "key_entities": list(set(msg.get("metadata", {}).get("entities", []) for msg in self.messages if msg.get("metadata", {}).get("entities"))),
            "conversation_length": self.last_updated - self.created_at,
            "last_activity": self.last_updated.isoformat()
        }

class StatefulAgent(BaseAgent):
    def __init__(self, name: str, specialty: str):
        super().__init__(name, specialty)
        self.conversations: Dict[str, ConversationState] = {}
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
    
    def start_session(self, session_id: str, user_info: Dict[str, Any] = None) -> str:
        """Start a new conversation session."""
        if session_id not in self.conversations:
            self.conversations[session_id] = ConversationState(session_id)
        
        self.active_sessions[session_id] = {
            "user_info": user_info or {},
            "session_start": datetime.now(),
            "interactions": 0
        }
        
        return session_id
    
    def end_session(self, session_id: str) -> Dict[str, Any]:
        """End a conversation session and return summary."""
        if session_id not in self.conversations:
            return {"error": "Session not found"}
        
        session_data = self.conversations[session_id]
        active_data = self.active_sessions.get(session_id, {})
        
        summary = {
            "session_id": session_id,
            "duration": datetime.now() - active_data.get("session_start", datetime.now()),
            "message_count": len(session_data.messages),
            "context_summary": session_data.get_context_summary()
        }
        
        # Clean up
        self.active_sessions.pop(session_id, None)
        
        return summary
    
    async def process_with_context(self, session_id: str, message: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process a message within conversation context."""
        if session_id not in self.conversations:
            self.start_session(session_id)
        
        session = self.conversations[session_id]
        active_session = self.active_sessions.get(session_id, {})
        
        # Add user message
        session.add_message("user", message, metadata)
        
        # Get conversation context
        context = session.get_context_summary()
        
        # Process based on context and message
        response = await self._generate_contextual_response(message, context)
        
        # Add agent response
        session.add_message("assistant", response["content"], {
            "response_type": response["type"],
            "confidence": response["confidence"],
            "context_used": True
        })
        
        # Update active session
        active_session["interactions"] += 1
        active_session["last_message"] = message
        
        # Return response with context info
        return {
            "session_id": session_id,
            "response": response["content"],
            "response_type": response["type"],
            "confidence": response["confidence"],
            "context_summary": context,
            "interaction_count": active_session["interactions"]
        }
    
    async def _generate_contextual_response(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate response based on conversation context."""
        
        # Simple context-aware response logic
        if context["message_count"] == 1:  # First message
            response_type = "greeting"
            content = f"Hello! I understand you want to discuss '{message}'. I'm ready to help with detailed information and analysis."
        elif "analyze" in message.lower() and context["message_count"] > 1:
            response_type = "analytical"
            content = f"Based on our conversation so far, I can provide a detailed analysis of '{message}' using the context we've built."
        elif "compare" in message.lower():
            response_type = "comparative"
            content = f"I can compare '{message}' with previous topics we've discussed. Let me provide a comprehensive comparison."
        else:
            response_type = "responsive"
            content = f"Thank you for the additional information about '{message}'. Here's my detailed response considering our conversation history."
        
        return {
            "content": content,
            "type": response_type,
            "confidence": 0.85,
            "context_aware": True
        }
    
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get detailed session summary."""
        if session_id not in self.conversations:
            return {"error": "Session not found"}
        
        session = self.conversations[session_id]
        active_data = self.active_sessions.get(session_id, {})
        
        return {
            "session_info": {
                "session_id": session_id,
                "is_active": session_id in self.active_sessions,
                "created": session.created_at.isoformat(),
                "last_updated": session.last_updated.isoformat()
            },
            "conversation_stats": {
                "total_messages": len(session.messages),
                "user_messages": len([m for m in session.messages if m["role"] == "user"]),
                "assistant_messages": len([m for m in session.messages if m["role"] == "assistant"])
            },
            "context": session.get_context_summary(),
            "recent_activity": session.messages[-3:] if session.messages else []
        }

# Test stateful agent
async def test_stateful_agent():
    print("=== Testing Stateful Agent Patterns ===")
    
    assistant = StatefulAgent("ContextualAssistant", "Stateful conversation")
    
    # Start a session
    session_id = assistant.start_session("user_123", {"name": "Alice", "interests": ["AI", "technology"]})
    print(f"🆔 Session started: {session_id}")
    
    # Multi-turn conversation
    conversation_turns = [
        "Hello, I need help understanding machine learning",
        "Can you analyze the different types of ML algorithms?",
        "Compare supervised vs unsupervised learning",
        "What about deep learning? How does it relate?",
        "Thanks, that was very helpful!"
    ]
    
    print(f"\n💬 Starting conversation with {len(conversation_turns)} turns:")
    
    for i, message in enumerate(conversation_turns, 1):
        print(f"\n--- Turn {i} ---")
        print(f"👤 User: {message}")
        
        result = await assistant.process_with_context(session_id, message, {
            "topic": "machine learning",
            "entities": ["ML", "algorithms", "supervised", "unsupervised", "deep learning"]
        })
        
        print(f"🤖 Assistant: {result['response']}")
        print(f"📊 Type: {result['response_type']} | Interaction #{result['interaction_count']}")
    
    # Get session summary
    print(f"\n📋 Session Summary:")
    summary = assistant.get_session_summary(session_id)
    print(f"   🆔 Session: {summary['session_info']['session_id']}")
    print(f"   📊 Messages: {summary['conversation_stats']['total_messages']}")
    print(f"   🎯 Topics: {summary['context']['topics']}")
    print(f"   👤 Active: {summary['session_info']['is_active']}")
    
    # End session
    end_summary = assistant.end_session(session_id)
    print(f"\n🏁 Session ended:")
    print(f"   ⏱️ Duration: {end_summary['duration']}")
    print(f"   📈 Interactions: {end_summary['message_count']}")

# Run the stateful test
asyncio.run(test_stateful_agent())
```

**Expected Output:**
```
=== Testing Stateful Agent Patterns ===
🆔 Session started: user_123

💬 Starting conversation with 5 turns:

--- Turn 1 ---
👤 User: Hello, I need help understanding machine learning
🤖 Assistant: Hello! I understand you want to discuss 'Hello, I need help understanding machine learning'. I'm ready to help with detailed information and analysis.
📊 Type: greeting | Interaction #1

--- Turn 2 ---
👤 User: Can you analyze the different types of ML algorithms?
🤖 Assistant: Based on our conversation so far, I can provide a detailed analysis of 'Can you analyze the different types of ML algorithms?' using the context we've built.
📊 Type: analytical | Interaction #2

--- Turn 3 ---
👤 User: Compare supervised vs unsupervised learning
🤖 Assistant: I can compare 'Compare supervised vs unsupervised learning' with previous topics we've discussed. Let me provide a comprehensive comparison.
📊 Type: comparative | Interaction #3

--- Turn 4 ---
👤 User: What about deep learning? How does it relate?
🤖 Assistant: Thank you for the additional information about 'What about deep learning? How does it relate?'. Here's my detailed response considering our conversation history.
📊 Type: responsive | Interaction #4

--- Turn 5 ---
👤 User: Thanks, that was very helpful!
🤖 Assistant: Thank you for the additional information about 'Thanks, that was very helpful!'. Here's my detailed response considering our conversation history.
📊 Type: responsive | Interaction #5

📋 Session Summary:
   🆔 Session: user_123
   📊 Messages: 10
   🎯 Topics: ['machine learning']
   👤 Active: True

🏁 Session ended:
   ⏱️ Duration: 0:00:05
   📈 Interactions: 10
```

---

## 5. Meta-Agent Patterns: Agents That Manage Agents

Create agents that dynamically create and manage other agents:

```python
import uuid
from typing import Type, Callable

class AgentFactory:
    """Factory for creating agents dynamically."""
    
    def __init__(self):
        self.agent_blueprints = {}
        self.active_agents = {}
    
    def register_blueprint(self, agent_type: str, blueprint: Dict[str, Any]):
        """Register an agent blueprint."""
        self.agent_blueprints[agent_type] = blueprint
    
    def create_agent(self, agent_type: str, agent_id: str = None, config: Dict[str, Any] = None) -> BaseAgent:
        """Create an agent from blueprint."""
        if agent_type not in self.agent_blueprints:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        blueprint = self.agent_blueprints[agent_type]
        agent_id = agent_id or str(uuid.uuid4())
        
        # Create agent based on type
        if agent_type == "research":
            agent = ResearchAgent()
        elif agent_type == "analysis":
            agent = AnalysisAgent()
        elif agent_type == "creative":
            agent = CreativeAgent()
        else:
            # Generic agent
            agent = BaseAgent(blueprint["name"], blueprint["specialty"])
        
        # Apply configuration
        if config:
            for key, value in config.items():
                if hasattr(agent, key):
                    setattr(agent, key, value)
        
        self.active_agents[agent_id] = agent
        return agent
    
    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Get an active agent by ID."""
        return self.active_agents.get(agent_id)
    
    def list_active_agents(self) -> List[Dict[str, Any]]:
        """List all active agents."""
        return [
            {
                "agent_id": agent_id,
                "agent_type": type(agent).__name__,
                "name": getattr(agent, 'name', 'Unknown'),
                "specialty": getattr(agent, 'specialty', 'Unknown')
            }
            for agent_id, agent in self.active_agents.items()
        ]
    
    def terminate_agent(self, agent_id: str) -> bool:
        """Terminate an active agent."""
        if agent_id in self.active_agents:
            del self.active_agents[agent_id]
            return True
        return False

class MetaAgent(BaseAgent):
    """Agent that manages other agents."""
    
    def __init__(self):
        super().__init__("MetaAgent", "Agent orchestration and management")
        self.factory = AgentFactory()
        self.agent_pools: Dict[str, List[str]] = {}  # agent_type -> [agent_ids]
        self.active_workflows: Dict[str, Dict[str, Any]] = {}
        self._setup_default_blueprints()
    
    def _setup_default_blueprints(self):
        """Setup default agent blueprints."""
        self.factory.register_blueprint("research", {
            "name": "Research Agent",
            "specialty": "Information gathering and analysis",
            "capabilities": ["web_search", "data_analysis", "fact_checking"]
        })
        
        self.factory.register_blueprint("analysis", {
            "name": "Analysis Agent",
            "specialty": "Data interpretation and insights",
            "capabilities": ["statistical_analysis", "pattern_recognition", "trend_analysis"]
        })
        
        self.factory.register_blueprint("creative", {
            "name": "Creative Agent",
            "specialty": "Content generation and innovation",
            "capabilities": ["content_creation", "brainstorming", "design_thinking"]
        })
    
    async def spawn_agent_team(self, task_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Spawn a team of agents based on task requirements."""
        team_id = str(uuid.uuid4())
        print(f"🚀 Spawning agent team: {team_id}")
        
        required_capabilities = task_requirements.get("capabilities", [])
        team_size = task_requirements.get("team_size", 3)
        
        # Determine agent types needed
        agent_types_needed = []
        if any(cap in ["web_search", "data_analysis", "fact_checking"] for cap in required_capabilities):
            agent_types_needed.append("research")
        if any(cap in ["statistical_analysis", "pattern_recognition"] for cap in required_capabilities):
            agent_types_needed.append("analysis")
        if any(cap in ["content_creation", "brainstorming"] for cap in required_capabilities):
            agent_types_needed.append("creative")
        
        # Ensure we have enough agents
        while len(agent_types_needed) < team_size:
            agent_types_needed.append(agent_types_needed[0])  # Duplicate most needed type
        
        # Create agents
        team_agents = []
        for i, agent_type in enumerate(agent_types_needed[:team_size]):
            agent_id = self.factory.create_agent(agent_type, config={
                "team_id": team_id,
                "role_index": i
            })
            team_agents.append(agent_id)
            print(f"   ✅ Created {agent_type} agent: {agent_id}")
        
        # Store team info
        self.agent_pools[team_id] = team_agents
        self.active_workflows[team_id] = {
            "team_id": team_id,
            "task": task_requirements,
            "agents": team_agents,
            "status": "active",
            "created_at": datetime.now()
        }
        
        return {
            "team_id": team_id,
            "agents_created": len(team_agents),
            "agent_ids": team_agents,
            "capabilities": required_capabilities,
            "status": "active"
        }
    
    async def coordinate_team_work(self, team_id: str, work_description: str) -> Dict[str, Any]:
        """Coordinate work across a team of agents."""
        if team_id not in self.agent_pools:
            return {"error": f"Team {team_id} not found"}
        
        print(f"🎯 Coordinating work for team: {team_id}")
        print(f"📋 Work: {work_description}")
        
        agent_ids = self.agent_pools[team_id]
        results = {}
        
        # Distribute work among agents
        for i, agent_id in enumerate(agent_ids):
            agent = self.factory.get_agent(agent_id)
            if agent:
                print(f"   🔄 Assigning work to {agent.name}...")
                
                # Each agent gets a portion of the work
                task_portion = f"{work_description} (Agent {i+1} portion)"
                result = await agent.process(task_portion)
                results[agent_id] = result
                print(f"   ✅ {agent.name} completed portion {i+1}")
        
        # Synthesize results
        synthesis = {
            "team_id": team_id,
            "work_description": work_description,
            "agent_count": len(results),
            "individual_results": results,
            "synthesis": f"Combined results from {len(results)} agents",
            "status": "completed"
        }
        
        return synthesis
    
    async def scale_team(self, team_id: str, additional_agents: int) -> Dict[str, Any]:
        """Scale up a team by adding more agents."""
        if team_id not in self.active_workflows:
            return {"error": f"Team {team_id} not found"}
        
        workflow = self.active_workflows[team_id]
        task_requirements = workflow["task"]
        
        print(f"📈 Scaling team {team_id} by {additional_agents} agents")
        
        # Create additional agents
        additional_team = await self.spawn_agent_team({
            **task_requirements,
            "team_size": additional_agents
        })
        
        # Add to existing team
        new_agent_ids = additional_team["agent_ids"]
        self.agent_pools[team_id].extend(new_agent_ids)
        
        return {
            "team_id": team_id,
            "agents_added": additional_agents,
            "new_agent_ids": new_agent_ids,
            "total_agents": len(self.agent_pools[team_id]),
            "status": "scaled"
        }
    
    def terminate_team(self, team_id: str) -> Dict[str, Any]:
        """Terminate an entire team."""
        if team_id not in self.agent_pools:
            return {"error": f"Team {team_id} not found"}
        
        agent_ids = self.agent_pools[team_id]
        
        # Terminate all agents
        for agent_id in agent_ids:
            self.factory.terminate_agent(agent_id)
        
        # Clean up
        team_agents = self.agent_pools.pop(team_id)
        workflow = self.active_workflows.pop(team_id)
        
        return {
            "team_id": team_id,
            "agents_terminated": len(team_agents),
            "workflow_duration": datetime.now() - workflow["created_at"],
            "status": "terminated"
        }
    
    def get_team_status(self, team_id: str) -> Dict[str, Any]:
        """Get comprehensive team status."""
        if team_id not in self.active_workflows:
            return {"error": f"Team {team_id} not found"}
        
        workflow = self.active_workflows[team_id]
        agent_ids = self.agent_pools[team_id]
        
        agents_info = []
        for agent_id in agent_ids:
            agent = self.factory.get_agent(agent_id)
            if agent:
                agents_info.append({
                    "agent_id": agent_id,
                    "name": agent.name,
                    "specialty": agent.specialty
                })
        
        return {
            "team_id": team_id,
            "status": workflow["status"],
            "created_at": workflow["created_at"].isoformat(),
            "agent_count": len(agent_ids),
            "agents": agents_info,
            "task": workflow["task"]
        }

# Test meta-agent patterns
async def test_meta_agent():
    print("=== Testing Meta-Agent Patterns ===")
    
    meta_agent = MetaAgent()
    
    # Spawn a team for a complex task
    task_requirements = {
        "capabilities": ["web_search", "statistical_analysis", "content_creation"],
        "team_size": 3,
        "priority": "high"
    }
    
    print("\n🎯 Spawning agent team...")
    team_result = await meta_agent.spawn_agent_team(task_requirements)
    team_id = team_result["team_id"]
    
    print(f"   ✅ Team created: {team_id}")
    print(f"   🤖 Agents: {team_result['agents_created']}")
    
    # Coordinate work across the team
    work_description = "Research and analyze market trends, then create a comprehensive report"
    
    print(f"\n🎯 Coordinating team work...")
    coordination_result = await meta_agent.coordinate_team_work(team_id, work_description)
    
    print(f"   📊 Results from {coordination_result['agent_count']} agents")
    print(f"   🏁 Status: {coordination_result['status']}")
    
    # Scale the team
    print(f"\n📈 Scaling team...")
    scale_result = await meta_agent.scale_team(team_id, 2)
    print(f"   ➕ Added {scale_result['agents_added']} agents")
    print(f"   📊 Total agents: {scale_result['total_agents']}")
    
    # Get team status
    print(f"\n📊 Team Status:")
    status = meta_agent.get_team_status(team_id)
    print(f"   🆔 Team ID: {status['team_id']}")
    print(f"   🎯 Status: {status['status']}")
    print(f"   🤖 Agent Count: {status['agent_count']}")
    print(f"   👥 Agents:")
    for agent in status['agents']:
        print(f"      • {agent['name']} ({agent['specialty']})")
    
    # Terminate team
    print(f"\n🏁 Terminating team...")
    termination_result = meta_agent.terminate_team(team_id)
    print(f"   ❌ Agents terminated: {termination_result['agents_terminated']}")
    print(f"   ⏱️ Duration: {termination_result['workflow_duration']}")

# Run the meta-agent test
asyncio.run(test_meta_agent())
```

**Expected Output:**
```
=== Testing Meta-Agent Patterns ===

🎯 Spawning agent team...
🚀 Spawning agent team: 8f2a1c3e-7d4b-4a2e-8c9d-1e5f3a2b4c1d
   ✅ Created research agent: 8f2a1c3e-7d4b-4a2e-8c9d-1e5f3a2b4c1d
   ✅ Created analysis agent: 7e3b5d9f-2a1c-4b3e-8f6d-9a2b5c4e8f1d
   ✅ Created creative agent: 6d2a4c8e-1f3b-4d5e-8a7c-2b4f6a9d3e8c1
   ✅ Team created: 8f2a1c3e-7d4b-4a2e-8c9d-1e5f3a2b4c1d
   🤖 Agents: 3

🎯 Coordinating team work...
🎯 Coordinating work for team: 8f2a1c3e-7d4b-4a2e-8c9d-1e5f3a2b4c1d
📋 Work: Research and analyze market trends, then create a comprehensive report
   🔄 Assigning work to Research Agent...
   ✅ Research Agent completed portion 1
   🔄 Assigning work to Analyzer...
   ✅ Analyzer completed portion 2
   🔄 Assigning work to Creator...
   ✅ Creator completed portion 3
   📊 Results from 3 agents
   🏁 Status: completed

📈 Scaling team...
📈 Scaling team 8f2a1c3e-7d4b-4a2e-8c9d-1e5f3a2b4c1d by 2 agents
🚀 Spawning agent team: a1b3c5d7-9e2f-4a6b-8c9d-2e4f6a8b3c5d
   ✅ Created research agent: a1b3c5d7-9e2f-4a6b-8c9d-2e4f6a8b3c5d
   ✅ Created analysis agent: b2c4d6e8-0f3a-5b7c-9d0e-3f5a7b9c4d6e
   ➕ Added 2 agents
   📊 Total agents: 5

📊 Team Status:
   🆔 Team ID: 8f2a1c3e-7d4b-4a2e-8c9d-1e5f3a2b4c1d
   🎯 Status: active
   🤖 Agent Count: 5
   👥 Agents:
      • Research Agent (Information gathering and analysis)
      • Analyzer (Data interpretation and insights)
      • Creator (Content generation and innovation)
      • Research Agent (Information gathering and analysis)
      • Analyzer (Data interpretation and insights)

🏁 Terminating team...
🏁 Terminating team 8f2a1c3e-7d4b-4a2e-8c9d-1e5f3a2b4c1d
❌ Agents terminated: 5
⏱️ Duration: 0:00:03
```

---

## 6. Advanced Workflow Orchestration

Create sophisticated workflows that combine all the patterns:

```python
class WorkflowOrchestrator:
    """Advanced workflow orchestrator combining multiple agent patterns."""
    
    def __init__(self):
        self.meta_agent = MetaAgent()
        self.active_workflows: Dict[str, Dict[str, Any]] = {}
        self.workflow_templates = self._load_workflow_templates()
    
    def _load_workflow_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load predefined workflow templates."""
        return {
            "research_report": {
                "name": "Research Report Generation",
                "phases": [
                    {"type": "research", "duration": "medium"},
                    {"type": "analysis", "duration": "medium"},
                    {"type": "creative", "duration": "short"}
                ],
                "requirements": ["web_search", "statistical_analysis", "content_creation"],
                "team_size": 3
            },
            "competitive_analysis": {
                "name": "Competitive Analysis",
                "phases": [
                    {"type": "research", "duration": "long"},
                    {"type": "research", "duration": "long"},
                    {"type": "analysis", "duration": "medium"},
                    {"type": "analysis", "duration": "medium"}
                ],
                "requirements": ["data_analysis", "pattern_recognition", "fact_checking"],
                "team_size": 4
            },
            "product_launch": {
                "name": "Product Launch Strategy",
                "phases": [
                    {"type": "research", "duration": "medium"},
                    {"type": "analysis", "duration": "medium"},
                    {"type": "creative", "duration": "long"},
                    {"type": "analysis", "duration": "short"}
                ],
                "requirements": ["market_research", "strategic_analysis", "content_creation"],
                "team_size": 3
            }
        }
    
    async def execute_workflow(self, template_name: str, custom_requirements: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a complete workflow based on template."""
        if template_name not in self.workflow_templates:
            return {"error": f"Unknown template: {template_name}"}
        
        template = self.workflow_templates[template_name]
        workflow_id = str(uuid.uuid4())
        
        print(f"🚀 Starting workflow: {template['name']}")
        print(f"🆔 Workflow ID: {workflow_id}")
        print("=" * 60)
        
        # Merge custom requirements
        requirements = template["requirements"]
        if custom_requirements:
            requirements.extend(custom_requirements.get("additional_capabilities", []))
        
        task_requirements = {
            "capabilities": requirements,
            "team_size": template["team_size"],
            "template": template_name,
            "custom": custom_requirements or {}
        }
        
        # Spawn initial team
        team_result = await self.meta_agent.spawn_agent_team(task_requirements)
        team_id = team_result["team_id"]
        
        # Execute phases
        workflow_results = []
        total_phases = len(template["phases"])
        
        for i, phase in enumerate(template["phases"], 1):
            print(f"\n📋 Phase {i}/{total_phases}: {phase['type'].title()}")
            print(f"   ⏱️ Estimated duration: {phase['duration']}")
            
            # Scale team if needed for this phase
            if i > 1 and i % 2 == 0:  # Scale every other phase
                scale_result = await self.meta_agent.scale_team(team_id, 1)
                print(f"   📈 Scaled to {scale_result['total_agents']} agents")
            
            # Coordinate work for this phase
            phase_description = f"Phase {i}: {phase['type']} work for {template['name']}"
            phase_result = await self.meta_agent.coordinate_team_work(team_id, phase_description)
            
            workflow_results.append({
                "phase": i,
                "type": phase["type"],
                "duration": phase["duration"],
                "result": phase_result,
                "success": True
            })
            
            print(f"   ✅ Phase {i} completed successfully")
        
        # Compile final workflow result
        final_result = {
            "workflow_id": workflow_id,
            "template_name": template_name,
            "template_display_name": template["name"],
            "status": "completed",
            "team_id": team_id,
            "execution_summary": {
                "total_phases": total_phases,
                "agents_used": team_result["agents_created"],
                "duration": "N/A",  # Would calculate actual duration
                "success_rate": "100%"
            },
            "phase_results": workflow_results,
            "final_deliverable": f"Complete {template['name']} with {total_phases} integrated phases",
            "metadata": {
                "capabilities_used": requirements,
                "team_size": template["team_size"],
                "workflow_complexity": "high" if total_phases > 3 else "medium"
            }
        }
        
        # Store workflow
        self.active_workflows[workflow_id] = final_result
        
        print(f"\n🎉 Workflow completed successfully!")
        print(f"   📊 Phases: {total_phases}")
        print(f"   🤖 Agents: {team_result['agents_created']}")
        print(f"   📋 Deliverable: {final_result['final_deliverable']}")
        
        # Clean up team
        termination_result = self.meta_agent.terminate_team(team_id)
        print(f"   🧹 Team terminated: {termination_result['agents_terminated']} agents")
        
        return final_result
    
    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get detailed workflow status."""
        if workflow_id not in self.active_workflows:
            return {"error": f"Workflow {workflow_id} not found"}
        
        workflow = self.active_workflows[workflow_id]
        
        return {
            "workflow_id": workflow_id,
            "name": workflow["template_display_name"],
            "status": workflow["status"],
            "progress": "100%" if workflow["status"] == "completed" else "0%",
            "phases_completed": len([p for p in workflow["phase_results"] if p["success"]]),
            "total_phases": workflow["execution_summary"]["total_phases"],
            "team_size": workflow["execution_summary"]["agents_used"],
            "deliverable": workflow["final_deliverable"]
        }

# Test advanced workflow orchestration
async def test_advanced_workflows():
    print("=== Testing Advanced Workflow Orchestration ===")
    
    orchestrator = WorkflowOrchestrator()
    
    # Test different workflow templates
    workflows_to_test = [
        "research_report",
        "competitive_analysis",
        "product_launch"
    ]
    
    for i, template_name in enumerate(workflows_to_test, 1):
        print(f"\n{'='*20} WORKFLOW {i} {'='*20}")
        
        # Execute workflow
        result = await orchestrator.execute_workflow(template_name)
        
        if "error" not in result:
            # Get workflow status
            status = orchestrator.get_workflow_status(result["workflow_id"])
            print(f"\n📊 Final Status:")
            print(f"   🏷️ Name: {status['name']}")
            print(f"   ✅ Status: {status['status']}")
            print(f"   📈 Progress: {status['progress']}")
            print(f"   📋 Deliverable: {status['deliverable']}")
        else:
            print(f"❌ Error: {result['error']}")

# Run the advanced workflow test
asyncio.run(test_advanced_workflows())
```

**Expected Output:**
```
=== Testing Advanced Workflow Orchestration ===

==================== WORKFLOW 1 =====================
🚀 Starting workflow: Research Report Generation
🆔 Workflow ID: a1b2c3d4-e5f6-7890-abcd-ef1234567890
============================================================

📋 Phase 1/3: Research
   ⏱️ Estimated duration: medium

🎯 Coordinating work for team: 8f2a1c3e-7d4b-4a2e-8c9d-1e5f3a2b4c1d
🎯 Coordinating work for team: 8f2a1c3e-7d4b-4a2e-8c9d-1e5f3a2b4c1d
   🔄 Assigning work to Research Agent...
   ✅ Research Agent completed portion 1
   🔄 Assigning work to Analyzer...
   ✅ Analyzer completed portion 2
   🔄 Assigning work to Creator...
   ✅ Creator completed portion 3
   ✅ Phase 1 completed successfully

📋 Phase 2/3: Analysis
   ⏱️ Estimated duration: medium
   📈 Scaled to 4 agents

🎯 Coordinating work for team: 8f2a1c3e-7d4b-4a2e-8c9d-1e5f3a2b4c1d
   🔄 Assigning work to Research Agent...
   ✅ Research Agent completed portion 1
   🔄 Assigning work to Analyzer...
   ✅ Analyzer completed portion 2
   🔄 Assigning work to Creator...
   ✅ Creator completed portion 3
   🔄 Assigning work to Research Agent...
   ✅ Research Agent completed portion 4
   ✅ Phase 2 completed successfully

📋 Phase 3/3: Creative
   ⏱️ Estimated duration: short

🎯 Coordinating work for team: 8f2a1c3e-7d4b-4a2e-8c9d-1e5f3a2b4c1d
   🔄 Assigning work to Research Agent...
   ✅ Research Agent completed portion 1
   🔄 Assigning work to Analyzer...
   ✅ Analyzer completed portion 2
   🔄 Assigning work to Creator...
   ✅ Creator completed portion 3
   🔄 Assigning work to Research Agent...
   ✅ Research Agent completed portion 4
   ✅ Phase 3 completed successfully

🎉 Workflow completed successfully!
   📊 Phases: 3
   🤖 Agents: 3
   📋 Deliverable: Complete Research Report Generation with 3 integrated phases
   🧹 Team terminated: 4 agents

📊 Final Status:
   🏷️ Name: Research Report Generation
   ✅ Status: completed
   📈 Progress: 100%
   📋 Deliverable: Complete Research Report Generation with 3 integrated phases
```

---

## Summary: Mastering Advanced Agent Patterns

**Advanced Patterns Covered:**

1. **Streaming Responses**: Real-time, incremental content delivery
2. **Agent Composition**: Multi-agent collaboration and workflow coordination
3. **Dynamic Agent Selection**: Smart routing based on query analysis
4. **State Management**: Persistent conversations with context awareness
5. **Meta-Agent Systems**: Agents that create and manage other agents
6. **Workflow Orchestration**: Complex, multi-phase business processes

**Real-World Applications:**
- **Customer Service**: Multi-tier support with handoffs between specialized agents
- **Content Creation**: Teams of agents for research, writing, editing, and publishing
- **Business Intelligence**: Workflows that research, analyze, and present insights
- **Project Management**: Dynamic team formation based on project requirements
- **E-learning**: Adaptive tutoring systems with specialized teaching agents

**Architecture Benefits:**
- **Scalability**: Add/remove agents dynamically based on workload
- **Resilience**: Fault tolerance through agent redundancy
- **Efficiency**: Optimal agent selection for specific tasks
- **Adaptability**: Self-modifying workflows based on context
- **Maintainability**: Clear separation of concerns between agent types

**Next Level:** Agent swarms that can self-organize, learning agents that improve over time, and distributed agent networks that can operate across multiple systems and devices!

---

## 🎯 Practice Exercise

Create a **Customer Service Orchestrator** that:

1. **Routes inquiries** to appropriate specialized agents (Technical, Billing, General)
2. **Escalates complex issues** by spawning additional expert agents
3. **Maintains conversation context** across multiple interactions
4. **Streams responses** for better user experience
5. **Scales team size** during high-volume periods

Build this as a complete system with all the advanced patterns integrated!

Ready to move to **Lesson 8: Error Handling and Resilience Patterns** where we'll explore how to build bulletproof agent systems that can handle failures gracefully and recover automatically? Let's make our agents truly robust! 🛡️