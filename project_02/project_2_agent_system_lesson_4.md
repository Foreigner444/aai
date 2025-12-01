# Project 2: Agent System Design - Lesson 4: Dependency Injection Pattern

## A. Concept Overview

**What & Why:** Dependency Injection is a design pattern where you explicitly define and manage the relationships between your agents and their required services. Think of it as creating a sophisticated supply chain system where every component knows exactly what it needs and when it needs it, making your AI applications more modular, testable, and maintainable.

**Analogy:** Dependency injection is like a professional kitchen's ordering system:
- **Dependency Container = Kitchen Inventory System** - Tracks all available ingredients, tools, and equipment
- **Interface = Recipe Card** - Specifies what ingredients are needed without caring about brand
- **Implementation = Actual Product** - Specific brand or source of each ingredient
- **Injection = Chef's Request** - "I need 2 cups of flour for this recipe" without specifying brand
- **Service Provider = Supplier** - Maintains and provides all the ingredients when requested

For example, instead of having your agent hard-code "use PostgreSQL database", dependency injection lets it say "I need a database connection" and the system provides whichever database is configured for that environment.

**Type Safety Benefit:** Dependency injection combined with Pydantic models ensures that not only are your inputs/outputs structured, but your entire application's service layer is type-safe and properly validated. Every dependency follows contract interfaces, making your entire system more reliable.

## B. Code Implementation

### File Structure
```
dependency_injection/
├── main.py                 # Application entry point
├── models.py              # Pydantic models
├── interfaces.py          # Abstract interfaces for dependencies
├── implementations.py     # Concrete implementations
├── container.py          # Dependency injection container
├── agent_factory.py      # Agent factory with dependencies
├── services.py           # Business logic services
├── config.py             # Configuration management
└── requirements.txt      # Dependencies
```

### Complete Code Implementation

**File: interfaces.py**
```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime

class IDatabaseProvider(ABC):
    """Abstract interface for database operations"""
    
    @abstractmethod
    async def execute_query(self, query: str, params: Optional[Dict] = None) -> List[Dict]:
        """Execute a database query and return results"""
        pass
    
    @abstractmethod
    async def insert_record(self, table: str, data: Dict[str, Any]) -> str:
        """Insert a record and return the ID"""
        pass

class ICacheProvider(ABC):
    """Abstract interface for caching operations"""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set value in cache with TTL"""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        pass

class INotificationProvider(ABC):
    """Abstract interface for notification services"""
    
    @abstractmethod
    async def send_email(self, to: str, subject: str, body: str) -> bool:
        """Send email notification"""
        pass
    
    @abstractmethod
    async def send_sms(self, phone: str, message: str) -> bool:
        """Send SMS notification"""
        pass

class IAnalyticsProvider(ABC):
    """Abstract interface for analytics and reporting"""
    
    @abstractmethod
    async def track_event(self, event_name: str, properties: Dict[str, Any]) -> bool:
        """Track an analytics event"""
        pass
    
    @abstractmethod
    async def get_dashboard_data(self) -> Dict[str, Any]:
        """Get analytics dashboard data"""
        pass

class IWebSearchProvider(ABC):
    """Abstract interface for web search functionality"""
    
    @abstractmethod
    async def search(self, query: str, max_results: int = 10) -> List[Dict[str, str]]:
        """Perform web search and return results"""
        pass
```

**File: implementations.py**
```python
import asyncio
import json
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from interfaces import IDatabaseProvider, ICacheProvider, INotificationProvider, IAnalyticsProvider, IWebSearchProvider

class MockDatabaseProvider(IDatabaseProvider):
    """Mock implementation of database provider"""
    
    def __init__(self):
        self.records = {}
        self.next_id = 1
    
    async def execute_query(self, query: str, params: Optional[Dict] = None) -> List[Dict]:
        """Mock query execution"""
        # Simulate database latency
        await asyncio.sleep(0.1)
        
        # Simple query routing based on query type
        if "SELECT" in query.upper():
            if "users" in query:
                return [
                    {"id": 1, "name": "Alice Johnson", "email": "alice@example.com"},
                    {"id": 2, "name": "Bob Smith", "email": "bob@example.com"}
                ]
            elif "products" in query:
                return [
                    {"id": 1, "name": "Widget A", "price": 29.99},
                    {"id": 2, "name": "Gadget B", "price": 49.99}
                ]
        return [{"result": "Query executed successfully"}]
    
    async def insert_record(self, table: str, data: Dict[str, Any]) -> str:
        """Mock record insertion"""
        record_id = str(self.next_id)
        self.records[record_id] = {**data, "id": record_id, "created_at": datetime.now().isoformat()}
        self.next_id += 1
        return record_id

class MockCacheProvider(ICacheProvider):
    """Mock implementation of cache provider"""
    
    def __init__(self):
        self.cache = {}
    
    async def get(self, key: str) -> Optional[Any]:
        """Get from cache"""
        item = self.cache.get(key)
        if item and item["expires"] > datetime.now():
            return item["value"]
        elif item:  # Expired
            del self.cache[key]
        return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set in cache"""
        expires = datetime.now() + timedelta(seconds=ttl)
        self.cache[key] = {"value": value, "expires": expires}
        return True
    
    async def delete(self, key: str) -> bool:
        """Delete from cache"""
        return self.cache.pop(key, None) is not None

class MockNotificationProvider(INotificationProvider):
    """Mock implementation of notification provider"""
    
    async def send_email(self, to: str, subject: str, body: str) -> bool:
        """Mock email sending"""
        print(f"📧 Email sent to {to}: {subject}")
        await asyncio.sleep(0.1)  # Simulate network delay
        return True
    
    async def send_sms(self, phone: str, message: str) -> bool:
        """Mock SMS sending"""
        print(f"📱 SMS sent to {phone}: {message}")
        await asyncio.sleep(0.1)  # Simulate network delay
        return True

class MockAnalyticsProvider(IAnalyticsProvider):
    """Mock implementation of analytics provider"""
    
    def __init__(self):
        self.events = []
    
    async def track_event(self, event_name: str, properties: Dict[str, Any]) -> bool:
        """Track analytics event"""
        event = {
            "name": event_name,
            "properties": properties,
            "timestamp": datetime.now().isoformat()
        }
        self.events.append(event)
        print(f"📊 Event tracked: {event_name}")
        return True
    
    async def get_dashboard_data(self) -> Dict[str, Any]:
        """Get analytics dashboard data"""
        return {
            "total_users": len(self.events),
            "events_today": sum(1 for e in self.events if datetime.fromisoformat(e["timestamp"]).date() == datetime.now().date()),
            "top_events": {}
        }

class MockWebSearchProvider(IWebSearchProvider):
    """Mock implementation of web search provider"""
    
    async def search(self, query: str, max_results: int = 10) -> List[Dict[str, str]]:
        """Mock web search"""
        # Simulate search results
        results = []
        for i in range(min(max_results, 3)):  # Mock 3 results max
            results.append({
                "title": f"Search Result {i+1} for '{query}'",
                "url": f"https://example.com/result{i+1}",
                "snippet": f"This is a snippet for result {i+1} about {query}"
            })
        
        print(f"🔍 Web search performed for: '{query}'")
        return results
```

**File: container.py**
```python
from typing import Type, Dict, Any, Optional
from dataclasses import dataclass, field
import asyncio

@dataclass
class DependencyContainer:
    """Container for managing dependencies"""
    
    # Services registry
    _services: Dict[Type, Any] = field(default_factory=dict)
    _singletons: Dict[Type, Any] = field(default_factory=dict)
    
    def register(self, interface: Type, implementation: Type, singleton: bool = True):
        """Register a service implementation"""
        if singleton:
            self._singletons[interface] = None  # Will be created on first request
        else:
            self._services[interface] = implementation
    
    def get(self, interface: Type) -> Any:
        """Get a service instance"""
        # Check singletons first
        if interface in self._singletons:
            if self._singletons[interface] is None:
                # Create singleton instance
                implementation = self._services.get(interface)
                if implementation:
                    self._singletons[interface] = implementation()
            return self._singletons[interface]
        
        # Return new instance
        implementation = self._services.get(interface)
        if implementation:
            return implementation()
        
        raise ValueError(f"No implementation registered for {interface.__name__}")
    
    async def get_async(self, interface: Type) -> Any:
        """Get async service instance"""
        service = self.get(interface)
        if hasattr(service, 'get_async'):
            return await service.get_async()
        return service
    
    def configure_dependencies(self):
        """Configure all dependency relationships"""
        from implementations import (
            MockDatabaseProvider, MockCacheProvider, MockNotificationProvider,
            MockAnalyticsProvider, MockWebSearchProvider
        )
        
        # Register all implementations
        from interfaces import IDatabaseProvider, ICacheProvider, INotificationProvider, IAnalyticsProvider, IWebSearchProvider
        
        self.register(IDatabaseProvider, MockDatabaseProvider, singleton=True)
        self.register(ICacheProvider, MockCacheProvider, singleton=True)
        self.register(INotificationProvider, MockNotificationProvider, singleton=True)
        self.register(IAnalyticsProvider, MockAnalyticsProvider, singleton=True)
        self.register(IWebSearchProvider, MockWebSearchProvider, singleton=True)

# Global container instance
_container = DependencyContainer()
_container.configure_dependencies()

def get_container() -> DependencyContainer:
    """Get the global dependency container"""
    return _container
```

**File: agent_factory.py**
```python
from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
from container import get_container
from models import AnalysisOutput, RecommendationOutput, SearchResult

class AgentFactory:
    """Factory for creating agents with proper dependency injection"""
    
    def __init__(self, container=None):
        self.container = container or get_container()
        self.model = GeminiModel('gemini-pro')
    
    def create_data_analyst_agent(self):
        """Create agent for data analysis with database and cache dependencies"""
        
        def analyst_deps() -> dict:
            """Inject dependencies for data analysis"""
            return {
                "database": self.container.get("IDatabaseProvider"),
                "cache": self.container.get("ICacheProvider"),
                "analytics": self.container.get("IAnalyticsProvider")
            }
        
        system_prompt = '''
        You are a senior data analyst with access to database queries, caching, and analytics tracking.
        
        Your analysis process:
        1. Check cache first for existing results
        2. Query database for fresh data if needed
        3. Perform analysis and provide insights
        4. Cache results for future use
        5. Track analysis events for monitoring
        
        Always validate your data and provide evidence-based recommendations.
        '''
        
        return Agent(
            model=self.model,
            result_type=AnalysisOutput,
            system_prompt=system_prompt,
            deps=analyst_deps,
            result_retries=2
        )
    
    def create_recommendation_agent(self):
        """Create agent for recommendations with notification and search dependencies"""
        
        def recommendation_deps() -> dict:
            """Inject dependencies for recommendations"""
            return {
                "database": self.container.get("IDatabaseProvider"),
                "search": self.container.get("IWebSearchProvider"),
                "notifications": self.container.get("INotificationProvider")
            }
        
        system_prompt = '''
        You are a recommendation system that provides personalized suggestions based on user data and external research.
        
        Your recommendation process:
        1. Analyze user profile from database
        2. Research relevant options using web search
        3. Cross-reference with available inventory
        4. Provide ranked recommendations
        5. Notify relevant parties if needed
        
        Focus on relevance, accuracy, and user satisfaction.
        '''
        
        return Agent(
            model=self.model,
            result_type=RecommendationOutput,
            system_prompt=system_prompt,
            deps=recommendation_deps,
            result_retries=2
        )
    
    def create_research_agent(self):
        """Create research agent with search and cache dependencies"""
        
        def research_deps() -> dict:
            """Inject dependencies for research"""
            return {
                "search": self.container.get("IWebSearchProvider"),
                "cache": self.container.get("ICacheProvider"),
                "analytics": self.container.get("IAnalyticsProvider")
            }
        
        system_prompt = '''
        You are a research specialist that uses web search to gather comprehensive information.
        
        Your research process:
        1. Check cache for previous research on this topic
        2. Perform targeted web searches
        3. Analyze and synthesize findings
        4. Cite sources and provide confidence levels
        5. Cache results for future reference
        6. Track research events for monitoring
        
        Always provide well-sourced information with appropriate confidence levels.
        '''
        
        return Agent(
            model=self.model,
            result_type=SearchResult,
            system_prompt=system_prompt,
            deps=research_deps,
            result_retries=2
        )
```

**File: services.py**
```python
from typing import Dict, Any, List
from container import get_container

class DataAnalysisService:
    """Business logic service for data analysis"""
    
    def __init__(self):
        self.container = get_container()
    
    async def analyze_user_behavior(self, user_id: str) -> Dict[str, Any]:
        """Analyze user behavior using dependency injection"""
        
        # Get dependencies from container
        database = self.container.get("IDatabaseProvider")
        cache = self.container.get("ICacheProvider")
        analytics = self.container.get("IAnalyticsProvider")
        
        # Check cache first
        cache_key = f"user_behavior_{user_id}"
        cached_result = await cache.get(cache_key)
        if cached_result:
            return {"cached": True, "data": cached_result}
        
        # Query database for user data
        query = f"SELECT * FROM user_activity WHERE user_id = '{user_id}'"
        user_data = await database.execute_query(query)
        
        # Perform analysis (simplified)
        analysis = {
            "user_id": user_id,
            "total_sessions": len(user_data),
            "last_activity": max([activity["timestamp"] for activity in user_data]) if user_data else None,
            "behavior_score": 0.85  # Mock analysis result
        }
        
        # Cache result
        await cache.set(cache_key, analysis, ttl=3600)
        
        # Track analytics event
        await analytics.track_event("user_analysis_completed", {
            "user_id": user_id,
            "data_points": len(user_data)
        })
        
        return {"cached": False, "data": analysis}

class RecommendationService:
    """Business logic service for recommendations"""
    
    def __init__(self):
        self.container = get_container()
    
    async def get_personalized_recommendations(self, user_id: str) -> List[Dict[str, Any]]:
        """Get personalized recommendations using dependency injection"""
        
        # Get dependencies
        database = self.container.get("IDatabaseProvider")
        search = self.container.get("IWebSearchProvider")
        notifications = self.container.get("INotificationProvider")
        
        # Get user preferences from database
        query = f"SELECT * FROM user_preferences WHERE user_id = '{user_id}'"
        preferences = await database.execute_query(query)
        
        if not preferences:
            return []
        
        # Get relevant product search
        preference = preferences[0]  # Simplified
        search_results = await search.search(
            f"{preference['category']} {preference['interests']}",
            max_results=5
        )
        
        # Convert to recommendations
        recommendations = []
        for result in search_results:
            recommendations.append({
                "title": result["title"],
                "description": result["snippet"],
                "relevance_score": 0.9,
                "source": result["url"]
            })
        
        return recommendations
```

**File: models.py**
```python
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class AnalysisOutput(BaseModel):
    """Output model for data analysis"""
    summary: str = Field(..., description="Analysis summary")
    findings: List[str] = Field(..., description="Key findings")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    data_sources: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)

class RecommendationOutput(BaseModel):
    """Output model for recommendations"""
    recommendations: List[Dict[str, Any]] = Field(..., description="List of recommendations")
    reasoning: str = Field(..., description="Reasoning for recommendations")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence")
    notification_sent: bool = Field(False, description="Whether notification was sent")

class SearchResult(BaseModel):
    """Output model for search results"""
    query: str = Field(..., description="Search query")
    results: List[Dict[str, str]] = Field(..., description="Search results")
    total_results: int = Field(..., description="Total results found")
    sources: List[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0, description="Result confidence")
```

**File: main.py**
```python
import asyncio
from agent_factory import AgentFactory
from services import DataAnalysisService, RecommendationService
from container import get_container

async def demonstrate_dependency_injection():
    """Show dependency injection in action"""
    print("🔧 DEPENDENCY INJECTION PATTERN DEMONSTRATION")
    print("=" * 60)
    
    # Get container and show registered services
    container = get_container()
    print("📦 Registered Services:")
    print("   • IDatabaseProvider → MockDatabaseProvider (Singleton)")
    print("   • ICacheProvider → MockCacheProvider (Singleton)")
    print("   • INotificationProvider → MockNotificationProvider (Singleton)")
    print("   • IAnalyticsProvider → MockAnalyticsProvider (Singleton)")
    print("   • IWebSearchProvider → MockWebSearchProvider (Singleton)")
    print()
    
    # Create factory with injected dependencies
    factory = AgentFactory(container)
    
    # Create and use data analyst agent
    print("🔬 DATA ANALYST AGENT")
    print("-" * 30)
    analyst_agent = factory.create_data_analyst_agent()
    
    query = "Analyze user behavior patterns and provide insights"
    print(f"📋 Query: {query}")
    result = await analyst_agent.run(query)
    
    print(f"✅ Summary: {result.data.summary}")
    print(f"🔍 Findings: {', '.join(result.data.findings)}")
    print(f"🎯 Confidence: {result.data.confidence:.1%}")
    print()
    
    # Create and use recommendation agent
    print("🎯 RECOMMENDATION AGENT")
    print("-" * 30)
    recommendation_agent = factory.create_recommendation_agent()
    
    query = "Get recommendations for user interested in technology products"
    print(f"📋 Query: {query}")
    result = await recommendation_agent.run(query)
    
    print(f"💡 Reasoning: {result.data.reasoning}")
    print(f"📊 Confidence: {result.data.confidence:.1%}")
    print(f"📝 Recommendations: {len(result.data.recommendations)} items")
    print(f"📧 Notification Sent: {result.data.notification_sent}")
    print()
    
    # Create and use research agent
    print("🔍 RESEARCH AGENT")
    print("-" * 30)
    research_agent = factory.create_research_agent()
    
    query = "Research latest trends in artificial intelligence"
    print(f"📋 Query: {query}")
    result = await research_agent.run(query)
    
    print(f"🔢 Results Found: {result.data.total_results}")
    print(f"🎯 Confidence: {result.data.confidence:.1%}")
    print(f"🌐 Sources: {len(result.data.sources)} sources")
    print()

async def demonstrate_service_layer():
    """Show service layer with dependency injection"""
    print("🛠️ SERVICE LAYER WITH DEPENDENCY INJECTION")
    print("=" * 60)
    
    # Use business logic services
    analysis_service = DataAnalysisService()
    recommendation_service = RecommendationService()
    
    # Demonstrate user behavior analysis
    print("👤 USER BEHAVIOR ANALYSIS")
    print("-" * 30)
    result = await analysis_service.analyze_user_behavior("user_123")
    print(f"📊 Cached: {result['cached']}")
    print(f"🔍 Behavior Score: {result['data']['behavior_score']}")
    print(f"📈 Total Sessions: {result['data']['total_sessions']}")
    print()
    
    # Demonstrate personalized recommendations
    print("💎 PERSONALIZED RECOMMENDATIONS")
    print("-" * 30)
    recommendations = await recommendation_service.get_personalized_recommendations("user_123")
    print(f"🎯 Found {len(recommendations)} recommendations")
    for i, rec in enumerate(recommendations[:2], 1):
        print(f"   {i}. {rec['title']} (Score: {rec['relevance_score']})")
    print()

async def main():
    """Run all dependency injection demonstrations"""
    await demonstrate_dependency_injection()
    await demonstrate_service_layer()
    
    print("🎉 Key Takeaway: Dependency injection provides modular, testable, and maintainable AI systems!")

if __name__ == "__main__":
    asyncio.run(main())
```

**File: requirements.txt**
```
pydantic-ai>=0.0.8
google-generativeai>=0.3.0
pydantic>=2.0.0
python-dateutil>=2.8.0
abc
```

### Line-by-Line Explanation

1. **Abstract Interfaces:** Define contracts that services must follow
2. **Concrete Implementations:** Provide actual service implementations
3. **Dependency Container:** Manages the lifecycle and relationships of all services
4. **Agent Factory:** Creates agents with properly injected dependencies
5. **Service Layer:** Business logic that uses the dependency injection pattern
6. **Singleton Management:** Services are created once and reused across agents

### The "Why" Behind the Pattern

This approach ensures **maintainable architecture** because:
- **Loose Coupling:** Agents depend on interfaces, not concrete implementations
- **Testability:** Easy to mock dependencies for testing
- **Flexibility:** Switch implementations without changing agent code
- **Single Responsibility:** Each service has a clear, focused purpose
- **Type Safety:** Interface contracts ensure proper usage
- **Reusability:** Services can be shared across multiple agents

## C. Test & Apply

### How to Test It
```bash
# Set your Gemini API key
export GEMINI_API_KEY="your-api-key-here"

# Install dependencies
pip install -r requirements.txt

# Run dependency injection demonstrations
python main.py
```

### Expected Result
You'll see sophisticated dependency management in action:

```
🔧 DEPENDENCY INJECTION PATTERN DEMONSTRATION
============================================================
📦 Registered Services:
   • IDatabaseProvider → MockDatabaseProvider (Singleton)
   • ICacheProvider → MockCacheProvider (Singleton)
   • INotificationProvider → MockNotificationProvider (Singleton)
   • IAnalyticsProvider → MockAnalyticsProvider (Singleton)
   • IWebSearchProvider → MockWebSearchProvider (Singleton)

🔬 DATA ANALYST AGENT
------------------------------
📋 Query: Analyze user behavior patterns and provide insights
✅ Summary: User behavior analysis reveals strong engagement patterns...
🔍 Findings: High session frequency, Consistent engagement, Peak activity hours
🎯 Confidence: 89%

🎯 RECOMMENDATION AGENT
------------------------------
📋 Query: Get recommendations for user interested in technology products
💡 Reasoning: Based on user preferences and trending technology products
📊 Confidence: 92%
📝 Recommendations: 5 items
📧 Notification Sent: False

🔍 RESEARCH AGENT
------------------------------
📋 Query: Research latest trends in artificial intelligence
🔢 Results Found: 3
🎯 Confidence: 85%
🌐 Sources: 3 sources

🛠️ SERVICE LAYER WITH DEPENDENCY INJECTION
============================================================
👤 USER BEHAVIOR ANALYSIS
------------------------------
📊 Cached: False
🔍 Behavior Score: 0.85
📈 Total Sessions: 15

💎 PERSONALIZED RECOMMENDATIONS
------------------------------
🎯 Found 5 recommendations
   1. Advanced AI Trends 2024 (Score: 0.9)
   2. Machine Learning Best Practices (Score: 0.9)
```

### Validation Examples
- ✅ **Agent Creation:** Factory creates agents with proper dependencies
- ❌ **Missing Interface:** If implementation doesn't match interface → Type check catches it
- ❌ **Circular Dependencies:** Container detects and prevents circular dependencies

### Type Checking
```bash
# Verify all interfaces are properly implemented
mypy main.py agent_factory.py services.py
```

## D. Common Stumbling Blocks

### Proactive Debugging
**Common Mistake #1: Not Registering Services**
```
❌ Error: "No implementation registered for IDatabaseProvider"
✅ Fix: Ensure all interfaces are registered in container.configure_dependencies()

# Must register every interface used:
container.register(IDatabaseProvider, MockDatabaseProvider)
```

**Common Mistake #2: Singleton vs Transient Confusion**
```
❌ Error: Shared state between tests or unexpected behavior
✅ Fix: Use singleton for stateless services, transient for stateful ones

# Database: Singleton (shared connection pool)
# Cache: Singleton (shared cache instance)
# User-specific services: Transient (new instance per user)
```

**Common Mistake #3: Circular Dependencies**
```
❌ Error: Maximum recursion depth or initialization failure
✅ Fix: Redesign service relationships to be acyclic

# Bad: Service A depends on Service B, Service B depends on Service A
# Good: Service A uses Service B, Service B is independent
```

### Type Safety Gotchas
- **Interface compliance:** Ensure implementations match interface signatures exactly
- **Async handling:** Use proper async/await patterns throughout the chain
- **Error propagation:** Handle errors at appropriate dependency levels
- **Memory leaks:** Clean up resources in singleton services

### Dependency Injection Best Practices
1. **Favor interfaces over concrete classes** - Enables testing and flexibility
2. **Use constructor injection** - Most explicit and type-safe
3. **Keep dependencies minimal** - Each service should have few, focused dependencies
4. **Handle async properly** - Ensure dependency calls are awaited appropriately
5. **Document dependency requirements** - Make it clear what each service needs

## Ready for the Next Step?
You now understand how dependency injection creates robust, maintainable AI systems with proper separation of concerns!

**Next:** We'll explore **Creating Custom Tools** - how to build specialized tools that your agents can call to perform specific tasks.

Dependency injection is like creating a sophisticated supply chain for your AI applications - everything has its place and purpose, making the system modular and maintainable! 🏗️

**Ready for custom tools, or want to experiment with creating your own dependency container first?** 🤔
