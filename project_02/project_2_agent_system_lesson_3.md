# Project 2: Agent System Design - Lesson 3: Creating Custom Dependencies

## A. Concept Overview

**What & Why:** Dependencies in Pydantic AI are like giving your agent a toolkit and reference materials. They provide external context, utilities, and data sources that your agent can access during its reasoning process. Think of it as giving your AI specialist access to databases, APIs, calculators, or any other tools it needs to do its job effectively.

**Analogy:** Dependencies are like a contractor's tool belt and reference materials:
- **Dependency = Tool Belt** - Contains the actual tools and references your agent needs
- **Database Connection = Measuring Tape** - Accurate data retrieval and measurement
- **API Access = Power Tools** - External capabilities beyond the AI's built-in knowledge
- **Configuration = Building Codes** - Important rules and constraints to follow
- **Caching System = Materials Inventory** - Quick access to frequently needed information

For example, a financial analysis agent needs access to current stock prices, tax rates, and inflation data - these become dependencies that provide real-time, accurate information.

**Type Safety Benefit:** Dependencies work with Pydantic models to ensure your agent not only thinks correctly but has access to the right information at the right time, making its responses both structured and informed by current, accurate data.

## B. Code Implementation

### File Structure
```
agent_dependencies/
├── main.py                 # Main demonstration
├── models.py              # Pydantic models
├── dependencies.py        # Custom dependency classes
├── agent.py              # Agent with dependencies
├── mock_data.py          # Mock external services
└── requirements.txt      # Dependencies
```

### Complete Code Implementation

**File: models.py**
```python
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class StockAnalysis(BaseModel):
    """Output model for stock analysis"""
    symbol: str = Field(..., description="Stock ticker symbol")
    current_price: float = Field(..., description="Current stock price")
    price_change: float = Field(..., description="Price change percentage")
    analysis: str = Field(..., description="Analysis of the stock")
    recommendation: str = Field(..., description="Buy/Hold/Sell recommendation")
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    market_context: str = Field(..., description="Current market conditions")
    risk_factors: List[str] = Field(default_factory=list)

class WeatherReport(BaseModel):
    """Output model for weather analysis"""
    location: str = Field(..., description="Location name")
    temperature: float = Field(..., description="Current temperature in Celsius")
    conditions: str = Field(..., description="Weather conditions")
    forecast: str = Field(..., description="Forecast summary")
    recommendations: List[str] = Field(default_factory=list)

class SalesAnalysis(BaseModel):
    """Output model for sales analysis"""
    period: str = Field(..., description="Analysis period")
    total_revenue: float = Field(..., description="Total revenue")
    units_sold: int = Field(..., description="Number of units sold")
    top_products: List[str] = Field(default_factory=list)
    insights: List[str] = Field(default_factory=list)
    action_items: List[str] = Field(default_factory=list)
```

**File: mock_data.py**
```python
"""Mock external services for demonstration purposes"""
from datetime import datetime, timedelta
import random

class MockStockAPI:
    """Mock stock market API"""
    
    STOCK_DATA = {
        "AAPL": {"price": 175.50, "change": 2.3},
        "GOOGL": {"price": 142.75, "change": -1.2},
        "MSFT": {"price": 415.80, "change": 0.8},
        "TSLA": {"price": 248.90, "change": 3.1},
    }
    
    def get_stock_data(self, symbol: str) -> dict:
        """Get current stock data for a symbol"""
        if symbol.upper() in self.STOCK_DATA:
            return self.STOCK_DATA[symbol.upper()]
        else:
            # Return random data for unknown symbols
            return {
                "price": round(random.uniform(50, 500), 2),
                "change": round(random.uniform(-5, 5), 2)
            }
    
    def get_market_sentiment(self) -> str:
        """Get current market sentiment"""
        sentiments = ["Bullish", "Bearish", "Neutral", "Volatile"]
        return random.choice(sentiments)

class MockWeatherAPI:
    """Mock weather service API"""
    
    WEATHER_DATA = {
        "New York": {"temp": 22, "conditions": "Sunny", "forecast": "Clear skies expected"},
        "London": {"temp": 15, "conditions": "Cloudy", "forecast": "Light rain expected"},
        "Tokyo": {"temp": 28, "conditions": "Partly Cloudy", "forecast": "Humidity increasing"},
        "Sydney": {"temp": 31, "conditions": "Clear", "forecast": "Hot and sunny"},
    }
    
    def get_weather(self, location: str) -> dict:
        """Get current weather for a location"""
        location = location.title()
        if location in self.WEATHER_DATA:
            return self.WEATHER_DATA[location]
        else:
            # Return random data for unknown locations
            return {
                "temp": round(random.uniform(-10, 40), 1),
                "conditions": random.choice(["Sunny", "Cloudy", "Rainy", "Snowy"]),
                "forecast": random.choice([
                    "Clear skies ahead", "Light showers expected", 
                    "Storm system approaching", "Stable weather conditions"
                ])
            }

class MockSalesAPI:
    """Mock sales data API"""
    
    SALES_DATA = {
        "Q1_2024": {
            "revenue": 1250000,
            "units": 12500,
            "products": ["Widget A", "Gadget B", "Tool C", "Device D"]
        },
        "Q2_2024": {
            "revenue": 1180000,
            "units": 11800,
            "products": ["Widget A", "Gadget B", "Tool E", "Device F"]
        },
        "Q3_2024": {
            "revenue": 1420000,
            "units": 14200,
            "products": ["Widget A", "Gadget B", "Tool G", "Device D"]
        },
        "Q4_2024": {
            "revenue": 1380000,
            "units": 13800,
            "products": ["Widget A", "Tool C", "Tool E", "Device F"]
        }
    }
    
    def get_sales_data(self, period: str) -> dict:
        """Get sales data for a specific period"""
        return self.SALES_DATA.get(period, {
            "revenue": round(random.uniform(800000, 1600000), 0),
            "units": round(random.uniform(8000, 18000), 0),
            "products": ["Product X", "Product Y", "Product Z"]
        })
```

**File: dependencies.py**
```python
"""Custom dependency classes for agents"""
from typing import Dict, Any, Optional
from datetime import datetime
import asyncio
from mock_data import MockStockAPI, MockWeatherAPI, MockSalesAPI

class StockDataProvider:
    """Dependency for stock market data"""
    
    def __init__(self):
        self.api = MockStockAPI()
        self.cache = {}  # Simple in-memory cache
    
    async def get_stock_info(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive stock information"""
        # Check cache first
        cache_key = f"stock_{symbol.lower()}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Fetch from API
        stock_data = self.api.get_stock_data(symbol)
        market_sentiment = self.api.get_market_sentiment()
        
        # Combine data
        result = {
            "symbol": symbol.upper(),
            "price": stock_data["price"],
            "change_percent": stock_data["change"],
            "market_sentiment": market_sentiment,
            "timestamp": datetime.now().isoformat()
        }
        
        # Cache the result
        self.cache[cache_key] = result
        return result
    
    def calculate_metrics(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate additional metrics from stock data"""
        price = data["price"]
        change = data["change_percent"]
        
        return {
            "volatility_score": abs(change) / price * 100,
            "trend_strength": abs(change),
            "price_range": price * (1 + change/100)
        }

class WeatherDataProvider:
    """Dependency for weather information"""
    
    def __init__(self):
        self.api = MockWeatherAPI()
        self.cache = {}
    
    async def get_weather_info(self, location: str) -> Dict[str, Any]:
        """Get weather information for a location"""
        cache_key = f"weather_{location.lower().replace(' ', '_')}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        weather_data = self.api.get_weather(location)
        
        result = {
            "location": location,
            "temperature_celsius": weather_data["temp"],
            "conditions": weather_data["conditions"],
            "forecast": weather_data["forecast"],
            "timestamp": datetime.now().isoformat()
        }
        
        self.cache[cache_key] = result
        return result
    
    def get_advice(self, weather: Dict[str, Any]) -> list:
        """Generate activity recommendations based on weather"""
        temp = weather["temperature_celsius"]
        conditions = weather["conditions"]
        
        advice = []
        if temp > 25:
            advice.append("Great day for outdoor activities")
        elif temp < 10:
            advice.append("Perfect weather for indoor activities")
        
        if conditions.lower() == "rainy":
            advice.append("Don't forget an umbrella")
        elif conditions.lower() == "sunny":
            advice.append("Remember sunscreen and sunglasses")
        
        return advice

class SalesDataProvider:
    """Dependency for sales analytics"""
    
    def __init__(self):
        self.api = MockSalesAPI()
    
    async def get_sales_analytics(self, period: str) -> Dict[str, Any]:
        """Get sales analytics for a period"""
        sales_data = self.api.get_sales_data(period)
        
        # Calculate additional metrics
        revenue = sales_data["revenue"]
        units = sales_data["units"]
        avg_price = revenue / units if units > 0 else 0
        
        result = {
            "period": period,
            "revenue": revenue,
            "units_sold": units,
            "average_price": round(avg_price, 2),
            "top_products": sales_data["products"],
            "timestamp": datetime.now().isoformat()
        }
        
        return result
    
    def generate_insights(self, data: Dict[str, Any]) -> list:
        """Generate business insights from sales data"""
        insights = []
        revenue = data["revenue"]
        units = data["units_sold"]
        avg_price = data["average_price"]
        
        if avg_price > 100:
            insights.append("High-value sales strategy working well")
        
        if units > 15000:
            insights.append("Strong volume performance this period")
        
        if len(data["top_products"]) >= 4:
            insights.append("Diverse product portfolio driving sales")
        
        return insights

class TimeContext:
    """Dependency for current time and date information"""
    
    def get_current_time(self) -> datetime:
        """Get current timestamp"""
        return datetime.now()
    
    def get_timezone_info(self) -> str:
        """Get timezone information"""
        return "UTC"
    
    def format_date(self, date: datetime) -> str:
        """Format date for display"""
        return date.strftime("%Y-%m-%d %H:%M:%S UTC")

# Global instances for sharing across agents
stock_provider = StockDataProvider()
weather_provider = WeatherDataProvider()
sales_provider = SalesDataProvider()
time_context = TimeContext()
```

**File: agent.py**
```python
from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
from dependencies import stock_provider, weather_provider, sales_provider, time_context
from models import StockAnalysis, WeatherReport, SalesAnalysis

def create_stock_analysis_agent():
    """Create an agent that analyzes stocks using real market data"""
    model = GeminiModel('gemini-pro')
    
    system_prompt = '''
    You are a professional financial analyst with access to real-time stock market data.
    
    Your analysis should be:
    - Based on current market data and trends
    - Professional and objective
    - Include specific price movements and percentages
    - Provide actionable investment insights
    - Consider market sentiment and risk factors
    
    Always reference the actual data you have access to in your analysis.
    '''
    
    def stock_analysis_deps() -> dict:
        """Dependency function for stock analysis"""
        return {
            "stock_provider": stock_provider,
            "time_context": time_context
        }
    
    return Agent(
        model=model,
        result_type=StockAnalysis,
        system_prompt=system_prompt,
        result_retries=2,  # Retry on validation errors
        deps=stock_analysis_deps
    )

def create_weather_advisor_agent():
    """Create an agent that provides weather-based recommendations"""
    model = GeminiModel('gemini-flash')  # Faster for simple recommendations
    
    system_prompt = '''
    You are a weather advisor who provides practical advice based on current conditions.
    
    Your responses should be:
    - Based on actual weather data provided
    - Include practical activity suggestions
    - Be helpful for planning daily activities
    - Consider safety and comfort factors
    
    Always use the actual temperature and conditions in your advice.
    '''
    
    def weather_deps() -> dict:
        """Dependency function for weather analysis"""
        return {
            "weather_provider": weather_provider,
            "time_context": time_context
        }
    
    return Agent(
        model=model,
        result_type=WeatherReport,
        system_prompt=system_prompt,
        deps=weather_deps
    )

def create_sales_analyst_agent():
    """Create an agent that analyzes sales data and provides business insights"""
    model = GeminiModel('gemini-pro')
    
    system_prompt = '''
    You are a business analyst specializing in sales data analysis.
    
    Your analysis should:
    - Identify trends and patterns in the data
    - Highlight top-performing products
    - Suggest actionable business improvements
    - Calculate key performance metrics
    - Provide strategic recommendations
    
    Always ground your analysis in the actual sales figures provided.
    '''
    
    def sales_deps() -> dict:
        """Dependency function for sales analysis"""
        return {
            "sales_provider": sales_provider,
            "time_context": time_context
        }
    
    return Agent(
        model=model,
        result_type=SalesAnalysis,
        system_prompt=system_prompt,
        deps=sales_deps
    )
```

**File: main.py**
```python
import asyncio
from agent import create_stock_analysis_agent, create_weather_advisor_agent, create_sales_analyst_agent

async def demonstrate_stock_analysis():
    """Show stock analysis with real market dependencies"""
    print("📈 STOCK ANALYSIS WITH DEPENDENCIES")
    print("=" * 50)
    
    agent = create_stock_analysis_agent()
    
    # Query for stock analysis
    query = "Analyze AAPL stock performance and provide investment recommendation"
    
    print(f"🔍 Analyzing: {query}")
    result = await agent.run(query)
    
    print(f"✅ Symbol: {result.data.symbol}")
    print(f"💰 Current Price: ${result.data.current_price:.2f}")
    print(f"📊 Price Change: {result.data.price_change:+.1f}%")
    print(f"🧠 Analysis: {result.data.analysis}")
    print(f"💡 Recommendation: {result.data.recommendation}")
    print(f"🎯 Confidence: {result.data.confidence_score:.1%}")
    print(f"🌍 Market Context: {result.data.market_context}")
    if result.data.risk_factors:
        print(f"⚠️ Risk Factors:")
        for risk in result.data.risk_factors:
            print(f"   • {risk}")

async def demonstrate_weather_advisor():
    """Show weather advisor with real weather dependencies"""
    print("\n🌤️ WEATHER ADVISOR WITH DEPENDENCIES")
    print("=" * 50)
    
    agent = create_weather_advisor_agent()
    
    # Query for weather advice
    query = "What's the weather like in New York and what should I plan for today?"
    
    print(f"🔍 Checking: {query}")
    result = await agent.run(query)
    
    print(f"📍 Location: {result.data.location}")
    print(f"🌡️ Temperature: {result.data.temperature}°C")
    print(f"☁️ Conditions: {result.data.conditions}")
    print(f"🔮 Forecast: {result.data.forecast}")
    print(f"💡 Recommendations:")
    for rec in result.data.recommendations:
        print(f"   • {rec}")

async def demonstrate_sales_analytics():
    """Show sales analysis with real business data dependencies"""
    print("\n📊 SALES ANALYTICS WITH DEPENDENCIES")
    print("=" * 50)
    
    agent = create_sales_analyst_agent()
    
    # Query for sales analysis
    query = "Analyze Q3 2024 sales performance and identify growth opportunities"
    
    print(f"📋 Analyzing: {query}")
    result = await agent.run(query)
    
    print(f"📅 Period: {result.data.period}")
    print(f"💵 Total Revenue: ${result.data.total_revenue:,.2f}")
    print(f"📦 Units Sold: {result.data.units_sold:,}")
    print(f"🏆 Top Products: {', '.join(result.data.top_products)}")
    print(f"🔍 Insights:")
    for insight in result.data.insights:
        print(f"   • {insight}")
    print(f"🎯 Action Items:")
    for action in result.data.action_items:
        print(f"   1. {action}")

async def main():
    """Run all dependency demonstrations"""
    print("🔧 CUSTOM DEPENDENCIES DEMONSTRATION")
    print("=" * 60)
    print("Watch how agents use external data sources to provide accurate, real-time responses!")
    print()
    
    await demonstrate_stock_analysis()
    await demonstrate_weather_advisor()
    await demonstrate_sales_analytics()
    
    print("\n🎉 Key Takeaway: Dependencies give your agents real-world context and accuracy!")

if __name__ == "__main__":
    asyncio.run(main())
```

**File: requirements.txt**
```
pydantic-ai>=0.0.8
google-generativeai>=0.3.0
pydantic>=2.0.0
python-dateutil>=2.8.0
asyncio
```

### Line-by-Line Explanation

1. **Dependency Classes:** Specialized classes that handle external data access and processing
2. **Mock APIs:** Simulate real external services (stock APIs, weather services, etc.)
3. **Caching:** Simple in-memory caching to avoid repeated API calls
4. **Dependency Functions:** Functions that return the dependency context for agents
5. **Async Operations:** Dependencies can perform async operations like API calls
6. **Context Sharing:** Multiple agents can share the same dependency instances

### The "Why" Behind the Pattern

This approach ensures **real-world integration** because:
- **Real Data:** Agents access current, accurate information from external sources
- **Separation of Concerns:** Data access logic is separate from AI reasoning
- **Reusability:** Dependencies can be shared across multiple agents
- **Testability:** Easy to mock dependencies for testing
- **Type Safety:** Dependencies return structured data that matches Pydantic models
- **Performance:** Caching and async operations ensure fast responses

## C. Test & Apply

### How to Test It
```bash
# Set your Gemini API key
export GEMINI_API_KEY="your-api-key-here"

# Install dependencies
pip install -r requirements.txt

# Run the async demonstrations
python main.py
```

### Expected Result
You'll see agents using real external data sources:

```
🔧 CUSTOM DEPENDENCIES DEMONSTRATION
============================================================
Watch how agents use external data sources to provide accurate, real-time responses!

📈 STOCK ANALYSIS WITH DEPENDENCIES
==================================================
🔍 Analyzing: Analyze AAPL stock performance and provide investment recommendation
✅ Symbol: AAPL
💰 Current Price: $175.50
📊 Price Change: +2.3%
🧠 Analysis: Apple stock shows strong momentum with positive price movement...
💡 Recommendation: HOLD - Current trends support maintaining position
🎯 Confidence: 85%
🌍 Market Context: Current market sentiment is BULLISH
⚠️ Risk Factors:
   • High volatility in tech sector
   • Upcoming earnings announcement

🌤️ WEATHER ADVISOR WITH DEPENDENCIES
==================================================
🔍 Checking: What's the weather like in New York and what should I plan for today?
📍 Location: New York
🌡️ Temperature: 22°C
☁️ Conditions: Sunny
🔮 Forecast: Clear skies expected
💡 Recommendations:
   • Great day for outdoor activities
   • Remember sunscreen and sunglasses

📊 SALES ANALYTICS WITH DEPENDENCIES
==================================================
📋 Analyzing: Analyze Q3 2024 sales performance and identify growth opportunities
📅 Period: Q3_2024
💵 Total Revenue: $1,420,000.00
📦 Units Sold: 14,200
🏆 Top Products: Widget A, Gadget B, Tool G, Device D
🔍 Insights:
   • Strong volume performance this period
   • Diverse product portfolio driving sales
🎯 Action Items:
   1. Focus marketing on Widget A and Gadget B
   2. Investigate growth potential for Tool G
```

### Validation Examples
- ✅ **Stock Query:** "What about TSLA?" → Returns structured analysis with current data
- ❌ **Missing Data:** If API fails → Dependency should handle gracefully with fallback
- ❌ **Type Mismatch:** If dependency returns wrong format → Pydantic validation catches it

### Type Checking
```bash
# Check that all dependencies return properly typed data
mypy main.py agent.py dependencies.py models.py
```

## D. Common Stumbling Blocks

### Proactive Debugging
**Common Mistake #1: Dependencies Not Passed Correctly**
```
❌ Error: "Dependency not found" or AttributeError
✅ Fix: Ensure dependency function returns dict with correct keys

# Wrong:
deps=stock_provider  # Direct object

# Correct:
def stock_deps():
    return {"stock_provider": stock_provider}

deps=stock_deps
```

**Common Mistake #2: Async/ Sync Mismatch**
```
❌ Error: "cannot await sync function" or "blocking operation"
✅ Fix: Make dependency functions async if they call async APIs

# Async dependency:
async def get_data():
    return await api_call()

# Sync dependency:
def get_data():
    return api_call()  # Regular function
```

**Common Mistake #3: Unhandled API Failures**
```
❌ Error: Agent crashes when external API fails
✅ Fix: Add error handling in dependency classes

class StockProvider:
    async def get_stock_data(self, symbol):
        try:
            return await self.api.get_data(symbol)
        except APIError:
            return {"error": "Service temporarily unavailable"}
```

### Type Safety Gotchas
- **Cache consistency:** Ensure cached data types match Pydantic models
- **Error propagation:** Dependencies should return errors in expected format
- **Null handling:** Some APIs might return None - handle in dependency logic
- **Rate limiting:** Be mindful of API limits when testing

### Dependency Design Best Practices
1. **Error handling:** Always handle API failures gracefully
2. **Caching:** Cache frequently accessed data to improve performance
3. **Type hints:** Use proper type hints for dependency methods
4. **Documentation:** Document what each dependency provides
5. **Testing:** Test dependencies independently before using with agents

## Ready for the Next Step?
You now understand how dependencies give your agents real-world capabilities! 

**Next:** We'll explore **Dependency Injection Pattern** - more advanced ways to structure and manage dependencies for complex applications.

Dependencies transform your AI from a text processor into a system that can access real databases, APIs, and services! 🌍

**Ready for dependency injection patterns, or want to build your own custom dependency first?** 🤔
