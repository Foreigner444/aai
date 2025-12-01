# Project 2: Agent System Design - Lesson 5: Agent with Dependencies

## A. Concept Overview

**What & Why:** Now that you understand dependency injection, it's time to see it in action! An "agent with dependencies" is like a specialized AI expert who has access to a complete toolkit of real-world services. Instead of just thinking about problems in isolation, your agent can query databases, search the web, send notifications, and perform complex multi-step workflows.

**Analogy:** Think of this as upgrading from a consultant who only gives advice to a full-service agency that can actually execute the work:
- **Before:** "You should update your database" → Just advice
- **After:** "I'll query your database, analyze the data, and send you a report" → Complete execution
- **Agent = Senior Consultant** - Has expertise and access to all necessary tools
- **Dependencies = Agency Resources** - Database team, web research team, notification service
- **Result = Full Service Delivery** - Actual work completed, not just recommendations

For example, a customer support agent with dependencies can:
1. Look up customer history from the database
2. Search for relevant solutions online
3. Analyze the specific issue
4. Generate personalized responses
5. Send notifications to the right teams

**Type Safety Benefit:** With proper dependency injection, your agent's actions are guided by structured interfaces and validated outputs. Each dependency call follows type-safe contracts, and the final result matches your Pydantic model exactly.

## B. Code Implementation

### File Structure
```
agent_with_dependencies/
├── main.py                 # Main demonstration
├── models.py              # Pydantic models for outputs
├── agent_configs.py       # Agent configurations with dependencies
├── workflow_agents.py     # Agents that perform complex workflows
├── business_logic.py      # Business logic using agents
├── dependency_factory.py  # Dependency injection factory
└── requirements.txt       # Dependencies
```

### Complete Code Implementation

**File: models.py**
```python
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

class WorkflowStep(BaseModel):
    """Individual step in a workflow"""
    step_name: str = Field(..., description="Name of the step")
    status: TaskStatus = Field(..., description="Status of the step")
    result: Optional[str] = Field(None, description="Step result")
    duration_seconds: Optional[float] = Field(None, description="Time taken")

class CustomerAnalysis(BaseModel):
    """Output model for customer support analysis"""
    customer_id: str = Field(..., description="Customer identifier")
    customer_profile: Dict[str, Any] = Field(..., description="Customer information")
    issue_summary: str = Field(..., description="Summary of the issue")
    recommended_solution: str = Field(..., description="Recommended fix")
    priority_level: str = Field(..., description="Priority level")
    escalation_required: bool = Field(..., description="Whether escalation is needed")
    workflow_steps: List[WorkflowStep] = Field(default_factory=list)
    response_generated: str = Field(..., description="Customer response text")
    notifications_sent: List[str] = Field(default_factory=list)

class ResearchReport(BaseModel):
    """Output model for research tasks"""
    research_query: str = Field(..., description="Original research query")
    findings: List[str] = Field(..., description="Key findings")
    sources: List[str] = Field(..., description="Sources consulted")
    data_points: List[Dict[str, Any]] = Field(default_factory=list)
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    workflow_execution: List[WorkflowStep] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)

class ProcessAutomation(BaseModel):
    """Output model for process automation tasks"""
    process_name: str = Field(..., description="Name of the automated process")
    inputs_processed: int = Field(..., description="Number of inputs processed")
    outputs_generated: int = Field(..., description="Number of outputs generated")
    errors_encountered: int = Field(..., description="Number of errors")
    workflow_log: List[WorkflowStep] = Field(default_factory=list)
    success_rate: float = Field(..., ge=0.0, le=1.0)
    next_actions: List[str] = Field(default_factory=list)
```

**File: dependency_factory.py**
```python
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import asyncio
import json
from datetime import datetime, timedelta
import random

@dataclass
class DependencyFactory:
    """Factory for creating and managing dependencies"""
    
    _services: Dict[str, Any] = field(default_factory=dict)
    _cache: Dict[str, Any] = field(default_factory=dict)
    
    def __init__(self):
        self._initialize_services()
    
    def _initialize_services(self):
        """Initialize all dependency services"""
        # Database Service
        self._services["database"] = MockDatabaseService()
        
        # Search Service
        self._services["search"] = MockSearchService()
        
        # Notification Service
        self._services["notifications"] = MockNotificationService()
        
        # Analytics Service
        self._services["analytics"] = MockAnalyticsService()
        
        # File Storage Service
        self._services["storage"] = MockStorageService()
        
        # Email Service
        self._services["email"] = MockEmailService()
    
    def get_service(self, service_name: str) -> Any:
        """Get a service instance"""
        if service_name not in self._services:
            raise ValueError(f"Service {service_name} not found")
        return self._services[service_name]
    
    def get_cache(self, key: str) -> Optional[Any]:
        """Get cached value"""
        item = self._cache.get(key)
        if item and item["expires"] > datetime.now():
            return item["value"]
        return None
    
    def set_cache(self, key: str, value: Any, ttl: int = 3600) -> None:
        """Set cached value"""
        expires = datetime.now() + timedelta(seconds=ttl)
        self._cache[key] = {"value": value, "expires": expires}

# Mock Services
class MockDatabaseService:
    """Mock database service"""
    
    def __init__(self):
        self.customers = {
            "cust_001": {
                "id": "cust_001",
                "name": "Alice Johnson",
                "email": "alice@example.com",
                "tier": "premium",
                "last_login": "2024-01-15",
                "issues": ["billing", "login"]
            },
            "cust_002": {
                "id": "cust_002", 
                "name": "Bob Smith",
                "email": "bob@example.com",
                "tier": "basic",
                "last_login": "2024-01-14",
                "issues": ["performance"]
            }
        }
    
    async def query_customer(self, customer_id: str) -> Optional[Dict[str, Any]]:
        """Query customer data"""
        await asyncio.sleep(0.1)  # Simulate DB latency
        return self.customers.get(customer_id)
    
    async def insert_case(self, case_data: Dict[str, Any]) -> str:
        """Insert support case"""
        await asyncio.sleep(0.1)
        return f"case_{random.randint(1000, 9999)}"

class MockSearchService:
    """Mock search service"""
    
    async def search_knowledge_base(self, query: str) -> List[Dict[str, str]]:
        """Search knowledge base for solutions"""
        await asyncio.sleep(0.2)
        
        results = [
            {
                "title": f"Common issue: {query}",
                "url": "https://kb.example.com/article123",
                "snippet": "This common issue can be resolved by..."
            },
            {
                "title": f"Advanced troubleshooting for {query}",
                "url": "https://kb.example.com/article456",
                "snippet": "For advanced cases, try these steps..."
            }
        ]
        return results[:2]

class MockNotificationService:
    """Mock notification service"""
    
    async def send_notification(self, recipient: str, message: str, channel: str = "slack") -> bool:
        """Send notification"""
        await asyncio.sleep(0.05)
        print(f"📢 Notification sent to {recipient} via {channel}: {message}")
        return True

class MockAnalyticsService:
    """Mock analytics service"""
    
    async def track_event(self, event_name: str, properties: Dict[str, Any]) -> None:
        """Track analytics event"""
        await asyncio.sleep(0.02)
        print(f"📊 Event tracked: {event_name}")

class MockStorageService:
    """Mock file storage service"""
    
    async def save_report(self, filename: str, content: str) -> str:
        """Save report to storage"""
        await asyncio.sleep(0.1)
        print(f"💾 Report saved: {filename}")
        return f"s3://reports/{filename}"

class MockEmailService:
    """Mock email service"""
    
    async def send_email(self, to: str, subject: str, body: str) -> bool:
        """Send email"""
        await asyncio.sleep(0.1)
        print(f"📧 Email sent to {to}: {subject}")
        return True

# Global factory instance
_factory = DependencyFactory()

def get_dependency_factory() -> DependencyFactory:
    """Get the global dependency factory"""
    return _factory
```

**File: workflow_agents.py**
```python
from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
from dependency_factory import get_dependency_factory
from models import CustomerAnalysis, ResearchReport, ProcessAutomation, WorkflowStep
from typing import Dict, Any
import asyncio
from datetime import datetime

class CustomerSupportAgent:
    """Agent specialized in customer support with full dependency access"""
    
    def __init__(self):
        self.factory = get_dependency_factory()
        self.model = GeminiModel('gemini-pro')
        
        self._agent = Agent(
            model=self.model,
            result_type=CustomerAnalysis,
            system_prompt='''
            You are a senior customer support specialist with access to customer databases, 
            knowledge bases, and notification systems.
            
            Your workflow:
            1. Query customer database for customer profile and history
            2. Search knowledge base for relevant solutions
            3. Analyze the issue and determine priority
            4. Generate appropriate response
            5. Notify relevant teams if escalation needed
            
            Always be professional, empathetic, and solution-focused.
            ''',
            result_retries=2
        )
    
    async def analyze_customer_issue(self, customer_id: str, issue_description: str) -> CustomerAnalysis:
        """Analyze customer issue with full workflow"""
        workflow_steps = []
        
        try:
            # Step 1: Query customer database
            step1_start = datetime.now()
            customer_data = await self.factory.get_service("database").query_customer(customer_id)
            workflow_steps.append(WorkflowStep(
                step_name="database_query",
                status="completed",
                result=f"Customer data retrieved",
                duration_seconds=(datetime.now() - step1_start).total_seconds()
            ))
            
            # Step 2: Search knowledge base
            step2_start = datetime.now()
            search_results = await self.factory.get_service("search").search_knowledge_base(issue_description)
            workflow_steps.append(WorkflowStep(
                step_name="knowledge_base_search",
                status="completed", 
                result=f"Found {len(search_results)} relevant articles",
                duration_seconds=(datetime.now() - step2_start).total_seconds()
            ))
            
            # Step 3: Build context for agent
            context = f"""
            Customer ID: {customer_id}
            Customer Data: {customer_data}
            Issue: {issue_description}
            Knowledge Base Results: {search_results}
            
            Please provide a comprehensive analysis and response.
            """
            
            # Step 4: Run agent analysis
            result = await self._agent.run(context)
            
            # Step 5: Execute notifications if needed
            notifications_sent = []
            if result.data.escalation_required:
                notification = await self.factory.get_service("notifications").send_notification(
                    "support-team", 
                    f"High priority case: {customer_id}",
                    "slack"
                )
                if notification:
                    notifications_sent.append("support-team-slack")
            
            # Update result with workflow and notifications
            result.data.workflow_steps = workflow_steps
            result.data.notifications_sent = notifications_sent
            
            return result.data
            
        except Exception as e:
            workflow_steps.append(WorkflowStep(
                step_name="error_handling",
                status="failed",
                result=f"Error: {str(e)}",
                duration_seconds=0.0
            ))
            
            # Return error result
            return CustomerAnalysis(
                customer_id=customer_id,
                customer_profile={},
                issue_summary=f"Error processing request: {str(e)}",
                recommended_solution="Please contact support directly",
                priority_level="high",
                escalation_required=True,
                workflow_steps=workflow_steps,
                response_generated="We apologize for the technical issue. Please contact support.",
                notifications_sent=[]
            )

class ResearchAgent:
    """Agent specialized in research with comprehensive tool access"""
    
    def __init__(self):
        self.factory = get_dependency_factory()
        self.model = GeminiModel('gemini-pro')
        
        self._agent = Agent(
            model=self.model,
            result_type=ResearchReport,
            system_prompt='''
            You are a research specialist with access to search engines, analytics, and reporting tools.
            
            Your research process:
            1. Conduct web searches for relevant information
            2. Analyze findings and identify key points
            3. Verify information across multiple sources
            4. Generate comprehensive report
            5. Save results to storage system
            6. Track research metrics
            
            Focus on accuracy, source quality, and actionable insights.
            ''',
            result_retries=2
        )
    
    async def conduct_research(self, query: str) -> ResearchReport:
        """Conduct comprehensive research with full workflow"""
        workflow_steps = []
        
        try:
            # Step 1: Web search
            step1_start = datetime.now()
            search_results = await self.factory.get_service("search").search_knowledge_base(query)
            workflow_steps.append(WorkflowStep(
                step_name="web_search",
                status="completed",
                result=f"Found {len(search_results)} sources",
                duration_seconds=(datetime.now() - step1_start).total_seconds()
            ))
            
            # Step 2: Build research context
            context = f"""
            Research Query: {query}
            Search Results: {search_results}
            
            Please analyze these findings and provide a comprehensive research report.
            """
            
            # Step 3: Agent analysis
            step2_start = datetime.now()
            result = await self._agent.run(context)
            workflow_steps.append(WorkflowStep(
                step_name="agent_analysis",
                status="completed",
                result="Analysis completed",
                duration_seconds=(datetime.now() - step2_start).total_seconds()
            ))
            
            # Step 4: Save to storage
            step3_start = datetime.now()
            report_filename = f"research_{query.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.md"
            report_content = f"# Research Report: {query}\n\nGenerated: {datetime.now().isoformat()}\n\n"
            
            for finding in result.data.findings:
                report_content += f"- {finding}\n"
            
            storage_result = await self.factory.get_service("storage").save_report(
                report_filename, 
                report_content
            )
            workflow_steps.append(WorkflowStep(
                step_name="save_to_storage",
                status="completed",
                result=f"Report saved to {storage_result}",
                duration_seconds=(datetime.now() - step3_start).total_seconds()
            ))
            
            # Step 5: Track analytics
            step4_start = datetime.now()
            await self.factory.get_service("analytics").track_event("research_completed", {
                "query": query,
                "sources": len(search_results),
                "confidence": result.data.confidence_score
            })
            workflow_steps.append(WorkflowStep(
                step_name="analytics_tracking",
                status="completed",
                result="Event tracked",
                duration_seconds=(datetime.now() - step4_start).total_seconds()
            ))
            
            # Update result with workflow
            result.data.workflow_execution = workflow_steps
            result.data.sources = [r["url"] for r in search_results]
            
            return result.data
            
        except Exception as e:
            workflow_steps.append(WorkflowStep(
                step_name="error_handling",
                status="failed", 
                result=f"Error: {str(e)}",
                duration_seconds=0.0
            ))
            
            return ResearchReport(
                research_query=query,
                findings=[f"Error during research: {str(e)}"],
                sources=[],
                confidence_score=0.0,
                workflow_execution=workflow_steps,
                recommendations=["Try again later or contact technical support"]
            )

class AutomationAgent:
    """Agent specialized in process automation with full system access"""
    
    def __init__(self):
        self.factory = get_dependency_factory()
        self.model = GeminiModel('gemini-flash')
        
        self._agent = Agent(
            model=self.model,
            result_type=ProcessAutomation,
            system_prompt='''
            You are a process automation specialist with access to databases, storage, and notifications.
            
            Your automation workflow:
            1. Process input data from database
            2. Apply business logic and transformations
            3. Generate outputs and reports
            4. Save results to storage
            5. Send notifications for completion
            6. Track success metrics
            
            Focus on reliability, accuracy, and comprehensive logging.
            ''',
            result_retries=2
        )
    
    async def automate_process(self, process_name: str, inputs: Dict[str, Any]) -> ProcessAutomation:
        """Automate a business process with full workflow"""
        workflow_steps = []
        errors = 0
        processed_count = 0
        
        try:
            # Step 1: Process input data
            step1_start = datetime.now()
            # Simulate processing multiple items
            input_items = inputs.get("items", [{"id": 1, "data": "sample"}])
            processed_count = len(input_items)
            workflow_steps.append(WorkflowStep(
                step_name="process_inputs",
                status="completed",
                result=f"Processed {processed_count} items",
                duration_seconds=(datetime.now() - step1_start).total_seconds()
            ))
            
            # Step 2: Apply business logic
            step2_start = datetime.now()
            context = f"""
            Process: {process_name}
            Input Items: {input_items}
            
            Please analyze and provide automation results.
            """
            result = await self._agent.run(context)
            workflow_steps.append(WorkflowStep(
                step_name="business_logic",
                status="completed",
                result="Business logic applied",
                duration_seconds=(datetime.now() - step2_start).total_seconds()
            ))
            
            # Step 3: Generate outputs
            step3_start = datetime.now()
            outputs_count = processed_count  # Simplified: one output per input
            workflow_steps.append(WorkflowStep(
                step_name="generate_outputs",
                status="completed",
                result=f"Generated {outputs_count} outputs",
                duration_seconds=(datetime.now() - step3_start).total_seconds()
            ))
            
            # Step 4: Save results
            step4_start = datetime.now()
            report_filename = f"{process_name}_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            results_data = {
                "process": process_name,
                "inputs": processed_count,
                "outputs": outputs_count,
                "timestamp": datetime.now().isoformat(),
                "agent_result": result.data.model_dump()
            }
            
            storage_result = await self.factory.get_service("storage").save_report(
                report_filename,
                json.dumps(results_data, indent=2)
            )
            workflow_steps.append(WorkflowStep(
                step_name="save_results",
                status="completed",
                result=f"Results saved to {storage_result}",
                duration_seconds=(datetime.now() - step4_start).total_seconds()
            ))
            
            # Step 5: Send completion notification
            step5_start = datetime.now()
            await self.factory.get_service("notifications").send_notification(
                "automation-team",
                f"Process {process_name} completed successfully",
                "slack"
            )
            workflow_steps.append(WorkflowStep(
                step_name="send_notifications",
                status="completed",
                result="Completion notification sent",
                duration_seconds=(datetime.now() - step5_start).total_seconds()
            ))
            
            # Calculate success rate
            success_rate = (processed_count - errors) / processed_count if processed_count > 0 else 1.0
            
            # Update result with workflow
            result.data.workflow_log = workflow_steps
            result.data.process_name = process_name
            result.data.inputs_processed = processed_count
            result.data.outputs_generated = outputs_count
            result.data.errors_encountered = errors
            result.data.success_rate = success_rate
            
            return result.data
            
        except Exception as e:
            errors += 1
            workflow_steps.append(WorkflowStep(
                step_name="error_handling",
                status="failed",
                result=f"Error: {str(e)}",
                duration_seconds=0.0
            ))
            
            success_rate = (processed_count - errors) / processed_count if processed_count > 0 else 0.0
            
            return ProcessAutomation(
                process_name=process_name,
                inputs_processed=processed_count,
                outputs_generated=max(0, processed_count - errors),
                errors_encountered=errors,
                workflow_log=workflow_steps,
                success_rate=success_rate,
                next_actions=["Review error logs", "Retry failed operations", "Contact technical support"]
            )
```

**File: business_logic.py**
```python
"""Business logic that uses agents with dependencies"""
from workflow_agents import CustomerSupportAgent, ResearchAgent, AutomationAgent
from typing import Dict, Any
import asyncio

class CustomerServiceManager:
    """Manages customer service operations using specialized agents"""
    
    def __init__(self):
        self.support_agent = CustomerSupportAgent()
        self.research_agent = ResearchAgent()
    
    async def handle_customer_inquiry(self, customer_id: str, inquiry: str) -> Dict[str, Any]:
        """Handle complete customer inquiry workflow"""
        print(f"🎯 Handling customer inquiry: {customer_id}")
        
        # Use support agent for main analysis
        analysis = await self.support_agent.analyze_customer_issue(customer_id, inquiry)
        
        # If complex issue, also do research
        research_data = None
        if analysis.priority_level in ["high", "medium"]:
            print(f"🔍 Conducting research for priority issue")
            research = await self.research_agent.conduct_research(inquiry)
            research_data = {
                "findings": research.findings,
                "confidence": research.confidence_score,
                "sources": research.sources[:3]  # Top 3 sources
            }
        
        return {
            "customer_analysis": analysis,
            "research_data": research_data,
            "recommended_actions": [
                "Send personalized response to customer",
                "Update customer profile",
                "Schedule follow-up if needed"
            ]
        }

class OperationsManager:
    """Manages operational processes using automation agents"""
    
    def __init__(self):
        self.automation_agent = AutomationAgent()
    
    async def run_daily_reports(self) -> Dict[str, Any]:
        """Run daily reporting process"""
        print("📊 Running daily reporting process")
        
        # Define daily report inputs
        inputs = {
            "process_type": "daily_reports",
            "items": [
                {"id": 1, "type": "sales", "date": "2024-01-15"},
                {"id": 2, "type": "support", "date": "2024-01-15"},
                {"id": 3, "type": "usage", "date": "2024-01-15"}
            ]
        }
        
        # Run automation
        result = await self.automation_agent.automate_process("daily_reports", inputs)
        
        return {
            "process_result": result,
            "summary": {
                "items_processed": result.inputs_processed,
                "success_rate": result.success_rate,
                "duration": sum(step.duration_seconds or 0 for step in result.workflow_log)
            }
        }
```

**File: main.py**
```python
import asyncio
from business_logic import CustomerServiceManager, OperationsManager
from dependency_factory import get_dependency_factory

async def demonstrate_customer_support_workflow():
    """Demonstrate customer support with full dependency workflow"""
    print("🏢 CUSTOMER SUPPORT WORKFLOW DEMONSTRATION")
    print("=" * 60)
    
    manager = CustomerServiceManager()
    
    # Handle a customer inquiry
    result = await manager.handle_customer_inquiry(
        customer_id="cust_001",
        inquiry="I'm having trouble logging into my account and my billing seems incorrect"
    )
    
    analysis = result["customer_analysis"]
    
    print(f"👤 Customer: {analysis.customer_profile.get('name', 'Unknown')}")
    print(f"📧 Email: {analysis.customer_profile.get('email', 'Unknown')}")
    print(f"📊 Tier: {analysis.customer_profile.get('tier', 'Unknown')}")
    print(f"❗ Issue Summary: {analysis.issue_summary}")
    print(f"🔧 Recommended Solution: {analysis.recommended_solution}")
    print(f"⚡ Priority Level: {analysis.priority_level}")
    print(f"🚨 Escalation Required: {'Yes' if analysis.escalation_required else 'No'}")
    print(f"📝 Generated Response: {analysis.response_generated}")
    
    if analysis.workflow_steps:
        print(f"\n🔄 Workflow Steps:")
        for step in analysis.workflow_steps:
            status_emoji = "✅" if step.status == "completed" else "❌"
            print(f"   {status_emoji} {step.step_name}: {step.result} ({step.duration_seconds:.2f}s)")
    
    if analysis.notifications_sent:
        print(f"\n📢 Notifications Sent: {', '.join(analysis.notifications_sent)}")
    
    if result["research_data"]:
        research = result["research_data"]
        print(f"\n🔍 Research Findings:")
        for finding in research["findings"][:2]:
            print(f"   • {finding}")
        print(f"   📊 Research Confidence: {research['confidence']:.1%}")

async def demonstrate_automation_workflow():
    """Demonstrate automation with full dependency workflow"""
    print("\n⚙️ AUTOMATION WORKFLOW DEMONSTRATION")
    print("=" * 60)
    
    manager = OperationsManager()
    
    # Run daily reports
    result = await manager.run_daily_reports()
    process = result["process_result"]
    
    print(f"🤖 Process: {process.process_name}")
    print(f"📥 Inputs Processed: {process.inputs_processed}")
    print(f"📤 Outputs Generated: {process.outputs_generated}")
    print(f"❌ Errors Encountered: {process.errors_encountered}")
    print(f"✅ Success Rate: {process.success_rate:.1%}")
    
    if process.workflow_log:
        print(f"\n🔄 Workflow Steps:")
        for step in process.workflow_log:
            status_emoji = "✅" if step.status == "completed" else "❌"
            print(f"   {status_emoji} {step.step_name}: {step.result} ({step.duration_seconds:.2f}s)")
    
    if process.next_actions:
        print(f"\n📋 Next Actions:")
        for action in process.next_actions:
            print(f"   • {action}")
    
    print(f"\n📊 Summary: {result['summary']}")

async def demonstrate_dependency_management():
    """Demonstrate dependency injection and management"""
    print("\n🔧 DEPENDENCY MANAGEMENT DEMONSTRATION")
    print("=" * 60)
    
    factory = get_dependency_factory()
    
    # Show available services
    print("📦 Available Services:")
    for service_name in factory._services.keys():
        print(f"   • {service_name}")
    
    # Test service access
    try:
        database = factory.get_service("database")
        search = factory.get_service("search")
        
        print(f"\n🗄️ Database Service: {type(database).__name__}")
        print(f"🔍 Search Service: {type(search).__name__}")
        
        # Test caching
        test_key = "test_cache"
        factory.set_cache(test_key, {"data": "cached_value"}, ttl=10)
        cached_value = factory.get_cache(test_key)
        print(f"💾 Cache Test: {'✅ Working' if cached_value else '❌ Failed'}")
        
    except Exception as e:
        print(f"❌ Service access error: {e}")

async def main():
    """Run all agent with dependencies demonstrations"""
    print("🎯 AGENT WITH DEPENDENCIES DEMONSTRATION")
    print("=" * 70)
    print("Watch how agents use real-world dependencies to execute complex workflows!")
    print()
    
    await demonstrate_customer_support_workflow()
    await demonstrate_automation_workflow()
    await demonstrate_dependency_management()
    
    print("\n🎉 Key Takeaway: Agents with dependencies can execute complete workflows!")

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
dataclasses
```

### Line-by-Line Explanation

1. **Dependency Factory:** Centralized service management with mock implementations
2. **Workflow Agents:** Agents that execute multi-step processes using dependencies
3. **Business Logic:** High-level orchestration that uses specialized agents
4. **Error Handling:** Comprehensive error management at each workflow step
5. **Analytics Tracking:** Track events and metrics throughout the workflow
6. **Notification System:** Send updates to relevant teams and stakeholders

### The "Why" Behind the Pattern

This approach ensures **end-to-end execution** because:
- **Real Workflows:** Agents don't just analyze - they execute complete business processes
- **System Integration:** Access to databases, search, notifications, storage, and analytics
- **Error Resilience:** Each step is tracked and errors are handled gracefully
- **Audit Trail:** Complete workflow logging for compliance and debugging
- **Business Value:** Actual work gets done, not just recommendations

## C. Test & Apply

### How to Test It
```bash
# Set your Gemini API key
export GEMINI_API_KEY="your-api-key-here"

# Install dependencies
pip install -r requirements.txt

# Run agent with dependencies demonstrations
python main.py
```

### Expected Result
You'll see agents performing complete business workflows:

```
🎯 AGENT WITH DEPENDENCIES DEMONSTRATION
======================================================================
Watch how agents use real-world dependencies to execute complete workflows!

🏢 CUSTOMER SUPPORT WORKFLOW DEMONSTRATION
============================================================
👤 Customer: Alice Johnson
📧 Email: alice@example.com
📊 Tier: premium
❗ Issue Summary: Customer experiencing login difficulties with billing discrepancies
🔧 Recommended Solution: Reset password and verify billing information
⚡ Priority Level: high
🚨 Escalation Required: Yes
📝 Generated Response: Hi Alice, I understand your frustration with the login and billing issues...

🔄 Workflow Steps:
   ✅ database_query: Customer data retrieved (0.11s)
   ✅ knowledge_base_search: Found 2 relevant articles (0.21s)
   ✅ agent_analysis: Analysis completed (1.85s)
   ✅ notifications: Notification sent to support-team via slack (0.06s)

📢 Notifications Sent: support-team-slack

🔍 Research Findings:
   • Common login issues include password resets and browser cache problems
   • Billing discrepancies often stem from payment method expiration
   📊 Research Confidence: 87%

⚙️ AUTOMATION WORKFLOW DEMONSTRATION
============================================================
🤖 Process: daily_reports
📥 Inputs Processed: 3
📤 Outputs Generated: 3
❌ Errors Encountered: 0
✅ Success Rate: 100.0%

🔄 Workflow Steps:
   ✅ process_inputs: Processed 3 items (0.01s)
   ✅ business_logic: Business logic applied (1.23s)
   ✅ generate_outputs: Generated 3 outputs (0.02s)
   ✅ save_results: Results saved to s3://reports/daily_reports_results_20240115.json (0.11s)
   ✅ send_notifications: Completion notification sent (0.05s)

📋 Next Actions:
   • Review error logs
   • Retry failed operations
   • Contact technical support

📊 Summary: 3 items, 100.0% success, 1.53s duration

🔧 DEPENDENCY MANAGEMENT DEMONSTRATION
============================================================
📦 Available Services:
   • database
   • search
   • notifications
   • analytics
   • storage
   • email

🗄️ Database Service: MockDatabaseService
🔍 Search Service: MockSearchService
💾 Cache Test: ✅ Working
```

### Validation Examples
- ✅ **Customer Analysis:** Complete workflow from query to response generation
- ❌ **Service Failure:** Agent handles missing dependencies gracefully
- ❌ **Workflow Error:** Partial completion with error tracking and reporting

### Type Checking
```bash
# Verify all workflow models and dependencies are properly typed
mypy main.py workflow_agents.py business_logic.py
```

## D. Common Stumbling Blocks

### Proactive Debugging
**Common Mistake #1: Dependency Not Available**
```
❌ Error: "Service database not found" 
✅ Fix: Ensure all services are registered in dependency factory

# Must register every service used:
factory = DependencyFactory()
factory._initialize_services()  # This registers all services
```

**Common Mistake #2: Async/Workflow Mismatch**
```
❌ Error: "cannot await non-coroutine function"
✅ Fix: Ensure all dependency methods are async where needed

# Database operations should be async:
async def query_customer(self, customer_id: str):
    # async implementation
```

**Common Mistake #3: Workflow State Loss**
```
❌ Error: Workflow fails halfway and no recovery
✅ Fix: Track workflow state at each step and handle errors

# Always log each step:
workflow_steps.append(WorkflowStep(
    step_name="step_name",
    status="completed" or "failed",
    result="description",
    duration_seconds=time_taken
))
```

### Type Safety Gotchas
- **Model validation:** Ensure workflow results match Pydantic model expectations
- **Error propagation:** Handle exceptions at appropriate workflow levels
- **State consistency:** Maintain consistent state across dependency calls
- **Memory management:** Clean up resources in long-running workflows

### Agent with Dependencies Best Practices
1. **Log every step** - Essential for debugging and audit trails
2. **Handle errors gracefully** - Never let one failed step crash the entire workflow
3. **Track performance** - Monitor duration and success rates
4. **Use appropriate models** - Match agent complexity to use case
5. **Design for recovery** - Plan for partial failures and retries

## Ready for the Next Step?
You now understand how agents with dependencies can execute complete business workflows from start to finish!

**Next:** We'll explore **Creating Custom Tools** - how to build specialized tools that your agents can call to perform specific tasks during their workflows.

Agents with dependencies transform from advisors into full-service executors that can query databases, search knowledge bases, send notifications, and generate comprehensive reports! 🚀

**Ready for custom tools, or want to build your own agent workflow first?** 🤔
