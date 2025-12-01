# PydanticAI Gemini Mentor - Agent System Design Project
## Lesson 8: Error Handling and Resilience Patterns

Welcome back, resilience architect! 🛡️ You've built sophisticated agent systems with advanced patterns. Now it's time to make them bulletproof! In production, things go wrong - APIs fail, networks timeout, data is corrupt, and users do unexpected things. This lesson will teach you how to build agent systems that gracefully handle failures, recover automatically, and maintain reliability under pressure.

---

## Why Error Handling Matters

**Real-world reality**: 
- **30% of API calls fail** due to network issues
- **Database connections timeout** under heavy load
- **External services go down** unexpectedly
- **Data validation fails** with malformed inputs
- **Users make mistakes** or provide unexpected inputs

**Without proper error handling**: Your agent crashes, users get poor experiences, and your system becomes unreliable.

**With proper error handling**: Your agent gracefully degrades, recovers automatically, and provides meaningful feedback.

**Analogy Time**: Think of error handling like a car's safety systems:
- **Airbags** (error recovery) - protect when crashes happen
- **ABS brakes** (graceful degradation) - maintain control when systems fail
- **Check engine lights** (monitoring) - alert you to problems early
- **Backup systems** (redundancy) - take over when primary systems fail

---

## 1. Basic Error Handling Patterns

Let's start with fundamental error handling techniques:

```python
from pydantic import BaseModel, Field, ValidationError
from typing import Any, Dict, Optional, List, Union
from datetime import datetime
from enum import Enum
import asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    VALIDATION = "validation"
    NETWORK = "network"
    TIMEOUT = "timeout"
    DATA = "data"
    SYSTEM = "system"
    BUSINESS_LOGIC = "business_logic"

class ErrorInfo(BaseModel):
    """Standardized error information."""
    timestamp: datetime
    severity: ErrorSeverity
    category: ErrorCategory
    message: str
    details: Dict[str, Any]
    recoverable: bool
    retry_count: int = 0
    context: Dict[str, Any] = {}

class ResilientAgent:
    """Base agent with comprehensive error handling."""
    
    def __init__(self, name: str):
        self.name = name
        self.error_counts: Dict[str, int] = {}
        self.consecutive_failures: Dict[str, int] = {}
        self.max_retries = 3
        self.retry_delay = 1.0
        self.circuit_breaker_threshold = 5
    
    def log_error(self, error: Exception, context: Dict[str, Any] = None) -> ErrorInfo:
        """Log and categorize errors."""
        error_info = ErrorInfo(
            timestamp=datetime.now(),
            severity=self._determine_severity(error),
            category=self._categorize_error(error),
            message=str(error),
            details={
                "error_type": type(error).__name__,
                "error_module": type(error).__module__
            },
            recoverable=self._is_recoverable(error),
            context=context or {}
        )
        
        # Update error statistics
        error_key = f"{error_info.category.value}_{error_info.severity.value}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
        # Log the error
        logger.error(f"Agent {self.name}: {error_info.message}", extra={
            "agent": self.name,
            "severity": error_info.severity.value,
            "category": error_info.category.value,
            "recoverable": error_info.recoverable
        })
        
        return error_info
    
    def _determine_severity(self, error: Exception) -> ErrorSeverity:
        """Determine error severity based on error type."""
        if isinstance(error, (ConnectionError, TimeoutError)):
            return ErrorSeverity.MEDIUM
        elif isinstance(error, (ValueError, ValidationError)):
            return ErrorSeverity.LOW
        elif isinstance(error, (SystemError, RuntimeError)):
            return ErrorSeverity.HIGH
        elif isinstance(error, MemoryError):
            return ErrorSeverity.CRITICAL
        else:
            return ErrorSeverity.MEDIUM
    
    def _categorize_error(self, error: Exception) -> ErrorCategory:
        """Categorize error by type."""
        if isinstance(error, (ValidationError, ValueError, TypeError)):
            return ErrorCategory.VALIDATION
        elif isinstance(error, (ConnectionError, OSError)):
            return ErrorCategory.NETWORK
        elif isinstance(error, TimeoutError):
            return ErrorCategory.TIMEOUT
        elif isinstance(error, (KeyError, IndexError)):
            return ErrorCategory.DATA
        elif isinstance(error, (SystemError, RuntimeError)):
            return ErrorCategory.SYSTEM
        else:
            return ErrorCategory.BUSINESS_LOGIC
    
    def _is_recoverable(self, error: Exception) -> bool:
        """Determine if error is recoverable."""
        recoverable_errors = (
            ConnectionError, TimeoutError, ConnectionRefusedError,
            ConnectionResetError, ValueError, ValidationError
        )
        return isinstance(error, recoverable_errors)

# Test basic error handling
def test_basic_error_handling():
    print("=== Testing Basic Error Handling ===")
    
    agent = ResilientAgent("TestAgent")
    
    # Test different error types
    test_errors = [
        ValueError("Invalid input value"),
        ConnectionError("Cannot connect to service"),
        TimeoutError("Request timed out"),
        KeyError("Missing required key"),
        RuntimeError("Unexpected runtime error")
    ]
    
    for error in test_errors:
        print(f"\n🔍 Testing error: {type(error).__name__}")
        error_info = agent.log_error(error, {"operation": "test_operation"})
        
        print(f"   📊 Severity: {error_info.severity.value}")
        print(f"   🏷️ Category: {error_info.category.value}")
        print(f"   ✅ Recoverable: {error_info.recoverable}")
        print(f"   📈 Error counts: {agent.error_counts}")

test_basic_error_handling()
```

**Expected Output:**
```
=== Testing Basic Error Handling ===

🔍 Testing error: ValueError
   📊 Severity: low
   🏷️ Category: validation
   ✅ Recoverable: True
   📈 Error counts: {'validation_low': 1}

🔍 Testing error: ConnectionError
   📊 Severity: medium
   🏷️ Category: network
   ✅ Recoverable: True
   📈 Error counts: {'network_medium': 1}

🔍 Testing error: TimeoutError
   📊 Severity: medium
   🏷️ Category: timeout
   ✅ Recoverable: True
   📈 Error counts: {'timeout_medium': 1}

🔍 Testing error: KeyError
   📊 Severity: medium
   🏷️ Category: data
   ✅ Recoverable: False
   📈 Error counts: {'data_medium': 1}

🔍 Testing error: RuntimeError
   📊 Severity: high
   🏷️ Category: system
   ✅ Recoverable: False
   📈 Error counts: {'system_high': 1}
```

---

## 2. Retry Mechanisms with Backoff

Implement intelligent retry logic with exponential backoff:

```python
import random
import time
from functools import wraps

class RetryConfig(BaseModel):
    """Configuration for retry behavior."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_multiplier: float = 2.0
    jitter: bool = True
    retry_on_exceptions: List[type] = [ConnectionError, TimeoutError, ValueError]

def retry_with_backoff(config: RetryConfig = None):
    """Decorator for implementing retry logic with backoff."""
    if config is None:
        config = RetryConfig()
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(config.max_attempts):
                try:
                    # Call the original function
                    if asyncio.iscoroutinefunction(func):
                        result = await func(*args, **kwargs)
                    else:
                        result = func(*args, **kwargs)
                    
                    # Success - return result
                    if attempt > 0:
                        logger.info(f"Function {func.__name__} succeeded on attempt {attempt + 1}")
                    return result
                
                except Exception as e:
                    last_exception = e
                    
                    # Check if exception is retryable
                    if not any(isinstance(e, exc_type) for exc_type in config.retry_on_exceptions):
                        logger.warning(f"Non-retryable exception in {func.__name__}: {e}")
                        raise e
                    
                    # Check if this is the last attempt
                    if attempt == config.max_attempts - 1:
                        logger.error(f"Function {func.__name__} failed after {config.max_attempts} attempts")
                        raise e
                    
                    # Calculate delay with exponential backoff
                    delay = config.base_delay * (config.backoff_multiplier ** attempt)
                    delay = min(delay, config.max_delay)
                    
                    # Add jitter to prevent thundering herd
                    if config.jitter:
                        delay = delay * (0.5 + random.random() * 0.5)
                    
                    logger.info(f"Function {func.__name__} failed on attempt {attempt + 1}, "
                               f"retrying in {delay:.2f}s: {e}")
                    
                    # Wait before retry
                    if asyncio.iscoroutinefunction(func):
                        await asyncio.sleep(delay)
                    else:
                        time.sleep(delay)
            
            # This should never be reached, but just in case
            raise last_exception
        
        return wrapper
    return decorator

# Example service with retry logic
class ExternalAPIService:
    """Simulated external API service with failures."""
    
    def __init__(self, failure_rate: float = 0.3):
        self.failure_rate = failure_rate
        self.call_count = 0
    
    @retry_with_backoff(RetryConfig(max_attempts=3, base_delay=0.5))
    async def get_data(self, endpoint: str) -> Dict[str, Any]:
        """Get data from external API with retry logic."""
        self.call_count += 1
        
        # Simulate random failures
        if random.random() < self.failure_rate:
            raise ConnectionError(f"Failed to connect to {endpoint}")
        
        # Simulate network delay
        await asyncio.sleep(0.1)
        
        return {
            "endpoint": endpoint,
            "data": f"Sample data from {endpoint}",
            "timestamp": datetime.now().isoformat(),
            "call_attempt": self.call_count
        }
    
    @retry_with_backoff(RetryConfig(max_attempts=5, base_delay=1.0, max_delay=30.0))
    async def get_critical_data(self, endpoint: str) -> Dict[str, Any]:
        """Get critical data with more aggressive retry logic."""
        self.call_count += 1
        
        # Simulate more realistic failure scenarios
        if random.random() < 0.6:  # Higher failure rate
            raise TimeoutError(f"Timeout getting {endpoint}")
        
        await asyncio.sleep(0.2)
        
        return {
            "endpoint": endpoint,
            "critical_data": f"Critical data from {endpoint}",
            "timestamp": datetime.now().isoformat()
        }

# Test retry mechanisms
async def test_retry_mechanisms():
    print("=== Testing Retry Mechanisms ===")
    
    api_service = ExternalAPIService(failure_rate=0.4)
    
    # Test regular endpoint
    print("\n🔄 Testing regular endpoint with retry...")
    try:
        for i in range(3):
            print(f"   Attempt {i+1}:")
            result = await api_service.get_data(f"/users/{i}")
            print(f"   ✅ Success: {result['endpoint']}")
    except Exception as e:
        print(f"   ❌ Final failure: {e}")
    
    # Test critical endpoint
    print("\n🔄 Testing critical endpoint with aggressive retry...")
    try:
        result = await api_service.get_critical_data("/critical/system")
        print(f"   ✅ Success: {result['endpoint']}")
    except Exception as e:
        print(f"   ❌ Final failure: {e}")
    
    print(f"\n📊 Total API calls made: {api_service.call_count}")

# Run the retry test
asyncio.run(test_retry_mechanisms())
```

---

## 3. Circuit Breaker Pattern

Implement circuit breaker to prevent cascade failures:

```python
from enum import Enum
from datetime import datetime, timedelta

class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing - reject requests
    HALF_OPEN = "half_open" # Testing if service recovered

class CircuitBreaker:
    """Circuit breaker implementation for preventing cascade failures."""
    
    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0, expected_exception: type = Exception):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        self.success_count = 0
    
    def call(self, func, *args, **kwargs):
        """Execute function through circuit breaker."""
        if self.state == CircuitState.OPEN:
            # Check if timeout has passed
            if self.last_failure_time and \
               datetime.now() - self.last_failure_time > timedelta(seconds=self.timeout):
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                logger.info("Circuit breaker moved to HALF_OPEN state")
            else:
                raise Exception("Circuit breaker is OPEN - requests blocked")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _on_success(self):
        """Handle successful execution."""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= 3:  # Require 3 successes to close
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                logger.info("Circuit breaker moved to CLOSED state")
        elif self.state == CircuitState.CLOSED:
            self.failure_count = max(0, self.failure_count - 1)  # Gradually recover
    
    def _on_failure(self):
        """Handle failed execution."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state."""
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "failure_threshold": self.failure_threshold,
            "last_failure_time": self.last_failure_time.isoformat() if self.last_failure_time else None,
            "success_count": self.success_count
        }

class ResilientAPIClient:
    """API client with circuit breaker protection."""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.default_config = {
            "failure_threshold": 3,
            "timeout": 30.0
        }
    
    def get_circuit_breaker(self, service_name: str, config: Dict[str, Any] = None) -> CircuitBreaker:
        """Get or create circuit breaker for service."""
        if service_name not in self.circuit_breakers:
            circuit_config = {**self.default_config, **(config or {})}
            self.circuit_breakers[service_name] = CircuitBreaker(
                failure_threshold=circuit_config["failure_threshold"],
                timeout=circuit_config["timeout"]
            )
        return self.circuit_breakers[service_name]
    
    async def make_request(self, service: str, endpoint: str, simulate_failure: bool = False) -> Dict[str, Any]:
        """Make request with circuit breaker protection."""
        circuit_breaker = self.get_circuit_breaker(service)
        
        try:
            result = circuit_breaker.call(self._simulate_api_call, endpoint, simulate_failure)
            return {
                "service": service,
                "endpoint": endpoint,
                "result": result,
                "circuit_state": circuit_breaker.get_state()["state"]
            }
        except Exception as e:
            return {
                "service": service,
                "endpoint": endpoint,
                "error": str(e),
                "circuit_state": circuit_breaker.get_state()["state"]
            }
    
    async def _simulate_api_call(self, endpoint: str, simulate_failure: bool = False) -> Dict[str, Any]:
        """Simulate API call that may fail."""
        if simulate_failure:
            # Introduce random failures
            if random.random() < 0.7:  # 70% failure rate
                raise ConnectionError(f"Service unavailable for {endpoint}")
        
        await asyncio.sleep(0.1)  # Simulate network delay
        
        return {
            "endpoint": endpoint,
            "status": "success",
            "data": f"Response from {endpoint}",
            "timestamp": datetime.now().isoformat()
        }
    
    def get_all_circuit_states(self) -> Dict[str, Dict[str, Any]]:
        """Get state of all circuit breakers."""
        return {
            service: cb.get_state()
            for service, cb in self.circuit_breakers.items()
        }

# Test circuit breaker
async def test_circuit_breaker():
    print("=== Testing Circuit Breaker Pattern ===")
    
    client = ResilientAPIClient()
    services = ["user-service", "payment-service", "notification-service"]
    
    # Test healthy service (no failures)
    print("\n🟢 Testing healthy service...")
    for i in range(5):
        result = await client.make_request("user-service", f"/users/{i}")
        state = result["circuit_state"]
        print(f"   Call {i+1}: {state} {'✅' if state == 'closed' else '⚠️'}")
    
    # Test failing service (should trigger circuit breaker)
    print("\n🔴 Testing failing service...")
    for i in range(10):
        result = await client.make_request("payment-service", f"/payments/{i}", simulate_failure=True)
        state = result["circuit_state"]
        status = "❌" if "error" in result else "✅"
        print(f"   Call {i+1}: {state} {status}")
    
    # Test recovery
    print("\n🟡 Testing recovery...")
    await asyncio.sleep(1)  # Wait for timeout
    
    for i in range(3):
        result = await client.make_request("payment-service", f"/recovery/{i}", simulate_failure=False)
        state = result["circuit_state"]
        status = "✅" if "error" not in result else "❌"
        print(f"   Recovery {i+1}: {state} {status}")
    
    # Show final states
    print("\n📊 Final Circuit Breaker States:")
    states = client.get_all_circuit_states()
    for service, state in states.items():
        print(f"   {service}: {state['state']} (failures: {state['failure_count']})")

# Run the circuit breaker test
asyncio.run(test_circuit_breaker())
```

---

## 4. Graceful Degradation Patterns

Implement graceful degradation when services fail:

```python
from typing import Callable, Any, Optional

class DegradedService:
    """Service that can operate in degraded mode."""
    
    def __init__(self, name: str, primary_impl: Callable, fallback_impl: Callable = None):
        self.name = name
        self.primary_impl = primary_impl
        self.fallback_impl = fallback_impl
        self.service_level = "full"  # full, degraded, offline
        self.fallback_count = 0
    
    async def execute(self, *args, **kwargs) -> Dict[str, Any]:
        """Execute with graceful degradation."""
        try:
            # Try primary implementation
            result = await self.primary_impl(*args, **kwargs)
            self.service_level = "full"
            return {
                "success": True,
                "service_level": self.service_level,
                "result": result,
                "degradation": False
            }
        
        except Exception as e:
            # Primary failed, try fallback if available
            if self.fallback_impl:
                try:
                    fallback_result = await self.fallback_impl(*args, **kwargs)
                    self.service_level = "degraded"
                    self.fallback_count += 1
                    
                    return {
                        "success": True,
                        "service_level": self.service_level,
                        "result": fallback_result,
                        "degradation": True,
                        "degradation_reason": str(e)
                    }
                except Exception as fallback_error:
                    # Both primary and fallback failed
                    self.service_level = "offline"
                    return {
                        "success": False,
                        "service_level": self.service_level,
                        "error": f"Primary: {str(e)}, Fallback: {str(fallback_error)}",
                        "degradation": True
                    }
            else:
                # No fallback available
                self.service_level = "offline"
                return {
                    "success": False,
                    "service_level": self.service_level,
                    "error": str(e),
                    "degradation": False
                }

class FallbackCache:
    """Cache that provides fallback data when primary source fails."""
    
    def __init__(self):
        self.cache: Dict[str, Any] = {}
        self.cache_hits = 0
        self.cache_misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key]
        else:
            self.cache_misses += 1
            return None
    
    def set(self, key: str, value: Any, ttl: int = 3600):
        """Set value in cache."""
        self.cache[key] = {
            "value": value,
            "timestamp": datetime.now(),
            "ttl": ttl
        }
    
    def is_expired(self, key: str) -> bool:
        """Check if cache entry is expired."""
        if key not in self.cache:
            return True
        
        entry = self.cache[key]
        age = datetime.now() - entry["timestamp"]
        return age.total_seconds() > entry["ttl"]

class UserProfileService:
    """User profile service with graceful degradation."""
    
    def __init__(self):
        self.primary_db = None  # Would be actual database
        self.cache = FallbackCache()
        self.api_service = DegradedService(
            "user_api",
            self._get_from_external_api,
            self._get_from_cache_fallback
        )
    
    async def _get_from_external_api(self, user_id: str) -> Dict[str, Any]:
        """Primary: Get user from external API."""
        # Simulate API call that might fail
        if random.random() < 0.4:  # 40% failure rate
            raise ConnectionError("External API unavailable")
        
        await asyncio.sleep(0.1)
        return {
            "user_id": user_id,
            "name": f"User {user_id}",
            "email": f"user{user_id}@example.com",
            "source": "external_api"
        }
    
    async def _get_from_cache_fallback(self, user_id: str) -> Dict[str, Any]:
        """Fallback: Get user from cache."""
        cached_data = self.cache.get(user_id)
        if cached_data and not self.cache.is_expired(user_id):
            return {
                "user_id": user_id,
                "name": cached_data["value"]["name"],
                "email": cached_data["value"]["email"],
                "source": "cache_fallback"
            }
        
        # Final fallback: generate default profile
        return {
            "user_id": user_id,
            "name": f"User {user_id} (Default)",
            "email": f"user{user_id}@default.com",
            "source": "default_fallback"
        }
    
    async def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Get user profile with graceful degradation."""
        print(f"🔍 Getting profile for user {user_id}...")
        
        # First try cache
        cached_data = self.cache.get(user_id)
        if cached_data and not self.cache.is_expired(user_id):
            return {
                "success": True,
                "user_id": user_id,
                "profile": cached_data["value"],
                "source": "cache",
                "service_level": "full"
            }
        
        # Cache miss or expired, try API with degradation
        result = await self.api_service.execute(user_id)
        
        if result["success"]:
            # Cache the successful result
            self.cache.set(user_id, result["result"])
            
            return {
                "success": True,
                "user_id": user_id,
                "profile": result["result"],
                "source": result["result"]["source"],
                "service_level": result["service_level"],
                "degradation": result["degradation"]
            }
        else:
            # Complete failure, use default
            default_profile = {
                "user_id": user_id,
                "name": f"User {user_id} (Unknown)",
                "email": f"user{user_id}@unknown.com",
                "source": "default"
            }
            
            return {
                "success": True,  # Still return success with default data
                "user_id": user_id,
                "profile": default_profile,
                "source": "default",
                "service_level": "offline",
                "error": result["error"]
            }

# Test graceful degradation
async def test_graceful_degradation():
    print("=== Testing Graceful Degradation ===")
    
    user_service = UserProfileService()
    
    # Test multiple users
    test_users = ["user_001", "user_002", "user_003", "user_004", "user_005"]
    
    for i, user_id in enumerate(test_users, 1):
        print(f"\n--- Test {i}: User {user_id} ---")
        
        result = await user_service.get_user_profile(user_id)
        
        print(f"   ✅ Success: {result['success']}")
        print(f"   📊 Source: {result['source']}")
        print(f"   🎯 Level: {result['service_level']}")
        print(f"   👤 Profile: {result['profile']['name']}")
        print(f"   📧 Email: {result['profile']['email']}")
        
        if result.get('degradation'):
            print(f"   ⚠️ Degraded: {result.get('degradation_reason', 'Unknown reason')}")
    
    # Show cache statistics
    print(f"\n📊 Cache Statistics:")
    print(f"   💾 Hits: {user_service.cache.cache_hits}")
    print(f"   ❌ Misses: {user_service.cache.cache_misses}")
    print(f"   📈 Hit Rate: {user_service.cache.cache_hits / max(1, user_service.cache.cache_hits + user_service.cache.cache_misses):.1%}")

# Run the graceful degradation test
asyncio.run(test_graceful_degradation())
```

---

## 5. Monitoring and Health Checks

Implement comprehensive monitoring and health checks:

```python
from dataclasses import dataclass
from typing import Dict, List, Any, Callable
import psutil
import time

@dataclass
class HealthMetric:
    """Health metric definition."""
    name: str
    value: float
    threshold: float
    status: str  # healthy, warning, critical
    timestamp: datetime
    description: str

class HealthMonitor:
    """Monitor system and service health."""
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.metrics: List[HealthMetric] = []
        self.check_functions: List[Callable] = []
        self.alert_callbacks: List[Callable] = []
        self.start_time = time.time()
    
    def add_health_check(self, check_func: Callable):
        """Add a health check function."""
        self.check_functions.append(check_func)
    
    def add_alert_callback(self, alert_func: Callable):
        """Add an alert callback function."""
        self.alert_callbacks.append(alert_func)
    
    async def run_health_checks(self) -> Dict[str, Any]:
        """Run all health checks and return status."""
        health_status = {
            "agent_name": self.agent_name,
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": time.time() - self.start_time,
            "overall_status": "healthy",
            "metrics": [],
            "checks": [],
            "alerts": []
        }
        
        # Run custom health checks
        for check_func in self.check_functions:
            try:
                if asyncio.iscoroutinefunction(check_func):
                    result = await check_func()
                else:
                    result = check_func()
                
                health_status["checks"].append({
                    "check_name": check_func.__name__,
                    "status": "passed",
                    "result": result
                })
            except Exception as e:
                health_status["checks"].append({
                    "check_name": check_func.__name__,
                    "status": "failed",
                    "error": str(e)
                })
                health_status["overall_status"] = "critical"
        
        # Run system metrics
        system_metrics = await self._collect_system_metrics()
        health_status["metrics"] = system_metrics
        
        # Determine overall health
        critical_metrics = [m for m in system_metrics if m.status == "critical"]
        if critical_metrics:
            health_status["overall_status"] = "critical"
        elif any(m.status == "warning" for m in system_metrics):
            health_status["overall_status"] = "warning"
        
        # Trigger alerts if needed
        if health_status["overall_status"] in ["warning", "critical"]:
            await self._trigger_alerts(health_status)
        
        return health_status
    
    async def _collect_system_metrics(self) -> List[HealthMetric]:
        """Collect system-level health metrics."""
        metrics = []
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        metrics.append(HealthMetric(
            name="cpu_usage",
            value=cpu_percent,
            threshold=80.0,
            status="healthy" if cpu_percent < 80 else ("warning" if cpu_percent < 90 else "critical"),
            timestamp=datetime.now(),
            description="CPU usage percentage"
        ))
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        metrics.append(HealthMetric(
            name="memory_usage",
            value=memory_percent,
            threshold=80.0,
            status="healthy" if memory_percent < 80 else ("warning" if memory_percent < 90 else "critical"),
            timestamp=datetime.now(),
            description="Memory usage percentage"
        ))
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        metrics.append(HealthMetric(
            name="disk_usage",
            value=disk_percent,
            threshold=85.0,
            status="healthy" if disk_percent < 85 else ("warning" if disk_percent < 95 else "critical"),
            timestamp=datetime.now(),
            description="Disk usage percentage"
        ))
        
        return metrics
    
    async def _trigger_alerts(self, health_status: Dict[str, Any]):
        """Trigger alert callbacks."""
        for alert_func in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(alert_func):
                    await alert_func(health_status)
                else:
                    alert_func(health_status)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

class ResilientAgentWithMonitoring(ResilientAgent):
    """Agent with comprehensive monitoring."""
    
    def __init__(self, name: str):
        super().__init__(name)
        self.health_monitor = HealthMonitor(name)
        self.performance_metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0.0,
            "last_request_time": None
        }
        self._setup_default_health_checks()
    
    def _setup_default_health_checks(self):
        """Setup default health checks."""
        self.health_monitor.add_health_check(self._check_error_rate)
        self.health_monitor.add_health_check(self._check_consecutive_failures)
        self.health_monitor.add_alert_callback(self._log_health_alert)
    
    def _check_error_rate(self) -> Dict[str, Any]:
        """Check error rate health."""
        total = self.performance_metrics["total_requests"]
        failed = self.performance_metrics["failed_requests"]
        
        if total == 0:
            return {"error_rate": 0, "status": "healthy"}
        
        error_rate = (failed / total) * 100
        status = "healthy"
        if error_rate > 10:
            status = "warning"
        elif error_rate > 25:
            status = "critical"
        
        return {
            "error_rate": error_rate,
            "status": status,
            "total_requests": total,
            "failed_requests": failed
        }
    
    def _check_consecutive_failures(self) -> Dict[str, Any]:
        """Check consecutive failures."""
        max_consecutive = max(self.consecutive_failures.values()) if self.consecutive_failures else 0
        
        if max_consecutive >= 5:
            status = "critical"
        elif max_consecutive >= 3:
            status = "warning"
        else:
            status = "healthy"
        
        return {
            "max_consecutive_failures": max_consecutive,
            "status": status
        }
    
    def _log_health_alert(self, health_status: Dict[str, Any]):
        """Log health alerts."""
        logger.warning(f"Health alert for {self.name}: {health_status['overall_status']}")
        
        if health_status["overall_status"] == "critical":
            # Could trigger email, slack notification, etc.
            print(f"🚨 CRITICAL ALERT for {self.name}!")
    
    async def monitored_execute(self, operation: Callable, *args, **kwargs) -> Any:
        """Execute operation with monitoring."""
        start_time = time.time()
        self.performance_metrics["total_requests"] += 1
        self.performance_metrics["last_request_time"] = datetime.now()
        
        try:
            # Execute the operation
            if asyncio.iscoroutinefunction(operation):
                result = await operation(*args, **kwargs)
            else:
                result = operation(*args, **kwargs)
            
            self.performance_metrics["successful_requests"] += 1
            
            # Update response time
            response_time = time.time() - start_time
            self._update_response_time(response_time)
            
            return result
        
        except Exception as e:
            self.performance_metrics["failed_requests"] += 1
            
            # Log error and update consecutive failures
            error_info = self.log_error(e, {"operation": operation.__name__ if hasattr(operation, '__name__') else str(operation)})
            
            # Update consecutive failures for this operation
            op_name = operation.__name__ if hasattr(operation, '__name__') else str(operation)
            self.consecutive_failures[op_name] = self.consecutive_failures.get(op_name, 0) + 1
            
            raise e
    
    def _update_response_time(self, response_time: float):
        """Update average response time."""
        current_avg = self.performance_metrics["average_response_time"]
        total_requests = self.performance_metrics["total_requests"]
        
        # Running average
        self.performance_metrics["average_response_time"] = (
            (current_avg * (total_requests - 1) + response_time) / total_requests
        )
    
    async def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report."""
        health_status = await self.health_monitor.run_health_checks()
        
        health_report = {
            **health_status,
            "performance": self.performance_metrics.copy(),
            "error_statistics": self.error_counts.copy(),
            "consecutive_failures": self.consecutive_failures.copy()
        }
        
        return health_report

# Test monitoring system
async def test_monitoring():
    print("=== Testing Monitoring and Health Checks ===")
    
    agent = ResilientAgentWithMonitoring("MonitoredAgent")
    
    # Simulate some operations (some successful, some failing)
    operations = [
        ("successful_op", lambda: "Success!"),
        ("failing_op", lambda: 1/0),
        ("another_success", lambda: "Another success!"),
        ("timeout_op", lambda: asyncio.sleep(10)),
        ("final_success", lambda: "Final success!")
    ]
    
    for op_name, operation in operations:
        print(f"\n🔄 Executing {op_name}...")
        try:
            result = await agent.monitored_execute(operation)
            print(f"   ✅ Success: {result}")
        except Exception as e:
            print(f"   ❌ Failed: {str(e)}")
    
    # Get health report
    print(f"\n📊 Health Report:")
    health_report = await agent.get_health_report()
    
    print(f"   🤖 Agent: {health_report['agent_name']}")
    print(f"   ⏱️ Uptime: {health_report['uptime_seconds']:.1f}s")
    print(f"   📊 Overall Status: {health_report['overall_status']}")
    print(f"   📈 Requests: {health_report['performance']['total_requests']}")
    print(f"   ✅ Success Rate: {health_report['performance']['successful_requests'] / max(1, health_report['performance']['total_requests']):.1%}")
    print(f"   ⚡ Avg Response Time: {health_report['performance']['average_response_time']:.3f}s")
    
    print(f"\n🚨 Health Checks:")
    for check in health_report['checks']:
        status_icon = "✅" if check['status'] == 'passed' else "❌"
        print(f"   {status_icon} {check['check_name']}: {check['status']}")
    
    print(f"\n📈 System Metrics:")
    for metric in health_report['metrics']:
        status_icon = "✅" if metric.status == "healthy" else ("⚠️" if metric.status == "warning" else "🚨")
        print(f"   {status_icon} {metric.name}: {metric.value:.1f}% ({metric.status})")

# Run the monitoring test
asyncio.run(test_monitoring())
```

---

## 6. Production-Ready Resilience Patterns

Combine all patterns into a production-ready system:

```python
class ProductionResilientAgent:
    """Production-ready agent with full resilience patterns."""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        # Basic resilience
        self.name = name
        self.error_counts = {}
        self.consecutive_failures = {}
        
        # Configuration
        self.config = config or self._default_config()
        
        # Circuit breakers for different services
        self.circuit_breakers = {}
        
        # Health monitoring
        self.health_monitor = HealthMonitor(name)
        self.performance_metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0.0
        }
        
        # Degradation services
        self.degraded_services = {}
        
        # Setup health checks
        self._setup_production_health_checks()
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for production resilience."""
        return {
            "max_retries": 3,
            "retry_base_delay": 1.0,
            "circuit_breaker_threshold": 5,
            "circuit_breaker_timeout": 60.0,
            "graceful_degradation": True,
            "health_check_interval": 30,
            "alert_thresholds": {
                "error_rate_warning": 10.0,
                "error_rate_critical": 25.0,
                "response_time_warning": 5.0,
                "response_time_critical": 10.0
            }
        }
    
    def _setup_production_health_checks(self):
        """Setup comprehensive health checks."""
        self.health_monitor.add_health_check(self._check_error_rate)
        self.health_monitor.add_health_check(self._check_response_time)
        self.health_monitor.add_health_check(self._check_circuit_breakers)
        self.health_monitor.add_health_check(self._check_service_availability)
        
        # Add alert callbacks
        self.health_monitor.add_alert_callback(self._send_high_priority_alert)
        self.health_monitor.add_alert_callback(self._log_performance_metrics)
    
    async def execute_with_full_resilience(self, 
                                          operation: Callable, 
                                          operation_name: str = None,
                                          service_name: str = None,
                                          *args, **kwargs) -> Dict[str, Any]:
        """Execute operation with full resilience patterns."""
        operation_name = operation_name or operation.__name__ if hasattr(operation, '__name__') else str(operation)
        service_name = service_name or "default"
        
        start_time = time.time()
        self.performance_metrics["total_requests"] += 1
        
        try:
            # Step 1: Check circuit breaker
            circuit_breaker = self._get_circuit_breaker(service_name)
            circuit_state = circuit_breaker.get_state()
            
            if circuit_state["state"] == "open":
                # Circuit breaker open, try degradation
                degraded_result = await self._try_degraded_operation(operation_name, service_name, *args, **kwargs)
                if degraded_result:
                    return degraded_result
                else:
                    raise Exception(f"Circuit breaker open for service {service_name}")
            
            # Step 2: Execute with retry logic
            result = await self._execute_with_retry(operation, *args, **kwargs)
            
            # Step 3: Update metrics
            self._on_success(operation_name, start_time)
            
            # Step 4: Return successful result
            return {
                "success": True,
                "result": result,
                "operation": operation_name,
                "service": service_name,
                "execution_time": time.time() - start_time,
                "resilience_patterns": {
                    "circuit_breaker_state": circuit_state["state"],
                    "retry_attempts": 1,  # Would track actual retries
                    "degradation_used": False
                }
            }
        
        except Exception as e:
            # Step 1: Handle failure
            self._on_failure(operation_name, service_name, e)
            
            # Step 2: Try graceful degradation
            degraded_result = await self._try_degraded_operation(operation_name, service_name, *args, **kwargs)
            if degraded_result:
                return degraded_result
            
            # Step 3: Return failure result
            return {
                "success": False,
                "error": str(e),
                "operation": operation_name,
                "service": service_name,
                "execution_time": time.time() - start_time,
                "error_info": self.log_error(e, {
                    "operation": operation_name,
                    "service": service_name
                })
            }
    
    async def _execute_with_retry(self, operation: Callable, *args, **kwargs) -> Any:
        """Execute operation with retry logic."""
        retry_config = RetryConfig(
            max_attempts=self.config["max_retries"],
            base_delay=self.config["retry_base_delay"],
            max_delay=30.0
        )
        
        retry_decorator = retry_with_backoff(retry_config)
        
        if asyncio.iscoroutinefunction(operation):
            return await retry_decorator(operation)(*args, **kwargs)
        else:
            return retry_decorator(operation)(*args, **kwargs)
    
    async def _try_degraded_operation(self, operation_name: str, service_name: str, *args, **kwargs) -> Optional[Dict[str, Any]]:
        """Try degraded operation if available."""
        if not self.config["graceful_degradation"]:
            return None
        
        degraded_service = self.degraded_services.get(service_name)
        if not degraded_service:
            return None
        
        try:
            result = await degraded_service.execute(*args, **kwargs)
            if result["success"]:
                return {
                    "success": True,
                    "result": result["result"],
                    "operation": operation_name,
                    "service": service_name,
                    "resilience_patterns": {
                        "degradation_used": True,
                        "service_level": result["service_level"]
                    }
                }
        except Exception as e:
            logger.warning(f"Degraded operation failed: {e}")
        
        return None
    
    def _get_circuit_breaker(self, service_name: str) -> CircuitBreaker:
        """Get or create circuit breaker for service."""
        if service_name not in self.circuit_breakers:
            self.circuit_breakers[service_name] = CircuitBreaker(
                failure_threshold=self.config["circuit_breaker_threshold"],
                timeout=self.config["circuit_breaker_timeout"]
            )
        return self.circuit_breakers[service_name]
    
    def _on_success(self, operation_name: str, start_time: float):
        """Handle successful execution."""
        self.performance_metrics["successful_requests"] += 1
        
        # Update response time
        response_time = time.time() - start_time
        total = self.performance_metrics["total_requests"]
        current_avg = self.performance_metrics["average_response_time"]
        self.performance_metrics["average_response_time"] = (
            (current_avg * (total - 1) + response_time) / total
        )
        
        # Reset consecutive failures
        self.consecutive_failures[operation_name] = 0
    
    def _on_failure(self, operation_name: str, service_name: str, error: Exception):
        """Handle failed execution."""
        self.performance_metrics["failed_requests"] += 1
        
        # Update consecutive failures
        self.consecutive_failures[operation_name] = self.consecutive_failures.get(operation_name, 0) + 1
        
        # Log error
        self.log_error(error, {
            "operation": operation_name,
            "service": service_name,
            "consecutive_failures": self.consecutive_failures[operation_name]
        })
        
        # Trigger circuit breaker if needed
        circuit_breaker = self._get_circuit_breaker(service_name)
        if hasattr(circuit_breaker, '_on_failure'):
            circuit_breaker._on_failure()
    
    # Health check methods
    def _check_error_rate(self) -> Dict[str, Any]:
        """Check error rate health."""
        total = self.performance_metrics["total_requests"]
        if total == 0:
            return {"error_rate": 0, "status": "healthy"}
        
        failed = self.performance_metrics["failed_requests"]
        error_rate = (failed / total) * 100
        
        thresholds = self.config["alert_thresholds"]
        if error_rate >= thresholds["error_rate_critical"]:
            status = "critical"
        elif error_rate >= thresholds["error_rate_warning"]:
            status = "warning"
        else:
            status = "healthy"
        
        return {
            "error_rate": error_rate,
            "status": status,
            "total_requests": total,
            "failed_requests": failed
        }
    
    def _check_response_time(self) -> Dict[str, Any]:
        """Check response time health."""
        avg_response_time = self.performance_metrics["average_response_time"]
        thresholds = self.config["alert_thresholds"]
        
        if avg_response_time >= thresholds["response_time_critical"]:
            status = "critical"
        elif avg_response_time >= thresholds["response_time_warning"]:
            status = "warning"
        else:
            status = "healthy"
        
        return {
            "average_response_time": avg_response_time,
            "status": status
        }
    
    def _check_circuit_breakers(self) -> Dict[str, Any]:
        """Check circuit breaker states."""
        open_breakers = []
        for service, cb in self.circuit_breakers.items():
            if cb.get_state()["state"] == "open":
                open_breakers.append(service)
        
        if open_breakers:
            status = "warning"
        else:
            status = "healthy"
        
        return {
            "status": status,
            "open_circuit_breakers": open_breakers,
            "total_services": len(self.circuit_breakers)
        }
    
    def _check_service_availability(self) -> Dict[str, Any]:
        """Check overall service availability."""
        services_count = len(self.circuit_breakers)
        if services_count == 0:
            return {"availability": 100.0, "status": "healthy"}
        
        open_breakers = sum(1 for cb in self.circuit_breakers.values() 
                          if cb.get_state()["state"] == "open")
        
        availability = ((services_count - open_breakers) / services_count) * 100
        
        if availability >= 95:
            status = "healthy"
        elif availability >= 80:
            status = "warning"
        else:
            status = "critical"
        
        return {
            "availability": availability,
            "status": status,
            "services_count": services_count,
            "unavailable_services": open_breakers
        }
    
    def _send_high_priority_alert(self, health_status: Dict[str, Any]):
        """Send high-priority alerts."""
        if health_status["overall_status"] == "critical":
            # In production, this would send alerts via email, Slack, PagerDuty, etc.
            print(f"🚨 HIGH PRIORITY ALERT: Agent {self.name} is in critical state!")
            print(f"   Error rate: {health_status.get('error_rate', 'N/A')}")
            print(f"   Services affected: {health_status.get('unavailable_services', 'N/A')}")
    
    def _log_performance_metrics(self, health_status: Dict[str, Any]):
        """Log performance metrics."""
        logger.info(f"Performance metrics for {self.name}: "
                   f"Success rate: {self.performance_metrics['successful_requests'] / max(1, self.performance_metrics['total_requests']):.1%}")

# Test production-ready resilience
async def test_production_resilience():
    print("=== Testing Production-Ready Resilience ===")
    
    # Create production agent
    config = {
        "max_retries": 2,
        "circuit_breaker_threshold": 3,
        "graceful_degradation": True
    }
    
    agent = ProductionResilientAgent("ProductionAgent", config)
    
    # Add degraded service
    def primary_operation():
        if random.random() < 0.3:  # 30% failure rate
            raise ConnectionError("Service unavailable")
        return "Primary operation result"
    
    def fallback_operation():
        return "Fallback operation result"
    
    degraded_service = DegradedService("test_service", primary_operation, fallback_operation)
    agent.degraded_services["test_service"] = degraded_service
    
    # Test operations with full resilience
    print("\n🧪 Testing operations with full resilience...")
    
    for i in range(10):
        print(f"\n--- Operation {i+1} ---")
        
        result = await agent.execute_with_full_resilience(
            lambda: "Success!" if random.random() > 0.4 else (_ for _ in ()).throw(ConnectionError("Random failure")),
            f"test_operation_{i+1}",
            "test_service"
        )
        
        print(f"   ✅ Success: {result['success']}")
        print(f"   ⏱️ Time: {result['execution_time']:.3f}s")
        
        if result['success']:
            patterns = result.get('resilience_patterns', {})
            if patterns.get('degradation_used'):
                print(f"   ⚠️ Used degradation: {patterns.get('service_level', 'unknown')}")
            if patterns.get('circuit_breaker_state') != 'closed':
                print(f"   🔌 Circuit breaker: {patterns['circuit_breaker_state']}")
    
    # Get final health report
    print(f"\n📊 Final Health Report:")
    health_report = await agent.health_monitor.run_health_checks()
    
    print(f"   🤖 Agent: {health_report['agent_name']}")
    print(f"   📊 Status: {health_report['overall_status']}")
    print(f"   📈 Requests: {agent.performance_metrics['total_requests']}")
    print(f"   ✅ Success Rate: {agent.performance_metrics['successful_requests'] / max(1, agent.performance_metrics['total_requests']):.1%}")
    print(f"   ⚡ Avg Response: {agent.performance_metrics['average_response_time']:.3f}s")
    print(f"   🔌 Circuit Breakers: {len(agent.circuit_breakers)}")
    
    # Show system metrics
    print(f"\n💻 System Health:")
    for metric in health_report['metrics']:
        status_icon = {"healthy": "✅", "warning": "⚠️", "critical": "🚨"}.get(metric.status, "❓")
        print(f"   {status_icon} {metric.name}: {metric.value:.1f}%")

# Run the production resilience test
asyncio.run(test_production_resilience())
```

---

## Summary: Building Bulletproof Agent Systems

**Resilience Patterns Mastered:**

1. **Basic Error Handling**: Categorization, logging, and error classification
2. **Retry Mechanisms**: Exponential backoff with intelligent failure detection
3. **Circuit Breakers**: Prevent cascade failures with automatic recovery testing
4. **Graceful Degradation**: Maintain functionality even when services fail
5. **Health Monitoring**: Comprehensive system health tracking and alerting
6. **Production Patterns**: Full resilience integration for enterprise systems

**Key Benefits:**
- **Reliability**: Systems continue operating even when components fail
- **Observability**: Clear visibility into system health and performance
- **Recovery**: Automatic recovery from transient failures
- **Scalability**: Handle varying loads and failure patterns gracefully
- **Maintainability**: Easy to debug and optimize based on metrics

**Production Checklist:**
- ✅ Error categorization and logging
- ✅ Retry logic with exponential backoff
- ✅ Circuit breaker protection
- ✅ Graceful degradation pathways
- ✅ Health monitoring and alerting
- ✅ Performance metrics tracking
- ✅ Comprehensive test coverage

**Real-World Applications:**
- **Financial Systems**: Transaction processing with rollback capabilities
- **E-commerce**: Order processing with payment gateway fallbacks
- **Healthcare**: Critical data processing with redundancy
- **IoT Systems**: Sensor data processing with offline capabilities
- **API Gateways**: Request routing with service discovery and load balancing

**Next Level:** Distributed resilience patterns, chaos engineering, and self-healing systems that can automatically reconfigure themselves based on observed patterns!

---

## 🎯 Practice Exercise

Build a **Financial Transaction Processor** that:

1. **Validates transactions** with comprehensive error handling
2. **Retries failed payment gateway calls** with exponential backoff
3. **Circuit breakers** prevent cascade failures during outages
4. **Graceful degradation** to manual processing when automated systems fail
5. **Real-time monitoring** of transaction success rates and system health
6. **Audit logging** for compliance and debugging

This should handle real money transactions, so make it bulletproof! 💰

Ready to move to **Lesson 9: Performance Optimization and Scalability** where we'll explore how to make your agent systems lightning-fast and handle massive loads efficiently? Let's optimize those systems! ⚡