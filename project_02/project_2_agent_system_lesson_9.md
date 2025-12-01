# Lesson 9: Performance Optimization and Scalability

## Introduction

Welcome to Lesson 9! In this lesson, we'll dive deep into making your agent systems lightning-fast and capable of handling massive workloads. Performance optimization isn't just about speed—it's about creating robust, scalable systems that can grow with your needs.

## Table of Contents
1. [Caching Strategies](#caching-strategies)
2. [Async Optimization](#async-optimization)
3. [Load Balancing](#load-balancing)
4. [Performance Monitoring](#performance-monitoring)
5. [Real-World Implementation](#real-world-implementation)
6. [Advanced Optimization Techniques](#advanced-optimization-techniques)

## Caching Strategies

### Understanding Cache Types

Let's start with the fundamental caching strategies that can dramatically improve your agent's performance:

```python
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List, Tuple
import hashlib
import json
import time
import asyncio
from dataclasses import dataclass
from abc import ABC, abstractmethod

# 1. In-Memory Cache Implementation
@dataclass
class CacheEntry:
    """Represents a cached item with metadata"""
    value: Any
    created_at: float
    accessed_at: float
    access_count: int
    ttl: Optional[float] = None
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired"""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
    
    def access(self) -> Any:
        """Access the cached value and update metadata"""
        self.accessed_at = time.time()
        self.access_count += 1
        return self.value

class InMemoryCache:
    """High-performance in-memory cache with TTL support"""
    
    def __init__(self, max_size: int = 1000, default_ttl: float = 3600):
        self._cache: Dict[str, CacheEntry] = {}
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._hit_count = 0
        self._miss_count = 0
    
    def _generate_key(self, key: str, params: Dict[str, Any]) -> str:
        """Generate a consistent cache key from parameters"""
        key_data = f"{key}:{json.dumps(params, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def get(self, key: str, params: Dict[str, Any] = None) -> Optional[Any]:
        """Retrieve value from cache"""
        cache_key = self._generate_key(key, params or {})
        
        if cache_key in self._cache:
            entry = self._cache[cache_key]
            if not entry.is_expired():
                self._hit_count += 1
                return entry.access()
            else:
                # Remove expired entry
                del self._cache[cache_key]
        
        self._miss_count += 1
        return None
    
    async def set(self, key: str, value: Any, ttl: float = None, params: Dict[str, Any] = None):
        """Store value in cache with optional TTL"""
        if len(self._cache) >= self._max_size:
            await self._evict_lru()
        
        cache_key = self._generate_key(key, params or {})
        entry = CacheEntry(
            value=value,
            created_at=time.time(),
            accessed_at=time.time(),
            access_count=0,
            ttl=ttl or self._default_ttl
        )
        self._cache[cache_key] = entry
    
    async def _evict_lru(self):
        """Evict least recently used item"""
        if not self._cache:
            return
        
        # Find least recently accessed item
        lru_key = min(self._cache.keys(), key=lambda k: self._cache[k].accessed_at)
        del self._cache[lru_key]
    
    async def clear(self):
        """Clear all cache entries"""
        self._cache.clear()
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_requests = self._hit_count + self._miss_count
        hit_rate = self._hit_count / total_requests if total_requests > 0 else 0
        
        return {
            "hit_count": self._hit_count,
            "miss_count": self._miss_count,
            "hit_rate": hit_rate,
            "cache_size": len(self._cache),
            "max_size": self._max_size
        }

# 2. Intelligent Cache Manager
class IntelligentCacheManager:
    """Advanced cache manager with automatic optimization"""
    
    def __init__(self, memory_cache: InMemoryCache):
        self.memory_cache = memory_cache
    
    async def get(self, key: str, params: Dict[str, Any] = None):
        """Get value from cache"""
        return await self.memory_cache.get(key, params)
    
    async def set(self, key: str, value: Any, ttl: float = 1800, params: Dict[str, Any] = None):
        """Set value in cache"""
        await self.memory_cache.set(key, value, ttl=ttl, params=params)
    
    async def optimize_cache(self):
        """Automatically optimize cache performance"""
        stats = await self.memory_cache.get_stats()
        
        # Adjust TTL based on access patterns
        if stats["hit_rate"] < 0.5:
            # Cache miss rate is high, reduce TTL
            self.memory_cache._default_ttl = max(300, self.memory_cache._default_ttl * 0.8)
        elif stats["hit_rate"] > 0.9:
            # Cache hit rate is excellent, can increase TTL
            self.memory_cache._default_ttl = min(7200, self.memory_cache._default_ttl * 1.1)

# Agent Caching Implementation
from pydantic_ai import Agent
import asyncio
from typing import TypeVar, Generic, Callable, Awaitable

T = TypeVar('T')

class CachedAgent(Generic[T]):
    """Wrapper for agents with intelligent caching"""
    
    def __init__(self, agent: Agent[T], cache_manager: IntelligentCacheManager, cache_prefix: str = "agent"):
        self.agent = agent
        self.cache_manager = cache_manager
        self.cache_prefix = cache_prefix
    
    async def run_cached(self, prompt: str, params: Dict[str, Any] = None, use_cache: bool = True, cache_ttl: float = 1800) -> T:
        """Run agent with intelligent caching"""
        
        # Generate cache key from prompt and parameters
        cache_key = f"{self.cache_prefix}:run"
        cache_params = {"prompt": prompt, "params": params or {}}
        
        if use_cache:
            # Try to get from cache first
            cached_result = await self.cache_manager.get(cache_key, cache_params)
            
            if cached_result is not None:
                print(f"✅ Cache hit for agent run (TTL: {cache_ttl}s)")
                return cached_result
        
        # Cache miss - run the agent
        print("🔄 Running agent (cache miss)")
        result = await self.agent.run(prompt, **params or {})
        
        # Store in cache
        if use_cache:
            await self.cache_manager.set(cache_key, result.data if hasattr(result, 'data') else result, ttl=cache_ttl, params=cache_params)
        
        return result

# Example usage with caching
async def create_cached_agent_system():
    """Example of a fully cached agent system"""
    
    # Initialize caches
    memory_cache = InMemoryCache(max_size=500, default_ttl=1800)
    cache_manager = IntelligentCacheManager(memory_cache)
    
    # Create agent with caching
    agent = Agent('gemini-1.5-flash', result_type=str, system_prompt='You are a helpful assistant.')
    cached_agent = CachedAgent(agent, cache_manager, "demo_agent")
    
    # Test caching performance
    test_prompt = "Explain quantum computing in simple terms"
    
    # First run (cache miss)
    start_time = time.time()
    result1 = await cached_agent.run_cached(test_prompt, cache_ttl=300)
    first_run_time = time.time() - start_time
    
    # Second run (should be cache hit)
    start_time = time.time()
    result2 = await cached_agent.run_cached(test_prompt, cache_ttl=300)
    second_run_time = time.time() - start_time
    
    print(f"First run time: {first_run_time:.2f}s")
    print(f"Second run time (cached): {second_run_time:.2f}s")
    print(f"Speed improvement: {first_run_time/second_run_time:.1f}x faster")
    
    # Get cache statistics
    stats = await memory_cache.get_stats()
    print(f"Cache hit rate: {stats['hit_rate']:.1%}")
    
    return cached_agent
```

## Async Optimization

### Concurrent Agent Operations

Let's implement sophisticated async patterns for maximum performance:

```python
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional

class AsyncAgentPool:
    """Pool of agents for concurrent processing"""
    
    def __init__(self, agent_creators: List[Callable[[], Agent]], max_concurrent: int = 10, timeout: float = 30.0):
        self.agent_creators = agent_creators
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.agents: List[Agent] = []
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent)
    
    async def initialize(self):
        """Initialize all agents in the pool"""
        self.agents = [creator() for creator in self.agent_creators]
    
    async def run_concurrent(self, requests: List[Dict[str, Any]], result_type: type) -> List[Any]:
        """Run multiple agent requests concurrently"""
        
        async def _run_single(request_data: Dict[str, Any]) -> Any:
            async with self.semaphore:
                # Select appropriate agent (round-robin for load distribution)
                agent_index = hash(request_data.get('prompt', '')) % len(self.agents)
                agent = self.agents[agent_index]
                
                try:
                    # Run with timeout
                    result = await asyncio.wait_for(
                        agent.run(request_data['prompt'], **request_data.get('params', {})),
                        timeout=self.timeout
                    )
                    return result.data if hasattr(result, 'data') else result
                except asyncio.TimeoutError:
                    return f"Request timed out after {self.timeout}s"
                except Exception as e:
                    return f"Error: {str(e)}"
        
        # Execute all requests concurrently
        tasks = [_run_single(request) for request in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return results

# Async Processing Pipeline
class AsyncProcessingPipeline:
    """Advanced async processing pipeline with multiple stages"""
    
    def __init__(self, stages: List[Callable], max_workers: int = 5):
        self.stages = stages
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
    
    async def process_batch(self, items: List[Any]) -> List[Any]:
        """Process items through all pipeline stages"""
        
        current_batch = items
        
        for stage_idx, stage in enumerate(self.stages):
            print(f"🔄 Processing stage {stage_idx + 1}/{len(self.stages)}")
            
            # Process items in parallel within each stage
            tasks = []
            for item in current_batch:
                if asyncio.iscoroutinefunction(stage):
                    task = stage(item)
                else:
                    task = asyncio.get_event_loop().run_in_executor(self.executor, stage, item)
                tasks.append(task)
            
            # Wait for all tasks in this stage to complete
            current_batch = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions
            current_batch = [item for item in current_batch if not isinstance(item, Exception)]
        
        return current_batch

# Example: High-Performance Agent System
async def create_high_performance_agent_system():
    """Complete high-performance agent system with all optimizations"""
    
    # 1. Initialize caching
    memory_cache = InMemoryCache(max_size=1000, default_ttl=3600)
    cache_manager = IntelligentCacheManager(memory_cache)
    
    # 2. Create agent pool
    def create_analysis_agent():
        return Agent('gemini-1.5-flash', result_type=str, system_prompt='You are a data analysis expert. Provide clear, actionable insights.')
    
    def create_writing_agent():
        return Agent('gemini-1.5-flash', result_type=str, system_prompt='You are a professional writer. Create engaging, well-structured content.')
    
    agent_pool = AsyncAgentPool([create_analysis_agent, create_writing_agent], max_concurrent=8, timeout=20.0)
    await agent_pool.initialize()
    
    # 3. Create processing pipeline
    pipeline = AsyncProcessingPipeline([
        # Stage 1: Data validation and preprocessing
        lambda x: validate_input(x),
        # Stage 2: Agent processing
        lambda x: agent_pool.run_concurrent([{'prompt': f"Analyze this data: {x}", 'params': {}}], str)[0] if isinstance(x, str) else x,
        # Stage 3: Result post-processing
        lambda x: post_process_result(x)
    ], max_workers=4)
    
    # 4. Test the system
    test_data = [
        "Sales data showing 25% growth in Q4",
        "Customer feedback analysis reveals UX issues",
        "Market research indicates new opportunities"
    ]
    
    print("🚀 Starting high-performance processing...")
    start_time = time.time()
    
    results = await pipeline.process_batch(test_data)
    
    processing_time = time.time() - start_time
    print(f"✅ Processed {len(test_data)} items in {processing_time:.2f}s")
    print(f"⚡ Average time per item: {processing_time/len(test_data):.2f}s")
    
    # Get performance stats
    cache_stats = await memory_cache.get_stats()
    print(f"📊 Cache hit rate: {cache_stats['hit_rate']:.1%}")
    
    return {"agent_pool": agent_pool, "cache_manager": cache_manager, "pipeline": pipeline, "results": results}

def validate_input(data: str) -> str:
    """Validate and clean input data"""
    if not data or len(data.strip()) < 10:
        raise ValueError("Input data too short or empty")
    return data.strip()

def post_process_result(result: Any) -> Dict[str, Any]:
    """Post-process agent results"""
    if isinstance(result, Exception):
        return {"error": str(result), "status": "failed"}
    
    return {"result": result, "status": "success", "processed_at": time.time()}
```

## Load Balancing

### Intelligent Load Distribution

Let's implement sophisticated load balancing strategies:

```python
import random
import statistics
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class LoadBalancingStrategy(Enum):
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_RESPONSE_TIME = "least_response_time"
    RANDOM = "random"

@dataclass
class AgentInstance:
    """Represents an agent instance with performance metrics"""
    id: str
    agent: Any
    weight: int = 1
    current_load: int = 0
    total_requests: int = 0
    total_response_time: float = 0.0
    successful_requests: int = 0
    failed_requests: int = 0
    last_used: float = 0.0
    status: str = "healthy"  # healthy, degraded, unhealthy
    
    @property
    def average_response_time(self) -> float:
        """Calculate average response time"""
        if self.successful_requests == 0:
            return 0.0
        return self.total_response_time / self.successful_requests
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        total_requests = self.successful_requests + self.failed_requests
        if total_requests == 0:
            return 1.0
        return self.successful_requests / total_requests

class IntelligentLoadBalancer:
    """Advanced load balancer with multiple strategies"""
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.LEAST_RESPONSE_TIME):
        self.strategy = strategy
        self.instances: Dict[str, AgentInstance] = {}
        self.request_history: List[Tuple[str, float, bool]] = []
    
    def add_instance(self, instance: AgentInstance):
        """Add agent instance to load balancer"""
        self.instances[instance.id] = instance
    
    def select_instance(self, request_context: Dict[str, Any] = None) -> Optional[AgentInstance]:
        """Select best instance based on current strategy"""
        
        healthy_instances = [instance for instance in self.instances.values() if instance.status == "healthy"]
        
        if not healthy_instances:
            healthy_instances = [instance for instance in self.instances.values() if instance.status != "unhealthy"]
        
        if not healthy_instances:
            return None
        
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return min(healthy_instances, key=lambda x: x.total_requests)
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin_select(healthy_instances)
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return min(healthy_instances, key=lambda x: x.current_load)
        elif self.strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
            return min(healthy_instances, key=lambda x: x.average_response_time)
        elif self.strategy == LoadBalancingStrategy.RANDOM:
            return random.choice(healthy_instances)
        else:
            return healthy_instances[0]
    
    def _weighted_round_robin_select(self, instances: List[AgentInstance]) -> AgentInstance:
        """Weighted round-robin selection strategy"""
        total_weight = sum(instance.weight for instance in instances)
        if total_weight == 0:
            return random.choice(instances)
        
        target = random.randint(1, total_weight)
        current_weight = 0
        
        for instance in instances:
            current_weight += instance.weight
            if current_weight >= target:
                return instance
        
        return instances[-1]
    
    def update_instance_metrics(self, instance_id: str, response_time: float, success: bool, load_increment: int = 1):
        """Update instance performance metrics"""
        if instance_id not in self.instances:
            return
        
        instance = self.instances[instance_id]
        instance.current_load += load_increment
        instance.total_requests += 1
        instance.last_used = time.time()
        
        if success:
            instance.successful_requests += 1
            instance.total_response_time += response_time
        else:
            instance.failed_requests += 1
        
        # Update status based on performance
        self._update_instance_status(instance)
    
    def _update_instance_status(self, instance: AgentInstance):
        """Update instance status based on performance metrics"""
        if instance.total_requests < 10:
            return  # Not enough data
        
        if instance.success_rate < 0.8:  # Less than 80% success rate
            instance.status = "unhealthy"
        elif instance.success_rate < 0.95 or instance.average_response_time > 5.0:
            instance.status = "degraded"
        else:
            instance.status = "healthy"
    
    async def run_with_load_balancing(self, request: Dict[str, Any], timeout: float = 30.0) -> Dict[str, Any]:
        """Execute request using load balancer"""
        
        instance = self.select_instance(request.get('context', {}))
        if not instance:
            return {"error": "No available instances", "status": "failed"}
        
        start_time = time.time()
        instance.current_load += 1
        
        try:
            # Execute request with timeout
            result = await asyncio.wait_for(
                instance.agent.run(request['prompt'], **request.get('params', {})),
                timeout=timeout
            )
            
            response_time = time.time() - start_time
            success = True
            output = result.data if hasattr(result, 'data') else result
            
        except asyncio.TimeoutError:
            response_time = timeout
            success = False
            output = f"Request timed out after {timeout}s"
            
        except Exception as e:
            response_time = time.time() - start_time
            success = False
            output = f"Error: {str(e)}"
        
        finally:
            instance.current_load = max(0, instance.current_load - 1)
            self.update_instance_metrics(instance.id, response_time, success)
        
        return {
            "result": output,
            "instance_id": instance.id,
            "response_time": response_time,
            "success": success,
            "status": "completed" if success else "failed"
        }
    
    def get_load_balancer_stats(self) -> Dict[str, Any]:
        """Get comprehensive load balancer statistics"""
        total_requests = sum(inst.total_requests for inst in self.instances.values())
        total_success = sum(inst.successful_requests for inst in self.instances.values())
        
        return {
            "total_instances": len(self.instances),
            "healthy_instances": len([i for i in self.instances.values() if i.status == "healthy"]),
            "degraded_instances": len([i for i in self.instances.values() if i.status == "degraded"]),
            "unhealthy_instances": len([i for i in self.instances.values() if i.status == "unhealthy"]),
            "total_requests": total_requests,
            "overall_success_rate": total_success / total_requests if total_requests > 0 else 1.0,
            "strategy": self.strategy.value
        }

# Example: Distributed Agent System with Load Balancing
async def create_distributed_agent_system():
    """Create a complete distributed agent system with load balancing"""
    
    # 1. Create multiple agent instances
    agents = []
    for i in range(3):
        agent = Agent('gemini-1.5-flash', result_type=str, system_prompt=f"You are agent #{i+1}. Provide helpful, accurate responses.")
        agents.append(agent)
    
    # 2. Create load balancer with intelligent strategy
    load_balancer = IntelligentLoadBalancer(strategy=LoadBalancingStrategy.LEAST_RESPONSE_TIME)
    
    # 3. Add instances to load balancer
    for i, agent in enumerate(agents):
        instance = AgentInstance(
            id=f"agent_{i+1}",
            agent=agent,
            weight=2 if i == 0 else 1  # First agent gets higher weight
        )
        load_balancer.add_instance(instance)
    
    # 4. Test load balancing with multiple requests
    test_requests = [
        {"prompt": "Explain machine learning", "context": {"session_id": f"session_{i}"}}
        for i in range(10)
    ]
    
    print("🚀 Testing distributed agent system...")
    results = []
    
    for request in test_requests:
        result = await load_balancer.run_with_load_balancing(request, timeout=15.0)
        results.append(result)
        print(f"✅ Request processed by {result['instance_id']} in {result['response_time']:.2f}s")
    
    # 5. Display load balancer statistics
    stats = load_balancer.get_load_balancer_stats()
    print("\n📊 Load Balancer Statistics:")
    print(f"Healthy instances: {stats['healthy_instances']}")
    print(f"Overall success rate: {stats['overall_success_rate']:.1%}")
    print(f"Strategy used: {stats['strategy']}")
    
    return load_balancer, results
```

## Performance Monitoring

### Real-Time Performance Tracking

Let's implement comprehensive monitoring and observability:

```python
import psutil
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
import threading

@dataclass
class PerformanceMetric:
    """Represents a single performance metric"""
    timestamp: float
    name: str
    value: float
    tags: Dict[str, str]

@dataclass
class SystemMetrics:
    """System-level performance metrics"""
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    timestamp: float

class PerformanceMonitor:
    """Comprehensive performance monitoring system"""
    
    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self.metrics: deque = deque(maxlen=max_history)
        self.system_metrics: deque = deque(maxlen=max_history)
        self.alert_callbacks: List[Callable] = []
        self.thresholds: Dict[str, float] = {}
        self._monitoring_active = False
        self._monitor_thread = None
        self._lock = threading.Lock()
    
    def add_metric(self, metric: PerformanceMetric):
        """Add a performance metric"""
        with self._lock:
            self.metrics.append(metric)
            self._check_thresholds(metric)
    
    def add_system_metric(self, metric: SystemMetrics):
        """Add system-level metric"""
        with self._lock:
            self.system_metrics.append(metric)
    
    def set_threshold(self, metric_name: str, threshold: float, comparison: str = "greater_than"):
        """Set alert threshold for a metric"""
        self.thresholds[f"{metric_name}:{comparison}"] = threshold
    
    def add_alert_callback(self, callback: Callable[[str, float, float], None]):
        """Add callback for alert notifications"""
        self.alert_callbacks.append(callback)
    
    def _check_thresholds(self, metric: PerformanceMetric):
        """Check if metric exceeds thresholds"""
        for threshold_key, threshold_value in self.thresholds.items():
            metric_name, comparison = threshold_key.split(":")
            
            if metric.name == metric_name:
                triggered = False
                if comparison == "greater_than" and metric.value > threshold_value:
                    triggered = True
                elif comparison == "less_than" and metric.value < threshold_value:
                    triggered = True
                
                if triggered:
                    for callback in self.alert_callbacks:
                        try:
                            callback(metric.name, metric.value, threshold_value)
                        except Exception as e:
                            print(f"Alert callback failed: {e}")
    
    def get_metric_history(self, metric_name: str, duration_minutes: int = 60) -> List[PerformanceMetric]:
        """Get metric history for specified duration"""
        cutoff_time = time.time() - (duration_minutes * 60)
        
        with self._lock:
            return [metric for metric in self.metrics if metric.name == metric_name and metric.timestamp >= cutoff_time]
    
    def get_current_system_metrics(self) -> SystemMetrics:
        """Get current system performance metrics"""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        
        return SystemMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_mb=memory.used / (1024 * 1024),
            timestamp=time.time()
        )
    
    def start_monitoring(self, interval_seconds: float = 5.0):
        """Start continuous monitoring"""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._monitor_thread = threading.Thread(target=self._monitoring_loop, args=(interval_seconds,), daemon=True)
        self._monitor_thread.start()
        print(f"🔍 Performance monitoring started (interval: {interval_seconds}s)")
    
    def stop_monitoring(self):
        """Stop continuous monitoring"""
        self._monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        print("🛑 Performance monitoring stopped")
    
    def _monitoring_loop(self, interval: float):
        """Main monitoring loop"""
        while self._monitoring_active:
            try:
                system_metric = self.get_current_system_metrics()
                self.add_system_metric(system_metric)
                time.sleep(interval)
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(interval)
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        
        recent_system_metrics = list(self.system_metrics)[-60:]  # Last 60 measurements
        
        if recent_system_metrics:
            avg_cpu = sum(m.cpu_percent for m in recent_system_metrics) / len(recent_system_metrics)
            avg_memory = sum(m.memory_percent for m in recent_system_metrics) / len(recent_system_metrics)
            max_cpu = max(m.cpu_percent for m in recent_system_metrics)
            max_memory = max(m.memory_percent for m in recent_system_metrics)
        else:
            avg_cpu = avg_memory = max_cpu = max_memory = 0
        
        # Group metrics by name
        metrics_by_name = defaultdict(list)
        with self._lock:
            for metric in self.metrics:
                metrics_by_name[metric.name].append(metric)
        
        # Calculate metric statistics
        metric_stats = {}
        for name, metric_list in metrics_by_name.items():
            if metric_list:
                values = [m.value for m in metric_list]
                metric_stats[name] = {
                    "count": len(values),
                    "average": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "latest": values[-1]
                }
        
        return {
            "timestamp": time.time(),
            "system": {
                "average_cpu_percent": avg_cpu,
                "maximum_cpu_percent": max_cpu,
                "average_memory_percent": avg_memory,
                "maximum_memory_percent": max_memory,
                "total_measurements": len(recent_system_metrics)
            },
            "application_metrics": metric_stats,
            "total_metrics_collected": len(self.metrics)
        }

# Agent Performance Tracking
class AgentPerformanceTracker:
    """Specialized performance tracker for AI agents"""
    
    def __init__(self, monitor: PerformanceMonitor, agent_id: str):
        self.monitor = monitor
        self.agent_id = agent_id
        self.request_times: deque = deque(maxlen=100)
        self.success_times: deque = deque(maxlen=100)
        self.error_times: deque = deque(maxlen=100)
    
    def track_request_start(self) -> float:
        """Mark the start of a request and return start time"""
        start_time = time.time()
        self.monitor.add_metric(PerformanceMetric(
            timestamp=start_time,
            name=f"agent_{self.agent_id}_request_started",
            value=1.0,
            tags={"agent_id": self.agent_id},
            metadata={}
        ))
        return start_time
    
    def track_request_end(self, start_time: float, success: bool, tokens_used: int = 0, error_message: str = None):
        """Mark the end of a request and record metrics"""
        end_time = time.time()
        response_time = end_time - start_time
        
        # Record response time
        self.monitor.add_metric(PerformanceMetric(
            timestamp=end_time,
            name=f"agent_{self.agent_id}_response_time",
            value=response_time,
            tags={"agent_id": self.agent_id, "success": str(success)},
            metadata={"tokens_used": tokens_used, "error": error_message}
        ))
        
        # Track request in deques
        self.request_times.append(response_time)
        if success:
            self.success_times.append(response_time)
        else:
            self.error_times.append(response_time)
        
        # Calculate and record success rate
        total_requests = len(self.request_times)
        successful_requests = len(self.success_times)
        success_rate = successful_requests / total_requests if total_requests > 0 else 0
        
        self.monitor.add_metric(PerformanceMetric(
            timestamp=end_time,
            name=f"agent_{self.agent_id}_success_rate",
            value=success_rate,
            tags={"agent_id": self.agent_id},
            metadata={"total_requests": total_requests}
        ))
    
    def get_agent_stats(self, duration_minutes: int = 60) -> Dict[str, Any]:
        """Get comprehensive agent performance statistics"""
        
        response_times = self.get_metric_history(f"agent_{self.agent_id}_response_time", duration_minutes)
        success_rates = self.get_metric_history(f"agent_{self.agent_id}_success_rate", duration_minutes)
        
        if not response_times:
            return {"status": "no_data"}
        
        times = [m.value for m in response_times]
        success_rate_values = [m.value for m in success_rates] if success_rates else []
        
        return {
            "agent_id": self.agent_id,
            "period_minutes": duration_minutes,
            "total_requests": len(times),
            "response_time": {
                "average": sum(times) / len(times),
                "min": min(times),
                "max": max(times),
                "p95": sorted(times)[int(len(times) * 0.95)] if times else 0
            },
            "success_rate": {
                "average": sum(success_rate_values) / len(success_rate_values) if success_rate_values else 0,
                "latest": success_rate_values[-1] if success_rate_values else 0
            }
        }
    
    def get_metric_history(self, metric_name: str, duration_minutes: int):
        """Get metric history from the main monitor"""
        return self.monitor.get_metric_history(metric_name, duration_minutes)

# Complete Monitoring System Example
async def create_monitoring_system():
    """Create a complete performance monitoring system"""
    
    # 1. Initialize performance monitor
    monitor = PerformanceMonitor(max_history=5000)
    
    # 2. Set up alert thresholds
    monitor.set_threshold("response_time", 5.0, "greater_than")
    monitor.set_threshold("cpu_percent", 80.0, "greater_than")
    monitor.set_threshold("memory_percent", 85.0, "greater_than")
    
    # 3. Add alert callback
    def alert_handler(metric_name: str, current_value: float, threshold: float):
        print(f"🚨 ALERT: {metric_name} = {current_value:.2f} (threshold: {threshold})")
    
    monitor.add_alert_callback(alert_handler)
    
    # 4. Create agent performance trackers
    agents = {}
    for i in range(3):
        agent_id = f"agent_{i+1}"
        agents[agent_id] = AgentPerformanceTracker(monitor, agent_id)
    
    # 5. Start monitoring
    monitor.start_monitoring(interval_seconds=2.0)
    
    # 6. Simulate agent workload
    print("🔄 Simulating agent workload with monitoring...")
    
    agent_instance = Agent('gemini-1.5-flash', result_type=str, system_prompt="You are helpful.")
    
    for i in range(20):
        agent_id = f"agent_{(i % 3) + 1}"
        tracker = agents[agent_id]
        
        # Track request
        start_time = tracker.track_request_start()
        
        try:
            # Simulate some requests failing
            if i % 7 == 0:  # ~14% failure rate
                raise Exception("Simulated failure")
            
            result = await agent_instance.run(f"Process request #{i}")
            tokens_used = len(result.data.split()) if result.data else 0
            tracker.track_request_end(start_time, success=True, tokens_used=tokens_used)
            
        except Exception as e:
            tracker.track_request_end(start_time, success=False, error_message=str(e))
        
        # Small delay between requests
        await asyncio.sleep(0.5)
    
    # 7. Generate performance report
    await asyncio.sleep(2)  # Let monitoring collect some data
    report = monitor.generate_performance_report()
    
    print("\n📊 Performance Report:")
    print(f"Average CPU: {report['system']['average_cpu_percent']:.1f}%")
    print(f"Average Memory: {report['system']['average_memory_percent']:.1f}%")
    print(f"Total metrics: {report['total_metrics_collected']}")
    
    # 8. Show per-agent stats
    print("\nPer-Agent Performance:")
    for agent_id, tracker in agents.items():
        stats = tracker.get_agent_stats(duration_minutes=10)
        if stats.get("status") != "no_data":
            print(f"{agent_id}:")
            print(f"  Requests: {stats['total_requests']}")
            print(f"  Avg Response Time: {stats['response_time']['average']:.2f}s")
            print(f"  Success Rate: {stats['success_rate']['average']:.1%}")
    
    # 9. Stop monitoring
    monitor.stop_monitoring()
    
    return monitor, agents
```

## Real-World Implementation

Let's put it all together with a complete production-ready system:

```python
class ProductionAgentSystem:
    """Complete production-ready agent system with all optimizations"""
    
    def __init__(self, agent_configs: List[Dict[str, Any]]):
        self.agent_configs = agent_configs
        self.memory_cache = InMemoryCache(max_size=2000, default_ttl=3600)
        self.cache_manager = IntelligentCacheManager(self.memory_cache)
        self.monitor = PerformanceMonitor(max_history=10000)
        self.load_balancer = IntelligentLoadBalancer()
        self.agent_trackers: Dict[str, AgentPerformanceTracker] = {}
        self.agents: List[Agent] = []
        self.agent_pool = None
        
        self._setup_monitoring()
    
    def _setup_monitoring(self):
        """Configure comprehensive monitoring"""
        # Set performance thresholds
        self.monitor.set_threshold("response_time", 10.0, "greater_than")
        self.monitor.set_threshold("cpu_percent", 75.0, "greater_than")
        self.monitor.set_threshold("memory_percent", 80.0, "greater_than")
        
        # Add notification callbacks
        self.monitor.add_alert_callback(self._handle_alert)
        
        # Start monitoring
        self.monitor.start_monitoring(interval_seconds=10.0)
    
    def _handle_alert(self, metric_name: str, current_value: float, threshold: float):
        """Handle alert notifications"""
        print(f"🚨 PERFORMANCE ALERT: {metric_name} = {current_value:.2f} (threshold: {threshold})")
    
    async def initialize(self):
        """Initialize the complete system"""
        print("🚀 Initializing Production Agent System...")
        
        # 1. Create agents
        for config in self.agent_configs:
            agent = Agent(
                model=config.get('model', 'gemini-1.5-flash'),
                result_type=config.get('result_type', str),
                system_prompt=config.get('system_prompt', ''),
                cache=config.get('cache', True),
                retries=config.get('retries', 2)
            )
            self.agents.append(agent)
        
        # 2. Create agent pool for load balancing
        def create_agent_wrapper(i):
            def wrapper():
                return self.agents[i]
            return wrapper
        
        self.agent_pool = AsyncAgentPool(
            [create_agent_wrapper(i) for i in range(len(self.agents))],
            max_concurrent=config.get('max_concurrent', 10),
            timeout=config.get('timeout', 30.0)
        )
        await self.agent_pool.initialize()
        
        # 3. Add instances to load balancer
        for i, agent in enumerate(self.agents):
            instance = AgentInstance(
                id=f"agent_{i+1}",
                agent=agent,
                weight=config.get('weight', 1)
            )
            self.load_balancer.add_instance(instance)
            
            # Create performance tracker
            self.agent_trackers[instance.id] = AgentPerformanceTracker(self.monitor, instance.id)
        
        print(f"✅ System initialized with {len(self.agents)} agents")
    
    async def process_request(self, prompt: str, params: Dict[str, Any] = None, session_id: str = None, priority: str = "normal") -> Dict[str, Any]:
        """Process a request with all optimizations"""
        
        start_time = time.time()
        request_context = {"session_id": session_id, "priority": priority}
        
        # Select agent using load balancer
        instance = self.load_balancer.select_instance(request_context)
        if not instance:
            return {"error": "No available agents", "status": "failed"}
        
        # Get performance tracker for this agent
        tracker = self.agent_trackers[instance.id]
        
        # Track request start
        tracking_start = tracker.track_request_start()
        
        try:
            # Check cache first
            cache_key = f"request:{hash(prompt)}"
            cached_result = await self.cache_manager.get(cache_key, {"params": params, "session_id": session_id})
            
            if cached_result is not None:
                # Cache hit - return cached result
                processing_time = time.time() - start_time
                tracker.track_request_end(tracking_start, success=True)
                
                return {
                    "result": cached_result,
                    "source": "cache",
                    "processing_time": processing_time,
                    "agent_id": instance.id,
                    "status": "completed"
                }
            
            # Cache miss - process request
            print(f"🔄 Processing request with {instance.id} (priority: {priority})")
            
            # Execute with load balancer
            result = await self.load_balancer.run_with_load_balancing({
                "prompt": prompt,
                "params": params or {},
                "context": request_context
            })
            
            processing_time = time.time() - start_time
            
            if result["success"]:
                # Store in cache for future requests
                await self.cache_manager.set(cache_key, result["result"], ttl=1800, params={"params": params, "session_id": session_id})
                
                # Track successful completion
                tracker.track_request_end(tracking_start, success=True)
                
                return {
                    "result": result["result"],
                    "source": "live",
                    "processing_time": processing_time,
                    "agent_id": instance.id,
                    "status": "completed"
                }
            else:
                # Request failed
                tracker.track_request_end(tracking_start, success=False, error_message=result.get("error"))
                return {
                    "error": result.get("error", "Unknown error"),
                    "processing_time": processing_time,
                    "agent_id": instance.id,
                    "status": "failed"
                }
        
        except Exception as e:
            processing_time = time.time() - start_time
            tracker.track_request_end(tracking_start, success=False, error_message=str(e))
            
            return {
                "error": str(e),
                "processing_time": processing_time,
                "agent_id": instance.id,
                "status": "failed"
            }
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        # Get system metrics
        report = self.monitor.generate_performance_report()
        
        # Get load balancer stats
        lb_stats = self.load_balancer.get_load_balancer_stats()
        
        # Get cache stats
        cache_stats = await self.memory_cache.get_stats()
        
        # Get per-agent performance
        agent_stats = {}
        for agent_id, tracker in self.agent_trackers.items():
            agent_stats[agent_id] = tracker.get_agent_stats(duration_minutes=60)
        
        return {
            "timestamp": time.time(),
            "system": report,
            "load_balancer": lb_stats,
            "cache": cache_stats,
            "agents": agent_stats,
            "health": {
                "total_agents": len(self.agents),
                "healthy_agents": lb_stats["healthy_instances"],
                "degraded_agents": lb_stats["degraded_instances"],
                "unhealthy_agents": lb_stats["unhealthy_instances"]
            }
        }
    
    async def shutdown(self):
        """Gracefully shutdown the system"""
        print("🛑 Shutting down Production Agent System...")
        self.monitor.stop_monitoring()
        print("✅ System shutdown complete")

# Complete Example Usage
async def demonstrate_production_system():
    """Demonstrate the complete production agent system"""
    
    # Define agent configurations
    agent_configs = [
        {
            "model": "gemini-1.5-flash",
            "result_type": str,
            "system_prompt": "You are a helpful data analysis expert.",
            "weight": 2,
            "retries": 2
        },
        {
            "model": "gemini-1.5-flash", 
            "result_type": str,
            "system_prompt": "You are a creative content writer.",
            "weight": 1,
            "retries": 2
        },
        {
            "model": "gemini-1.5-flash",
            "result_type": str, 
            "system_prompt": "You are a technical documentation specialist.",
            "weight": 1,
            "retries": 3
        }
    ]
    
    # Create and initialize system
    system = ProductionAgentSystem(agent_configs=agent_configs)
    await system.initialize()
    
    # Test with various requests
    test_requests = [
        {"prompt": "Analyze this sales data: 25% growth in Q4", "priority": "high"},
        {"prompt": "Write a creative story about AI", "priority": "normal"},
        {"prompt": "Explain machine learning algorithms", "priority": "normal"},
        {"prompt": "Create API documentation", "priority": "low"},
        {"prompt": "What is the capital of France?", "priority": "normal"}
    ]
    
    print("\n🚀 Processing test requests...")
    
    for i, request in enumerate(test_requests):
        session_id = f"session_{i // 2}"  # Group some requests
        result = await system.process_request(
            prompt=request["prompt"],
            priority=request["priority"],
            session_id=session_id
        )
        
        print(f"Request {i+1}: {result['status']} "
              f"(source: {result['source']}, "
              f"time: {result['processing_time']:.2f}s, "
              f"agent: {result['agent_id']})")
    
    # Get system status
    print("\n📊 System Status Report:")
    status = await system.get_system_status()
    
    print(f"Healthy agents: {status['health']['healthy_agents']}/{status['health']['total_agents']}")
    print(f"Cache hit rate: {status['cache']['hit_rate']:.1%}")
    print(f"Overall success rate: {status['load_balancer']['overall_success_rate']:.1%}")
    
    # Wait a bit for monitoring data
    await asyncio.sleep(5)
    
    # Shutdown
    await system.shutdown()

# Run the demonstration
if __name__ == "__main__":
    asyncio.run(demonstrate_production_system())
```

## Advanced Optimization Techniques

### Circuit Breaker and Memory Management

```python
import gc
import weakref
from typing import Set, Dict, Any
import threading
from contextlib import asynccontextmanager

class CircuitBreaker:
    """Circuit breaker for fault tolerance"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0, expected_exception: type = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half-open
    
    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        
        if self.state == "open":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half-open"
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            if self.state == "half-open":
                self.state = "closed"
                self.failure_count = 0
            return result
            
        except self.expected_exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
            
            raise e

class ResourceManager:
    """Advanced resource management for agent systems"""
    
    def __init__(self, max_memory_mb: int = 1024):
        self.max_memory_mb = max_memory_mb
        self._memory_check_interval = 60  # seconds
        self._monitoring = False
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "rss_mb": memory_info.rss / (1024 * 1024),  # Resident Set Size
            "percent": process.memory_percent()
        }
    
    def check_memory_pressure(self) -> bool:
        """Check if system is under memory pressure"""
        memory_usage = self.get_memory_usage()
        return memory_usage["rss_mb"] > self.max_memory_mb * 0.8
    
    async def optimize_memory(self):
        """Perform memory optimization"""
        print("🧹 Performing memory optimization...")
        
        # Force garbage collection
        collected = gc.collect()
        print(f"   Collected {collected} objects")
        
        # Log memory usage after optimization
        usage = self.get_memory_usage()
        print(f"   Memory usage: {usage['rss_mb']:.1f}MB ({usage['percent']:.1f}%)")
        
        return usage

# Performance Profiling
class PerformanceProfiler:
    """Advanced performance profiling for agent operations"""
    
    def __init__(self):
        self.profiles: Dict[str, List[float]] = defaultdict(list)
        self.active_profiles: Dict[str, float] = {}
    
    def start_profile(self, operation_name: str):
        """Start profiling an operation"""
        self.active_profiles[operation_name] = time.time()
    
    def end_profile(self, operation_name: str):
        """End profiling and record duration"""
        if operation_name in self.active_profiles:
            duration = time.time() - self.active_profiles[operation_name]
            self.profiles[operation_name].append(duration)
            del self.active_profiles[operation_name]
            return duration
        return 0
    
    def get_profile_stats(self, operation_name: str) -> Dict[str, float]:
        """Get profiling statistics for an operation"""
        if operation_name not in self.profiles or not self.profiles[operation_name]:
            return {"count": 0}
        
        durations = self.profiles[operation_name]
        durations.sort()
        
        return {
            "count": len(durations),
            "average": sum(durations) / len(durations),
            "min": min(durations),
            "max": max(durations),
            "p95": durations[int(len(durations) * 0.95)]
        }

@asynccontextmanager
async def profile_operation(profiler: PerformanceProfiler, operation_name: str):
    """Context manager for profiling operations"""
    profiler.start_profile(operation_name)
    try:
        yield
    finally:
        duration = profiler.end_profile(operation_name)
        print(f"⏱️  {operation_name}: {duration:.3f}s")
```

## Key Performance Indicators (KPIs)

### Measuring Success

Here are the critical metrics to track in your optimized agent system:

1. **Response Time**: Target < 2 seconds for 95% of requests
2. **Throughput**: Requests per second the system can handle
3. **Cache Hit Rate**: Target > 80% for frequently accessed content
4. **Error Rate**: Target < 1% for production systems
5. **Resource Utilization**: CPU < 70%, Memory < 80%
6. **Availability**: Target 99.9% uptime

## Best Practices Summary

### 1. Caching Strategy
- Use multi-level caching (memory + distributed)
- Implement TTL based on data volatility
- Monitor cache hit rates and adjust accordingly
- Use consistent hashing for cache keys

### 2. Async Optimization
- Use async/await for I/O-bound operations
- Implement connection pooling
- Set appropriate timeouts
- Use semaphore for concurrency control

### 3. Load Balancing
- Choose strategy based on workload patterns
- Monitor instance health continuously
- Implement circuit breakers for fault tolerance
- Use weighted distribution for heterogeneous instances

### 4. Monitoring & Alerting
- Track both system and application metrics
- Set up automated alerting with escalation
- Implement performance profiling
- Monitor resource utilization trends

### 5. Production Readiness
- Implement comprehensive error handling
- Use graceful degradation strategies
- Plan for horizontal scaling
- Implement security best practices

## Next Steps

Congratulations! You've completed Lesson 9 and learned how to build high-performance, scalable agent systems. You now understand:

✅ **Caching strategies** - From in-memory to distributed caching
✅ **Async optimization** - Concurrent processing and pipeline patterns  
✅ **Load balancing** - Intelligent distribution algorithms
✅ **Performance monitoring** - Real-time observability and alerting
✅ **Production patterns** - Circuit breakers, resource management, and profiling

In the next lesson, we'll cover **Advanced Agent Patterns** including multi-agent coordination, conversation management, and sophisticated reasoning techniques.

**Ready for the final stretch? Let's build something amazing!** 🚀

---

*Performance optimization is an ongoing process. Start with these fundamentals, measure everything, and continuously iterate based on real-world usage patterns.*