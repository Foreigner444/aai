# Lesson 12: Streaming Responses

## A. Concept Overview

### What & Why
**Streaming Responses** allow your agent to return results incrementally as they're generated, rather than waiting for the entire response to complete. Users see text appearing in real-time, tool calls as they happen, and can cancel long-running operations. This is crucial for user experience - nobody wants to stare at a blank screen for 30 seconds waiting for a response. Streaming makes agents feel responsive, interactive, and modern.

### Analogy
Think of streaming responses like watching a live sports broadcast vs waiting for a recorded replay:

**Non-Streaming (Blocking)** = Recorded Replay:
- Game happens (you wait)
- Commentators analyze (you wait more)
- Highlights are compiled (still waiting)
- Finally, 3 hours later, you get to watch
- If you get bored, tough luck - can't stop it once started

**Streaming** = Live Broadcast:
- Play happens → you see it immediately
- Commentator speaks → you hear it in real-time
- Something boring? Switch channels or pause
- Exciting moment? You're experiencing it NOW
- Interactive, responsive, engaging

Streaming transforms static responses into dynamic experiences!

### Type Safety Benefit
Streaming with type safety provides:
- **Typed stream chunks**: Each chunk validated against expected types
- **Async type checking**: Proper async/await patterns with mypy
- **Error handling**: Typed exceptions during streaming
- **Cancellation safety**: Type-safe stream termination
- **Backpressure handling**: Type-checked flow control
- **Testing**: Mock streams with typed chunks

Your streaming pipeline becomes fully type-checked and reliable!

---

## B. Code Implementation

### File Structure
```
agent_system_design/
├── lesson_12_streaming_responses.py
├── requirements.txt
└── .env
```

### Complete Code Snippet

**lesson_12_streaming_responses.py**
```python
"""
Lesson 12: Streaming Responses
Learn to stream agent responses in real-time
"""

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from dataclasses import dataclass
from typing import AsyncIterator, Optional
from datetime import datetime
import asyncio
from dotenv import load_dotenv

load_dotenv()


# Dependencies
@dataclass
class StreamingDeps:
    """Dependencies for streaming examples"""
    user_id: str
    session_id: str


# Result models
class StreamingAnalysis(BaseModel):
    """Analysis that will be streamed"""
    summary: str
    findings: list[str]
    recommendations: list[str]
    data_sources: list[str]


# EXAMPLE 1: Basic Streaming Agent

basic_agent = Agent(
    model='gemini-1.5-flash',
    result_type=str,  # Simple string response
    system_prompt="""
You are a helpful assistant that provides detailed explanations.

Provide comprehensive answers with:
- Clear introduction
- Detailed explanation with examples
- Practical applications
- Summary conclusion

Write naturally and conversationally.
""",
)


async def basic_streaming_example():
    """
    Demonstrate basic streaming with simple text output
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Text Streaming")
    print("="*70)
    print("\nQuery: Explain async programming in Python\n")
    print("Response (streaming):")
    print("-" * 70)
    
    query = "Explain async programming in Python with examples"
    
    # Use run_stream() instead of run_sync()
    async with basic_agent.run_stream(query) as stream:
        # Iterate over chunks as they arrive
        full_response = []
        async for chunk in stream.stream_text():
            # Print each chunk as it arrives
            print(chunk, end='', flush=True)
            full_response.append(chunk)
        
        # Get final validated result
        result = await stream.get_data()
    
    print("\n" + "-" * 70)
    print(f"✅ Streaming complete. Total length: {len(''.join(full_response))} chars")


# EXAMPLE 2: Streaming with Tools

tools_agent = Agent(
    model='gemini-1.5-flash',
    result_type=StreamingAnalysis,
    deps_type=StreamingDeps,
    system_prompt="""
You are a data analyst with access to tools.

Use tools to gather data, then provide comprehensive analysis.

Available tools:
- fetch_sales_data: Get sales information
- fetch_customer_data: Get customer metrics
- calculate_growth: Calculate growth rates
""",
)


@tools_agent.tool
async def fetch_sales_data(
    ctx: RunContext[StreamingDeps],
    year: int
) -> dict[str, any]:
    """
    Fetch sales data for a year (simulated slow operation).
    
    Args:
        year: Year to fetch data for
    
    Returns:
        Sales data dictionary
    """
    print(f"\n🔧 [Tool] fetch_sales_data(year={year}) - Starting...")
    
    # Simulate slow database query
    await asyncio.sleep(1.5)
    
    data = {
        "year": year,
        "total_sales": 15_750_000,
        "quarters": [3_500_000, 4_200_000, 3_900_000, 4_150_000]
    }
    
    print(f"✅ [Tool] fetch_sales_data complete: ${data['total_sales']:,}")
    return data


@tools_agent.tool
async def fetch_customer_data(
    ctx: RunContext[StreamingDeps]
) -> dict[str, any]:
    """
    Fetch customer metrics (simulated slow operation).
    
    Returns:
        Customer metrics dictionary
    """
    print(f"\n🔧 [Tool] fetch_customer_data() - Starting...")
    
    # Simulate slow API call
    await asyncio.sleep(1.0)
    
    data = {
        "total_customers": 1250,
        "new_customers": 180,
        "retention_rate": 0.89
    }
    
    print(f"✅ [Tool] fetch_customer_data complete: {data['total_customers']} customers")
    return data


@tools_agent.tool
def calculate_growth(
    previous: float,
    current: float
) -> dict[str, float]:
    """
    Calculate growth rate between two values.
    
    Args:
        previous: Previous period value
        current: Current period value
    
    Returns:
        Growth rate and percentage
    """
    print(f"\n🔧 [Tool] calculate_growth({previous:,.0f} → {current:,.0f})")
    
    if previous == 0:
        return {"growth_rate": 0, "growth_percent": 0}
    
    growth = current - previous
    growth_percent = (growth / previous) * 100
    
    result = {
        "growth_amount": growth,
        "growth_percent": round(growth_percent, 2)
    }
    
    print(f"✅ [Tool] calculate_growth complete: {result['growth_percent']}% growth")
    return result


async def tool_streaming_example():
    """
    Demonstrate streaming with tool calls visible in real-time
    """
    print("\n\n" + "="*70)
    print("EXAMPLE 2: Streaming with Tool Calls")
    print("="*70)
    print("\nQuery: Analyze our 2024 sales and customer growth\n")
    
    deps = StreamingDeps(user_id="user_123", session_id="sess_456")
    query = "Analyze our 2024 sales and customer growth"
    
    print("🔄 Streaming agent execution (watch tools being called):")
    print("-" * 70)
    
    async with tools_agent.run_stream(query, deps=deps) as stream:
        # Stream will show tool calls as they happen
        async for chunk in stream.stream():
            # Chunks can be text or structured data
            if hasattr(chunk, 'content'):
                # Text content chunk
                print(f"💬 [Text]: {chunk.content}", flush=True)
            elif hasattr(chunk, 'tool_name'):
                # Tool call chunk
                print(f"⚙️  [Tool Call]: {chunk.tool_name}", flush=True)
        
        # Get final validated result
        result = await stream.get_data()
    
    print("\n" + "-" * 70)
    print("\n📊 FINAL ANALYSIS:")
    print(f"   Summary: {result.summary}")
    print(f"   Findings: {len(result.findings)} key findings")
    print(f"   Recommendations: {len(result.recommendations)} recommendations")
    print(f"   Data Sources: {', '.join(result.data_sources)}")


# EXAMPLE 3: Streaming with Progress Updates

progress_agent = Agent(
    model='gemini-1.5-flash',
    result_type=str,
    system_prompt="""
You are a research assistant that provides detailed, well-researched answers.

Provide your answer in clear sections:
1. Introduction and overview
2. Key concepts and definitions
3. Detailed analysis
4. Examples and applications
5. Conclusion and summary
""",
)


async def streaming_with_progress():
    """
    Demonstrate streaming with progress tracking
    """
    print("\n\n" + "="*70)
    print("EXAMPLE 3: Streaming with Progress Tracking")
    print("="*70)
    print("\nQuery: Explain machine learning model training\n")
    
    query = "Explain the process of training a machine learning model in detail"
    
    print("🔄 Streaming response with progress:")
    print("-" * 70)
    
    chunks_received = 0
    total_chars = 0
    start_time = datetime.now()
    
    async with progress_agent.run_stream(query) as stream:
        async for chunk in stream.stream_text():
            chunks_received += 1
            total_chars += len(chunk)
            
            # Print the text
            print(chunk, end='', flush=True)
            
            # Show progress every 10 chunks
            if chunks_received % 10 == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                chars_per_sec = total_chars / elapsed if elapsed > 0 else 0
                print(f"\n[Progress: {chunks_received} chunks, {total_chars} chars, {chars_per_sec:.1f} chars/sec]", flush=True)
        
        result = await stream.get_data()
    
    elapsed = (datetime.now() - start_time).total_seconds()
    
    print("\n" + "-" * 70)
    print(f"\n✅ Streaming complete!")
    print(f"   Total chunks: {chunks_received}")
    print(f"   Total characters: {total_chars}")
    print(f"   Elapsed time: {elapsed:.2f} seconds")
    print(f"   Average speed: {total_chars/elapsed:.1f} chars/sec")


# EXAMPLE 4: Cancellable Streaming

async def cancellable_streaming():
    """
    Demonstrate cancelling a stream mid-execution
    """
    print("\n\n" + "="*70)
    print("EXAMPLE 4: Cancellable Streaming")
    print("="*70)
    print("\nQuery: Write a very long essay about Python\n")
    print("(Will cancel after 3 seconds)\n")
    
    query = "Write a very long, detailed essay about Python programming with many examples"
    
    print("🔄 Streaming response (will cancel after 3 seconds):")
    print("-" * 70)
    
    try:
        async with basic_agent.run_stream(query) as stream:
            start_time = datetime.now()
            chars_received = 0
            
            async for chunk in stream.stream_text():
                print(chunk, end='', flush=True)
                chars_received += len(chunk)
                
                # Cancel after 3 seconds
                elapsed = (datetime.now() - start_time).total_seconds()
                if elapsed > 3:
                    print("\n\n⏹️  [Cancelling stream after 3 seconds...]")
                    break
            
            print("\n" + "-" * 70)
            print(f"✅ Stream cancelled successfully after receiving {chars_received} characters")
    
    except asyncio.CancelledError:
        print("\n❌ Stream was cancelled")


# EXAMPLE 5: Server-Sent Events (SSE) Pattern for Web Apps

async def sse_streaming_example():
    """
    Demonstrate SSE pattern for web applications
    
    This shows how to format streaming responses for web frontends
    """
    print("\n\n" + "="*70)
    print("EXAMPLE 5: Server-Sent Events (SSE) Pattern")
    print("="*70)
    print("\nThis demonstrates the SSE format used in web applications\n")
    
    query = "Explain REST APIs"
    
    print("SSE Stream Output:")
    print("-" * 70)
    
    async with basic_agent.run_stream(query) as stream:
        event_id = 0
        
        async for chunk in stream.stream_text():
            event_id += 1
            
            # Format as SSE event
            sse_message = f"id: {event_id}\nevent: message\ndata: {chunk}\n\n"
            print(sse_message, end='', flush=True)
            
            # Small delay to simulate network
            await asyncio.sleep(0.05)
        
        # Send completion event
        print(f"id: {event_id + 1}\nevent: complete\ndata: {{\"status\": \"done\"}}\n\n")
    
    print("-" * 70)
    print("✅ SSE stream complete (this format works with frontend EventSource API)")


# EXAMPLE 6: Error Handling in Streams

error_agent = Agent(
    model='gemini-1.5-flash',
    result_type=str,
    system_prompt="You are a helpful assistant.",
)


@error_agent.tool
async def flaky_tool(should_fail: bool = False) -> dict:
    """
    Tool that can fail (for demonstrating error handling)
    
    Args:
        should_fail: Whether to simulate a failure
    """
    print(f"\n🔧 [Tool] flaky_tool(should_fail={should_fail})")
    
    await asyncio.sleep(0.5)
    
    if should_fail:
        print("❌ [Tool] flaky_tool failed!")
        raise ValueError("Simulated tool failure")
    
    print("✅ [Tool] flaky_tool succeeded")
    return {"status": "success"}


async def error_handling_example():
    """
    Demonstrate error handling in streaming
    """
    print("\n\n" + "="*70)
    print("EXAMPLE 6: Error Handling in Streams")
    print("="*70)
    print("\nDemonstrating graceful error handling during streaming\n")
    
    query = "Test the flaky tool"
    
    print("🔄 Streaming with potential errors:")
    print("-" * 70)
    
    try:
        async with error_agent.run_stream(query) as stream:
            async for chunk in stream.stream_text():
                print(chunk, end='', flush=True)
            
            result = await stream.get_data()
            print("\n✅ Stream completed successfully")
    
    except ValueError as e:
        print(f"\n⚠️  Caught error during streaming: {e}")
        print("✅ Error handled gracefully - stream terminated cleanly")
    
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")


# EXAMPLE 7: Buffered Streaming

async def buffered_streaming():
    """
    Demonstrate buffered streaming (accumulate chunks before sending)
    """
    print("\n\n" + "="*70)
    print("EXAMPLE 7: Buffered Streaming")
    print("="*70)
    print("\nBuffering chunks to reduce network overhead\n")
    
    query = "Explain database indexing"
    
    print("🔄 Buffered streaming (chunks combined):")
    print("-" * 70)
    
    buffer = []
    buffer_size = 50  # Send every 50 characters
    
    async with basic_agent.run_stream(query) as stream:
        async for chunk in stream.stream_text():
            buffer.append(chunk)
            
            # Send buffer when it reaches threshold
            if len(''.join(buffer)) >= buffer_size:
                buffered_text = ''.join(buffer)
                print(f"[Buffer {len(buffered_text)} chars] {buffered_text}", flush=True)
                buffer = []
        
        # Send remaining buffer
        if buffer:
            buffered_text = ''.join(buffer)
            print(f"[Buffer {len(buffered_text)} chars] {buffered_text}", flush=True)
    
    print("\n" + "-" * 70)
    print("✅ Buffered streaming complete (reduces number of updates)")


# Main demonstration runner

async def main():
    """
    Run all streaming examples
    """
    print("\n" + "="*70)
    print("STREAMING RESPONSES - COMPREHENSIVE DEMONSTRATION")
    print("="*70)
    print("\nThis lesson demonstrates 7 streaming patterns:\n")
    print("1. Basic text streaming")
    print("2. Streaming with tool calls")
    print("3. Progress tracking during streaming")
    print("4. Cancellable streams")
    print("5. Server-Sent Events (SSE) format")
    print("6. Error handling in streams")
    print("7. Buffered streaming")
    print("\nNote: Streaming provides real-time feedback and better UX!")
    
    # Run all examples
    await basic_streaming_example()
    await tool_streaming_example()
    await streaming_with_progress()
    await cancellable_streaming()
    await sse_streaming_example()
    await error_handling_example()
    await buffered_streaming()
    
    print("\n\n" + "="*70)
    print("ALL STREAMING EXAMPLES COMPLETE")
    print("="*70)
    print("\n🎯 Key Takeaways:")
    print("   ✅ Streaming provides real-time feedback")
    print("   ✅ Users can see progress and cancel if needed")
    print("   ✅ Tool calls are visible as they happen")
    print("   ✅ Better UX for long-running operations")
    print("   ✅ Works with SSE for web applications")
    print("   ✅ Error handling is critical in streams")
    print("   ✅ Buffering can optimize network usage")
    print("\n💡 Always use streaming for user-facing agents!")


if __name__ == "__main__":
    asyncio.run(main())
```

### Line-by-Line Explanation

**Basic Streaming (Lines 39-75)**:
- Use `run_stream()` instead of `run_sync()`
- `async with` context manager for stream
- `stream.stream_text()` yields text chunks
- Print chunks as they arrive
- `await stream.get_data()` gets final validated result

**Streaming with Tools (Lines 78-196)**:
- Tools are async functions
- Tools execute during streaming
- Users see tool calls in real-time
- Tool results flow into response
- Final result is validated Pydantic model

**Progress Tracking (Lines 199-250)**:
- Track chunks, characters, timing
- Show progress indicators
- Calculate streaming speed
- Useful for long-running operations

**Cancellable Streams (Lines 253-291)**:
- Can break out of stream loop
- Stream terminates gracefully
- Useful for user-initiated cancellations
- Prevents wasted compute

**SSE Pattern (Lines 294-333)**:
- Format for web applications
- `id: event: data:` format
- Works with browser EventSource API
- Real-time updates to frontend

**Error Handling (Lines 336-398)**:
- Try/except around streaming
- Graceful failure handling
- Stream terminates cleanly on errors
- Important for production reliability

**Buffered Streaming (Lines 401-443)**:
- Accumulate chunks before sending
- Reduces update frequency
- Optimizes network overhead
- Trade-off: slightly higher latency

### The "Why" Behind the Pattern

**Why use streaming instead of blocking?**

❌ **Non-Streaming** (Poor UX):
```python
# User waits 30 seconds staring at blank screen
result = agent.run_sync(long_query)  # Blocking!
print(result.data)  # Finally appears
# User: "Is this thing broken??"
```

✅ **Streaming** (Great UX):
```python
# User sees response appear word by word
async with agent.run_stream(long_query) as stream:
    async for chunk in stream.stream_text():
        print(chunk, end='', flush=True)  # Real-time!
    result = await stream.get_data()
# User: "Wow, this feels responsive!"
```

**Benefits of Streaming**:
1. **Immediate feedback**: Users know something is happening
2. **Perceived performance**: Feels faster even if total time is same
3. **Cancellable**: Users can stop if not getting what they want
4. **Progressive disclosure**: Users process information as it arrives
5. **Better engagement**: Interactive feel keeps users engaged

---

## C. Test & Apply

### How to Test It

1. **Run the streaming demo**:
```bash
python lesson_12_streaming_responses.py
```

2. **Observe real-time output**

3. **Try your own streaming agent**:
```python
my_agent = Agent(
    model='gemini-1.5-flash',
    result_type=str,
    system_prompt="You are a helpful assistant."
)

async def my_stream():
    async with my_agent.run_stream("Explain Python") as stream:
        async for chunk in stream.stream_text():
            print(chunk, end='', flush=True)
        result = await stream.get_data()

asyncio.run(my_stream())
```

### Expected Result

You should see output appearing progressively:

```
======================================================================
EXAMPLE 1: Basic Text Streaming
======================================================================

Query: Explain async programming in Python

Response (streaming):
----------------------------------------------------------------------
Async programming in Python allows you to write concurrent code that can handle multiple operations simultaneously without blocking. Instead of waiting for one operation to complete before starting the next, async code can switch between tasks efficiently...

[Text appears word by word in real-time]

...This makes Python excellent for I/O-bound applications like web servers and API clients.
----------------------------------------------------------------------
✅ Streaming complete. Total length: 847 chars

======================================================================
EXAMPLE 2: Streaming with Tool Calls
======================================================================

Query: Analyze our 2024 sales and customer growth

🔄 Streaming agent execution (watch tools being called):
----------------------------------------------------------------------

🔧 [Tool] fetch_sales_data(year=2024) - Starting...
✅ [Tool] fetch_sales_data complete: $15,750,000

🔧 [Tool] fetch_customer_data() - Starting...
✅ [Tool] fetch_customer_data complete: 1250 customers

💬 [Text]: Based on the data gathered...
💬 [Text]: Sales in 2024 totaled $15.75 million...
💬 [Text]: Customer base grew to 1,250 with strong retention...

----------------------------------------------------------------------

📊 FINAL ANALYSIS:
   Summary: Strong performance in 2024 with solid growth
   Findings: 3 key findings
   Recommendations: 4 recommendations
   Data Sources: sales_database, customer_api
```

### Validation Examples

**Streaming Patterns**:

```python
# ✅ Basic text streaming
async with agent.run_stream(query) as stream:
    async for chunk in stream.stream_text():
        print(chunk, end='')

# ✅ Streaming with structured data
async with agent.run_stream(query, deps=deps) as stream:
    async for chunk in stream.stream():
        # Handle different chunk types
        pass

# ✅ Cancellable streaming
async with agent.run_stream(query) as stream:
    async for chunk in stream.stream_text():
        if should_cancel():
            break  # Exit cleanly

# ✅ Error handling
try:
    async with agent.run_stream(query) as stream:
        async for chunk in stream.stream_text():
            print(chunk, end='')
except Exception as e:
    print(f"Stream error: {e}")
```

### Type Checking

```bash
mypy lesson_12_streaming_responses.py
```

Expected: `Success: no issues found`

---

## D. Common Stumbling Blocks

### 1. Forgetting async/await

**The Problem**:
```python
# ❌ Using sync methods with streaming
with agent.run_stream(query) as stream:  # Wrong!
    for chunk in stream.stream_text():  # Wrong!
        print(chunk)
```

**The Fix**:
Always use async/await:
```python
# ✅ Proper async streaming
async with agent.run_stream(query) as stream:  # ✅
    async for chunk in stream.stream_text():  # ✅
        print(chunk, end='')
```

### 2. Not Using Context Manager

**The Problem**:
```python
# ❌ Not using context manager
stream = agent.run_stream(query)  # Resources not cleaned up!
async for chunk in stream.stream_text():
    print(chunk)
```

**The Fix**:
Always use `async with`:
```python
# ✅ Proper resource management
async with agent.run_stream(query) as stream:
    async for chunk in stream.stream_text():
        print(chunk, end='')
# ✅ Resources automatically cleaned up
```

### 3. Blocking Operations in Stream Loop

**The Problem**:
```python
async with agent.run_stream(query) as stream:
    async for chunk in stream.stream_text():
        # ❌ Blocking operation!
        time.sleep(0.1)  # Blocks entire event loop!
        print(chunk)
```

**The Fix**:
Use async operations:
```python
async with agent.run_stream(query) as stream:
    async for chunk in stream.stream_text():
        # ✅ Non-blocking
        await asyncio.sleep(0.1)  # Yields control
        print(chunk, end='')
```

### 4. Not Handling Stream Errors

**The Problem**:
```python
# ❌ No error handling
async with agent.run_stream(query) as stream:
    async for chunk in stream.stream_text():
        print(chunk, end='')
# What if tool fails mid-stream?
```

**The Fix**:
Wrap in try/except:
```python
# ✅ Proper error handling
try:
    async with agent.run_stream(query) as stream:
        async for chunk in stream.stream_text():
            print(chunk, end='')
except Exception as e:
    print(f"\n⚠️  Stream error: {e}")
    # Handle gracefully
```

### 5. Type Safety Gotcha: Mixing stream_text() and stream()

**The Problem**:
```python
async with agent.run_stream(query) as stream:
    # ❌ Using both methods
    async for chunk in stream.stream_text():
        print(chunk)
    async for chunk in stream.stream():  # Already consumed!
        print(chunk)
```

**The Fix**:
Choose one streaming method:
```python
# ✅ Option 1: Text only
async with agent.run_stream(query) as stream:
    async for text_chunk in stream.stream_text():
        print(text_chunk, end='')

# ✅ Option 2: All chunks (text + structured)
async with agent.run_stream(query) as stream:
    async for chunk in stream.stream():
        if hasattr(chunk, 'content'):
            print(chunk.content, end='')
```

---

## Ready for the Next Lesson?

🎉 **Fantastic work!** You now understand streaming responses:

✅ Basic text streaming with `run_stream()`  
✅ Streaming with tool calls visible in real-time  
✅ Progress tracking during long operations  
✅ Cancellable streams for user control  
✅ SSE format for web applications  
✅ Error handling in streaming contexts  
✅ Buffered streaming for optimization  

**Streaming transforms static agents into responsive, interactive experiences!** Users stay engaged, can provide feedback, and have control over long-running operations.

In the next lesson, we'll explore **Handling Tool Errors** - you'll learn comprehensive error handling strategies for production agents, including graceful degradation, retry logic, and user-friendly error messages!

**Ready for Lesson 13, or would you like to practice streaming patterns first?** 🚀
