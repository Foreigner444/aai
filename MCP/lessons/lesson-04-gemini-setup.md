# Lesson 4: Google Gemini Setup

## Getting Your API Key

### Step 1: Visit Google AI Studio

Go to: https://makersuite.google.com/app/apikey

Or: https://aistudio.google.com/apikey

### Step 2: Sign In

Use your Google account. If you don't have one, create a free account.

### Step 3: Create API Key

1. Click "Create API Key"
2. Select a Google Cloud project (or create a new one)
3. Copy the generated API key

**Important**: Save this key securely! You won't be able to see it again.

### Step 4: Set Environment Variable

**Linux/Mac**:
```bash
export GEMINI_API_KEY='your-api-key-here'
```

Add to `~/.bashrc` or `~/.zshrc` for persistence:
```bash
echo 'export GEMINI_API_KEY="your-api-key-here"' >> ~/.bashrc
source ~/.bashrc
```

**Windows (PowerShell)**:
```powershell
$env:GEMINI_API_KEY='your-api-key-here'
```

For persistence:
```powershell
[System.Environment]::SetEnvironmentVariable('GEMINI_API_KEY', 'your-api-key-here', 'User')
```

**Windows (CMD)**:
```cmd
set GEMINI_API_KEY=your-api-key-here
```

## Gemini Models Overview

### gemini-1.5-flash

**Best for**: Most applications

**Features**:
- Fast responses (< 1 second typically)
- Cost-effective
- Good reasoning capabilities
- Long context window (1M tokens)
- Multimodal (text, images, video, audio)

**Use cases**:
- Chatbots
- Data extraction
- Content generation
- Quick analysis

**Pricing** (as of Dec 2024):
- Input: $0.075 per 1M tokens
- Output: $0.30 per 1M tokens

### gemini-1.5-pro

**Best for**: Complex reasoning tasks

**Features**:
- Superior reasoning
- Better at complex instructions
- Longer context (2M tokens)
- More accurate
- Multimodal

**Use cases**:
- Code generation
- Complex analysis
- Research tasks
- Detailed summaries

**Pricing**:
- Input: $1.25 per 1M tokens
- Output: $5.00 per 1M tokens

### gemini-2.0-flash-exp

**Best for**: Experimental features

**Features**:
- Cutting-edge capabilities
- Newest features
- May change frequently
- Free during preview

**Use cases**:
- Testing new features
- Experimentation
- Non-production projects

**Note**: Experimental models may have rate limits and can change without notice.

## Installation

```bash
pip install google-generativeai pydantic-ai
```

Verify installation:
```bash
python -c "import google.generativeai as genai; print('Installed!')"
```

## Testing Your Setup

### Test 1: Direct Gemini API

```python
import os
import google.generativeai as genai

api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    raise ValueError("GEMINI_API_KEY not set!")

genai.configure(api_key=api_key)

model = genai.GenerativeModel('gemini-1.5-flash')
response = model.generate_content('Say hello!')

print(response.text)
```

Save as `test_gemini.py` and run:
```bash
python test_gemini.py
```

Expected output: A friendly greeting from Gemini.

### Test 2: Pydantic AI with Gemini

```python
import os
import asyncio
from pydantic_ai import Agent

api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    raise ValueError("GEMINI_API_KEY not set!")

agent = Agent('gemini-1.5-flash')

async def main():
    result = await agent.run('Count from 1 to 5')
    print(result.data)

asyncio.run(main())
```

Save as `test_pydantic_gemini.py` and run:
```bash
python test_pydantic_gemini.py
```

Expected output: Numbers 1 through 5.

## Configuration Options

### Setting API Key in Code

For testing only (not recommended for production):

```python
from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel

model = GeminiModel(
    'gemini-1.5-flash',
    api_key='your-api-key-here'
)

agent = Agent(model)
```

### Using .env Files

Create `.env` file:
```
GEMINI_API_KEY=your-api-key-here
```

Install python-dotenv:
```bash
pip install python-dotenv
```

Load in code:
```python
from dotenv import load_dotenv
load_dotenv()

from pydantic_ai import Agent

agent = Agent('gemini-1.5-flash')
```

### Model Parameters

```python
from pydantic_ai import Agent

agent = Agent(
    'gemini-1.5-flash',
    model_settings={
        'temperature': 0.7,
        'max_tokens': 1000,
        'top_p': 0.9,
    }
)
```

**Parameters**:
- `temperature`: 0.0-1.0 (higher = more creative)
- `max_tokens`: Maximum response length
- `top_p`: Nucleus sampling (0.0-1.0)

## Rate Limits and Quotas

### Free Tier

As of Dec 2024:
- 15 requests per minute
- 1 million tokens per minute
- 1500 requests per day

### Paid Tier

- Higher rate limits
- Pay-as-you-go pricing
- No daily request limits
- Better support

### Handling Rate Limits

```python
import asyncio
from pydantic_ai import Agent

agent = Agent('gemini-1.5-flash')

async def make_request_with_retry(prompt: str, max_retries=3):
    for attempt in range(max_retries):
        try:
            result = await agent.run(prompt)
            return result.data
        except Exception as e:
            if "rate limit" in str(e).lower() and attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"Rate limited. Waiting {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                raise
    
    raise Exception("Max retries exceeded")
```

## Security Best Practices

### 1. Never Hardcode API Keys

❌ **Bad**:
```python
api_key = "AIzaSyC_abc123xyz"
```

✅ **Good**:
```python
api_key = os.getenv('GEMINI_API_KEY')
```

### 2. Add .env to .gitignore

```bash
echo ".env" >> .gitignore
```

### 3. Use Separate Keys

- Development key for testing
- Production key for live applications
- Rotate keys periodically

### 4. Restrict API Key

In Google Cloud Console:
- Restrict key to specific APIs
- Limit by IP address if possible
- Set usage quotas

### 5. Monitor Usage

Check usage regularly:
- Google Cloud Console → APIs & Services → Credentials
- Set up billing alerts
- Monitor for unusual activity

## Troubleshooting

### Error: "API key not valid"

**Check**:
1. Key is correctly copied (no extra spaces)
2. API key is enabled for Generative AI API
3. Billing is set up (even for free tier)

**Solution**:
```bash
python -c "import os; print(f'Key length: {len(os.getenv(\"GEMINI_API_KEY\", \"\"))}')"
```

Should be around 39 characters.

### Error: "Resource exhausted"

**Meaning**: Rate limit exceeded

**Solution**:
- Wait a few minutes
- Implement retry logic
- Upgrade to paid tier

### Error: "Permission denied"

**Meaning**: API not enabled

**Solution**:
1. Go to Google Cloud Console
2. Enable "Generative Language API"
3. Wait a few minutes and retry

### Error: "Model not found"

**Meaning**: Model name incorrect

**Solution**:
Check model name:
```python
valid_models = [
    'gemini-1.5-flash',
    'gemini-1.5-pro',
    'gemini-2.0-flash-exp'
]
```

## Cost Management

### Estimate Costs

**Example calculation**:
```
Input: 1000 tokens
Output: 500 tokens
Model: gemini-1.5-flash

Cost = (1000 * $0.075 / 1M) + (500 * $0.30 / 1M)
     = $0.000075 + $0.00015
     = $0.000225 per request

1000 requests = $0.225
```

### Tracking Usage

```python
import asyncio
from pydantic_ai import Agent

class UsageTracker:
    def __init__(self):
        self.total_tokens = 0
        self.total_requests = 0
    
    def track(self, input_tokens: int, output_tokens: int):
        self.total_tokens += input_tokens + output_tokens
        self.total_requests += 1
    
    def estimate_cost(self, model='gemini-1.5-flash'):
        prices = {
            'gemini-1.5-flash': {'input': 0.075, 'output': 0.30},
            'gemini-1.5-pro': {'input': 1.25, 'output': 5.00},
        }
        price = prices.get(model, prices['gemini-1.5-flash'])
        cost = (self.total_tokens * (price['input'] + price['output']) / 2) / 1_000_000
        return cost

tracker = UsageTracker()

agent = Agent('gemini-1.5-flash')

async def main():
    result = await agent.run("Hello")
    tracker.track(input_tokens=10, output_tokens=20)
    
    print(f"Requests: {tracker.total_requests}")
    print(f"Tokens: {tracker.total_tokens}")
    print(f"Est. cost: ${tracker.estimate_cost():.6f}")

asyncio.run(main())
```

## Practice Exercise

Set up and test your Gemini integration:

1. **Get API Key**: Follow the steps above

2. **Create test script** (`gemini_test.py`):
   - Set up environment variable
   - Create agent with gemini-1.5-flash
   - Test structured output with a simple Pydantic model
   - Print results

3. **Test different models**: Compare responses from:
   - gemini-1.5-flash
   - gemini-1.5-pro
   
   Same prompt, different quality/speed?

4. **Test rate limiting**: Make 20 rapid requests
   - Do you hit limits?
   - Implement retry logic

5. **Calculate costs**: Estimate cost for your use case
   - How many requests per day?
   - Average tokens per request?
   - Monthly estimate?

## Summary

- Get API key from Google AI Studio
- Store in environment variable (never hardcode)
- Three main models: flash (fast), pro (powerful), experimental (new)
- Free tier: 15 RPM, 1M TPM, 1500 RPD
- Monitor usage and costs
- Implement retry logic for rate limits
- Follow security best practices

---

**Next**: [Lesson 5: Building Your First MCP Server](lesson-05-first-mcp-server.md)
