# 📘 Lesson 16: Complete Structured Output App

Congratulations! 🎉 This is the final lesson where we bring everything together into a production-ready application!

---

## A. Concept Overview

### What You've Learned

Over the past 15 lessons, you've mastered:
- ✅ Pydantic AI fundamentals and type safety
- ✅ Structured outputs vs raw text
- ✅ Python environment setup
- ✅ Pydantic models and validation
- ✅ Google Gemini API integration
- ✅ Creating and running agents
- ✅ Error handling and debugging
- ✅ Prompt engineering

### What You'll Build

A complete **User Profile Extraction System** that:
- Extracts structured user information from text
- Validates all data with Pydantic
- Handles errors gracefully
- Provides useful feedback
- Is ready for production use

### Type Safety Benefits Realized

Your complete application will have:
- 100% type-safe AI outputs
- Automatic validation of all data
- Clear error messages
- IDE autocomplete everywhere
- Reliable, testable code

---

## B. Code Implementation

### Project Structure

```
pydantic-ai-project/
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── user_profile.py     # Pydantic models
│   ├── agents/
│   │   ├── __init__.py
│   │   └── profile_extractor.py # Agent definition
│   ├── utils/
│   │   ├── __init__.py
│   │   └── error_handler.py    # Error handling
│   └── config.py               # Configuration
├── main.py                     # Main application
├── .env                        # API key (not committed)
├── .env.example               # Template
├── requirements.txt           # Dependencies
└── README.md                  # Documentation
```

### Step 1: Create the Models

`src/models/user_profile.py`:

```python
"""
Pydantic models for user profile extraction.
"""
from pydantic import BaseModel, Field, EmailStr
from typing import Optional, Literal
from datetime import date
from enum import Enum


class EmploymentStatus(str, Enum):
    """Employment status options."""
    EMPLOYED = "employed"
    SELF_EMPLOYED = "self_employed"
    UNEMPLOYED = "unemployed"
    STUDENT = "student"
    RETIRED = "retired"
    UNKNOWN = "unknown"


class Address(BaseModel):
    """A physical address."""
    city: str = Field(description="City name")
    state: Optional[str] = Field(default=None, description="State or province")
    country: str = Field(default="USA", description="Country name")
    
    def __str__(self) -> str:
        parts = [self.city]
        if self.state:
            parts.append(self.state)
        parts.append(self.country)
        return ", ".join(parts)


class SocialProfile(BaseModel):
    """A social media profile."""
    platform: str = Field(description="Platform name (Twitter, LinkedIn, etc.)")
    username: Optional[str] = Field(default=None, description="Username on the platform")
    url: Optional[str] = Field(default=None, description="Profile URL")


class UserProfile(BaseModel):
    """
    Complete user profile extracted from text.
    This is the main model used by the extraction agent.
    """
    # Basic Information
    name: str = Field(description="Full name of the person")
    age: Optional[int] = Field(
        default=None, 
        ge=0, 
        le=150,
        description="Age in years"
    )
    
    # Contact Information
    email: Optional[str] = Field(
        default=None,
        description="Email address"
    )
    phone: Optional[str] = Field(
        default=None,
        description="Phone number"
    )
    
    # Location
    location: Optional[Address] = Field(
        default=None,
        description="Current location"
    )
    
    # Professional Information
    occupation: Optional[str] = Field(
        default=None,
        description="Current job title or profession"
    )
    company: Optional[str] = Field(
        default=None,
        description="Current employer or company"
    )
    employment_status: EmploymentStatus = Field(
        default=EmploymentStatus.UNKNOWN,
        description="Current employment status"
    )
    
    # Skills and Interests
    skills: list[str] = Field(
        default_factory=list,
        description="Professional skills mentioned"
    )
    interests: list[str] = Field(
        default_factory=list,
        description="Personal interests or hobbies"
    )
    
    # Social Media
    social_profiles: list[SocialProfile] = Field(
        default_factory=list,
        description="Social media profiles mentioned"
    )
    
    # Additional Info
    bio: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Brief bio or summary"
    )
    
    def summary(self) -> str:
        """Generate a human-readable summary."""
        parts = [f"👤 {self.name}"]
        
        if self.age:
            parts.append(f"📅 {self.age} years old")
        
        if self.occupation:
            job = self.occupation
            if self.company:
                job += f" at {self.company}"
            parts.append(f"💼 {job}")
        
        if self.location:
            parts.append(f"📍 {self.location}")
        
        if self.email:
            parts.append(f"📧 {self.email}")
        
        if self.skills:
            parts.append(f"🛠️ Skills: {', '.join(self.skills[:5])}")
        
        return "\n".join(parts)
```

### Step 2: Create the Agent

`src/agents/profile_extractor.py`:

```python
"""
User profile extraction agent.
"""
from pydantic_ai import Agent
from src.models.user_profile import UserProfile


# System prompt optimized through testing
SYSTEM_PROMPT = """
You are an expert at extracting personal and professional information from text.

Your task is to identify and extract user profile information including:
- Name (required)
- Age (if mentioned)
- Contact information (email, phone)
- Location (city, state, country)
- Professional details (job, company, skills)
- Interests and hobbies
- Social media profiles

Guidelines:
1. Only extract information that is explicitly stated or clearly implied
2. For employment status, infer from context (e.g., "CEO" implies employed)
3. Extract all mentioned skills as a list
4. Social media: look for @usernames, profile URLs, or platform mentions
5. If location is partial (just city), include what's available
6. Keep the bio field as a brief summary of the person

Examples of what to extract:
- "John Smith, 35, works as a software engineer at Google"
  → name="John Smith", age=35, occupation="software engineer", company="Google"
  
- "Contact Sarah at sarah@email.com or @sarahdev on Twitter"
  → name="Sarah", email="sarah@email.com", social_profiles=[{platform="Twitter", username="sarahdev"}]

Be accurate and don't invent information not present in the text.
"""


def create_profile_agent(model: str = 'gemini-1.5-flash') -> Agent[None, UserProfile]:
    """
    Create a profile extraction agent.
    
    Args:
        model: The Gemini model to use
    
    Returns:
        Configured Agent instance
    """
    return Agent(
        model,
        result_type=UserProfile,
        system_prompt=SYSTEM_PROMPT,
        retries=2
    )


# Default agent instance
profile_agent = create_profile_agent()


def extract_profile(text: str) -> UserProfile:
    """
    Extract a user profile from text.
    
    Args:
        text: The text to extract from
    
    Returns:
        Extracted UserProfile
    
    Raises:
        ValidationError: If extraction fails validation
        Exception: For other errors
    """
    result = profile_agent.run_sync(text)
    return result.data
```

### Step 3: Create Error Handler

`src/utils/error_handler.py`:

```python
"""
Error handling utilities for the application.
"""
from pydantic import ValidationError
from dataclasses import dataclass
from enum import Enum
from typing import Any
import logging

logger = logging.getLogger(__name__)


class ErrorType(str, Enum):
    """Types of errors that can occur."""
    VALIDATION = "validation"
    API = "api"
    NETWORK = "network"
    RATE_LIMIT = "rate_limit"
    AUTH = "authentication"
    UNKNOWN = "unknown"


@dataclass
class AppError:
    """Structured application error."""
    error_type: ErrorType
    message: str
    details: dict[str, Any] | None = None
    recoverable: bool = True
    
    def to_user_message(self) -> str:
        """Create a user-friendly error message."""
        messages = {
            ErrorType.VALIDATION: "The AI returned invalid data. Please try again.",
            ErrorType.API: "There was an issue with the AI service. Please try later.",
            ErrorType.NETWORK: "Network error. Please check your connection.",
            ErrorType.RATE_LIMIT: "Too many requests. Please wait a moment.",
            ErrorType.AUTH: "Authentication failed. Please check your API key.",
            ErrorType.UNKNOWN: "An unexpected error occurred.",
        }
        return messages.get(self.error_type, self.message)


def handle_extraction_error(exception: Exception) -> AppError:
    """
    Convert an exception to an AppError.
    
    Args:
        exception: The caught exception
    
    Returns:
        Structured AppError
    """
    # Log the error
    logger.error(f"Extraction error: {type(exception).__name__}: {exception}")
    
    # Classify the error
    error_str = str(exception).lower()
    
    if isinstance(exception, ValidationError):
        return AppError(
            error_type=ErrorType.VALIDATION,
            message="Validation failed",
            details={"errors": exception.errors()},
            recoverable=True
        )
    
    if "rate limit" in error_str or "429" in error_str:
        return AppError(
            error_type=ErrorType.RATE_LIMIT,
            message="Rate limit exceeded",
            recoverable=True
        )
    
    if "api key" in error_str or "401" in error_str or "403" in error_str:
        return AppError(
            error_type=ErrorType.AUTH,
            message="Authentication failed",
            recoverable=False
        )
    
    if "connect" in error_str or "timeout" in error_str:
        return AppError(
            error_type=ErrorType.NETWORK,
            message="Network error",
            recoverable=True
        )
    
    return AppError(
        error_type=ErrorType.UNKNOWN,
        message=str(exception),
        recoverable=False
    )
```

### Step 4: Create Configuration

`src/config.py`:

```python
"""
Application configuration.
"""
import os
from dotenv import load_dotenv
from dataclasses import dataclass

load_dotenv()


@dataclass
class Config:
    """Application configuration."""
    gemini_api_key: str
    gemini_model: str
    debug: bool
    
    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not found in environment. "
                "Please create a .env file with your API key."
            )
        
        return cls(
            gemini_api_key=api_key,
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
            debug=os.getenv("DEBUG", "false").lower() == "true"
        )


# Global config instance
config = Config.from_env()
```

### Step 5: Create Main Application

`main.py`:

```python
"""
Main application - User Profile Extraction System

This application extracts structured user profile information
from unstructured text using Pydantic AI and Google Gemini.
"""
from src.agents.profile_extractor import extract_profile
from src.utils.error_handler import handle_extraction_error
from src.models.user_profile import UserProfile
import json


def print_header():
    """Print application header."""
    print("=" * 60)
    print("👤 USER PROFILE EXTRACTION SYSTEM")
    print("Powered by Pydantic AI & Google Gemini")
    print("=" * 60)


def print_profile(profile: UserProfile):
    """Print extracted profile in a nice format."""
    print("\n✅ EXTRACTION SUCCESSFUL!")
    print("-" * 40)
    print(profile.summary())
    print("-" * 40)
    
    # Full details
    print("\n📋 Full Profile Data:")
    print(json.dumps(profile.model_dump(), indent=2, default=str))


def interactive_mode():
    """Run in interactive mode."""
    print_header()
    print("\nEnter text to extract profile information.")
    print("Type 'quit' to exit, 'example' for sample text.\n")
    
    example_text = """
    Meet Sarah Chen, a 32-year-old senior software engineer at Meta.
    She's based in San Francisco, CA and has been coding for over 10 years.
    Her expertise includes Python, machine learning, and distributed systems.
    In her free time, Sarah enjoys rock climbing, photography, and contributing
    to open source projects. You can reach her at sarah.chen@email.com or
    find her on LinkedIn at linkedin.com/in/sarahchen and Twitter @sarahcodes.
    She's passionate about making AI more accessible to everyone.
    """
    
    while True:
        user_input = input("\n📝 Enter text (or 'quit'/'example'): ").strip()
        
        if user_input.lower() == 'quit':
            print("\nGoodbye! 👋")
            break
        
        if user_input.lower() == 'example':
            user_input = example_text
            print(f"\nUsing example text:\n{user_input[:100]}...")
        
        if not user_input:
            print("Please enter some text to analyze.")
            continue
        
        try:
            print("\n⏳ Extracting profile...")
            profile = extract_profile(user_input)
            print_profile(profile)
        
        except Exception as e:
            error = handle_extraction_error(e)
            print(f"\n❌ Error: {error.to_user_message()}")
            if error.details:
                print(f"   Details: {error.details}")


def process_single(text: str) -> UserProfile | None:
    """
    Process a single text input.
    
    Args:
        text: Text to extract profile from
    
    Returns:
        UserProfile if successful, None if failed
    """
    try:
        return extract_profile(text)
    except Exception as e:
        error = handle_extraction_error(e)
        print(f"Error: {error.to_user_message()}")
        return None


def batch_process(texts: list[str]) -> list[UserProfile | None]:
    """
    Process multiple texts.
    
    Args:
        texts: List of texts to process
    
    Returns:
        List of UserProfiles (None for failed extractions)
    """
    results = []
    for i, text in enumerate(texts, 1):
        print(f"\nProcessing {i}/{len(texts)}...")
        results.append(process_single(text))
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Process command line argument
        text = " ".join(sys.argv[1:])
        profile = process_single(text)
        if profile:
            print_profile(profile)
    else:
        # Run interactive mode
        interactive_mode()
```

### Step 6: Create Requirements File

`requirements.txt`:

```
pydantic-ai[google]>=0.0.20
pydantic>=2.5.0
python-dotenv>=1.0.0
httpx>=0.25.0
```

### Step 7: Create README

`README.md`:

```markdown
# User Profile Extraction System

A type-safe AI application that extracts structured user profiles from text
using Pydantic AI and Google Gemini.

## Features

- ✅ Extracts comprehensive user profile information
- ✅ Type-safe with Pydantic validation
- ✅ Handles errors gracefully
- ✅ Interactive and batch processing modes
- ✅ Production-ready code structure

## Quick Start

1. **Clone and setup:**
   ```bash
   git clone <repo>
   cd pydantic-ai-project
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Add your API key:**
   ```bash
   cp .env.example .env
   # Edit .env and add your GEMINI_API_KEY
   ```

3. **Run the application:**
   ```bash
   python main.py
   ```

## Usage

### Interactive Mode
```bash
python main.py
```

### Single Text Processing
```bash
python main.py "John Smith is a 35-year-old developer at Google"
```

### Programmatic Usage
```python
from src.agents.profile_extractor import extract_profile

profile = extract_profile("Sarah Chen, engineer at Meta, sarah@email.com")
print(profile.name)  # "Sarah Chen"
print(profile.company)  # "Meta"
print(profile.email)  # "sarah@email.com"
```

## Extracted Fields

| Field | Type | Description |
|-------|------|-------------|
| name | str | Full name (required) |
| age | int | Age in years |
| email | str | Email address |
| phone | str | Phone number |
| location | Address | City, state, country |
| occupation | str | Job title |
| company | str | Employer |
| skills | list[str] | Professional skills |
| interests | list[str] | Hobbies/interests |
| social_profiles | list | Social media profiles |
| bio | str | Brief summary |

## Project Structure

```
├── src/
│   ├── models/         # Pydantic models
│   ├── agents/         # AI agents
│   └── utils/          # Utilities
├── main.py             # Application entry
├── .env               # Configuration
└── requirements.txt   # Dependencies
```

## License

MIT
```

---

## C. Test & Apply

### Running Your Application

```bash
# Activate environment
source venv/bin/activate

# Run interactive mode
python main.py

# Run with text argument
python main.py "John Smith, 30, software engineer at Google in NYC"
```

### Testing Your Application

```python
"""
Tests for the profile extraction system.
"""
import pytest
from src.models.user_profile import UserProfile, Address
from src.agents.profile_extractor import extract_profile


def test_user_profile_model():
    """Test the UserProfile model directly."""
    profile = UserProfile(
        name="Test User",
        age=30,
        occupation="Developer",
        skills=["Python", "AI"]
    )
    assert profile.name == "Test User"
    assert profile.age == 30
    assert len(profile.skills) == 2


def test_address_model():
    """Test the Address model."""
    addr = Address(city="NYC", state="NY", country="USA")
    assert str(addr) == "NYC, NY, USA"


def test_extraction_basic():
    """Test basic extraction."""
    text = "John Smith is a developer."
    profile = extract_profile(text)
    assert profile.name == "John Smith"
    assert profile.occupation == "developer"


def test_extraction_with_contact():
    """Test extraction with contact info."""
    text = "Contact Sarah at sarah@test.com"
    profile = extract_profile(text)
    assert profile.name == "Sarah"
    assert profile.email == "sarah@test.com"
```

---

## D. What You've Accomplished! 🏆

### Skills You've Mastered

1. **Pydantic AI Framework** - Creating agents with typed outputs
2. **Pydantic Models** - Defining and validating data structures
3. **Google Gemini** - Integrating with powerful AI models
4. **Type Safety** - Ensuring reliable, predictable AI outputs
5. **Error Handling** - Building robust applications
6. **Prompt Engineering** - Crafting effective AI instructions
7. **Production Patterns** - Structuring code for real-world use

### Your Complete Application Features

- ✅ Structured data extraction from text
- ✅ Comprehensive user profile model
- ✅ Type-safe validation at every step
- ✅ Graceful error handling
- ✅ Interactive and batch processing
- ✅ Clean, modular code structure
- ✅ Ready for production deployment

---

## ✅ Project 1 Complete!

### 🎉 Congratulations!

You've completed the **Structured Output Basics** project! You now have:
- A solid foundation in Pydantic AI
- Understanding of type-safe AI development
- A working, production-ready application
- Skills to build more complex AI systems

### What's Next?

Ready for more? Here are the other projects waiting for you:

1. ~~**Structured Output Basics**~~ ✅ COMPLETED!
2. **Agent System Design** - Build agents with custom tools and dependencies
3. **Data Extraction Pipeline** - Extract complex nested data
4. **RAG System with Gemini** - Retrieval-augmented generation
5. **Conversational AI** - Chat systems with memory
6. **Multi-Agent Orchestration** - Agents working together
7. **Production Patterns** - Async, streaming, error handling
8. **FastAPI Integration** - REST API with AI
9. **Testing & Validation** - Comprehensive testing
10. **Advanced Gemini Features** - Multimodal and more

### Key Takeaways from Project 1

1. **Type safety transforms AI development** - No more parsing nightmares
2. **Pydantic models are contracts** - The AI must follow them
3. **Good prompts = good outputs** - Invest in prompt engineering
4. **Error handling is essential** - Things will go wrong
5. **Structure your code** - Clean architecture scales better

---

*🚀 You're now a Pydantic AI developer! Ready for Project 2?*
