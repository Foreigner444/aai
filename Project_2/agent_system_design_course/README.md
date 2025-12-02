# Agent System Design Course

Welcome to the **Agent System Design** project! This course teaches you to build sophisticated, production-ready AI agents using Pydantic AI and Google Gemini.

## 📚 Course Overview

This project contains **16 comprehensive lessons** that will take you from basic agent architecture to advanced multi-tool agent systems with streaming, error handling, and production patterns.

## 🎯 What You'll Build

By the end of this course, you'll have built a **complete multi-tool agent system** with:
- Custom tools and dependencies
- Optimized system prompts
- Dynamic tool selection
- Streaming responses
- Comprehensive error handling
- Retry and fallback mechanisms
- Result validation

## 📖 Lessons

### **Part 1: Foundations** (Lessons 1-4)
1. **Understanding Agent Architecture** - Core concepts of Pydantic AI agents
2. **System Prompts and Instructions** - Crafting effective prompts
3. **Creating Custom Dependencies** - Dependency injection basics
4. **Dependency Injection Pattern** - Advanced dependency management

### **Part 2: Tools** (Lessons 5-8)
5. **Agent with Dependencies** - Using dependencies in execution
6. **Creating Custom Tools** - Building powerful agent tools
7. **Tool Function Signatures** - Designing robust tool interfaces
8. **Tool Descriptions for Gemini** - Writing effective tool documentation

### **Part 3: Multi-Tool Systems** (Lessons 9-12)
9. **Multi-Tool Agents** - Agents with multiple tools
10. **Tool Context and Parameters** - Advanced tool parameter patterns
11. **Dynamic Tool Selection** - Helping Gemini choose the right tools
12. **Streaming Responses** - Real-time response streaming

### **Part 4: Production Patterns** (Lessons 13-16)
13. **Handling Tool Errors** - Robust error handling in tools
14. **Retries and Fallbacks** - Reliability patterns
15. **Agent Result Validation** - Validating agent outputs
16. **Complete Multi-Tool Agent System** - Final project integration

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- Google Gemini API key (get one at https://ai.google.dev/)

### Installation

1. **Create a virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies**:
```bash
pip install pydantic-ai pydantic google-generativeai python-dotenv
```

3. **Set up your API key**:
```bash
echo "GOOGLE_API_KEY=your_api_key_here" > .env
```

### Running Lessons

Each lesson is a self-contained markdown file with:
- Concept explanations
- Complete code examples
- Testing instructions
- Common pitfalls

Start with **Lesson 1** and progress sequentially:
```bash
# Read the lesson
cat lesson_01_understanding_agent_architecture.md

# Create and run the example code
# (Code examples are provided in each lesson)
```

## 💡 Key Concepts

### Type Safety First
Every lesson emphasizes type safety through:
- Pydantic model validation
- Python type hints
- mypy type checking
- IDE autocomplete support

### Production-Ready Patterns
You'll learn patterns used in real production systems:
- Dependency injection
- Resource management
- Error handling
- Async operations
- Testing strategies

### Gemini Integration
Optimized for Google Gemini models:
- Gemini Flash (fast, cost-effective)
- Gemini Pro (balanced performance)
- Massive context windows (1M+ tokens)
- Advanced features (multimodal, function calling)

## 📝 Course Structure

Each lesson follows a consistent structure:

1. **Concept Overview**
   - What & Why
   - Real-world analogy
   - Type safety benefits

2. **Code Implementation**
   - File structure
   - Complete, runnable code
   - Line-by-line explanation
   - Pattern rationale

3. **Test & Apply**
   - How to test
   - Expected results
   - Validation examples
   - Type checking

4. **Common Stumbling Blocks**
   - Common errors
   - What causes them
   - How to fix them
   - Type safety gotchas

## 🎓 Learning Path

**Recommended approach**:
1. Read the lesson thoroughly
2. Type out the code examples (don't just copy-paste!)
3. Run the code and observe the output
4. Experiment with modifications
5. Try the practice exercises
6. Move to the next lesson

**Take your time!** Each lesson builds on previous concepts. Master each one before moving forward.

## 🛠️ Tools & Technologies

- **Pydantic AI**: Type-safe AI agent framework
- **Pydantic v2**: Data validation library
- **Google Gemini**: Advanced AI models
- **Python 3.9+**: Type hints and modern features
- **mypy**: Static type checker
- **python-dotenv**: Environment variable management

## 📦 Project Structure

```
agent_system_design_course/
├── README.md (this file)
├── lesson_01_understanding_agent_architecture.md
├── lesson_02_system_prompts_and_instructions.md
├── lesson_03_creating_custom_dependencies.md
├── lesson_04_dependency_injection_pattern.md
├── lesson_05_agent_with_dependencies.md
├── lesson_06_creating_custom_tools.md
├── lesson_07_tool_function_signatures.md
├── lesson_08_tool_descriptions_for_gemini.md
├── lesson_09_multi_tool_agents.md
├── lesson_10_tool_context_and_parameters.md
├── lesson_11_dynamic_tool_selection.md
├── lesson_12_streaming_responses.md
├── lesson_13_handling_tool_errors.md
├── lesson_14_retries_and_fallbacks.md
├── lesson_15_agent_result_validation.md
└── lesson_16_complete_multi_tool_agent_system.md
```

## 🎯 Learning Objectives

By completing this course, you will:

✅ Understand Pydantic AI agent architecture  
✅ Create type-safe agents with structured outputs  
✅ Design and implement custom tools  
✅ Manage dependencies with dependency injection  
✅ Write effective system prompts  
✅ Handle errors robustly  
✅ Implement streaming responses  
✅ Build production-ready multi-tool systems  
✅ Validate AI outputs automatically  
✅ Integrate with Google Gemini models  

## 🌟 Best Practices

Throughout the course, you'll learn:

- **Type Safety**: Using Python type hints and Pydantic validation
- **Error Handling**: Graceful degradation and user-friendly errors
- **Testing**: Mocking dependencies and validating outputs
- **Documentation**: Self-documenting code with docstrings
- **Performance**: Async operations and streaming
- **Security**: Safe handling of API keys and sensitive data

## 🤝 Support

If you have questions or get stuck:

1. Re-read the lesson's "Common Stumbling Blocks" section
2. Check the code examples carefully
3. Try running with verbose logging to see what's happening
4. Experiment with simpler versions of the code
5. Review earlier lessons for foundational concepts

## 🎉 Ready to Start?

Jump into **Lesson 1: Understanding Agent Architecture** and begin your journey to mastering Pydantic AI with Gemini!

Remember: The best way to learn is by doing. Type out the code, run it, break it, fix it, and experiment!

**Happy coding!** 🚀
