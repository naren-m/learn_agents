"""
=============================================================================
SHARED TOOLS - Reusable Tool Definitions for Agent Examples
=============================================================================

This module contains tool definitions that are shared across different agent
implementations (LangChain, LangGraph, etc.).

WHAT ARE TOOLS?
---------------
Tools are functions that an AI agent can call to interact with the world.
They must have:
- A clear name
- A description (the LLM reads this to decide when to use the tool!)
- Type hints for inputs

The @tool decorator from langchain_core converts a function into a tool
that can be used by agents.
"""

from langchain_core.tools import tool


# =============================================================================
# MATH TOOLS
# =============================================================================

@tool
def add(a: int, b: int) -> int:
    """Add two numbers together. Use this when you need to perform addition."""
    print(f"  [TOOL CALLED] add({a}, {b})")
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers together. Use this when you need to perform multiplication."""
    print(f"  [TOOL CALLED] multiply({a}, {b})")
    return a * b


# =============================================================================
# WEATHER TOOLS
# =============================================================================

@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city. Use this when asked about weather."""
    print(f"  [TOOL CALLED] get_weather({city})")
    # In a real app, this would call a weather API
    weather_data = {
        "new york": "Sunny, 72F",
        "london": "Cloudy, 58F",
        "tokyo": "Rainy, 65F",
    }
    return weather_data.get(city.lower(), f"Weather data not available for {city}")


# =============================================================================
# TOOL COLLECTION
# =============================================================================

# Default collection of all tools for easy import
tools = [add, multiply, get_weather]
