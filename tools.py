"""
=============================================================================
SHARED TOOLS - OOP-Based Tool Definitions for Agent Examples
=============================================================================

This module contains tool definitions organized using Object-Oriented Programming
principles. Tools are grouped into logical classes and managed by a ToolRegistry.

DESIGN PRINCIPLES:
------------------
1. Single Responsibility: Each tool class handles one domain (math, weather, etc.)
2. Encapsulation: Tool logic is encapsulated within classes
3. Open/Closed: Easy to extend with new tool classes without modifying existing code
4. Registry Pattern: ToolRegistry provides centralized tool management
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from langchain_core.tools import tool, BaseTool


# =============================================================================
# BASE TOOL PROVIDER (Abstract Base Class)
# =============================================================================

class BaseToolProvider(ABC):
    """
    Abstract base class for tool providers.

    Each tool provider is responsible for a specific domain of functionality
    (e.g., math operations, weather lookups, etc.).
    """

    @abstractmethod
    def get_tools(self) -> List[BaseTool]:
        """Return a list of tools provided by this provider."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this tool provider."""
        pass


# =============================================================================
# MATH TOOLS PROVIDER
# =============================================================================

class MathToolProvider(BaseToolProvider):
    """
    Provides mathematical operation tools.

    Available tools:
    - add: Addition of two numbers
    - multiply: Multiplication of two numbers
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize the math tool provider.

        Args:
            verbose: If True, print tool calls for debugging
        """
        self._verbose = verbose
        self._tools = self._create_tools()

    @property
    def name(self) -> str:
        return "MathTools"

    def _create_tools(self) -> List[BaseTool]:
        """Create and return the math tools."""
        verbose = self._verbose

        @tool
        def add(a: int, b: int) -> int:
            """Add two numbers together. Use this when you need to perform addition."""
            if verbose:
                print(f"  [TOOL CALLED] add({a}, {b})")
            return a + b

        @tool
        def multiply(a: int, b: int) -> int:
            """Multiply two numbers together. Use this when you need to perform multiplication."""
            if verbose:
                print(f"  [TOOL CALLED] multiply({a}, {b})")
            return a * b

        return [add, multiply]

    def get_tools(self) -> List[BaseTool]:
        """Return the list of math tools."""
        return self._tools


# =============================================================================
# WEATHER TOOLS PROVIDER
# =============================================================================

class WeatherToolProvider(BaseToolProvider):
    """
    Provides weather-related tools.

    Available tools:
    - get_weather: Get current weather for a city
    """

    # Default weather data (in a real app, this would come from an API)
    DEFAULT_WEATHER_DATA: Dict[str, str] = {
        "new york": "Sunny, 72F",
        "london": "Cloudy, 58F",
        "tokyo": "Rainy, 65F",
    }

    def __init__(self, verbose: bool = True, weather_data: Dict[str, str] = None):
        """
        Initialize the weather tool provider.

        Args:
            verbose: If True, print tool calls for debugging
            weather_data: Optional custom weather data dictionary
        """
        self._verbose = verbose
        self._weather_data = weather_data or self.DEFAULT_WEATHER_DATA
        self._tools = self._create_tools()

    @property
    def name(self) -> str:
        return "WeatherTools"

    def _create_tools(self) -> List[BaseTool]:
        """Create and return the weather tools."""
        verbose = self._verbose
        weather_data = self._weather_data

        @tool
        def get_weather(city: str) -> str:
            """Get the current weather for a city. Use this when asked about weather."""
            if verbose:
                print(f"  [TOOL CALLED] get_weather({city})")
            return weather_data.get(
                city.lower(),
                f"Weather data not available for {city}"
            )

        return [get_weather]

    def get_tools(self) -> List[BaseTool]:
        """Return the list of weather tools."""
        return self._tools


# =============================================================================
# TOOL REGISTRY (Centralized Tool Management)
# =============================================================================

class ToolRegistry:
    """
    Centralized registry for managing tool providers.

    The registry pattern allows:
    - Easy registration of new tool providers
    - Centralized access to all tools
    - Flexible tool selection by provider name

    Usage:
        registry = ToolRegistry()
        registry.register(MathToolProvider())
        registry.register(WeatherToolProvider())

        all_tools = registry.get_all_tools()
        math_tools = registry.get_tools_by_provider("MathTools")
    """

    def __init__(self):
        """Initialize an empty tool registry."""
        self._providers: Dict[str, BaseToolProvider] = {}

    def register(self, provider: BaseToolProvider) -> "ToolRegistry":
        """
        Register a tool provider with the registry.

        Args:
            provider: The tool provider to register

        Returns:
            Self for method chaining
        """
        self._providers[provider.name] = provider
        return self

    def unregister(self, provider_name: str) -> bool:
        """
        Unregister a tool provider from the registry.

        Args:
            provider_name: Name of the provider to unregister

        Returns:
            True if provider was removed, False if not found
        """
        if provider_name in self._providers:
            del self._providers[provider_name]
            return True
        return False

    def get_tools_by_provider(self, provider_name: str) -> List[BaseTool]:
        """
        Get tools from a specific provider.

        Args:
            provider_name: Name of the provider

        Returns:
            List of tools from the specified provider

        Raises:
            KeyError: If provider is not registered
        """
        if provider_name not in self._providers:
            raise KeyError(f"Provider '{provider_name}' not registered")
        return self._providers[provider_name].get_tools()

    def get_all_tools(self) -> List[BaseTool]:
        """
        Get all tools from all registered providers.

        Returns:
            Combined list of all tools
        """
        all_tools = []
        for provider in self._providers.values():
            all_tools.extend(provider.get_tools())
        return all_tools

    def list_providers(self) -> List[str]:
        """
        List all registered provider names.

        Returns:
            List of provider names
        """
        return list(self._providers.keys())

    def __len__(self) -> int:
        """Return the number of registered providers."""
        return len(self._providers)


# =============================================================================
# DEFAULT REGISTRY FACTORY
# =============================================================================

def create_default_registry(verbose: bool = True) -> ToolRegistry:
    """
    Create a registry with all default tool providers.

    Args:
        verbose: If True, tools will print when called

    Returns:
        ToolRegistry with MathToolProvider and WeatherToolProvider registered
    """
    registry = ToolRegistry()
    registry.register(MathToolProvider(verbose=verbose))
    registry.register(WeatherToolProvider(verbose=verbose))
    return registry


# =============================================================================
# CONVENIENCE EXPORTS (Backward Compatibility)
# =============================================================================

# Create default registry and export tools for backward compatibility
_default_registry = create_default_registry(verbose=True)
tools = _default_registry.get_all_tools()
