# Suppress urllib3 LibreSSL warning (must be before any imports)
import warnings
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

"""
=============================================================================
LANGCHAIN TOOL-CALLING AGENT - OOP Implementation
=============================================================================

This script demonstrates the fundamental concepts of an AI agent using LangChain
with Object-Oriented Programming principles.

DESIGN PRINCIPLES:
------------------
1. Single Responsibility: Each class has one clear purpose
2. Encapsulation: Agent internals are hidden behind a clean interface
3. Dependency Injection: Tools, LLM, and prompts can be injected
4. Open/Closed: Easy to extend without modifying existing code

WHAT IS AN AGENT?
-----------------
An agent is an LLM that can:
1. REASON about what to do (think about the problem)
2. ACT by calling tools (execute actions)
3. OBSERVE the results (see what happened)
4. REPEAT until the task is complete

CLASS STRUCTURE:
----------------
- AgentConfig: Configuration dataclass for agent settings
- LangChainAgent: Main agent class using tool-calling approach
- AgentDemo: Demo runner for showcasing the agent

FLOW:
-----
User Question --> Agent --> LLM thinks --> Decides to use tool -->
Tool executes --> Result returned --> LLM thinks again -->
Either use another tool OR give final answer
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from langchain_ollama import ChatOllama
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool

from tools import tools as default_tools, ToolRegistry, create_default_registry


# =============================================================================
# AGENT CONFIGURATION
# =============================================================================

@dataclass
class AgentConfig:
    """
    Configuration settings for the LangChain agent.

    Attributes:
        model_name: The Ollama model to use
        temperature: LLM temperature (0 = deterministic)
        verbose: Whether to print agent's thinking process
        max_iterations: Maximum number of tool-call loops
        handle_parsing_errors: Whether to gracefully handle bad LLM output
    """
    model_name: str = "llama3.2:3b-instruct-q8_0"
    temperature: float = 0.0
    verbose: bool = True
    max_iterations: int = 10
    handle_parsing_errors: bool = True


# =============================================================================
# DEFAULT PROMPT TEMPLATE
# =============================================================================

DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant that can use tools to answer questions.
When you need to perform calculations, use the appropriate math tools.
When asked about weather, use the get_weather tool.
Always use tools when they are relevant to the question.
After getting tool results, provide a natural language response to the user."""


# =============================================================================
# LANGCHAIN AGENT CLASS
# =============================================================================

class LangChainAgent:
    """
    A tool-calling agent using LangChain.

    This agent processes user queries by:
    1. Receiving a question
    2. Using an LLM to decide if tools are needed
    3. Executing tools if required
    4. Looping until a final answer is ready

    The tool-calling approach uses native function calling (JSON-based)
    instead of text parsing, making it more reliable with modern LLMs.

    Example:
        agent = LangChainAgent()
        result = agent.run("What is 5 + 3?")
        print(result)
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        tools: Optional[List[BaseTool]] = None,
        tool_registry: Optional[ToolRegistry] = None,
        system_prompt: Optional[str] = None,
    ):
        """
        Initialize the LangChain agent.

        Args:
            config: Agent configuration settings
            tools: List of tools for the agent to use
            tool_registry: Optional ToolRegistry for advanced tool management
            system_prompt: Custom system prompt for the agent
        """
        self._config = config or AgentConfig()
        self._tool_registry = tool_registry
        self._tools = self._resolve_tools(tools)
        self._system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

        self._llm = self._create_llm()
        self._prompt = self._create_prompt()
        self._agent = self._create_agent()
        self._executor = self._create_executor()

    def _resolve_tools(self, tools: Optional[List[BaseTool]]) -> List[BaseTool]:
        """Resolve which tools to use based on provided arguments."""
        if tools is not None:
            return tools
        if self._tool_registry is not None:
            return self._tool_registry.get_all_tools()
        return default_tools

    def _create_llm(self) -> ChatOllama:
        """Create and configure the LLM instance."""
        return ChatOllama(
            model=self._config.model_name,
            temperature=self._config.temperature,
        )

    def _create_prompt(self) -> ChatPromptTemplate:
        """
        Create the prompt template for the agent.

        The prompt includes:
        - System message with instructions
        - Human message placeholder for user input
        - Agent scratchpad for tool calls/results
        """
        return ChatPromptTemplate.from_messages([
            ("system", self._system_prompt),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

    def _create_agent(self):
        """Create the tool-calling agent."""
        return create_tool_calling_agent(
            llm=self._llm,
            tools=self._tools,
            prompt=self._prompt,
        )

    def _create_executor(self) -> AgentExecutor:
        """
        Create the agent executor.

        The executor runs the agent in a loop:
        1. Agent decides what to do
        2. If tool call: execute tool, feed result back to agent
        3. If final answer: return to user
        4. Repeat until done or max iterations reached
        """
        return AgentExecutor(
            agent=self._agent,
            tools=self._tools,
            verbose=self._config.verbose,
            max_iterations=self._config.max_iterations,
            handle_parsing_errors=self._config.handle_parsing_errors,
        )

    def run(self, question: str) -> Dict[str, Any]:
        """
        Run the agent with a question.

        Args:
            question: The user's question

        Returns:
            Dictionary containing 'input' and 'output' keys
        """
        return self._executor.invoke({"input": question})

    def get_answer(self, question: str) -> str:
        """
        Run the agent and return only the answer string.

        Args:
            question: The user's question

        Returns:
            The agent's response as a string
        """
        result = self.run(question)
        return result.get("output", "")

    @property
    def config(self) -> AgentConfig:
        """Get the agent configuration."""
        return self._config

    @property
    def tools(self) -> List[BaseTool]:
        """Get the list of tools available to the agent."""
        return self._tools

    @property
    def executor(self) -> AgentExecutor:
        """Get the underlying AgentExecutor."""
        return self._executor


# =============================================================================
# DEMO RUNNER CLASS
# =============================================================================

class AgentDemo:
    """
    Demo runner for showcasing the LangChain agent.

    This class encapsulates the demo logic, making it easy to run
    different test scenarios.
    """

    def __init__(self, agent: LangChainAgent):
        """
        Initialize the demo runner.

        Args:
            agent: The LangChainAgent instance to demo
        """
        self._agent = agent

    def run_test(self, name: str, question: str) -> Optional[Dict[str, Any]]:
        """
        Run a single test case.

        Args:
            name: Name of the test
            question: Question to ask the agent

        Returns:
            The agent's response dictionary, or None if error
        """
        print(f"\n{'='*60}")
        print(f"TEST: {name}")
        print(f"Question: {question}")
        print(f"{'='*60}")

        try:
            result = self._agent.run(question)
            print(f"\n>>> Final Answer: {result['output']}")
            return result
        except Exception as e:
            print(f"\n>>> Error: {e}")
            return None

    def run_all_demos(self):
        """Run all demonstration test cases."""
        self._print_header()

        # Run test cases
        self.run_test("Simple Multiplication", "What is 7 multiplied by 8?")
        self.run_test("Multi-Step Math", "What is 5 plus 3, then multiply that result by 2?")
        self.run_test("Weather Query", "What's the weather like in Tokyo?")

        self._print_summary()

    def _print_header(self):
        """Print the demo header."""
        print("=" * 60)
        print("LANGCHAIN TOOL-CALLING AGENT DEMO (OOP VERSION)")
        print("=" * 60)
        print("""
    This agent can:
    - Do math (add, multiply)
    - Check weather (for NY, London, Tokyo)

    Watch the [TOOL CALLED] messages to see when tools are used!
        """)

    def _print_summary(self):
        """Print the demo summary."""
        print("\n" + "=" * 60)
        print("DEMO COMPLETE")
        print("=" * 60)
        print("""
    KEY CONCEPTS DEMONSTRATED:
    --------------------------
    1. TOOLS: Functions the agent can call (add, multiply, get_weather)
    2. LLM: The brain that decides which tools to use
    3. AGENT: Combines LLM + tools into a decision-making system
    4. EXECUTOR: Runs the agent loop until task is complete
    5. OOP DESIGN: Clean, modular, and testable code structure

    The agent:
    - Receives a question
    - Decides if a tool is needed
    - Calls the tool with appropriate arguments
    - Uses the result to formulate a response
        """)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Create agent with default configuration
    agent = LangChainAgent(
        config=AgentConfig(
            model_name="llama3.2:3b-instruct-q8_0",
            temperature=0,
            verbose=True,
        )
    )

    # Run the demo
    demo = AgentDemo(agent)
    demo.run_all_demos()
