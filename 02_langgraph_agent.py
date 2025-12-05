# Suppress urllib3 LibreSSL warning (must be before any imports)
import warnings
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

"""
=============================================================================
LANGGRAPH AGENT - OOP Implementation of Graph-Based Agents
=============================================================================

This script demonstrates the LangGraph agent concept using Object-Oriented
Programming principles.

DESIGN PRINCIPLES:
------------------
1. Single Responsibility: Each class has one clear purpose
2. Encapsulation: Agent internals are hidden behind a clean interface
3. Dependency Injection: Tools and LLM can be injected for flexibility
4. Open/Closed: Easy to extend without modifying existing code

CLASS STRUCTURE:
----------------
- AgentConfig: Configuration dataclass for agent settings
- LangGraphAgent: Main agent class encapsulating the graph-based workflow

FLOW:
-----
       +----------+
       |  START   |
       +----+-----+
            |
            v
    +-------+--------+
    |  call_model    |  <-- LLM decides: use tool or respond?
    +-------+--------+
            |
            v
    +-------+--------+
    | should_continue |  <-- Conditional edge: tool_calls exist?
    +-------+--------+
           /  \
          /    \
         v      v
   +------+   +------+
   | tools |  | END  |  <-- Either call tools OR finish
   +---+---+  +------+
       |
       +-----> back to call_model (loop!)
"""

from dataclasses import dataclass, field
from typing import TypedDict, Annotated, Sequence, List, Optional
import operator

from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from tools import tools as default_tools, ToolRegistry, create_default_registry


# =============================================================================
# AGENT STATE (TypedDict for Graph State)
# =============================================================================

class AgentState(TypedDict):
    """
    The state that flows through the agent graph.

    Attributes:
        messages: List of all messages in the conversation.
                  The Annotated[..., operator.add] means new messages get
                  APPENDED to the existing list, not replaced.
    """
    messages: Annotated[Sequence[BaseMessage], operator.add]


# =============================================================================
# AGENT CONFIGURATION
# =============================================================================

@dataclass
class AgentConfig:
    """
    Configuration settings for the LangGraph agent.

    Attributes:
        model_name: The Ollama model to use
        temperature: LLM temperature (0 = deterministic)
        verbose: Whether to print debug information
        max_iterations: Maximum number of tool-call loops
    """
    model_name: str = "llama3.2:3b-instruct-q8_0"
    temperature: float = 0.0
    verbose: bool = True
    max_iterations: int = 10


# =============================================================================
# LANGGRAPH AGENT CLASS
# =============================================================================

class LangGraphAgent:
    """
    A graph-based agent using LangGraph.

    This agent processes user queries by:
    1. Receiving a question
    2. Using an LLM to decide if tools are needed
    3. Executing tools if required
    4. Looping until a final answer is ready

    Example:
        agent = LangGraphAgent()
        result = agent.run("What is 5 + 3?")
        print(result)
    """

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        tools: Optional[List[BaseTool]] = None,
        tool_registry: Optional[ToolRegistry] = None,
    ):
        """
        Initialize the LangGraph agent.

        Args:
            config: Agent configuration settings
            tools: List of tools for the agent to use
            tool_registry: Optional ToolRegistry for advanced tool management
        """
        self._config = config or AgentConfig()
        self._tool_registry = tool_registry
        self._tools = self._resolve_tools(tools)
        self._llm = self._create_llm()
        self._llm_with_tools = self._llm.bind_tools(self._tools)
        self._graph = self._build_graph()

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

    def _build_graph(self) -> StateGraph:
        """
        Build the agent workflow graph.

        Returns:
            Compiled StateGraph ready for execution
        """
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("call_model", self._call_model_node)
        workflow.add_node("tools", ToolNode(self._tools))

        # Set entry point
        workflow.set_entry_point("call_model")

        # Add conditional edges
        workflow.add_conditional_edges(
            "call_model",
            self._should_continue,
            {
                "tools": "tools",
                END: END,
            }
        )

        # Add edge from tools back to call_model (creates the loop)
        workflow.add_edge("tools", "call_model")

        return workflow.compile()

    def _call_model_node(self, state: AgentState) -> dict:
        """
        Node that calls the LLM.

        This node:
        1. Takes current messages from state
        2. Passes them to the LLM
        3. Returns the response to be added to state
        """
        if self._config.verbose:
            print("\n[NODE: call_model] Thinking...")

        messages = state["messages"]
        response = self._llm_with_tools.invoke(messages)

        if self._config.verbose:
            content_preview = response.content[:100] if response.content else '[Tool Call]'
            print(f"  LLM Response: {content_preview}...")

        return {"messages": [response]}

    def _should_continue(self, state: AgentState) -> str:
        """
        Conditional edge: Decide whether to continue with tools or end.

        Returns:
            "tools" if tool calls are detected, END otherwise
        """
        last_message = state["messages"][-1]

        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            if self._config.verbose:
                print("  [DECISION] Tool calls detected, routing to tools node")
            return "tools"

        if self._config.verbose:
            print("  [DECISION] No tool calls, finishing")
        return END

    def run(self, question: str) -> str:
        """
        Run the agent with a question and return the final answer.

        Args:
            question: The user's question

        Returns:
            The agent's final response
        """
        if self._config.verbose:
            print(f"\n{'='*60}")
            print(f"Question: {question}")
            print(f"{'='*60}")

        initial_state = {
            "messages": [HumanMessage(content=question)]
        }

        final_state = self._graph.invoke(initial_state)
        final_message = final_state["messages"][-1]

        return final_message.content

    @property
    def config(self) -> AgentConfig:
        """Get the agent configuration."""
        return self._config

    @property
    def tools(self) -> List[BaseTool]:
        """Get the list of tools available to the agent."""
        return self._tools

    @staticmethod
    def print_graph_structure():
        """Print an ASCII representation of the graph structure."""
        print("""
    LANGGRAPH AGENT STRUCTURE:
    ==========================

                    +---------+
                    |  START  |
                    +----+----+
                         |
                         v
                +--------+--------+
                |   call_model    |  <-- LLM processes messages
                +--------+--------+
                         |
                         v
                +--------+--------+
                | should_continue |  <-- Check for tool calls
                +--------+--------+
                        / \\
                       /   \\
            tool_calls?     no tool_calls?
                     /           \\
                    v             v
            +-------+      +-----+
            | tools | ---->| END |
            +-------+      +-----+
                 |
                 +---> (loops back to call_model)
        """)


# =============================================================================
# DEMO RUNNER CLASS
# =============================================================================

class AgentDemo:
    """
    Demo runner for showcasing the LangGraph agent.

    This class encapsulates the demo logic, making it easy to run
    different test scenarios.
    """

    def __init__(self, agent: LangGraphAgent):
        """
        Initialize the demo runner.

        Args:
            agent: The LangGraphAgent instance to demo
        """
        self._agent = agent

    def run_test(self, name: str, question: str) -> str:
        """
        Run a single test case.

        Args:
            name: Name of the test
            question: Question to ask the agent

        Returns:
            The agent's response
        """
        print(f"\n--- {name} ---")
        result = self._agent.run(question)
        print(f"\n>>> Final Answer: {result}")
        return result

    def run_all_demos(self):
        """Run all demonstration test cases."""
        print("=" * 60)
        print("LANGGRAPH AGENT DEMO (OOP VERSION)")
        print("=" * 60)

        # Show graph structure
        LangGraphAgent.print_graph_structure()

        # Run test cases
        self.run_test("Test 1: Math Question", "What is 7 multiplied by 8?")
        self.run_test("Test 2: Multi-Step Math", "What is 5 plus 3, then multiply that result by 2?")
        self.run_test("Test 3: Weather Query", "What's the weather like in Tokyo?")

        self._print_summary()

    def _print_summary(self):
        """Print the demo summary."""
        print("\n" + "=" * 60)
        print("DEMO COMPLETE")
        print("=" * 60)
        print("""
    KEY TAKEAWAYS:
    --------------
    1. LangGraph gives you EXPLICIT CONTROL over the agent flow
    2. State is SHARED across all nodes
    3. Conditional edges let you create BRANCHING LOGIC
    4. The loop (tools -> call_model) continues until no more tool calls
    5. OOP design makes the code modular and testable
        """)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Create agent with default configuration
    agent = LangGraphAgent(
        config=AgentConfig(
            model_name="llama3.2:3b-instruct-q8_0",
            temperature=0,
            verbose=True,
        )
    )

    # Run the demo
    demo = AgentDemo(agent)
    demo.run_all_demos()
