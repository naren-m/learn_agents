# Suppress urllib3 LibreSSL warning (must be before any imports)
import warnings
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

"""
=============================================================================
LANGGRAPH AGENT - Understanding Graph-Based Agents
=============================================================================

This script demonstrates the same agent concept using LangGraph.

LANGGRAPH vs LANGCHAIN AGENTS:
------------------------------
LangChain Agent (01_langchain_agent.py):
  - Uses a predefined ReAct loop
  - Less control over the flow
  - Simpler to set up
  - Good for standard use cases

LangGraph Agent (this file):
  - You define the graph (nodes + edges) explicitly
  - Full control over the agent flow
  - Can create complex, multi-step workflows
  - Better for custom agent behaviors

KEY CONCEPTS IN LANGGRAPH:
--------------------------
1. STATE    - A shared data structure that flows through the graph
2. NODES    - Functions that process/modify the state
3. EDGES    - Connections between nodes (can be conditional!)
4. GRAPH    - The complete workflow definition

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

from typing import TypedDict, Annotated, Sequence
import operator

from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# Import shared tools from tools.py
from tools import tools


# =============================================================================
# STEP 1: Define the State
# =============================================================================
# State is a TypedDict that holds all the data flowing through the graph.
# The key insight: messages accumulate (operator.add) as the agent runs.

class AgentState(TypedDict):
    """
    The state of our agent.

    - messages: A list of all messages in the conversation
                The Annotated[..., operator.add] means new messages get
                APPENDED to the existing list, not replaced.
    """
    messages: Annotated[Sequence[BaseMessage], operator.add]


# =============================================================================
# STEP 2: Create the LLM with Tool Binding
# =============================================================================
# In LangGraph, we "bind" tools to the LLM. This tells the LLM what tools
# are available and how to call them.

llm = ChatOllama(
    model="llama3.2:3b-instruct-q8_0",  # Has native tool calling support
    temperature=0,
)

# Bind tools to the LLM - this is different from LangChain!
# The LLM now knows about the tools and can generate tool calls.
llm_with_tools = llm.bind_tools(tools)


# =============================================================================
# STEP 3: Define Graph Nodes (Functions)
# =============================================================================

def call_model(state: AgentState) -> dict:
    """
    NODE 1: Call the LLM

    This node:
    1. Takes the current messages from state
    2. Passes them to the LLM
    3. Gets a response (which might include tool calls!)
    4. Returns the new message to be added to state
    """
    print("\n[NODE: call_model] Thinking...")
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    print(f"  LLM Response: {response.content[:100] if response.content else '[Tool Call]'}...")
    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    """
    CONDITIONAL EDGE: Decide what to do next

    Looks at the last message:
    - If it has tool_calls -> route to "tools" node
    - Otherwise -> route to END

    This is the "decision point" in our graph.
    """
    last_message = state["messages"][-1]

    # Check if the LLM wants to use a tool
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        print(f"  [DECISION] Tool calls detected, routing to tools node")
        return "tools"

    print(f"  [DECISION] No tool calls, finishing")
    return END


# =============================================================================
# STEP 4: Build the Graph
# =============================================================================

# Create the graph with our state schema
workflow = StateGraph(AgentState)

# Add nodes (the "boxes" in our flowchart)
workflow.add_node("call_model", call_model)
workflow.add_node("tools", ToolNode(tools))  # ToolNode is a prebuilt helper

# Set the entry point (where execution starts)
workflow.set_entry_point("call_model")

# Add conditional edges (the "arrows" in our flowchart)
# From "call_model", check should_continue to decide where to go
workflow.add_conditional_edges(
    "call_model",      # Source node
    should_continue,   # Function that returns the next node name
    {
        "tools": "tools",  # If function returns "tools", go to tools node
        END: END,          # If function returns END, stop execution
    }
)

# After tools run, always go back to call_model (this creates the loop!)
workflow.add_edge("tools", "call_model")

# Compile the graph into a runnable
agent = workflow.compile()


# =============================================================================
# STEP 5: Helper Function to Run the Agent
# =============================================================================

def run_agent(question: str) -> str:
    """
    Run the agent with a question and return the final answer.

    This shows the complete flow:
    1. Create initial state with user's question
    2. Run the graph (it loops until should_continue returns END)
    3. Extract the final response
    """
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print(f"{'='*60}")

    # Create initial state with the user's message
    initial_state = {
        "messages": [HumanMessage(content=question)]
    }

    # Run the agent graph
    # The graph will execute nodes and follow edges until it reaches END
    final_state = agent.invoke(initial_state)

    # The last message in the state is the agent's final response
    final_message = final_state["messages"][-1]
    return final_message.content


# =============================================================================
# STEP 6: Visualize the Graph (Optional but Educational!)
# =============================================================================

def print_graph_structure():
    """Print an ASCII representation of our graph structure."""
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
# STEP 7: Run the Agent!
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("LANGGRAPH AGENT DEMO")
    print("=" * 60)

    # Show the graph structure
    print_graph_structure()

    # Test 1: Simple math
    print("\n--- Test 1: Math Question ---")
    result = run_agent("What is 7 multiplied by 8?")
    print(f"\n>>> Final Answer: {result}")

    # Test 2: Multi-step reasoning
    print("\n--- Test 2: Multi-Step Math ---")
    result = run_agent("What is 5 plus 3, then multiply that result by 2?")
    print(f"\n>>> Final Answer: {result}")

    # Test 3: Weather query
    print("\n--- Test 3: Weather Query ---")
    result = run_agent("What's the weather like in Tokyo?")
    print(f"\n>>> Final Answer: {result}")

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
    5. You can add MORE NODES for complex workflows (e.g., human approval)
    """)
