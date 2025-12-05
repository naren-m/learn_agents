# Suppress urllib3 LibreSSL warning (must be before any imports)
import warnings
warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

"""
=============================================================================
LANGCHAIN TOOL-CALLING AGENT - Understanding Agents Step by Step
=============================================================================

This script demonstrates the fundamental concepts of an AI agent using LangChain.

WHAT IS AN AGENT?
-----------------
An agent is an LLM that can:
1. REASON about what to do (think about the problem)
2. ACT by calling tools (execute actions)
3. OBSERVE the results (see what happened)
4. REPEAT until the task is complete

KEY COMPONENTS:
---------------
1. LLM (Brain)        - The AI model that thinks and decides
2. Tools (Hands)      - Functions the agent can call to interact with the world
3. Agent (Controller) - Orchestrates the LLM and tools
4. AgentExecutor      - Runs the agent in a loop until task is complete

FLOW:
-----
User Question --> Agent --> LLM thinks --> Decides to use tool -->
Tool executes --> Result returned --> LLM thinks again -->
Either use another tool OR give final answer

NOTE: This uses the modern "tool-calling" agent which works better with local
models like Ollama compared to the older ReAct text-parsing approach.
"""

from langchain_ollama import ChatOllama
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Import shared tools from tools.py
from tools import tools


# =============================================================================
# STEP 1: Create the LLM (Brain)
# =============================================================================
# Using Ollama to run a local LLM. No API keys needed!
# The llama3.2 model has good native tool calling support.

llm = ChatOllama(
    model="llama3.2:3b-instruct-q8_0",  # Has native tool calling support
    temperature=0,  # 0 = deterministic, predictable responses
)


# =============================================================================
# STEP 2: Create the Prompt Template
# =============================================================================
# This is the "instruction manual" that tells the LLM how to be an agent.
# The tool-calling agent uses a chat-style prompt with message placeholders.

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful assistant that can use tools to answer questions.
When you need to perform calculations, use the appropriate math tools.
When asked about weather, use the get_weather tool.
Always use tools when they are relevant to the question.
After getting tool results, provide a natural language response to the user."""
    ),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),  # Where tool calls/results go
])


# =============================================================================
# STEP 3: Create the Agent
# =============================================================================
# The tool-calling agent uses native function calling (JSON-based)
# instead of text parsing. This is more reliable with modern LLMs.

agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
)


# =============================================================================
# STEP 4: Create the Agent Executor
# =============================================================================
# The executor runs the agent in a loop:
# 1. Agent decides what to do
# 2. If tool call: execute tool, feed result back to agent
# 3. If final answer: return to user
# 4. Repeat until done or max iterations reached

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,      # Set to True to see the agent's thinking process!
    max_iterations=10, # Safety limit to prevent infinite loops
    handle_parsing_errors=True,  # Gracefully handle if LLM outputs bad format
)


# =============================================================================
# STEP 5: Run the Agent!
# =============================================================================

def run_test(name: str, question: str):
    """Helper function to run a test and display results."""
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"Question: {question}")
    print(f"{'='*60}")
    try:
        result = agent_executor.invoke({"input": question})
        print(f"\n>>> Final Answer: {result['output']}")
        return result
    except Exception as e:
        print(f"\n>>> Error: {e}")
        return None


if __name__ == "__main__":
    print("=" * 60)
    print("LANGCHAIN TOOL-CALLING AGENT DEMO")
    print("=" * 60)
    print("""
    This agent can:
    - Do math (add, multiply)
    - Check weather (for NY, London, Tokyo)

    Watch the [TOOL CALLED] messages to see when tools are used!
    """)

    # Test 1: Simple math (should use multiply tool)
    run_test("Simple Multiplication", "What is 7 multiplied by 8?")

    # Test 2: Multi-step reasoning (should use multiple tools)
    run_test("Multi-Step Math", "What is 5 plus 3, then multiply that result by 2?")

    # Test 3: Weather query
    run_test("Weather Query", "What's the weather like in Tokyo?")

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

    The agent:
    - Receives a question
    - Decides if a tool is needed
    - Calls the tool with appropriate arguments
    - Uses the result to formulate a response
    """)
