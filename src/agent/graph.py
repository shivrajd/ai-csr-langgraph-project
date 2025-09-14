"""LangGraph supervisor-based multi-agent system.

Implements the official agent supervisor pattern from LangGraph tutorial
with research and math agents to test infinite loop fix.
Reference: https://langchain-ai.github.io/langgraph/tutorials/multi_agent/agent_supervisor/
"""

from __future__ import annotations

import os
from typing import Any, Dict, TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor

# Load environment variables
load_dotenv()


class Context(TypedDict):
    """Context parameters for the agent.

    Set these when creating assistants OR when invoking the graph.
    See: https://langchain-ai.github.io/langgraph/cloud/how-tos/configuration_cloud/
    """

    model: str
    temperature: float


# Define tools for the agents
@tool
def web_search(query: str) -> str:
    """Search the web for information."""
    # Mock implementation for testing
    return f"Here are the search results for '{query}': Mock research data about {query}. This would normally return real web search results."


@tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


def create_research_agent():
    """Create a research agent specialized in web search and information gathering."""
    research_agent = create_react_agent(
        model=init_chat_model("openai:gpt-4o-mini", temperature=0.3),
        tools=[web_search],
        prompt=(
            "You are a research specialist. Your role is to search for information "
            "and provide comprehensive answers based on your findings. "
            "Use the web_search tool to gather relevant information when needed. "
            "Be thorough and accurate in your research."
        ),
        name="research_agent"
    )
    return research_agent


def create_math_agent():
    """Create a math agent specialized in mathematical calculations."""
    math_agent = create_react_agent(
        model=init_chat_model("openai:gpt-4o-mini", temperature=0.1),
        tools=[add, multiply],
        prompt=(
            "You are a math specialist. Your role is to help with mathematical "
            "calculations and problem solving. Use the available tools for "
            "computations when needed. Be precise and clear in your explanations."
        ),
        name="math_agent"
    )
    return math_agent


def create_agent_supervisor():
    """Create a supervisor to manage research and math agents."""
    # Create the specialized agents
    research_agent = create_research_agent()
    math_agent = create_math_agent()

    # Create supervisor with proper multi-agent configuration
    supervisor = create_supervisor(
        [research_agent, math_agent],  # Pass agents as first positional argument
        model=init_chat_model("openai:gpt-4o-mini", temperature=0.3),
        prompt=(
            "You are a supervisor managing two agents:\n"
            "- a research agent. Assign research-related tasks to this agent\n"
            "- a math agent. Assign math-related tasks to this agent\n"
            "Assign work to one agent at a time, do not call agents in parallel.\n"
            "Do not do any work yourself."
        ),
        output_mode="last_message",
        add_handoff_back_messages=True,  # Enable proper handoff tracking
    )

    return supervisor


# Define the main graph
graph = create_agent_supervisor().compile(name="Multi-Agent Supervisor")