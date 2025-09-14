"""LangGraph supervisor-based chat agent.

Uses the langgraph-supervisor library for a simple conversational AI agent
that can chat naturally with users. Designed to be easily extensible with
additional workers in the future.
"""

from __future__ import annotations

import os
from typing import Any, Dict, TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
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


def create_chat_agent():
    """Create a simple conversational agent using ReAct pattern."""
    # TODO(human): Simplify by passing model string directly to create_react_agent
    # The create_react_agent can accept model strings and handle initialization internally

    # Initialize the language model using init_chat_model
    llm = init_chat_model(
        "openai:gpt-4o-mini",
        temperature=0.7
    )

    # Create a simple chat agent with no tools for now
    chat_agent = create_react_agent(
        model=llm,
        tools=[],  # No tools for simple chat - can be extended later
        prompt=(
            "You are a helpful AI assistant. Have natural conversations with users. "
            "Be friendly, informative, and engaging. Respond thoughtfully to questions "
            "and provide helpful information when requested."
        ),
        name="chat_assistant"
    )

    return chat_agent


def create_chat_supervisor():
    """Create a supervisor to manage the chat agent."""
    # Initialize the language model for the supervisor using init_chat_model
    supervisor_llm = init_chat_model(
        "openai:gpt-4o-mini",
        temperature=0.3  # Lower temperature for more consistent routing
    )

    # Create the chat agent
    chat_agent = create_chat_agent()

    # Create supervisor with a single chat agent
    supervisor = create_supervisor(
        agents=[chat_agent],
        model=supervisor_llm,
        prompt=(
            "You are a supervisor managing a chat assistant. "
            "For any user message, delegate to the chat_assistant to provide "
            "a helpful and natural response. The chat assistant is designed "
            "to handle general conversation and questions."
        ),
        output_mode="last_message",  # Only return the final response
        add_handoff_messages=False,  # Keep responses clean for simple chat
    )

    return supervisor


# Define the main graph
graph = create_chat_supervisor().compile(name="Chat Agent with Supervisor")