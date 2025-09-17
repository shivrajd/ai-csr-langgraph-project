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
# Removed MemorySaver - LangGraph API handles persistence automatically

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


def create_knowledge_agent():
    """Create a knowledge agent specialized in RAG-based knowledge retrieval."""
    from src.agent.tools.rag_tools import retrieve_knowledge

    knowledge_agent = create_react_agent(
        model=init_chat_model("openai:gpt-4o-mini", temperature=0.3),
        tools=[retrieve_knowledge],
        prompt=(
            "You are a knowledge specialist with access to the company's comprehensive knowledge base. "
            "Your primary responsibility is to provide accurate, authoritative information about products, "
            "policies, procedures, and frequently asked questions.\n\n"
            "ALWAYS use the retrieve_knowledge tool FIRST for any informational queries about:\n"
            "- Product specifications, features, and details\n"
            "- Company policies (shipping, returns, warranty, etc.)\n"
            "- Technical documentation and troubleshooting\n"
            "- Frequently asked questions and procedures\n\n"
            "Guidelines:\n"
            "- Use retrieve_knowledge before providing any product or policy information\n"
            "- Base your responses primarily on retrieved knowledge base content\n"
            "- If knowledge base lacks specific information, clearly state this limitation\n"
            "- Provide comprehensive, accurate answers with specific details when available\n"
            "- Cite or reference the knowledge base content when applicable\n\n"
            "Be thorough, precise, and always prioritize knowledge base information over general knowledge."
        ),
        name="knowledge_agent"
    )
    return knowledge_agent


def create_orders_agent():
    """Create an orders management agent specialized in order lookup and status tracking."""
    from src.agent.tools.order_tools import lookup_order, get_order_status, get_tracking_number

    orders_agent = create_react_agent(
        model=init_chat_model("openai:gpt-4o-mini", temperature=0.3),
        tools=[lookup_order, get_order_status, get_tracking_number],
        prompt=(
            "You are an orders specialist responsible for helping customers with order-related inquiries. "
            "Your primary functions include order lookups, status updates, and tracking information.\n\n"
            "ALWAYS use the appropriate order tool for customer inquiries about:\n"
            "- Order details and information (use lookup_order)\n"
            "- Current order status (use get_order_status)\n"
            "- Shipping and tracking information (use get_tracking_number)\n\n"
            "Guidelines:\n"
            "- Ask for the order ID if not provided by the customer\n"
            "- Use lookup_order for comprehensive order details\n"
            "- Use get_order_status for quick status checks\n"
            "- Use get_tracking_number specifically for tracking requests\n"
            "- Present information clearly and offer additional help when appropriate\n"
            "- If an order isn't found, suggest double-checking the order ID format\n"
            "- Be empathetic and helpful, especially for issues like cancellations or delays\n\n"
            "Be accurate, helpful, and always use the specific tools available to provide precise order information."
        ),
        name="orders_agent"
    )
    return orders_agent


def create_agent_supervisor():
    """Create a supervisor to manage research, math, knowledge, and orders agents."""
    # Create the specialized agents
    research_agent = create_research_agent()
    math_agent = create_math_agent()
    knowledge_agent = create_knowledge_agent()
    orders_agent = create_orders_agent()

    # Create supervisor with proper multi-agent configuration
    supervisor = create_supervisor(
        [research_agent, math_agent, knowledge_agent, orders_agent],  # Pass agents as first positional argument
        model=init_chat_model("openai:gpt-4o-mini", temperature=0.3),
        prompt=(
            "You are a supervisor managing four specialized agents:\n\n"
            "- **orders_agent**: For order-specific inquiries including order lookup, status checks, "
            "tracking information, and any questions about specific customer orders. "
            "PRIORITIZE this agent for order-related questions.\n\n"
            "- **knowledge_agent**: For product information, company policies, shipping/returns, "
            "warranty details, FAQ content, and any questions requiring company knowledge base lookup. "
            "Use for general policy questions but NOT specific order inquiries.\n\n"
            "- **research_agent**: For web searches and general research tasks that require "
            "external information not in the knowledge base.\n\n"
            "- **math_agent**: For mathematical calculations and problem solving.\n\n"
            "Routing Strategy:\n"
            "1. For specific order inquiries (lookup, status, tracking) → orders_agent\n"
            "2. For customer service questions about products, policies, or procedures → knowledge_agent\n"
            "3. For general research or web search needs → research_agent\n"
            "4. For calculations or mathematical problems → math_agent\n\n"
            "CRITICAL RESPONSIBILITY:\n"
            "After a worker agent provides information, you MUST synthesize and present "
            "the actual details to the user. Never just acknowledge that information was provided.\n\n"
            "Response Guidelines:\n"
            "- Include specific details: prices, timelines, policies, numbers, calculations\n"
            "- Present information clearly and completely in your response\n"
            "- If calculations were done, state the actual numerical results\n"
            "- If policies were retrieved, summarize the key points with specifics\n"
            "- If shipping rates were found, include the actual prices and timeframes\n\n"
            "Examples of GOOD vs BAD responses:\n"
            "❌ BAD: 'I've looked up your order for you'\n"
            "✅ GOOD: 'Order ORD-001 was delivered on January 21st. It contained 2 Chrome Battery CB12-7.5 units totaling $149.99'\n"
            "❌ BAD: 'I've provided the shipping rates for you'\n"
            "✅ GOOD: 'Our international shipping rates are: Standard shipping (10-21 days) costs $19.99, Express shipping (5-10 days) costs $39.99'\n"
            "❌ BAD: 'The calculation is complete. The answer is provided above'\n"
            "✅ GOOD: 'The result of 2 + 2 is 4'\n"
            "❌ BAD: 'I've checked your tracking information'\n"
            "✅ GOOD: 'Your order ORD-002 tracking number is 1Z999AA1012345676 with UPS. Expected delivery is January 25th'\n\n"
            "Always assign work to ONE agent at a time. Do not call agents in parallel.\n"
            "Delegate to the appropriate specialist, then synthesize their response with full details."
        ),
        output_mode="last_message",
        add_handoff_back_messages=True,  # Enable proper handoff tracking
    )

    return supervisor


# Define the main graph - persistence is handled automatically by LangGraph API
graph = create_agent_supervisor().compile(
    name="Multi-Agent Supervisor"
)