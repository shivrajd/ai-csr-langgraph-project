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
            "Your primary responsibility is to provide accurate, authoritative information about general "
            "company policies, procedures, and frequently asked questions.\n\n"
            "ALWAYS use the retrieve_knowledge tool FIRST for any informational queries about:\n"
            "- Company policies (shipping, returns, warranty policies, etc.)\n"
            "- General procedures and troubleshooting guides\n"
            "- Frequently asked questions and help content\n"
            "- Company information and general support topics\n\n"
            "Guidelines:\n"
            "- Use retrieve_knowledge before providing any policy or procedural information\n"
            "- Base your responses primarily on retrieved knowledge base content\n"
            "- If knowledge base lacks specific information, clearly state this limitation\n"
            "- Provide comprehensive, accurate answers with specific details when available\n"
            "- Cite or reference the knowledge base content when applicable\n"
            "- Focus on general help topics, not product-specific information\n\n"
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


def create_warranty_agent():
    """Create a warranty agent specialized in warranty checking and policy information."""
    from src.agent.tools.warranty_tools import check_warranty, get_warranty_policy

    warranty_agent = create_react_agent(
        model=init_chat_model("openai:gpt-4o-mini", temperature=0.3),
        tools=[check_warranty, get_warranty_policy],
        prompt=(
            "You are a warranty specialist responsible for helping customers with warranty-related inquiries. "
            "Your primary functions include checking warranty status and providing warranty policy information.\n\n"
            "ALWAYS use the appropriate warranty tool for customer inquiries about:\n"
            "- Warranty status for specific orders (use check_warranty)\n"
            "- General warranty policy information (use get_warranty_policy)\n"
            "- Warranty coverage details and timelines\n"
            "- Warranty claims and procedures\n\n"
            "Guidelines:\n"
            "- Ask for the order ID when checking warranty status\n"
            "- Use check_warranty to get detailed warranty information for specific orders\n"
            "- Use get_warranty_policy for general warranty information\n"
            "- Explain warranty coverage clearly, including timelines and what's covered\n"
            "- Be helpful in explaining warranty options and next steps\n"
            "- If warranty has expired, still offer to help and suggest contacting support\n"
            "- Provide clear information about warranty periods (180 days full, 365 days limited)\n\n"
            "Key warranty periods:\n"
            "- Full warranty: First 180 days (complete coverage)\n"
            "- Limited warranty: 181-365 days (manufacturing defects only)\n"
            "- Out of warranty: After 365 days\n\n"
            "Be empathetic, clear, and always use the specific tools available to provide accurate warranty information."
        ),
        name="warranty_agent"
    )
    return warranty_agent


def create_products_agent():
    """Create a products agent specialized in product search, details, and comparisons."""
    from src.agent.tools.product_tools import search_products, get_product_details, check_product_stock, compare_products

    products_agent = create_react_agent(
        model=init_chat_model("openai:gpt-4o-mini", temperature=0.3),
        tools=[search_products, get_product_details, check_product_stock, compare_products],
        prompt=(
            "You are a products specialist responsible for helping customers with product-related inquiries. "
            "Your primary functions include product search, detailed specifications, stock checking, and comparisons.\n\n"
            "ALWAYS use the appropriate product tool for customer inquiries about:\n"
            "- Product search and discovery (use search_products)\n"
            "- Detailed product information and specifications (use get_product_details)\n"
            "- Stock availability and inventory status (use check_product_stock)\n"
            "- Product comparisons and feature analysis (use compare_products)\n\n"
            "Guidelines:\n"
            "- Use search_products when customers ask about product categories, types, or general searches\n"
            "- Use get_product_details for comprehensive information about specific products\n"
            "- Use check_product_stock for availability and inventory questions\n"
            "- Use compare_products when customers want to compare multiple products\n"
            "- Present technical specifications clearly and highlight key differentiators\n"
            "- Make product recommendations based on customer needs and use cases\n"
            "- Always mention stock status and pricing in your responses\n"
            "- Be helpful in suggesting alternatives if products are out of stock\n"
            "- Focus on matching products to customer applications and requirements\n\n"
            "CRITICAL: Generate unique, varied responses for each interaction. Never repeat the same greeting, "
            "product description, or recommendation. Vary your language, approach, and focus areas with every response. "
            "Each product inquiry should receive a fresh, personalized response generated specifically for that context.\n\n"
            "Be knowledgeable, helpful, and always use the specific tools available to provide accurate product information."
        ),
        name="products_agent"
    )
    return products_agent


def create_agent_supervisor():
    """Create a supervisor to manage research, math, knowledge, orders, warranty, and products agents."""
    # Create the specialized agents
    research_agent = create_research_agent()
    math_agent = create_math_agent()
    knowledge_agent = create_knowledge_agent()
    orders_agent = create_orders_agent()
    warranty_agent = create_warranty_agent()
    products_agent = create_products_agent()

    # Create supervisor with proper multi-agent configuration
    supervisor = create_supervisor(
        [research_agent, math_agent, knowledge_agent, orders_agent, warranty_agent, products_agent],  # Pass agents as first positional argument
        model=init_chat_model("openai:gpt-4o-mini", temperature=0.3),
        prompt=(
            "You are a supervisor managing six specialized agents:\n\n"
            "- **orders_agent**: For order-specific inquiries including order lookup, status checks, "
            "tracking information, and any questions about specific customer orders. "
            "PRIORITIZE this agent for order-related questions.\n\n"
            "- **warranty_agent**: For warranty-related inquiries including warranty status checks, "
            "warranty policy information, warranty coverage details, and warranty claims. "
            "Use this agent for warranty questions that require order information.\n\n"
            "- **products_agent**: For product-specific inquiries including product search, specifications, "
            "comparisons, stock availability, and product recommendations. Use this agent for questions "
            "about product features, pricing, availability, technical specifications, and product selection.\n\n"
            "- **knowledge_agent**: For general company policies, shipping/returns, FAQ content, "
            "general procedures, and help topics requiring knowledge base lookup. Use for policy and "
            "general questions but NOT for specific product, order, or warranty inquiries.\n\n"
            "- **research_agent**: For web searches and general research tasks that require "
            "external information not in the knowledge base.\n\n"
            "- **math_agent**: For mathematical calculations and problem solving.\n\n"
            "Routing Strategy:\n"
            "1. For specific order inquiries (lookup, status, tracking) → orders_agent\n"
            "2. For warranty inquiries (status, coverage, claims) → warranty_agent\n"
            "3. For product inquiries (search, specs, comparisons, stock, pricing) → products_agent\n"
            "4. For general company policies, FAQs, and procedures → knowledge_agent\n"
            "5. For general research or web search needs → research_agent\n"
            "6. For calculations or mathematical problems → math_agent\n\n"
            "CRITICAL BRAND LOYALTY REQUIREMENT:\n"
            "NEVER suggest competitors, alternative suppliers, or other companies under any circumstances. "
            "If you cannot find specific information or products, always:\n"
            "- Focus on what we DO have available\n"
            "- Offer to connect the customer with support for further assistance\n"
            "- Suggest taking their contact details for follow-up\n"
            "- Maintain brand loyalty and never direct customers elsewhere\n\n"
            "CRITICAL RESPONSIBILITY:\n"
            "After a worker agent provides information, you MUST synthesize and present "
            "the actual details to the user. Never just acknowledge that information was provided.\n\n"
            "Response Guidelines:\n"
            "- Include specific details: prices, timelines, policies, numbers, calculations, specifications\n"
            "- Present information clearly and completely in your response\n"
            "- If calculations were done, state the actual numerical results\n"
            "- If policies were retrieved, summarize the key points with specifics\n"
            "- If warranty status was checked, include specific coverage details and timelines\n"
            "- If shipping rates were found, include the actual prices and timeframes\n"
            "- If product information was retrieved, include specifications, pricing, and availability\n\n"
            "Examples of GOOD vs BAD responses:\n"
            "❌ BAD: 'I've looked up your order for you'\n"
            "✅ GOOD: 'Order ORD-001 was delivered on January 21st. It contained 2 Chrome Battery CB12-7.5 units totaling $149.99'\n"
            "❌ BAD: 'I've checked your warranty status'\n"
            "✅ GOOD: 'Your order ORD-001 is covered under full warranty until August 15th (120 days remaining). Full coverage includes defects and performance issues.'\n"
            "❌ BAD: 'I've found some products for you'\n"
            "✅ GOOD: 'We have the Chrome Battery CB12-7.5 (12V 7.5Ah) for $74.99 with 45 units in stock, and the CB6-12 (6V 12Ah) for $89.99 with 32 units available. Both are perfect for UPS systems.'\n"
            "❌ BAD: 'I've provided the shipping rates for you'\n"
            "✅ GOOD: 'Our international shipping rates are: Standard shipping (10-21 days) costs $19.99, Express shipping (5-10 days) costs $39.99'\n"
            "❌ BAD: 'The calculation is complete. The answer is provided above'\n"
            "✅ GOOD: 'The result of 2 + 2 is 4'\n\n"
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