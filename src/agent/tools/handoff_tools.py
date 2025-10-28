"""
Bot-to-Human Handoff Tools for Customer Service Escalation.

Implements tools for detecting and managing escalations from AI bot to human agents.
Integrates with Chatwoot via the integration server to update conversation status.
"""

import logging
from typing import Literal
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def detect_escalation_need(
    customer_message: str,
    sentiment: Literal["frustrated", "angry", "urgent", "neutral"] = "neutral",
    issue_complexity: Literal["simple", "moderate", "complex"] = "simple"
) -> str:
    """
    Analyze if the customer's message requires escalation to a human agent.

    USE THIS TOOL to determine if handoff to human is needed based on:
    - Explicit requests for human agent ("speak to a human", "talk to someone")
    - Customer frustration or anger signals
    - Complex issues beyond bot capabilities
    - Repeated failed attempts to resolve issue
    - Complaints or sensitive matters

    Args:
        customer_message: The customer's message to analyze
        sentiment: Customer's emotional state (frustrated, angry, urgent, neutral)
        issue_complexity: Complexity of the issue (simple, moderate, complex)

    Returns:
        Analysis result indicating if escalation is needed
    """
    # Explicit human request keywords - use specific phrases to avoid false positives
    human_keywords = [
        "human", "human agent", "real agent", "live agent", "speak to an agent",
        "talk to an agent", "representative", "manager",
        "supervisor", "real person", "actual person"
    ]

    # Frustration/anger indicators
    frustration_keywords = [
        "frustrated", "angry", "ridiculous", "useless", "terrible",
        "awful", "horrible", "can't believe", "fed up", "enough"
    ]

    message_lower = customer_message.lower()

    # Check for explicit human request
    has_human_request = any(keyword in message_lower for keyword in human_keywords)

    # Check for frustration
    has_frustration = any(keyword in message_lower for keyword in frustration_keywords)

    # Filter out false positives - questions about order lookup methods
    # These are normal informational questions, not escalation requests
    order_method_questions = [
        "email address", "use email", "share email", "provide email",
        "send email", "enter email", "give email", "email instead",
        "order number", "order id", "give order", "provide order"
    ]

    is_order_method_question = any(phrase in message_lower for phrase in order_method_questions)

    # Don't escalate if customer is just asking about how to provide information
    # UNLESS they're also expressing frustration
    if is_order_method_question and not has_frustration and sentiment == "neutral":
        return "NO_ESCALATION_NEEDED|Customer asking about order lookup methods - not an escalation"

    # Determine escalation need
    needs_escalation = (
        has_human_request or
        sentiment in ["angry", "frustrated"] or
        (has_frustration and sentiment != "neutral") or
        issue_complexity == "complex"
    )

    if needs_escalation:
        # Determine urgency
        if sentiment == "angry" or "now" in message_lower or "immediately" in message_lower:
            urgency = "high"
        elif has_human_request or sentiment == "frustrated":
            urgency = "medium"
        else:
            urgency = "low"

        # Determine reason
        if has_human_request:
            reason = "Explicit request for human agent"
        elif sentiment == "angry":
            reason = "Customer is angry and needs immediate attention"
        elif sentiment == "frustrated":
            reason = "Customer is frustrated with current assistance"
        elif issue_complexity == "complex":
            reason = "Issue complexity requires human expertise"
        else:
            reason = "Customer needs human assistance"

        return f"ESCALATION_NEEDED|Reason: {reason}|Urgency: {urgency}"
    else:
        return "NO_ESCALATION_NEEDED|Customer query can be handled by bot"


@tool
def request_human_handoff(
    escalation_reason: str,
    urgency_level: Literal["high", "medium", "low"] = "medium",
    customer_context: str = ""
) -> str:
    """
    Initiate handoff from bot to human agent.

    USE THIS TOOL when escalation has been determined necessary to:
    - Transfer conversation to human agent queue
    - Update Chatwoot conversation status
    - Provide context to human agent

    Args:
        escalation_reason: Why the handoff is needed
        urgency_level: Priority level (high, medium, low)
        customer_context: Brief context for the human agent

    Returns:
        Handoff signal with customer-facing message
    """
    # Generate customer-facing message based on urgency and reason
    if urgency_level == "high":
        if "angry" in escalation_reason.lower():
            customer_message = "I completely understand your frustration. Let me connect you with one of our team members right away who can give this their immediate attention."
        else:
            customer_message = "I'm connecting you with one of our specialists right now who can assist you directly."
    elif urgency_level == "medium":
        if "request" in escalation_reason.lower():
            customer_message = "Of course! I'm transferring you to one of our team members who will be happy to help you personally."
        else:
            customer_message = "I'd like to connect you with one of our team members who can better assist you with this."
    else:
        customer_message = "Let me transfer you to one of our team members for further assistance."

    # Format response for integration server detection
    # Format: HANDOFF_REQUESTED: {reason} | Urgency: {level} | Status: {customer_message}
    handoff_signal = f"HANDOFF_REQUESTED: {escalation_reason} | Urgency: {urgency_level} | Status: {customer_message}"

    logger.info(f"🔄 Human handoff requested: {escalation_reason} (Urgency: {urgency_level})")

    return handoff_signal


# Available handoff tools for the handoff agent
HANDOFF_TOOLS = [
    detect_escalation_need,
    request_human_handoff
]

__all__ = [
    "detect_escalation_need",
    "request_human_handoff",
    "HANDOFF_TOOLS"
]
