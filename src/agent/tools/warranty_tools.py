"""
Warranty Management Tools for the Warranty Agent.

This module provides warranty checking functionality that integrates with the orders agent
to determine warranty status based on order date and product information.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Warranty thresholds (in days)
FULL_WARRANTY_DAYS = 180
LIMITED_WARRANTY_DAYS = 365

def _parse_order_date(date_str: str) -> datetime:
    """Parse order date string to datetime object."""
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        logger.error(f"Invalid date format: {date_str}")
        raise

def _calculate_warranty_status(order_date: datetime) -> Dict[str, Any]:
    """Calculate warranty status based on order date."""
    today = datetime.now()
    days_since_order = (today - order_date).days

    if days_since_order <= FULL_WARRANTY_DAYS:
        status = "full_warranty"
        coverage = "Full warranty coverage"
        description = "Your order is covered under full warranty with complete protection for defects and performance issues."
        days_remaining = FULL_WARRANTY_DAYS - days_since_order
    elif days_since_order <= LIMITED_WARRANTY_DAYS:
        status = "limited_warranty"
        coverage = "Limited warranty coverage"
        description = "Your order has limited warranty coverage for manufacturing defects only."
        days_remaining = LIMITED_WARRANTY_DAYS - days_since_order
    else:
        status = "out_of_warranty"
        coverage = "Out of warranty"
        description = "Your order is no longer covered under warranty."
        days_remaining = 0

    return {
        "status": status,
        "coverage": coverage,
        "description": description,
        "days_since_order": days_since_order,
        "days_remaining": max(0, days_remaining),
        "full_warranty_expires": order_date + timedelta(days=FULL_WARRANTY_DAYS),
        "limited_warranty_expires": order_date + timedelta(days=LIMITED_WARRANTY_DAYS)
    }

@tool
def check_warranty(order_id: str) -> str:
    """
    Check warranty status for an order by looking up order details and calculating warranty coverage.

    This tool first retrieves order information, then determines warranty status based on
    the order date and current warranty thresholds.

    Args:
        order_id: The order ID to check warranty for (e.g., "ORD-001", "ORD-002")

    Returns:
        Detailed warranty status information or error message if order not found
    """
    try:
        logger.info(f"Checking warranty for order: {order_id}")

        # Import the order lookup tool from orders agent
        from src.agent.tools.order_tools import TEST_ORDERS

        # Normalize order ID
        order_id = order_id.strip().upper()

        if order_id not in TEST_ORDERS:
            return f"Order {order_id} not found. Please verify the order ID and try again. To check warranty status, I need to first locate your order information."

        order = TEST_ORDERS[order_id]
        order_date_str = order.get('order_date')

        if not order_date_str:
            return f"Order {order_id} found but order date information is missing. Cannot determine warranty status without order date."

        # Parse order date and calculate warranty status
        order_date = _parse_order_date(order_date_str)
        warranty_info = _calculate_warranty_status(order_date)

        # Build comprehensive warranty response
        response_parts = []
        response_parts.append(f"Warranty Status for Order {order_id}:")
        response_parts.append(f"Order Date: {order_date_str}")
        response_parts.append(f"Days Since Order: {warranty_info['days_since_order']} days")
        response_parts.append(f"Warranty Status: {warranty_info['coverage']}")
        response_parts.append(f"Description: {warranty_info['description']}")

        # Add warranty timeline information
        if warranty_info['status'] == 'full_warranty':
            response_parts.append(f"Full Warranty Valid Until: {warranty_info['full_warranty_expires'].strftime('%Y-%m-%d')} ({warranty_info['days_remaining']} days remaining)")
            response_parts.append(f"Limited Warranty Valid Until: {warranty_info['limited_warranty_expires'].strftime('%Y-%m-%d')}")
        elif warranty_info['status'] == 'limited_warranty':
            response_parts.append(f"Full Warranty Expired: {warranty_info['full_warranty_expires'].strftime('%Y-%m-%d')}")
            response_parts.append(f"Limited Warranty Valid Until: {warranty_info['limited_warranty_expires'].strftime('%Y-%m-%d')} ({warranty_info['days_remaining']} days remaining)")
        else:
            response_parts.append(f"Full Warranty Expired: {warranty_info['full_warranty_expires'].strftime('%Y-%m-%d')}")
            response_parts.append(f"Limited Warranty Expired: {warranty_info['limited_warranty_expires'].strftime('%Y-%m-%d')}")

        # Add order details for context
        if 'items' in order:
            response_parts.append(f"\nOrder Items:")
            for item in order['items']:
                response_parts.append(f"- {item['name']} (Qty: {item['quantity']})")

        # Add warranty policy information
        response_parts.append(f"\nWarranty Policy:")
        response_parts.append(f"- Full warranty covers defects and performance issues for {FULL_WARRANTY_DAYS} days")
        response_parts.append(f"- Limited warranty covers manufacturing defects only for {LIMITED_WARRANTY_DAYS} days")

        # Add action items based on warranty status
        if warranty_info['status'] == 'out_of_warranty':
            response_parts.append(f"\nWhile your warranty has expired, we may still be able to help with certain issues. Please contact customer support for assistance.")
        elif warranty_info['days_remaining'] <= 30:
            response_parts.append(f"\nNotice: Your warranty will expire in {warranty_info['days_remaining']} days. If you have any issues, please report them soon.")

        # TODO(human): Add specific next steps based on warranty status
        # Consider what actionable advice would be most helpful for customers

        return "\n".join(response_parts)

    except Exception as e:
        logger.error(f"Error checking warranty for order {order_id}: {e}")
        return f"I'm having trouble checking warranty information for order {order_id}. Please try again in a moment or contact customer support for assistance."

@tool
def get_warranty_policy() -> str:
    """
    Get information about the general warranty policy and coverage details.

    Returns:
        Detailed warranty policy information
    """
    try:
        logger.info("Retrieving warranty policy information")

        policy_info = [
            "Chrome Battery Warranty Policy",
            "=" * 35,
            "",
            "Full Warranty Coverage (First 180 days):",
            "- Complete protection against defects and performance issues",
            "- Free replacement for any manufacturing defects",
            "- Performance guarantee coverage",
            "- Free shipping for warranty replacements",
            "",
            "Limited Warranty Coverage (181-365 days):",
            "- Manufacturing defect coverage only",
            "- Replacement available for verified manufacturing issues",
            "- Customer pays shipping costs",
            "- Performance issues may not be covered",
            "",
            "After 365 Days:",
            "- Warranty coverage expires",
            "- Support available on case-by-case basis",
            "- Replacement parts may be available for purchase",
            "",
            "To check your specific order's warranty status, please provide your order ID.",
            "",
            "For warranty claims or questions, contact our customer support team."
        ]

        return "\n".join(policy_info)

    except Exception as e:
        logger.error(f"Error retrieving warranty policy: {e}")
        return "I'm having trouble accessing warranty policy information. Please contact customer support for detailed warranty information."

# List of available tools for the warranty agent
available_tools = [check_warranty, get_warranty_policy]

__all__ = ["check_warranty", "get_warranty_policy", "available_tools"]