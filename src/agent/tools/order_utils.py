"""
Shared Order Data Utilities

This module provides reusable helper functions for querying order data from Supabase.
These are NOT tools - they are utility functions that can be imported and used by
any agent that needs order information.

Following LangGraph best practices: when multiple agents need the same data source,
create shared utility functions to avoid code duplication while maintaining proper
separation of concerns.

Usage:
    from src.agent.tools.order_utils import get_order_with_items

    order_data = get_order_with_items("417698")
    if order_data:
        order_date = order_data["order_date"]
        items = order_data["items"]
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

# Import Supabase client utilities
from src.agent.tools.supabase_client import (
    get_supabase_client,
    query_orders_table,
    query_order_items_by_order_id
)

logger = logging.getLogger(__name__)


def get_order_with_items(order_identifier: str) -> Optional[Dict[str, Any]]:
    """
    Get order information with items.

    This is a shared utility function that retrieves order details including
    order date and all items. Can be used by any agent that needs this data.

    Args:
        order_identifier: Order number (OrderNumberComplete) or email address

    Returns:
        Dictionary with order info and items, or None if not found:
        {
            "order_id": int,
            "order_number": str,
            "order_date": str,
            "customer_name": str,
            "customer_email": str,
            "order_total": float,
            "status": str,
            "items": [
                {
                    "name": str,
                    "sku": str,
                    "quantity": int,
                    "unit_price": float,
                    ...
                },
                ...
            ]
        }
    """
    try:
        logger.info(f"[order_utils] Getting order with items for: {order_identifier}")

        # Try as order number first
        orders = query_orders_table("OrderNumberComplete", str(order_identifier))

        # If not found and looks like email, try email lookup
        if (not orders or len(orders) == 0) and "@" in order_identifier:
            orders = query_orders_table("BillEmail", order_identifier)
            if orders and len(orders) > 0:
                # Return most recent order for this email
                orders = sorted(
                    orders,
                    key=lambda x: x.get("OrderDate", ""),
                    reverse=True
                )

        if not orders or len(orders) == 0:
            logger.warning(f"[order_utils] No order found for: {order_identifier}")
            return None

        order = orders[0]
        order_id = order.get("OrderID")

        # Get order items
        items = query_order_items_by_order_id(order_id)

        # Build structured result
        result = {
            "order_id": order_id,
            "order_number": order.get("OrderNumberComplete") or order.get("OrderNumber"),
            "order_date": order.get("OrderDate"),
            "customer_name": order.get("BillFirstName", "") + " " + order.get("BillLastName", ""),
            "customer_email": order.get("BillEmail"),
            "order_total": order.get("OrderTotal"),
            "status": order.get("OnlineStatus", "Unknown"),
            "items": items or []
        }

        logger.info(f"[order_utils] Found order {result['order_number']} with {len(result['items'])} items")
        return result

    except Exception as e:
        logger.error(f"[order_utils] Error getting order {order_identifier}: {e}")
        return None


def get_order_date_and_skus(order_number: str) -> Optional[Dict[str, Any]]:
    """
    Get minimal order info: just date and product SKUs.

    Lightweight utility for agents that only need date and SKU info
    (e.g., for warranty calculations).

    Args:
        order_number: The order number

    Returns:
        Dictionary with minimal info, or None if not found:
        {
            "order_date": str,
            "skus": ["ZB-12R-35", "PB-ABC-123", ...]
        }
    """
    try:
        order_data = get_order_with_items(order_number)
        if not order_data:
            return None

        skus = [item.get("SKU", "") for item in order_data["items"] if item.get("SKU")]

        return {
            "order_date": order_data["order_date"],
            "skus": skus,
            "items": order_data["items"]  # Include full items for detailed warranty checks
        }

    except Exception as e:
        logger.error(f"[order_utils] Error getting date/SKUs for order {order_number}: {e}")
        return None


def format_order_date(date_value: Any) -> str:
    """
    Format order date for display.

    Args:
        date_value: Date value from database

    Returns:
        Formatted date string
    """
    if not date_value:
        return "Not available"

    try:
        if isinstance(date_value, str):
            # Try ISO format
            try:
                dt = datetime.fromisoformat(date_value.replace('Z', '+00:00'))
                return dt.strftime("%B %d, %Y")
            except:
                # Try other formats
                for fmt in ["%Y-%m-%d", "%m/%d/%Y", "%Y-%m-%d %H:%M:%S"]:
                    try:
                        dt = datetime.strptime(date_value, fmt)
                        return dt.strftime("%B %d, %Y")
                    except:
                        continue
                return date_value
        elif isinstance(date_value, datetime):
            return date_value.strftime("%B %d, %Y")
        else:
            return str(date_value)
    except Exception as e:
        logger.warning(f"[order_utils] Error formatting date {date_value}: {e}")
        return str(date_value)
