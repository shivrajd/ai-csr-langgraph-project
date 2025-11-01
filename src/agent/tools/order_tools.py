"""
Order Management Tools for the Orders Agent.

This module provides order management functionality using real data from Supabase
tables (shipworks_order and shipworks_shipment).
"""

import logging
import os
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from langchain_core.tools import tool

# Load environment variables if not already loaded
from dotenv import load_dotenv
load_dotenv()

# Import Supabase client utilities
from src.agent.tools.supabase_client import (
    get_supabase_client,
    query_orders_table,
    query_shipments_by_order_id,
    query_order_items_by_order_id
)

logger = logging.getLogger(__name__)

# Helper functions for data formatting and retrieval

def format_date(date_value: Any) -> str:
    """Format a date value from database to readable string."""
    if not date_value:
        return "Not available"

    try:
        if isinstance(date_value, str):
            # Parse ISO format date
            dt = datetime.fromisoformat(date_value.replace('Z', '+00:00'))
            return dt.strftime("%B %d, %Y")
        elif isinstance(date_value, datetime):
            return date_value.strftime("%B %d, %Y")
        else:
            return str(date_value)
    except Exception as e:
        logger.warning(f"Error formatting date {date_value}: {e}")
        return str(date_value)


def format_currency(amount: Any) -> str:
    """Format a currency amount from database."""
    if amount is None:
        return "$0.00"

    try:
        return f"${float(amount):.2f}"
    except (ValueError, TypeError) as e:
        logger.warning(f"Error formatting currency {amount}: {e}")
        return str(amount)


def get_order_by_id(order_identifier: str) -> Optional[Dict[str, Any]]:
    """
    Look up an order by OrderNumberComplete or email address.

    Args:
        order_identifier: Either an order number (OrderNumberComplete) or email address

    Returns:
        Order record dict or None if not found
    """
    try:
        # First try as order number
        orders = query_orders_table("OrderNumberComplete", order_identifier)

        if orders and len(orders) > 0:
            return orders[0]

        # If not found and looks like an email, try email lookup
        if "@" in order_identifier:
            orders = query_orders_table("BillEmail", order_identifier)
            if orders and len(orders) > 0:
                # Return most recent order for this email
                sorted_orders = sorted(
                    orders,
                    key=lambda x: x.get("OrderDate", ""),
                    reverse=True
                )
                return sorted_orders[0]

        return None

    except Exception as e:
        logger.error(f"Error looking up order {order_identifier}: {e}")
        return None


def get_orders_by_email(email: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Get all orders for a customer email.

    Args:
        email: Customer email address
        limit: Maximum number of orders to return

    Returns:
        List of order records, sorted by date (most recent first)
    """
    try:
        orders = query_orders_table("BillEmail", email)

        if not orders:
            return []

        # Sort by OrderDate, most recent first
        sorted_orders = sorted(
            orders,
            key=lambda x: x.get("OrderDate", ""),
            reverse=True
        )

        return sorted_orders[:limit]

    except Exception as e:
        logger.error(f"Error getting orders for email {email}: {e}")
        return []


@tool
def lookup_order(order_id: Union[str, int]) -> str:
    """
    Look up comprehensive order details by order number or email address.

    This tool retrieves complete order information including customer details,
    pricing, shipping information, tracking, and current status from the Supabase database.

    Args:
        order_id: The order number (e.g., "12345" or 12345) or customer email address

    Returns:
        Complete order details or error message if order not found
    """
    try:
        logger.info(f"Looking up order: {order_id}")

        # Normalize input - CRITICAL: Ensure string type (LangGraph may pass integers)
        order_identifier = str(order_id).strip()

        # Check if it's an email (multiple orders) or single order lookup
        if "@" in order_identifier:
            # Email lookup - get all orders for this customer
            orders = get_orders_by_email(order_identifier, limit=5)

            if not orders:
                return f"No orders found for email {order_identifier}. Please verify the email address and try again."

            # Format summary of all orders
            details = [f"Found {len(orders)} order(s) for {order_identifier}:\n"]

            for idx, order in enumerate(orders, 1):
                order_num = order.get("OrderNumberComplete") or order.get("OrderNumber") or order.get("OrderID")
                order_date = format_date(order.get("OrderDate"))
                total = format_currency(order.get("OrderTotal"))
                status = order.get("OnlineStatus", "Unknown")

                details.append(f"{idx}. Order #{order_num}")
                details.append(f"   Date: {order_date}")
                details.append(f"   Total: {total}")
                details.append(f"   Status: {status}")
                details.append("")

            details.append("To get more details about a specific order, please provide the order number.")
            return "\n".join(details)

        # Single order lookup
        order = get_order_by_id(order_identifier)

        if not order:
            return f"Order {order_identifier} not found. Please verify the order number and try again. If you continue to have issues, please contact customer support."

        # Format order details for customer display
        details = []

        # Basic order information
        order_num = order.get("OrderNumberComplete") or order.get("OrderNumber") or order.get("OrderID")
        details.append(f"Order Number: {order_num}")
        details.append(f"Order Date: {format_date(order.get('OrderDate'))}")

        # Customer information
        customer_name = f"{order.get('BillFirstName', '')} {order.get('BillLastName', '')}".strip()
        if customer_name:
            details.append(f"Customer: {customer_name}")
        if order.get("BillEmail"):
            details.append(f"Email: {order.get('BillEmail')}")

        # Order status and amount
        status = order.get("OnlineStatus", "Unknown")
        details.append(f"Status: {status}")
        details.append(f"Total Amount: {format_currency(order.get('OrderTotal'))}")

        # Item count (no line items per requirements)
        item_count = order.get("RollupItemCount")
        if item_count:
            details.append(f"Number of Items: {item_count}")

        # Shipping address
        ship_name = f"{order.get('ShipFirstName', '')} {order.get('ShipLastName', '')}".strip()
        ship_city = order.get("ShipCity", "")
        ship_state = order.get("ShipStateProvCode", "")
        ship_country = order.get("ShipCountryCode", "")

        if ship_name or ship_city:
            shipping_parts = []
            if ship_name:
                shipping_parts.append(ship_name)
            location_parts = [part for part in [ship_city, ship_state, ship_country] if part]
            if location_parts:
                shipping_parts.append(", ".join(location_parts))

            details.append(f"\nShipping Address: {', '.join(shipping_parts)}")

        # Get shipment/tracking information
        try:
            order_id_val = order.get("OrderID")
            if order_id_val:
                shipments = query_shipments_by_order_id(order_id_val)

                if shipments:
                    details.append(f"\nShipment Information:")
                    for idx, shipment in enumerate(shipments, 1):
                        tracking = shipment.get("TrackingNumber")
                        carrier = shipment.get("Carrier", "")
                        service = shipment.get("Service", "")
                        ship_date = format_date(shipment.get("ShipDate"))

                        if len(shipments) > 1:
                            details.append(f"\n  Shipment {idx}:")
                            prefix = "    "
                        else:
                            prefix = "  "

                        if tracking:
                            details.append(f"{prefix}Tracking Number: {tracking}")
                        if carrier:
                            carrier_service = f"{carrier}"
                            if service:
                                carrier_service += f" - {service}"
                            details.append(f"{prefix}Carrier: {carrier_service}")
                        if ship_date != "Not available":
                            details.append(f"{prefix}Ship Date: {ship_date}")
                else:
                    # No shipments yet
                    if status.lower() in ["shipped", "delivered"]:
                        details.append(f"\nTracking information is being updated. Please check back shortly.")
                    else:
                        details.append(f"\nThis order has not shipped yet.")

        except Exception as e:
            logger.error(f"Error fetching shipment info: {e}")
            details.append(f"\nUnable to retrieve shipment information at this time.")

        return "\n".join(details)

    except Exception as e:
        logger.error(f"Error looking up order {order_id}: {e}")
        return "I'm having trouble accessing order information right now. Please try again in a moment or contact customer support for assistance."


@tool
def get_order_status(order_id: Union[str, int]) -> str:
    """
    Get the current status of an order from the database.

    This tool provides quick status information for an order without full details.

    Args:
        order_id: The order number (string or integer) or customer email to check status for

    Returns:
        Current order status or error message if order not found
    """
    try:
        logger.info(f"Getting status for order: {order_id}")

        # Normalize input - CRITICAL: Ensure string type (LangGraph may pass integers)
        order_identifier = str(order_id).strip()

        # Look up the order
        order = get_order_by_id(order_identifier)

        if not order:
            return f"Order {order_identifier} not found. Please check your order number and try again."

        # Extract order information
        order_num = order.get("OrderNumberComplete") or order.get("OrderNumber") or order.get("OrderID")
        status = order.get("OnlineStatus", "Unknown")
        order_date = format_date(order.get("OrderDate"))

        # Build status message
        message = f"Order {order_num} (placed {order_date})\nStatus: {status}"

        # Try to get shipment information for additional context
        try:
            order_id_val = order.get("OrderID")
            if order_id_val:
                shipments = query_shipments_by_order_id(order_id_val)

                if shipments:
                    # Add tracking information if available
                    tracking_numbers = [
                        s.get("TrackingNumber")
                        for s in shipments
                        if s.get("TrackingNumber")
                    ]

                    if tracking_numbers:
                        if len(tracking_numbers) == 1:
                            message += f"\nTracking: {tracking_numbers[0]}"
                        else:
                            message += f"\nTracking numbers: {', '.join(tracking_numbers)}"

                        # Get most recent ship date
                        ship_dates = [
                            s.get("ShipDate")
                            for s in shipments
                            if s.get("ShipDate")
                        ]
                        if ship_dates:
                            latest_ship_date = max(ship_dates)
                            message += f"\nLast shipped: {format_date(latest_ship_date)}"

        except Exception as e:
            logger.warning(f"Could not fetch shipment info for status check: {e}")

        return message

    except Exception as e:
        logger.error(f"Error getting status for order {order_id}: {e}")
        return "I'm having trouble checking order status right now. Please try again in a moment."


@tool
def get_tracking_number(order_id: Union[str, int]) -> str:
    """
    Get tracking information for a shipped order from the database.

    This tool provides all tracking numbers and carrier information for orders
    that have been shipped. An order may have multiple shipments.

    Args:
        order_id: The order number (string or integer) or email address to get tracking information for

    Returns:
        Tracking information or appropriate message based on order status
    """
    try:
        logger.info(f"Getting tracking info for order: {order_id}")

        # Normalize input - CRITICAL: Ensure string type (LangGraph may pass integers)
        order_identifier = str(order_id).strip()

        # Look up the order
        order = get_order_by_id(order_identifier)

        if not order:
            return f"Order {order_identifier} not found. Please verify your order number and try again."

        order_num = order.get("OrderNumberComplete") or order.get("OrderNumber") or order.get("OrderID")
        status = order.get("OnlineStatus", "Unknown")

        # Get shipments for this order
        order_id_val = order.get("OrderID")
        if not order_id_val:
            return f"Order {order_num}: Unable to retrieve tracking information at this time."

        shipments = query_shipments_by_order_id(order_id_val)

        if not shipments:
            # No shipments found
            return f"Order {order_num} (Status: {status})\n\nThis order has not shipped yet. Tracking information will be available once your order ships."

        # Format tracking information for all shipments
        result = [f"Order {order_num} - Tracking Information:\n"]

        for idx, shipment in enumerate(shipments, 1):
            tracking = shipment.get("TrackingNumber")
            carrier = shipment.get("Carrier", "")
            service = shipment.get("Service", "")
            ship_date = format_date(shipment.get("ShipDate"))

            if len(shipments) > 1:
                result.append(f"Shipment {idx}:")

            if tracking:
                result.append(f"  Tracking Number: {tracking}")
            else:
                result.append(f"  Tracking Number: Not yet available")

            if carrier:
                carrier_info = carrier
                if service:
                    carrier_info += f" - {service}"
                result.append(f"  Carrier: {carrier_info}")

            if ship_date != "Not available":
                result.append(f"  Ship Date: {ship_date}")

            if idx < len(shipments):
                result.append("")  # Add blank line between shipments

        # Add helpful context based on tracking availability
        has_tracking = any(s.get("TrackingNumber") for s in shipments)
        if not has_tracking:
            result.append("\nNote: Tracking numbers are typically available within a few hours of shipment. Please check back shortly.")

        return "\n".join(result)

    except Exception as e:
        logger.error(f"Error getting tracking for order {order_id}: {e}")
        return "I'm having trouble retrieving tracking information right now. Please try again in a moment."


@tool
def get_order_items(order_id: Union[str, int]) -> str:
    """
    Get detailed product information for items in an order.

    This tool retrieves the list of products/items that were ordered, including
    product names, SKUs, quantities, prices, and descriptions. Use this when customers
    ask about what products are in their order, what they ordered, battery models,
    or specific product details.

    Args:
        order_id: The order number (e.g., "12345" or 12345) or customer email address

    Returns:
        Detailed list of products in the order(s) or error message if not found
    """
    try:
        logger.info(f"Getting order items for: {order_id}")

        # Normalize input - CRITICAL: Ensure string type (LangGraph may pass integers)
        order_identifier = str(order_id).strip()

        # Check if it's an email (multiple orders) or single order lookup
        if "@" in order_identifier:
            # Email lookup - get all orders for this customer
            orders = get_orders_by_email(order_identifier, limit=10)

            if not orders:
                return f"No orders found for email {order_identifier}. Please verify the email address and try again."

            # Collect items from all orders
            all_items_details = []
            all_items_details.append(f"Products ordered by {order_identifier}:\n")

            for order in orders:
                order_num = order.get("OrderNumberComplete") or order.get("OrderNumber") or order.get("OrderID")
                order_date = format_date(order.get("OrderDate"))
                order_id_val = order.get("OrderID")

                if not order_id_val:
                    continue

                # Get items for this order
                items = query_order_items_by_order_id(order_id_val)

                if items:
                    all_items_details.append(f"\nOrder #{order_num} ({order_date}):")
                    for idx, item in enumerate(items, 1):
                        item_name = item.get("Name", "Unknown Product")
                        sku = item.get("SKU", "")
                        quantity = item.get("Quantity", 0)
                        unit_price = item.get("UnitPrice", 0)
                        description = item.get("Description", "").strip()

                        all_items_details.append(f"\n{idx}. {item_name}")
                        if sku:
                            all_items_details.append(f"   SKU: {sku}")
                        all_items_details.append(f"   Quantity: {int(float(quantity))}")
                        all_items_details.append(f"   Price: {format_currency(unit_price)} each")
                        if description:
                            all_items_details.append(f"   Description: {description}")

            if len(all_items_details) == 1:  # Only the header was added
                return f"No items found in orders for {order_identifier}."

            return "\n".join(all_items_details)

        # Single order lookup
        order = get_order_by_id(order_identifier)

        if not order:
            return f"Order {order_identifier} not found. Please verify the order number and try again."

        order_id_val = order.get("OrderID")
        if not order_id_val:
            return f"Unable to retrieve items for order {order_identifier}."

        # Get items for this order
        items = query_order_items_by_order_id(order_id_val)

        if not items:
            return f"No items found in order {order_identifier}. This may indicate a data sync issue."

        # Format item details
        order_num = order.get("OrderNumberComplete") or order.get("OrderNumber") or order.get("OrderID")
        details = []
        details.append(f"Order #{order_num} contains {len(items)} item(s):\n")

        for idx, item in enumerate(items, 1):
            item_name = item.get("Name", "Unknown Product")
            sku = item.get("SKU", "")
            quantity = item.get("Quantity", 0)
            unit_price = item.get("UnitPrice", 0)
            description = item.get("Description", "").strip()
            weight = item.get("Weight")

            details.append(f"{idx}. {item_name}")
            if sku:
                details.append(f"   SKU: {sku}")
            details.append(f"   Quantity: {int(float(quantity))}")
            details.append(f"   Price: {format_currency(unit_price)} each")
            if description:
                details.append(f"   Description: {description}")
            if weight:
                try:
                    weight_val = float(weight)
                    if weight_val > 0:
                        details.append(f"   Weight: {weight_val} lbs")
                except (ValueError, TypeError):
                    pass

            # Add spacing between items (except after last item)
            if idx < len(items):
                details.append("")

        return "\n".join(details)

    except Exception as e:
        logger.error(f"Error getting order items for {order_id}: {e}")
        return "I'm having trouble retrieving product information right now. Please try again in a moment."


# List of available tools for the orders agent
available_tools = [lookup_order, get_order_status, get_tracking_number, get_order_items]

__all__ = ["lookup_order", "get_order_status", "get_tracking_number", "get_order_items", "available_tools"]