"""
Order Management Tools for the Orders Agent.

This module provides placeholder order management functionality with static test data
for various order scenarios. This will be iteratively built into a fully functional
system in future phases.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Static test order data with various scenarios
TEST_ORDERS = {
    "ORD-001": {
        "order_id": "ORD-001",
        "customer_name": "John Smith",
        "customer_email": "john.smith@example.com",
        "order_date": "2024-01-15",
        "status": "delivered",
        "total_amount": "$149.99",
        "items": [
            {"name": "Chrome Battery CB12-7.5", "quantity": 2, "price": "$74.99"}
        ],
        "shipping_address": "123 Main St, Anytown, ST 12345",
        "tracking_number": "1Z999AA1012345675",
        "carrier": "UPS",
        "estimated_delivery": "2024-01-22",
        "actual_delivery": "2024-01-21"
    },
    "ORD-002": {
        "order_id": "ORD-002",
        "customer_name": "Sarah Johnson",
        "customer_email": "sarah.j@example.com",
        "order_date": "2024-01-18",
        "status": "shipped",
        "total_amount": "$89.99",
        "items": [
            {"name": "Chrome Battery CB6-12", "quantity": 1, "price": "$89.99"}
        ],
        "shipping_address": "456 Oak Ave, Springfield, IL 62701",
        "tracking_number": "1Z999AA1012345676",
        "carrier": "UPS",
        "estimated_delivery": "2024-01-25",
        "ship_date": "2024-01-20"
    },
    "ORD-003": {
        "order_id": "ORD-003",
        "customer_name": "Mike Davis",
        "customer_email": "mike.davis@example.com",
        "order_date": "2024-01-20",
        "status": "processing",
        "total_amount": "$299.97",
        "items": [
            {"name": "Chrome Battery CB12-100", "quantity": 1, "price": "$199.99"},
            {"name": "Battery Charger Pro", "quantity": 1, "price": "$99.98"}
        ],
        "shipping_address": "789 Pine St, Portland, OR 97201",
        "estimated_ship_date": "2024-01-23"
    },
    "ORD-004": {
        "order_id": "ORD-004",
        "customer_name": "Lisa Chen",
        "customer_email": "lisa.chen@example.com",
        "order_date": "2024-01-10",
        "status": "cancelled",
        "total_amount": "$59.99",
        "items": [
            {"name": "Chrome Battery CB6-4.5", "quantity": 1, "price": "$59.99"}
        ],
        "cancellation_reason": "Customer requested cancellation",
        "cancellation_date": "2024-01-12",
        "refund_status": "processed",
        "refund_amount": "$59.99"
    },
    "ORD-005": {
        "order_id": "ORD-005",
        "customer_name": "Robert Wilson",
        "customer_email": "rob.wilson@example.com",
        "order_date": "2024-01-22",
        "status": "pending",
        "total_amount": "$179.98",
        "items": [
            {"name": "Chrome Battery CB12-9", "quantity": 2, "price": "$89.99"}
        ],
        "shipping_address": "321 Elm St, Austin, TX 73301",
        "payment_status": "pending_verification"
    }
}


@tool
def lookup_order(order_id: str) -> str:
    """
    Look up comprehensive order details by order ID.

    This tool retrieves complete order information including customer details,
    items, pricing, shipping information, and current status.

    Args:
        order_id: The order ID to look up (e.g., "ORD-001", "ORD-002")

    Returns:
        Complete order details or error message if order not found
    """
    try:
        logger.info(f"Looking up order: {order_id}")

        # Normalize order ID
        order_id = order_id.strip().upper()

        if order_id in TEST_ORDERS:
            order = TEST_ORDERS[order_id]

            # Format order details for customer display
            details = []
            details.append(f"Order ID: {order['order_id']}")
            details.append(f"Order Date: {order['order_date']}")
            details.append(f"Status: {order['status'].title()}")
            details.append(f"Total Amount: {order['total_amount']}")

            # Add items
            details.append("\nItems Ordered:")
            for item in order['items']:
                details.append(f"- {item['name']} (Qty: {item['quantity']}) - {item['price']}")

            # Add shipping information
            if 'shipping_address' in order:
                details.append(f"\nShipping Address: {order['shipping_address']}")

            # Status-specific information
            if order['status'] == 'delivered':
                if 'actual_delivery' in order:
                    details.append(f"Delivered On: {order['actual_delivery']}")
                if 'tracking_number' in order:
                    details.append(f"Tracking Number: {order['tracking_number']}")

            elif order['status'] == 'shipped':
                if 'tracking_number' in order:
                    details.append(f"Tracking Number: {order['tracking_number']}")
                if 'estimated_delivery' in order:
                    details.append(f"Expected Delivery: {order['estimated_delivery']}")
                if 'ship_date' in order:
                    details.append(f"Ship Date: {order['ship_date']}")

            elif order['status'] == 'processing':
                if 'estimated_ship_date' in order:
                    details.append(f"Expected Ship Date: {order['estimated_ship_date']}")

            elif order['status'] == 'cancelled':
                details.append(f"Cancellation Date: {order['cancellation_date']}")
                details.append(f"Reason: {order['cancellation_reason']}")
                if 'refund_status' in order:
                    details.append(f"Refund Status: {order['refund_status'].title()}")
                    details.append(f"Refund Amount: {order['refund_amount']}")

            elif order['status'] == 'pending':
                if 'payment_status' in order:
                    details.append(f"Payment Status: {order['payment_status'].replace('_', ' ').title()}")

            return "\n".join(details)
        else:
            return f"Order {order_id} not found. Please verify the order ID and try again. If you continue to have issues, please contact customer support."

    except Exception as e:
        logger.error(f"Error looking up order {order_id}: {e}")
        return "I'm having trouble accessing order information right now. Please try again in a moment or contact customer support for assistance."


@tool
def get_order_status(order_id: str) -> str:
    """
    Get the current status of an order.

    This tool provides quick status information for an order without full details.

    Args:
        order_id: The order ID to check status for

    Returns:
        Current order status or error message if order not found
    """
    try:
        logger.info(f"Getting status for order: {order_id}")

        # Normalize order ID
        order_id = order_id.strip().upper()

        if order_id in TEST_ORDERS:
            order = TEST_ORDERS[order_id]
            status = order['status']

            status_messages = {
                'pending': f"Order {order_id} is pending. We're processing your payment and preparing your order.",
                'processing': f"Order {order_id} is being processed. Your items are being prepared for shipment.",
                'shipped': f"Order {order_id} has been shipped! Your tracking number is {order.get('tracking_number', 'not available yet')}.",
                'delivered': f"Order {order_id} was successfully delivered on {order.get('actual_delivery', 'an earlier date')}.",
                'cancelled': f"Order {order_id} was cancelled on {order.get('cancellation_date', 'a previous date')}."
            }

            return status_messages.get(status, f"Order {order_id} status: {status.title()}")
        else:
            return f"Order {order_id} not found. Please check your order ID and try again."

    except Exception as e:
        logger.error(f"Error getting status for order {order_id}: {e}")
        return "I'm having trouble checking order status right now. Please try again in a moment."


@tool
def get_tracking_number(order_id: str) -> str:
    """
    Get tracking information for a shipped order.

    This tool provides tracking numbers and carrier information for orders
    that have been shipped.

    Args:
        order_id: The order ID to get tracking information for

    Returns:
        Tracking information or appropriate message based on order status
    """
    try:
        logger.info(f"Getting tracking info for order: {order_id}")

        # Normalize order ID
        order_id = order_id.strip().upper()

        if order_id in TEST_ORDERS:
            order = TEST_ORDERS[order_id]
            status = order['status']

            if status in ['shipped', 'delivered']:
                tracking_num = order.get('tracking_number')
                carrier = order.get('carrier', 'the carrier')

                if tracking_num:
                    result = f"Order {order_id} tracking number: {tracking_num}"
                    if carrier:
                        result += f" (Carrier: {carrier})"

                    if status == 'delivered':
                        result += f"\n\nThis order was delivered on {order.get('actual_delivery', 'an earlier date')}."
                    elif 'estimated_delivery' in order:
                        result += f"\n\nEstimated delivery: {order['estimated_delivery']}"

                    return result
                else:
                    return f"Order {order_id} has been shipped but tracking information is not yet available. Please check back in a few hours."

            elif status == 'processing':
                ship_date = order.get('estimated_ship_date', 'soon')
                return f"Order {order_id} is still being processed and hasn't shipped yet. Expected ship date: {ship_date}. Tracking information will be available once the order ships."

            elif status == 'pending':
                return f"Order {order_id} is still pending and hasn't shipped yet. We'll provide tracking information once your order ships."

            elif status == 'cancelled':
                return f"Order {order_id} was cancelled and no shipment was made."

            else:
                return f"Order {order_id} status is {status}. Tracking information is not applicable for this status."
        else:
            return f"Order {order_id} not found. Please verify your order ID and try again."

    except Exception as e:
        logger.error(f"Error getting tracking for order {order_id}: {e}")
        return "I'm having trouble retrieving tracking information right now. Please try again in a moment."


# List of available tools for the orders agent
available_tools = [lookup_order, get_order_status, get_tracking_number]

__all__ = ["lookup_order", "get_order_status", "get_tracking_number", "available_tools"]