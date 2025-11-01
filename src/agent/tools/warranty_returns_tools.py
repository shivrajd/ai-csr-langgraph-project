"""
Warranty and Returns Management Tools for the Warranty Returns Agent.

This module provides comprehensive warranty status checking and RMA (Return Merchandise
Authorization) tracking functionality. It implements brand-specific warranty periods
and integrates with the chromeinventory_rma table for return/replacement status tracking.

Key Features:
- Brand-specific warranty periods (ZB, PB, PRO, BT brands)
- Warranty eligibility calculation (refund vs replacement windows)
- RMA status tracking and lookup
- Integration with order data for warranty verification
"""

import logging
import os
import re
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from langchain_core.tools import tool

# Load environment variables if not already loaded
from dotenv import load_dotenv
load_dotenv()

# Import Supabase client utilities (for RMA queries only)
from src.agent.tools.supabase_client import get_supabase_client

logger = logging.getLogger(__name__)

# Brand-Specific Warranty Configuration
# Based on ChromeBattery warranty policy effective 2024-03-01
# Format: {brand_prefix: {refund_days, replacement_days}}
BRAND_WARRANTY_CONFIG = {
    "ZB": {
        "refund_days": 30,
        "replacement_days": 365,
        "name": "Standard (ZB)"
    },
    "PB": {
        "refund_days": 45,
        "replacement_days": 365,
        "name": "Performance (PB)"
    },
    "PRO": {
        "refund_days": 75,
        "replacement_days": 732,  # 2 years
        "name": "Professional (PRO)"
    },
    "BT": {
        "refund_days": 90,
        "replacement_days": 732,  # 2 years
        "name": "Bluetooth (BT)"
    },
    "default": {
        "refund_days": 60,
        "replacement_days": 549,  # ~1.5 years
        "name": "Default"
    }
}

# Helper Functions

def format_date(date_value: Any) -> str:
    """Format a date value from database to readable string."""
    if not date_value:
        return "Not available"

    try:
        if isinstance(date_value, str):
            # Handle various date formats
            # Try ISO format first
            try:
                dt = datetime.fromisoformat(date_value.replace('Z', '+00:00'))
                return dt.strftime("%B %d, %Y")
            except:
                # Try other common formats
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
        logger.warning(f"Error formatting date {date_value}: {e}")
        return str(date_value)


def parse_date(date_value: Any) -> Optional[datetime]:
    """Parse a date value to datetime object (timezone-naive for calculations)."""
    if not date_value:
        return None

    try:
        if isinstance(date_value, datetime):
            # Convert to naive datetime if it has timezone info
            if date_value.tzinfo is not None:
                return date_value.replace(tzinfo=None)
            return date_value
        elif isinstance(date_value, str):
            # Try ISO format first
            try:
                dt = datetime.fromisoformat(date_value.replace('Z', '+00:00'))
                # Convert to naive datetime
                return dt.replace(tzinfo=None) if dt.tzinfo else dt
            except:
                # Try other common formats
                for fmt in ["%Y-%m-%d", "%m/%d/%Y", "%Y-%m-%d %H:%M:%S"]:
                    try:
                        return datetime.strptime(date_value, fmt)
                    except:
                        continue
        return None
    except Exception as e:
        logger.warning(f"Error parsing date {date_value}: {e}")
        return None


def extract_brand_from_sku(sku: str) -> str:
    """
    Extract brand prefix from SKU.

    Examples:
        "ZB-12R-35" -> "ZB"
        "PB-ABC-123" -> "PB"
        "PRO-XYZ-456" -> "PRO"
        "BT-DEF-789" -> "BT"

    Args:
        sku: Product SKU string

    Returns:
        Brand prefix (uppercase) or "default" if no match
    """
    if not sku:
        return "default"

    sku_upper = sku.upper().strip()

    # Try exact prefix matches first
    for brand in ["PRO", "BT", "ZB", "PB"]:
        if sku_upper.startswith(f"{brand}-") or sku_upper.startswith(brand):
            return brand

    return "default"


def get_brand_warranty_periods(brand: str) -> Dict[str, Any]:
    """
    Get warranty periods for a specific brand.

    Args:
        brand: Brand prefix (ZB, PB, PRO, BT, or default)

    Returns:
        Dictionary with refund_days, replacement_days, and name
    """
    brand_upper = brand.upper() if brand else "default"
    return BRAND_WARRANTY_CONFIG.get(brand_upper, BRAND_WARRANTY_CONFIG["default"])


def calculate_warranty_status(
    order_date: datetime,
    brand: str,
    current_date: Optional[datetime] = None
) -> Dict[str, Any]:
    """
    Calculate comprehensive warranty status for a product.

    Args:
        order_date: Date of original purchase (timezone-naive)
        brand: Brand prefix (ZB, PB, PRO, BT, etc.)
        current_date: Current date (defaults to now, timezone-naive)

    Returns:
        Dictionary with warranty status details:
        - within_refund_period: bool
        - within_replacement_period: bool
        - days_since_purchase: int
        - refund_days_remaining: int
        - replacement_days_remaining: int
        - brand_info: dict
        - status_message: str
    """
    if current_date is None:
        # Use timezone-naive datetime for calculations
        current_date = datetime.now()

    # Ensure both dates are timezone-naive for calculation
    if hasattr(order_date, 'tzinfo') and order_date.tzinfo is not None:
        order_date = order_date.replace(tzinfo=None)
    if hasattr(current_date, 'tzinfo') and current_date.tzinfo is not None:
        current_date = current_date.replace(tzinfo=None)

    # Get brand-specific warranty periods
    brand_info = get_brand_warranty_periods(brand)

    # Calculate days since purchase
    days_since_purchase = (current_date - order_date).days

    # Calculate warranty eligibility
    refund_days_remaining = brand_info["refund_days"] - days_since_purchase
    replacement_days_remaining = brand_info["replacement_days"] - days_since_purchase

    within_refund_period = refund_days_remaining > 0
    within_replacement_period = replacement_days_remaining > 0

    # Generate status message
    if within_refund_period:
        status_message = f"✅ Within refund period ({refund_days_remaining} days remaining)"
    elif within_replacement_period:
        status_message = f"✅ Within replacement period ({replacement_days_remaining} days remaining)"
    else:
        status_message = f"❌ Warranty expired ({abs(replacement_days_remaining)} days ago)"

    return {
        "within_refund_period": within_refund_period,
        "within_replacement_period": within_replacement_period,
        "days_since_purchase": days_since_purchase,
        "refund_days_remaining": refund_days_remaining,
        "replacement_days_remaining": replacement_days_remaining,
        "brand_info": brand_info,
        "status_message": status_message
    }


def query_rma_table(filter_field: str, filter_value: str) -> List[Dict[str, Any]]:
    """
    Query the chromeinventory_rma table.

    Args:
        filter_field: Field name to filter on (e.g., "OrderNumber", "Email", "RmaNumber")
        filter_value: Value to match

    Returns:
        List of matching RMA records
    """
    try:
        client = get_supabase_client()
        response = (
            client.table("chromeinventory_rma")
            .select("*")
            .eq(filter_field, filter_value)
            .order("RmaDate", desc=True)
            .execute()
        )
        return response.data
    except Exception as e:
        logger.error(f"❌ Error querying RMA table: {e}")
        raise


def format_rma_status(rma_record: Dict[str, Any]) -> str:
    """
    Format a single RMA record into a user-friendly status string.

    Args:
        rma_record: RMA record dictionary from database

    Returns:
        Formatted status string
    """
    rma_number = rma_record.get("RmaNumber", "Unknown")
    item_name = rma_record.get("ItemName", "Unknown Item")
    return_type = rma_record.get("ReturnType", "Unknown")
    return_status = rma_record.get("ReturnStatus", "Pending")
    approved = rma_record.get("Approved", 0)
    rma_date = format_date(rma_record.get("RmaDate"))

    # Approval status
    approval_status = "✅ Approved" if approved == 1 else "⏳ Pending Approval"

    # Build status message
    status_parts = [
        f"RMA #{rma_number}",
        f"Item: {item_name}",
        f"Type: {return_type}",
        f"Status: {return_status}",
        f"Approval: {approval_status}",
        f"Created: {rma_date}"
    ]

    # Add tracking info if available
    return_tracking = rma_record.get("ReturnTracking")
    if return_tracking:
        status_parts.append(f"Return Tracking: {return_tracking}")

    # Add return label info
    return_label_sent = rma_record.get("ReturnLabelSent")
    if return_label_sent:
        status_parts.append("Return Label: Sent")

    # Add received date if available
    return_received = rma_record.get("ReturnReceived")
    if return_received:
        status_parts.append(f"Received: {format_date(return_received)}")

    # Add action taken
    return_action = rma_record.get("ReturnAction")
    if return_action:
        status_parts.append(f"Action: {return_action}")

    # Add results/resolution
    results = rma_record.get("Results")
    if results:
        status_parts.append(f"Resolution: {results}")

    return "\n".join(status_parts)


# Tool Functions (exposed to LangGraph)

@tool
def check_product_warranty_status(order_number: str) -> str:
    """
    Check warranty status for all products in an order using brand-specific periods.

    This tool uses shared order utilities to get order data, then calculates warranty
    eligibility based on brand-specific warranty policies. Different brands (ZB, PB, PRO, BT)
    have different warranty periods.

    Args:
        order_number: The order number to check warranty for (e.g., "417698")

    Returns:
        Formatted warranty status information for all items in the order,
        including refund/replacement eligibility and days remaining.

    Examples:
        >>> check_product_warranty_status("417698")
        "Order #417698 Warranty Status:
        Purchase Date: January 15, 2024

        Item 1: ZB-12R-35 Battery
        Brand: Standard (ZB)
        ✅ Within replacement period (245 days remaining)
        ..."
    """
    try:
        logger.info(f"Checking warranty status for order: {order_number}")

        # Use shared order utilities - NO database duplication
        from src.agent.tools.order_utils import get_order_with_items

        order_data = get_order_with_items(order_number)

        if not order_data:
            return f"❌ Order #{order_number} not found. Please verify the order number and try again."

        order_date_str = order_data["order_date"]
        items = order_data["items"]

        # Parse order date
        order_date = parse_date(order_date_str)
        if not order_date:
            return f"❌ Unable to determine order date for order #{order_number}. Cannot calculate warranty status."

        if not items or len(items) == 0:
            return f"❌ No items found for order #{order_number}."

        # Build warranty status response
        response_parts = [
            f"📦 Order #{order_number} Warranty Status",
            f"📅 Purchase Date: {format_date(order_date_str)}",
            ""
        ]

        for idx, item in enumerate(items, 1):
            item_name = item.get("Name", "Unknown Item")
            sku = item.get("SKU", "")
            quantity = item.get("Quantity", 1)

            # Extract brand and calculate warranty
            brand = extract_brand_from_sku(sku)
            warranty_status = calculate_warranty_status(order_date, brand)

            # Format item status
            response_parts.append(f"Item {idx}: {item_name}")
            if sku:
                response_parts.append(f"  SKU: {sku}")
            response_parts.append(f"  Brand: {warranty_status['brand_info']['name']}")
            response_parts.append(f"  Quantity: {quantity}")
            response_parts.append(f"  {warranty_status['status_message']}")

            # Add detailed period info
            if warranty_status['within_refund_period']:
                response_parts.append(f"  💰 Refund eligible: {warranty_status['refund_days_remaining']} days remaining")
                response_parts.append(f"  🔄 Replacement eligible: {warranty_status['replacement_days_remaining']} days remaining")
            elif warranty_status['within_replacement_period']:
                response_parts.append(f"  🔄 Replacement eligible: {warranty_status['replacement_days_remaining']} days remaining")
                response_parts.append(f"  ❌ Refund period expired")
            else:
                response_parts.append(f"  ❌ Both refund and replacement periods expired")

            response_parts.append("")  # Blank line between items

        # Add warranty policy info
        response_parts.append("📋 Warranty Policy by Brand:")
        response_parts.append("  • Standard (ZB): 30-day refund, 365-day replacement")
        response_parts.append("  • Performance (PB): 45-day refund, 365-day replacement")
        response_parts.append("  • Professional (PRO): 75-day refund, 732-day replacement")
        response_parts.append("  • Bluetooth (BT): 90-day refund, 732-day replacement")
        response_parts.append("  • Default: 60-day refund, 549-day replacement")

        return "\n".join(response_parts)

    except Exception as e:
        logger.error(f"Error checking warranty status for order {order_number}: {e}")
        return f"❌ An error occurred while checking warranty status. Please try again or contact support."


@tool
def lookup_rma_by_order(order_number: str) -> str:
    """
    Look up RMA (Return Merchandise Authorization) records by order number.

    This tool searches for existing return or replacement requests associated with
    a specific order number and provides detailed status information.

    Args:
        order_number: The order number to search for RMA records (e.g., "417698")

    Returns:
        Formatted RMA status information including authorization number, return type,
        approval status, tracking information, and resolution details.

    Examples:
        >>> lookup_rma_by_order("417698")
        "Found 2 RMA record(s) for order #417698:

        RMA #51d1899142a83
        Item: Battery Pack XYZ
        Type: Replacement
        Status: Approved
        ..."
    """
    try:
        logger.info(f"Looking up RMA records for order: {order_number}")

        # Query RMA table by order number
        rma_records = query_rma_table("OrderNumber", str(order_number))

        if not rma_records or len(rma_records) == 0:
            return f"📋 No RMA records found for order #{order_number}. If you need to initiate a return or replacement, please contact our support team."

        # Build response
        response_parts = [
            f"📋 Found {len(rma_records)} RMA record(s) for order #{order_number}:",
            ""
        ]

        for idx, rma in enumerate(rma_records, 1):
            response_parts.append(f"--- RMA Record {idx} ---")
            response_parts.append(format_rma_status(rma))
            response_parts.append("")

        return "\n".join(response_parts)

    except Exception as e:
        logger.error(f"Error looking up RMA for order {order_number}: {e}")
        return f"❌ An error occurred while looking up RMA records. Please try again or contact support."


@tool
def lookup_rma_by_email(email: str) -> str:
    """
    Look up all RMA records associated with a customer's email address.

    This tool finds all return and replacement requests for a specific customer,
    which is useful when the customer doesn't remember their order number.

    Args:
        email: Customer's email address (e.g., "customer@example.com")

    Returns:
        Formatted list of all RMA records for the customer, including order numbers,
        items, return types, and current status of each RMA.

    Examples:
        >>> lookup_rma_by_email("customer@example.com")
        "Found 3 RMA record(s) for customer@example.com:

        RMA #51d1899142a83 (Order #417698)
        Item: Battery Pack XYZ
        Type: Replacement
        ..."
    """
    try:
        logger.info(f"Looking up RMA records for email: {email}")

        # Query RMA table by email
        rma_records = query_rma_table("Email", email.lower().strip())

        if not rma_records or len(rma_records) == 0:
            return f"📋 No RMA records found for {email}. If you need to initiate a return or replacement, please contact our support team."

        # Build response
        response_parts = [
            f"📋 Found {len(rma_records)} RMA record(s) for {email}:",
            ""
        ]

        for idx, rma in enumerate(rma_records, 1):
            order_number = rma.get("OrderNumber", "Unknown")
            response_parts.append(f"--- RMA Record {idx} (Order #{order_number}) ---")
            response_parts.append(format_rma_status(rma))
            response_parts.append("")

        return "\n".join(response_parts)

    except Exception as e:
        logger.error(f"Error looking up RMA for email {email}: {e}")
        return f"❌ An error occurred while looking up RMA records. Please try again or contact support."


@tool
def get_rma_status(rma_number: str) -> str:
    """
    Get detailed status of a specific RMA by RMA number.

    This tool provides comprehensive information about a specific return or replacement
    request, including approval status, tracking information, and resolution details.

    Args:
        rma_number: The RMA authorization number (e.g., "51d1899142a83")

    Returns:
        Detailed status information including item details, return type, approval status,
        shipping tracking, return label status, and resolution information.

    Examples:
        >>> get_rma_status("51d1899142a83")
        "RMA #51d1899142a83 Status:

        Item: Battery Pack XYZ
        Order: #417698
        Type: Replacement
        Status: Approved
        Return Tracking: 1Z999AA10123456789
        ..."
    """
    try:
        logger.info(f"Getting status for RMA: {rma_number}")

        # Query RMA table by RMA number
        rma_records = query_rma_table("RmaNumber", rma_number.strip())

        if not rma_records or len(rma_records) == 0:
            return f"❌ RMA #{rma_number} not found. Please verify the RMA number and try again."

        rma = rma_records[0]

        # Build detailed response
        response_parts = [
            f"📋 RMA #{rma_number} Detailed Status:",
            ""
        ]

        # Add formatted status
        response_parts.append(format_rma_status(rma))

        # Add additional details if available
        order_number = rma.get("OrderNumber")
        if order_number:
            response_parts.append("")
            response_parts.append(f"💡 To check warranty status for this order, use order number: {order_number}")

        return "\n".join(response_parts)

    except Exception as e:
        logger.error(f"Error getting RMA status for {rma_number}: {e}")
        return f"❌ An error occurred while retrieving RMA status. Please try again or contact support."


@tool
def get_brand_warranty_info(brand: str = "all") -> str:
    """
    Get warranty period information for specific brand(s).

    This tool provides details about warranty policies for different product brands,
    including refund and replacement periods.

    Args:
        brand: Brand prefix to get info for (ZB, PB, PRO, BT, default, or "all")
              Default is "all" which shows all brand warranty policies.

    Returns:
        Formatted warranty policy information for the specified brand(s).

    Examples:
        >>> get_brand_warranty_info("ZB")
        "Warranty Policy for Standard (ZB):
        • Refund Period: 30 days from purchase
        • Replacement Period: 365 days from purchase
        ..."

        >>> get_brand_warranty_info("all")
        "ChromeBattery Warranty Policies by Brand:

        Standard (ZB):
        • Refund Period: 30 days
        • Replacement Period: 365 days
        ..."
    """
    try:
        logger.info(f"Getting warranty info for brand: {brand}")

        brand_upper = brand.upper().strip()

        if brand_upper == "ALL":
            # Show all brand warranties
            response_parts = [
                "🛡️ ChromeBattery Warranty Policies by Brand:",
                "",
                "Our warranty coverage varies by product line to ensure the best",
                "protection for your specific battery type.",
                ""
            ]

            for brand_key in ["ZB", "PB", "PRO", "BT", "default"]:
                brand_info = BRAND_WARRANTY_CONFIG[brand_key]
                response_parts.append(f"📦 {brand_info['name']}:")
                response_parts.append(f"   • Refund Period: {brand_info['refund_days']} days from purchase")
                response_parts.append(f"   • Replacement Period: {brand_info['replacement_days']} days from purchase")
                response_parts.append("")

            response_parts.append("💡 Tips:")
            response_parts.append("   • Refund period allows full money-back returns")
            response_parts.append("   • Replacement period covers warranty exchanges")
            response_parts.append("   • Brand is determined by product SKU (e.g., ZB-12R-35 is Standard)")

        else:
            # Show specific brand warranty
            brand_info = get_brand_warranty_periods(brand_upper)

            response_parts = [
                f"🛡️ Warranty Policy for {brand_info['name']}:",
                "",
                f"📅 Refund Period: {brand_info['refund_days']} days from purchase date",
                f"   Full refund available for returns within this period",
                "",
                f"🔄 Replacement Period: {brand_info['replacement_days']} days from purchase date",
                f"   Warranty replacement available for defective items",
                "",
                "💡 Note:",
                "   • Warranty period starts from the original purchase date",
                "   • Refund period is shorter than replacement period",
                "   • Items must meet return conditions (see our return policy)",
            ]

        return "\n".join(response_parts)

    except Exception as e:
        logger.error(f"Error getting brand warranty info: {e}")
        return "❌ An error occurred while retrieving warranty information. Please try again."


# Export all tools
__all__ = [
    "check_product_warranty_status",
    "lookup_rma_by_order",
    "lookup_rma_by_email",
    "get_rma_status",
    "get_brand_warranty_info"
]
