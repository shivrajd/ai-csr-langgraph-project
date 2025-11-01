"""Supabase connection manager for LangGraph Order Agent.

This module provides a lightweight Supabase client for querying order and shipment data
from the shipworks_order and shipworks_shipment tables.
"""

import os
import logging
from typing import Optional
from functools import lru_cache

try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    Client = None

logger = logging.getLogger(__name__)


class SupabaseConnection:
    """Supabase connection manager for order data queries.

    Creates a Supabase client using credentials from environment variables.

    Usage:
        with SupabaseConnection() as supabase:
            result = supabase.table("shipworks_order").select("*").eq("OrderNumberComplete", "12345").execute()
    """

    def __init__(self):
        self.url = os.getenv("SUPABASE_URL")
        self.key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")

        if not self.url:
            raise ValueError(
                "SUPABASE_URL environment variable must be set. "
                "Please add it to your .env file."
            )

        if not self.key:
            raise ValueError(
                "SUPABASE_SERVICE_ROLE_KEY (or SUPABASE_KEY) environment variable must be set. "
                "Please add it to your .env file."
            )

        if not SUPABASE_AVAILABLE:
            raise ImportError(
                "supabase-py library is not installed. "
                "Please install it: pip install supabase"
            )

        self.client: Optional[Client] = None

    def __enter__(self) -> Client:
        """Create and return Supabase client."""
        logger.info(f"🔌 Connecting to Supabase at {self.url}")

        try:
            self.client = create_client(self.url, self.key)
            logger.info("✅ Supabase connection established")
            return self.client
        except Exception as e:
            logger.error(f"❌ Failed to connect to Supabase: {e}")
            raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up Supabase client (no explicit cleanup needed for HTTP client)."""
        if exc_type:
            logger.error(f"⚠️ Error during Supabase operation: {exc_val}")
        logger.info("✅ Supabase connection closed")
        self.client = None


@lru_cache(maxsize=1)
def get_supabase_client() -> Client:
    """Get a cached Supabase client instance.

    This function creates and caches a single Supabase client for reuse across tool calls.
    The client is thread-safe and can be shared across multiple requests.

    Returns:
        Client: Supabase client instance

    Raises:
        ValueError: If environment variables are not set
        ImportError: If supabase-py library is not installed
    """
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")

    if not url:
        raise ValueError(
            "SUPABASE_URL environment variable must be set. "
            "Please add it to your .env file."
        )

    if not key:
        raise ValueError(
            "SUPABASE_SERVICE_ROLE_KEY (or SUPABASE_KEY) environment variable must be set. "
            "Please add it to your .env file."
        )

    if not SUPABASE_AVAILABLE:
        raise ImportError(
            "supabase-py library is not installed. "
            "Please install it: pip install supabase"
        )

    logger.info(f"🔌 Creating Supabase client for {url}")
    client = create_client(url, key)
    logger.info("✅ Supabase client created")

    return client


# Convenience function for simple queries without context manager
def query_orders_table(
    filter_field: str,
    filter_value: str,
    select_fields: str = "*"
) -> list:
    """Query the shipworks_order table with a single filter.

    Args:
        filter_field: Field name to filter on (e.g., "OrderNumberComplete", "BillEmail")
        filter_value: Value to match
        select_fields: Fields to select (default: "*")

    Returns:
        List of matching order records
    """
    try:
        client = get_supabase_client()
        response = (
            client.table("shipworks_order")
            .select(select_fields)
            .eq(filter_field, str(filter_value))  # Explicit string cast for type safety
            .execute()
        )
        return response.data
    except Exception as e:
        logger.error(f"❌ Error querying orders table: {e}")
        raise


def query_shipments_by_order_id(order_id: int) -> list:
    """Query all non-voided shipments for a given order ID.

    Args:
        order_id: The OrderID to look up shipments for

    Returns:
        List of matching shipment records
    """
    try:
        client = get_supabase_client()
        response = (
            client.table("shipworks_shipment")
            .select("*")
            .eq("OrderID", order_id)
            .eq("Voided", False)
            .order("ShipDate", desc=True)
            .execute()
        )
        return response.data
    except Exception as e:
        logger.error(f"❌ Error querying shipments table: {e}")
        raise


def query_order_items_by_order_id(order_id: int) -> list:
    """Query all order items for a given order ID.

    Args:
        order_id: The OrderID to look up items for

    Returns:
        List of matching order item records with product details
    """
    try:
        client = get_supabase_client()
        response = (
            client.table("shipworks_order_item")
            .select("OrderItemID, OrderID, Name, SKU, Quantity, UnitPrice, Description, Weight")
            .eq("OrderID", order_id)
            .order("OrderItemID")
            .execute()
        )
        return response.data
    except Exception as e:
        logger.error(f"❌ Error querying order items table: {e}")
        raise
