"""
Product Management Tools for the Products Agent.

This module provides product search, details, and comparison functionality with static test data
for various product scenarios. This will be iteratively built into a fully functional
product catalog system in future phases.
"""

import logging
from typing import Dict, Any, List
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Static test product data with comprehensive details
TEST_PRODUCTS = {
    "CB12-7.5": {
        "product_id": "CB12-7.5",
        "sku": "CB12-7.5",
        "name": "Chrome Battery CB12-7.5",
        "category": "Sealed Lead Acid Batteries",
        "price": "$74.99",
        "description": "12V 7.5Ah sealed lead-acid battery with F1 terminals. Perfect for UPS systems, emergency lighting, and security systems.",
        "specifications": {
            "voltage": "12V",
            "capacity": "7.5Ah",
            "terminal_type": "F1 (0.187 inch)",
            "dimensions": "5.94 x 2.56 x 3.70 inches",
            "weight": "5.5 lbs",
            "chemistry": "Sealed Lead Acid (SLA)",
            "life_expectancy": "3-5 years"
        },
        "stock_status": "in_stock",
        "stock_quantity": 45,
        "applications": ["UPS Systems", "Emergency Lighting", "Security Systems", "Fire Alarms"],
        "features": ["Maintenance-free", "Leak-proof", "AGM technology", "Wide temperature range"],
        "warranty": "2 years replacement warranty"
    },
    "CB6-12": {
        "product_id": "CB6-12",
        "sku": "CB6-12",
        "name": "Chrome Battery CB6-12",
        "category": "Deep Cycle Batteries",
        "price": "$89.99",
        "description": "6V 12Ah deep cycle battery designed for renewable energy systems, golf carts, and marine applications.",
        "specifications": {
            "voltage": "6V",
            "capacity": "12Ah",
            "terminal_type": "F2 (0.250 inch)",
            "dimensions": "5.94 x 3.86 x 3.70 inches",
            "weight": "8.2 lbs",
            "chemistry": "Deep Cycle AGM",
            "life_expectancy": "5-7 years"
        },
        "stock_status": "in_stock",
        "stock_quantity": 32,
        "applications": ["Solar Power Systems", "Golf Carts", "Marine Equipment", "RV Systems"],
        "features": ["Deep discharge capability", "Maintenance-free", "Vibration resistant", "Long cycle life"],
        "warranty": "3 years replacement warranty"
    },
    "BCP-SMART": {
        "product_id": "BCP-SMART",
        "sku": "BCP-SMART",
        "name": "Battery Charger Pro",
        "category": "Battery Chargers",
        "price": "$149.99",
        "description": "Smart multi-stage battery charger compatible with 6V and 12V lead-acid batteries. Features automatic detection and multiple charging modes.",
        "specifications": {
            "input_voltage": "110-240V AC",
            "output_voltage": "6V/12V (automatic detection)",
            "charging_current": "2A/6A/10A (selectable)",
            "compatible_batteries": "Lead-acid, AGM, Gel, Flooded",
            "dimensions": "8.5 x 4.2 x 2.8 inches",
            "weight": "2.1 lbs",
            "display": "LED status indicators"
        },
        "stock_status": "low_stock",
        "stock_quantity": 8,
        "applications": ["Battery Maintenance", "Automotive", "Marine", "Recreational Vehicles"],
        "features": ["Multi-stage charging", "Automatic voltage detection", "Reverse polarity protection", "Overcharge protection"],
        "warranty": "1 year replacement warranty"
    }
}

# Category mapping for easier searching
CATEGORY_MAPPING = {
    "batteries": ["CB12-7.5", "CB6-12"],
    "sealed lead acid": ["CB12-7.5"],
    "deep cycle": ["CB6-12"],
    "chargers": ["BCP-SMART"],
    "accessories": ["BCP-SMART"]
}


@tool
def search_products(query: str) -> str:
    """
    Search for products by name, category, or keywords.

    This tool searches the product catalog for matching products based on
    name, category, applications, or features.

    Args:
        query: Search term (product name, category, or keywords)

    Returns:
        List of matching products with basic information
    """
    try:
        logger.info(f"Searching products for: {query}")

        query_lower = query.lower().strip()
        matching_products = []

        # Search through all products
        for product_id, product in TEST_PRODUCTS.items():
            # Check if query matches product name, category, applications, features, or specifications
            # TODO(human): Add search logic to check within product specifications
            specs_match = any(query_lower in str(spec_value).lower() for spec_value in product["specifications"].values())

            if (query_lower in product["name"].lower() or
                query_lower in product["category"].lower() or
                any(query_lower in app.lower() for app in product["applications"]) or
                any(query_lower in feature.lower() for feature in product["features"]) or
                specs_match):

                matching_products.append(product)

        # Also check category mapping
        if query_lower in CATEGORY_MAPPING:
            for product_id in CATEGORY_MAPPING[query_lower]:
                if product_id in TEST_PRODUCTS:
                    product = TEST_PRODUCTS[product_id]
                    if product not in matching_products:
                        matching_products.append(product)

        if matching_products:
            results = []
            results.append(f"Found {len(matching_products)} product(s) matching '{query}':\n")

            for product in matching_products:
                stock_emoji = "✅" if product["stock_status"] == "in_stock" else "⚠️"
                results.append(f"{stock_emoji} **{product['name']}** (SKU: {product['sku']})")
                results.append(f"   Price: {product['price']}")
                results.append(f"   Category: {product['category']}")
                results.append(f"   Stock: {product['stock_quantity']} units ({product['stock_status'].replace('_', ' ')})")
                results.append(f"   Description: {product['description'][:100]}...")
                results.append("")

            return "\n".join(results)
        else:
            return f"No products found matching '{query}'. Try searching for 'batteries', 'chargers', '12V', '6V', or specific product names like 'CB12-7.5'."

    except Exception as e:
        logger.error(f"Error searching products for '{query}': {e}")
        return "I'm having trouble searching the product catalog right now. Please try again in a moment."


@tool
def get_product_details(product_id: str) -> str:
    """
    Get comprehensive details for a specific product.

    This tool retrieves complete product information including specifications,
    pricing, stock status, and technical details.

    Args:
        product_id: Product ID or SKU to retrieve details for

    Returns:
        Complete product details or error message if product not found
    """
    try:
        logger.info(f"Getting product details for: {product_id}")

        # Normalize product ID
        product_id = product_id.strip().upper()

        if product_id in TEST_PRODUCTS:
            product = TEST_PRODUCTS[product_id]

            details = []
            details.append(f"📦 **{product['name']}**")
            details.append(f"SKU: {product['sku']}")
            details.append(f"Category: {product['category']}")
            details.append(f"Price: {product['price']}")
            details.append("")

            # Description
            details.append("**Description:**")
            details.append(product['description'])
            details.append("")

            # Specifications
            details.append("**Technical Specifications:**")
            for spec_name, spec_value in product['specifications'].items():
                details.append(f"• {spec_name.replace('_', ' ').title()}: {spec_value}")
            details.append("")

            # Stock Information
            stock_emoji = "✅" if product["stock_status"] == "in_stock" else "⚠️" if product["stock_status"] == "low_stock" else "❌"
            details.append("**Availability:**")
            details.append(f"{stock_emoji} Stock Status: {product['stock_status'].replace('_', ' ').title()}")
            details.append(f"• Quantity Available: {product['stock_quantity']} units")
            details.append("")

            # Applications
            details.append("**Recommended Applications:**")
            for app in product['applications']:
                details.append(f"• {app}")
            details.append("")

            # Features
            details.append("**Key Features:**")
            for feature in product['features']:
                details.append(f"• {feature}")
            details.append("")

            # Warranty
            details.append(f"**Warranty:** {product['warranty']}")

            return "\n".join(details)
        else:
            return f"Product '{product_id}' not found. Available products: CB12-7.5, CB6-12, BCP-SMART. Please check the product ID and try again."

    except Exception as e:
        logger.error(f"Error getting product details for '{product_id}': {e}")
        return "I'm having trouble retrieving product details right now. Please try again in a moment."


@tool
def check_product_stock(product_id: str) -> str:
    """
    Check stock availability and quantity for a specific product.

    This tool provides current stock status and quantity information
    for inventory planning and availability checks.

    Args:
        product_id: Product ID or SKU to check stock for

    Returns:
        Current stock status and quantity information
    """
    try:
        logger.info(f"Checking stock for product: {product_id}")

        # Normalize product ID
        product_id = product_id.strip().upper()

        if product_id in TEST_PRODUCTS:
            product = TEST_PRODUCTS[product_id]

            stock_status = product['stock_status']
            stock_quantity = product['stock_quantity']

            # Determine stock emoji and message
            if stock_status == "in_stock":
                stock_emoji = "✅"
                availability_msg = "Available for immediate shipment"
            elif stock_status == "low_stock":
                stock_emoji = "⚠️"
                availability_msg = "Limited quantity available - order soon"
            else:
                stock_emoji = "❌"
                availability_msg = "Currently out of stock"

            result = []
            result.append(f"{stock_emoji} **Stock Status for {product['name']}**")
            result.append(f"Product ID: {product['sku']}")
            result.append(f"Current Stock: {stock_quantity} units")
            result.append(f"Status: {stock_status.replace('_', ' ').title()}")
            result.append(f"Availability: {availability_msg}")
            result.append(f"Price: {product['price']}")

            # Add restocking information for low/out of stock
            if stock_status == "low_stock":
                result.append("")
                result.append("💡 **Recommendation:** Consider ordering soon as stock is running low.")
            elif stock_status == "out_of_stock":
                result.append("")
                result.append("📞 **Next Steps:** Contact customer service for restocking timeline and backorder options.")

            return "\n".join(result)
        else:
            return f"Product '{product_id}' not found. Available products: CB12-7.5, CB6-12, BCP-SMART. Please verify the product ID."

    except Exception as e:
        logger.error(f"Error checking stock for '{product_id}': {e}")
        return "I'm having trouble checking product stock right now. Please try again in a moment."


@tool
def compare_products(product_ids: str) -> str:
    """
    Compare features and specifications of multiple products.

    This tool provides a side-by-side comparison of product features,
    specifications, and pricing to help with decision making.

    Args:
        product_ids: Comma-separated list of product IDs to compare (e.g., "CB12-7.5,CB6-12")

    Returns:
        Detailed comparison table of the specified products
    """
    try:
        # Parse and normalize product IDs
        ids = [pid.strip().upper() for pid in product_ids.split(",")]
        logger.info(f"Comparing products: {ids}")

        valid_products = []
        invalid_ids = []

        for product_id in ids:
            if product_id in TEST_PRODUCTS:
                valid_products.append(TEST_PRODUCTS[product_id])
            else:
                invalid_ids.append(product_id)

        if invalid_ids:
            return f"Product(s) not found: {', '.join(invalid_ids)}. Available products: CB12-7.5, CB6-12, BCP-SMART."

        if len(valid_products) < 2:
            return "Please provide at least 2 valid product IDs for comparison."

        # Build comparison
        comparison = []
        comparison.append(f"🔍 **Product Comparison** ({len(valid_products)} products)")
        comparison.append("=" * 50)
        comparison.append("")

        # Basic Information
        comparison.append("**📋 Basic Information:**")
        for product in valid_products:
            comparison.append(f"• **{product['name']}** - {product['price']} - {product['category']}")
        comparison.append("")

        # Specifications Comparison
        comparison.append("**⚙️ Key Specifications:**")

        # Get all unique specification keys
        all_spec_keys = set()
        for product in valid_products:
            all_spec_keys.update(product['specifications'].keys())

        for spec_key in sorted(all_spec_keys):
            spec_name = spec_key.replace('_', ' ').title()
            comparison.append(f"**{spec_name}:**")
            for product in valid_products:
                spec_value = product['specifications'].get(spec_key, "N/A")
                comparison.append(f"  • {product['name']}: {spec_value}")
            comparison.append("")

        # Stock and Availability
        comparison.append("**📦 Availability:**")
        for product in valid_products:
            stock_emoji = "✅" if product["stock_status"] == "in_stock" else "⚠️" if product["stock_status"] == "low_stock" else "❌"
            comparison.append(f"  • {product['name']}: {stock_emoji} {product['stock_quantity']} units ({product['stock_status'].replace('_', ' ')})")
        comparison.append("")

        # Applications
        comparison.append("**🎯 Applications:**")
        for product in valid_products:
            comparison.append(f"**{product['name']}:**")
            for app in product['applications']:
                comparison.append(f"  • {app}")
            comparison.append("")

        # Features
        comparison.append("**✨ Key Features:**")
        for product in valid_products:
            comparison.append(f"**{product['name']}:**")
            for feature in product['features']:
                comparison.append(f"  • {feature}")
            comparison.append("")

        # Warranty
        comparison.append("**🛡️ Warranty:**")
        for product in valid_products:
            comparison.append(f"  • {product['name']}: {product['warranty']}")

        return "\n".join(comparison)

    except Exception as e:
        logger.error(f"Error comparing products '{product_ids}': {e}")
        return "I'm having trouble comparing these products right now. Please try again in a moment."


# List of available tools for the products agent
available_tools = [search_products, get_product_details, check_product_stock, compare_products]

__all__ = ["search_products", "get_product_details", "check_product_stock", "compare_products", "available_tools"]