"""
Product Management Tools for the Products Agent.

This module provides product search, details, and comparison functionality with static test data
for various product scenarios. This will be iteratively built into a fully functional
product catalog system in future phases.
"""

import logging
import os
import requests
from typing import Dict, Any, List
from langchain_core.tools import tool
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Static test product data with comprehensive details
TEST_PRODUCTS = {
    "CB12-7.5": {
        "product_id": "CB12-7.5",
        "sku": "CB12-7.5",
        "handle": "cb12-7-5",
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
        "handle": "cb6-12",
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
        "handle": "battery-charger-pro",
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


def _construct_product_url(handle: str) -> str:
    """
    Construct a customer-facing product URL from a product handle.

    Args:
        handle: Product handle (URL-safe identifier)

    Returns:
        Full product URL on the public store
    """
    store_url = os.getenv('SHOPIFY_STORE_URL', 'https://chromebattery.com')
    # Remove trailing slash if present
    store_url = store_url.rstrip('/')
    return f"{store_url}/products/{handle}"


@tool
def search_products(query: str) -> str:
    """
    Search for products by name, category, or keywords from the Shopify store.

    This tool searches the live Shopify product catalog for matching products based on
    name, category, vendor, product type, and tags.

    Args:
        query: Search term (product name, category, or keywords)

    Returns:
        List of matching products with basic information
    """
    try:
        logger.info(f"Searching Shopify products for: {query}")

        # Check if Shopify credentials are configured
        store_domain = os.getenv('SHOPIFY_STORE_DOMAIN')
        access_token = os.getenv('SHOPIFY_ACCESS_TOKEN')
        api_version = os.getenv('SHOPIFY_API_VERSION', '2024-10')

        if not store_domain or not access_token:
            logger.warning("Shopify credentials not configured, falling back to test data")
            return _search_test_products(query)

        # GraphQL query for product search
        graphql_query = """
        query SearchProducts($query: String!) {
            products(first: 20, query: $query) {
                edges {
                    node {
                        id
                        title
                        handle
                        description(truncateAt: 200)
                        productType
                        vendor
                        status
                        totalInventory
                        priceRangeV2 {
                            minVariantPrice {
                                amount
                                currencyCode
                            }
                        }
                        variants(first: 1) {
                            edges {
                                node {
                                    sku
                                    price
                                    inventoryQuantity
                                    availableForSale
                                }
                            }
                        }
                        tags
                    }
                }
            }
        }
        """

        # Make GraphQL request to Shopify
        response = requests.post(
            f"https://{store_domain}/admin/api/{api_version}/graphql.json",
            json={"query": graphql_query, "variables": {"query": query}},
            headers={
                "X-Shopify-Access-Token": access_token,
                "Content-Type": "application/json"
            },
            timeout=30
        )

        if response.status_code != 200:
            logger.error(f"Shopify API error: {response.status_code} - {response.text}")
            return f"Unable to search products right now. API returned status {response.status_code}."

        data = response.json()

        # Check for GraphQL errors
        if 'errors' in data:
            logger.error(f"GraphQL errors: {data['errors']}")
            return "There was an issue with the product search query. Please try again."

        products = data.get('data', {}).get('products', {}).get('edges', [])

        if not products:
            return f"No products found matching '{query}'. Try searching for different keywords or product categories."

        # Format results
        results = []
        results.append(f"Found {len(products)} product(s) matching '{query}':\n")

        for edge in products:
            product = edge['node']

            # Extract basic info
            title = product.get('title', 'Unknown Product')
            handle = product.get('handle', '')
            description = product.get('description', 'No description available')
            product_type = product.get('productType', 'Uncategorized')
            vendor = product.get('vendor', 'Unknown Vendor')
            status = product.get('status', 'UNKNOWN')
            total_inventory = product.get('totalInventory', 0) or 0

            # Extract price info
            price_range = product.get('priceRangeV2', {})
            min_price = price_range.get('minVariantPrice', {})
            price_amount = min_price.get('amount', '0.00')
            currency = min_price.get('currencyCode', 'USD')
            formatted_price = f"${price_amount} {currency}"

            # Extract variant info
            variants = product.get('variants', {}).get('edges', [])
            sku = "N/A"
            if variants:
                first_variant = variants[0]['node']
                sku = first_variant.get('sku') or 'N/A'

            # Determine stock status
            if total_inventory > 20:
                stock_status = "in_stock"
                stock_emoji = "✅"
            elif total_inventory > 0:
                stock_status = "low_stock"
                stock_emoji = "⚠️"
            else:
                stock_status = "out_of_stock"
                stock_emoji = "❌"

            # Construct product URL
            product_url = _construct_product_url(handle) if handle else None

            # Format output with clickable title
            if product_url:
                results.append(f"{stock_emoji} [**{title}**]({product_url}) (SKU: {sku})")
            else:
                results.append(f"{stock_emoji} **{title}** (SKU: {sku})")
            results.append(f"   Price: {formatted_price}")
            results.append(f"   Category: {product_type}")
            results.append(f"   Vendor: {vendor}")
            results.append(f"   Stock: {total_inventory} units ({stock_status.replace('_', ' ')})")
            results.append(f"   Description: {description}")
            results.append("")

        return "\n".join(results)

    except requests.exceptions.RequestException as e:
        logger.error(f"Network error searching Shopify products: {e}")
        return "Unable to connect to the product catalog right now. Please try again in a moment."
    except Exception as e:
        logger.error(f"Error searching Shopify products for '{query}': {e}")
        return "I'm having trouble searching the product catalog right now. Please try again in a moment."


def _search_test_products(query: str) -> str:
    """Fallback function to search test products when Shopify is not configured."""
    try:
        logger.info(f"Using test data for product search: {query}")

        query_lower = query.lower().strip()
        matching_products = []

        # Search through test products
        for product_id, product in TEST_PRODUCTS.items():
            if (query_lower in product["name"].lower() or
                query_lower in product["category"].lower() or
                any(query_lower in app.lower() for app in product["applications"]) or
                any(query_lower in feature.lower() for feature in product["features"])):
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
            results.append(f"Found {len(matching_products)} product(s) matching '{query}' (using test data):\n")

            for product in matching_products:
                stock_emoji = "✅" if product["stock_status"] == "in_stock" else "⚠️"
                # Construct product URL
                product_url = _construct_product_url(product['handle'])
                results.append(f"{stock_emoji} [**{product['name']}**]({product_url}) (SKU: {product['sku']})")
                results.append(f"   Price: {product['price']}")
                results.append(f"   Category: {product['category']}")
                results.append(f"   Stock: {product['stock_quantity']} units ({product['stock_status'].replace('_', ' ')})")
                results.append(f"   Description: {product['description'][:100]}...")
                results.append("")

            return "\n".join(results)
        else:
            return f"No products found matching '{query}'. Try searching for 'batteries', 'chargers', '12V', '6V', or specific product names."

    except Exception as e:
        logger.error(f"Error in fallback product search: {e}")
        return "I'm having trouble searching the product catalog right now."


@tool
def get_product_details(product_id: str) -> str:
    """
    Get comprehensive details for a specific product from the Shopify store.

    This tool retrieves complete product information including specifications,
    pricing, stock status, variants, and technical details.

    Args:
        product_id: Product ID, SKU, or handle to retrieve details for

    Returns:
        Complete product details or error message if product not found
    """
    try:
        logger.info(f"Getting product details for: {product_id}")

        # Check if Shopify credentials are configured
        store_domain = os.getenv('SHOPIFY_STORE_DOMAIN')
        access_token = os.getenv('SHOPIFY_ACCESS_TOKEN')
        api_version = os.getenv('SHOPIFY_API_VERSION', '2024-10')

        if not store_domain or not access_token:
            logger.warning("Shopify credentials not configured, falling back to test data")
            return _get_test_product_details(product_id)

        # Step 1: Determine if we have a GID, SKU, or handle
        product_gid = None
        if product_id.startswith('gid://shopify/Product/'):
            product_gid = product_id
        else:
            # Search for product by SKU or handle
            search_query = f'sku:{product_id}' if not product_id.islower() else f'handle:{product_id}'

            find_product_query = """
            query FindProduct($query: String!) {
                products(first: 1, query: $query) {
                    edges {
                        node {
                            id
                        }
                    }
                }
            }
            """

            try:
                response = requests.post(
                    f"https://{store_domain}/admin/api/{api_version}/graphql.json",
                    json={"query": find_product_query, "variables": {"query": search_query}},
                    headers={
                        "X-Shopify-Access-Token": access_token,
                        "Content-Type": "application/json"
                    },
                    timeout=30
                )

                if response.status_code == 200:
                    data = response.json()
                    if 'errors' not in data:
                        products = data.get('data', {}).get('products', {}).get('edges', [])
                        if products:
                            product_gid = products[0]['node']['id']

            except Exception as e:
                logger.error(f"Error finding product by {product_id}: {e}")
                return f"Unable to find product '{product_id}'. Please try again in a moment."

        if not product_gid:
            return f"Product '{product_id}' not found. Please check the product ID, SKU, or handle and try again."

        # Step 2: Fetch detailed product information
        detailed_query = """
        query GetProductDetails($id: ID!) {
            product(id: $id) {
                id
                title
                description
                descriptionHtml
                handle
                productType
                vendor
                status
                tags
                seo {
                    title
                    description
                }
                totalInventory
                priceRangeV2 {
                    minVariantPrice {
                        amount
                        currencyCode
                    }
                    maxVariantPrice {
                        amount
                        currencyCode
                    }
                }
                variants(first: 10) {
                    edges {
                        node {
                            id
                            title
                            sku
                            price
                            compareAtPrice
                            inventoryQuantity
                            availableForSale
                            selectedOptions {
                                name
                                value
                            }
                            inventoryItem {
                                tracked
                            }
                        }
                    }
                }
                collections(first: 5) {
                    edges {
                        node {
                            title
                        }
                    }
                }
                media(first: 5) {
                    edges {
                        node {
                            alt
                            mediaContentType
                            ... on MediaImage {
                                image {
                                    url
                                    altText
                                }
                            }
                        }
                    }
                }
                metafields(first: 20, namespace: "custom") {
                    edges {
                        node {
                            key
                            value
                            type
                        }
                    }
                }
            }
        }
        """

        try:
            response = requests.post(
                f"https://{store_domain}/admin/api/{api_version}/graphql.json",
                json={"query": detailed_query, "variables": {"id": product_gid}},
                headers={
                    "X-Shopify-Access-Token": access_token,
                    "Content-Type": "application/json"
                },
                timeout=30
            )

            if response.status_code != 200:
                logger.error(f"Shopify API error: {response.status_code} - {response.text}")
                return f"Unable to get product details right now. API returned status {response.status_code}."

            data = response.json()

            # Check for GraphQL errors
            if 'errors' in data:
                logger.error(f"GraphQL errors: {data['errors']}")
                return "There was an issue fetching product details. Please try again."

            product = data.get('data', {}).get('product')
            if not product:
                return f"Product '{product_id}' not found."

            # Format the response
            return _format_shopify_product_details(product)

        except requests.exceptions.RequestException as e:
            logger.error(f"Network error fetching product details: {e}")
            return "Unable to connect to retrieve product details right now. Please try again in a moment."
        except Exception as e:
            logger.error(f"Error fetching product details for '{product_id}': {e}")
            return "I'm having trouble retrieving product details right now. Please try again in a moment."

    except Exception as e:
        logger.error(f"Error getting product details for '{product_id}': {e}")
        return "I'm having trouble retrieving product details right now. Please try again in a moment."


def _format_shopify_product_details(product: dict) -> str:
    """Format Shopify product data to match the existing structure."""
    try:
        details = []

        # Basic product information
        title = product.get('title', 'Unknown Product')
        handle = product.get('handle', '')
        product_type = product.get('productType', 'Uncategorized')
        vendor = product.get('vendor', 'Unknown Vendor')
        description = product.get('description', 'No description available')
        total_inventory = product.get('totalInventory', 0) or 0

        # Get primary variant info
        variants = product.get('variants', {}).get('edges', [])
        primary_sku = "N/A"
        if variants:
            primary_sku = variants[0]['node'].get('sku') or 'N/A'

        # Price information
        price_range = product.get('priceRangeV2', {})
        min_price = price_range.get('minVariantPrice', {})
        max_price = price_range.get('maxVariantPrice', {})

        price_amount = min_price.get('amount', '0.00')
        currency = min_price.get('currencyCode', 'USD')

        if min_price.get('amount') == max_price.get('amount'):
            formatted_price = f"${price_amount} {currency}"
        else:
            formatted_price = f"${min_price.get('amount', '0.00')} - ${max_price.get('amount', '0.00')} {currency}"

        # Stock status
        if total_inventory > 20:
            stock_status = "in_stock"
            stock_emoji = "✅"
        elif total_inventory > 0:
            stock_status = "low_stock"
            stock_emoji = "⚠️"
        else:
            stock_status = "out_of_stock"
            stock_emoji = "❌"

        # Build the response with clickable title
        product_url = _construct_product_url(handle) if handle else None
        if product_url:
            details.append(f"📦 [**{title}**]({product_url})")
        else:
            details.append(f"📦 **{title}**")
        details.append(f"SKU: {primary_sku}")
        details.append(f"Handle: {handle}")
        details.append(f"Category: {product_type}")
        details.append(f"Vendor: {vendor}")
        details.append(f"Price: {formatted_price}")
        details.append("")

        # Description
        details.append("**Description:**")
        details.append(description)
        details.append("")

        # Product specifications from metafields and variants
        details.append("**Technical Specifications:**")

        # Extract specifications from metafields
        metafields = product.get('metafields', {}).get('edges', [])
        spec_found = False
        for edge in metafields:
            field = edge['node']
            key = field.get('key', '').replace('_', ' ').title()
            value = field.get('value', '')
            if key and value:
                details.append(f"• {key}: {value}")
                spec_found = True

        # If no metafields, extract from variant options
        if not spec_found and variants:
            variant_specs = {}
            for edge in variants:
                variant = edge['node']
                options = variant.get('selectedOptions', [])
                for option in options:
                    name = option.get('name', '')
                    value = option.get('value', '')
                    if name and value and name.lower() not in ['title', 'default title']:
                        variant_specs[name] = variant_specs.get(name, set())
                        variant_specs[name].add(value)

            for spec_name, values in variant_specs.items():
                if len(values) == 1:
                    details.append(f"• {spec_name}: {next(iter(values))}")
                else:
                    details.append(f"• {spec_name}: {', '.join(sorted(values))}")
                spec_found = True

        if not spec_found:
            details.append("• Product Type: " + product_type)
            if vendor != "Unknown Vendor":
                details.append("• Brand: " + vendor)

        details.append("")

        # Stock Information
        details.append("**Availability:**")
        details.append(f"{stock_emoji} Stock Status: {stock_status.replace('_', ' ').title()}")
        details.append(f"• Total Inventory: {total_inventory} units")

        # Variant details if multiple variants
        if len(variants) > 1:
            details.append(f"• Available Variants: {len(variants)}")
            for i, edge in enumerate(variants[:5], 1):  # Show up to 5 variants
                variant = edge['node']
                variant_title = variant.get('title', f'Variant {i}')
                variant_price = variant.get('price', '0.00')
                variant_inventory = variant.get('inventoryQuantity', 0) or 0
                variant_sku = variant.get('sku') or 'N/A'

                details.append(f"  - {variant_title}: ${variant_price} {currency} (SKU: {variant_sku}, Stock: {variant_inventory})")

        details.append("")

        # Applications from collections and tags
        collections = product.get('collections', {}).get('edges', [])
        tags = product.get('tags', [])

        applications = []
        for edge in collections:
            collection_title = edge['node'].get('title', '')
            if collection_title and collection_title not in applications:
                applications.append(collection_title)

        # Add relevant tags as applications
        relevant_tags = [tag for tag in tags if any(keyword in tag.lower() for keyword in
                        ['battery', 'power', 'energy', 'backup', 'solar', 'marine', 'automotive', 'ups'])]
        applications.extend(relevant_tags[:3])  # Add up to 3 relevant tags

        if applications:
            details.append("**Recommended Applications:**")
            for app in applications[:5]:  # Show up to 5 applications
                details.append(f"• {app}")
        else:
            details.append("**Recommended Applications:**")
            details.append(f"• {product_type}")
            details.append("• General Use")

        details.append("")

        # Key Features from tags and product info
        features = []

        # Add relevant tags as features
        feature_tags = [tag for tag in tags if tag not in applications and len(tag) > 2]
        features.extend(feature_tags[:4])  # Add up to 4 feature tags

        # Add default features based on product type
        if not features:
            if 'battery' in product_type.lower():
                features.extend(['Reliable Performance', 'Long-lasting', 'Professional Grade'])
            elif 'charger' in product_type.lower():
                features.extend(['Smart Charging', 'Multiple Modes', 'Safety Features'])
            else:
                features.extend(['High Quality', 'Professional Grade', 'Durable Construction'])

        details.append("**Key Features:**")
        for feature in features[:5]:  # Show up to 5 features
            details.append(f"• {feature}")
        details.append("")

        # SEO and additional info
        seo = product.get('seo', {})
        if seo.get('description'):
            details.append("**Additional Information:**")
            details.append(seo['description'])
            details.append("")

        # Warranty info (placeholder - would come from metafields in real implementation)
        details.append("**Warranty:** Standard manufacturer warranty applies")

        return "\n".join(details)

    except Exception as e:
        logger.error(f"Error formatting Shopify product details: {e}")
        return "Product details retrieved but could not be formatted properly."


def _get_test_product_details(product_id: str) -> str:
    """Fallback function to get test product details when Shopify is not configured."""
    try:
        logger.info(f"Using test data for product details: {product_id}")

        # Normalize product ID
        product_id = product_id.strip().upper()

        if product_id in TEST_PRODUCTS:
            product = TEST_PRODUCTS[product_id]

            details = []
            # Make title clickable
            product_url = _construct_product_url(product['handle'])
            details.append(f"📦 [**{product['name']}**]({product_url}) (Test Data)")
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
            return f"Product '{product_id}' not found in test data. Available products: CB12-7.5, CB6-12, BCP-SMART. Please check the product ID and try again."

    except Exception as e:
        logger.error(f"Error in fallback product details: {e}")
        return "I'm having trouble retrieving product details right now."


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