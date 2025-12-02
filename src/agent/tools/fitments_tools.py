"""Vehicle-Battery Fitment Tools for the Fitments Agent.

This module provides tools to look up battery compatibility for vehicles
and find compatible vehicles for specific battery models using ChromaDB
semantic search.

The fitments data is synced from Supabase to ChromaDB Cloud for semantic
search capabilities, enabling natural language queries like:
- "What battery fits my 2020 Honda CBR600?"
- "Which vehicles use the YTZ7S battery?"
"""

import logging
from typing import List, Dict, Any
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


def _format_battery_results(results: List[Dict[str, Any]]) -> str:
    """Format battery search results for customer presentation.

    Args:
        results: List of battery matches from retriever

    Returns:
        Formatted string with battery recommendations
    """
    if not results:
        return "No matching batteries found for this vehicle."

    # Deduplicate by chrome_model (same battery may appear multiple times)
    seen_models = set()
    unique_batteries = []
    for r in results:
        model = r.get("chrome_model", "")
        if model and model not in seen_models:
            seen_models.add(model)
            unique_batteries.append(r)

    if not unique_batteries:
        return "No matching batteries found for this vehicle."

    lines = ["**Compatible Batteries Found:**\n"]

    for i, battery in enumerate(unique_batteries[:5], 1):  # Limit to top 5
        chrome_model = battery.get("chrome_model", "Unknown")
        chrome_sku = battery.get("chrome_sku", "")
        make = battery.get("make", "")
        model = battery.get("model", "")
        year = battery.get("year", "")

        vehicle_info = " ".join(filter(None, [make, model, year]))

        lines.append(f"{i}. **{chrome_model}**")
        if chrome_sku:
            lines.append(f"   - SKU: `{chrome_sku}`")
        if vehicle_info:
            lines.append(f"   - Fits: {vehicle_info}")
        lines.append("")

    # Add note about Shopify lookup
    if unique_batteries:
        primary_sku = unique_batteries[0].get("chrome_sku", "")
        primary_model = unique_batteries[0].get("chrome_model", "")
        lines.append(f"\n**Recommended:** {primary_model} (SKU: {primary_sku})")
        lines.append("\nFor pricing, availability, and purchase link, search for this product in our store.")

    return "\n".join(lines)


def _format_vehicle_results(results: List[Dict[str, Any]], battery_model: str) -> str:
    """Format vehicle search results for customer presentation.

    Args:
        results: List of vehicle matches from retriever
        battery_model: The battery model searched for

    Returns:
        Formatted string with compatible vehicles
    """
    if not results:
        return f"No compatible vehicles found for battery {battery_model}."

    # Deduplicate by make/model/year
    seen_vehicles = set()
    unique_vehicles = []
    for r in results:
        key = (r.get("make", ""), r.get("model", ""), r.get("year", ""))
        if key not in seen_vehicles:
            seen_vehicles.add(key)
            unique_vehicles.append(r)

    if not unique_vehicles:
        return f"No compatible vehicles found for battery {battery_model}."

    lines = [f"**Vehicles Compatible with {battery_model}:**\n"]

    # Group by make for better organization
    vehicles_by_make: Dict[str, List[Dict]] = {}
    for v in unique_vehicles:
        make = v.get("make", "Other")
        if make not in vehicles_by_make:
            vehicles_by_make[make] = []
        vehicles_by_make[make].append(v)

    for make in sorted(vehicles_by_make.keys()):
        lines.append(f"\n**{make}:**")
        for v in vehicles_by_make[make][:10]:  # Limit per make
            model = v.get("model", "")
            year = v.get("year", "")
            if model:
                lines.append(f"  - {model} ({year})" if year else f"  - {model}")

    total = len(unique_vehicles)
    if total > 30:
        lines.append(f"\n*...and {total - 30} more vehicles*")

    return "\n".join(lines)


@tool
def find_battery_for_vehicle(vehicle_query: str) -> str:
    """
    Find compatible batteries for a vehicle based on make, model, and year.

    Use this tool when a customer asks which battery fits their vehicle.
    The tool searches our fitment database to find matching batteries.

    Args:
        vehicle_query: Description of the vehicle - can include make, model, year,
                       and vehicle type. Examples:
                       - "2020 Honda CBR600"
                       - "Arctic Cat DVX50 ATV 2006"
                       - "Yamaha YZF-R6 2019 motorcycle"

    Returns:
        List of compatible batteries with model names and SKUs.
        The primary recommendation includes the SKU for Shopify product lookup.
    """
    try:
        from src.agent.tools.chroma_retriever import ChromaFitmentsRetriever

        logger.info(f"Searching batteries for vehicle: {vehicle_query}")

        # Initialize retriever and search
        retriever = ChromaFitmentsRetriever()
        results = retriever.search_battery_for_vehicle(query=vehicle_query, top_k=10)

        # Format and return results
        formatted = _format_battery_results(results)
        logger.info(f"Found {len(results)} battery matches")

        return formatted

    except Exception as e:
        logger.error(f"Error finding battery for vehicle: {e}")
        return (
            "I'm having trouble searching our fitment database right now. "
            "Please try again or contact our support team for assistance with "
            "finding the right battery for your vehicle."
        )


@tool
def find_vehicles_for_battery(battery_model: str) -> str:
    """
    Find vehicles that are compatible with a specific battery model.

    Use this tool when a customer asks which vehicles use a particular battery,
    or to verify if their vehicle is compatible with a battery they're considering.

    Args:
        battery_model: The battery model name to search for. Examples:
                       - "YTZ7S"
                       - "YTX14-BS"
                       - "YB16CL-B"

    Returns:
        List of compatible vehicles organized by make, with model and year information.
    """
    try:
        from src.agent.tools.chroma_retriever import ChromaFitmentsRetriever

        logger.info(f"Searching vehicles for battery: {battery_model}")

        # Initialize retriever and search
        retriever = ChromaFitmentsRetriever()
        results = retriever.search_vehicles_for_battery(
            battery_model=battery_model,
            top_k=50  # Get more results for vehicle listings
        )

        # Format and return results
        formatted = _format_vehicle_results(results, battery_model)
        logger.info(f"Found {len(results)} vehicle matches")

        return formatted

    except Exception as e:
        logger.error(f"Error finding vehicles for battery: {e}")
        return (
            f"I'm having trouble searching for vehicles compatible with {battery_model}. "
            "Please try again or contact our support team for assistance."
        )
