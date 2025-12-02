"""ChromaDB Cloud retriever for vehicle-battery fitment queries.

This module provides semantic search functionality for the chrome_fitments collection
stored in ChromaDB Cloud, supporting bidirectional queries:
- Vehicle → Battery: "What battery fits my 2020 Honda CBR600?"
- Battery → Vehicle: "What vehicles use battery YTZ7S?"
"""

import os
import logging
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from chromadb import CloudClient

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Constants
COLLECTION_NAME = "chrome_fitments"


class ChromaFitmentsRetriever:
    """ChromaDB Cloud retriever for vehicle-battery fitment lookups.

    Uses semantic search with metadata filtering to find:
    - Batteries that fit a specific vehicle (make/model/year)
    - Vehicles compatible with a specific battery model
    """

    def __init__(self):
        """Initialize ChromaDB Cloud connection and get collection."""
        api_key = os.getenv("CHROMA_API_KEY")
        tenant = os.getenv("CHROMA_TENANT")
        database = os.getenv("CHROMA_DATABASE")

        if not all([api_key, tenant, database]):
            raise ValueError(
                "Missing ChromaDB Cloud credentials. "
                "Required: CHROMA_API_KEY, CHROMA_TENANT, CHROMA_DATABASE"
            )

        self.client = CloudClient(
            tenant=tenant,
            database=database,
            api_key=api_key
        )

        # Get the fitments collection
        self.collection = self.client.get_collection(name=COLLECTION_NAME)
        logger.info(f"Connected to ChromaDB Cloud collection: {COLLECTION_NAME}")

    def search_battery_for_vehicle(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search for batteries that fit a vehicle.

        Args:
            query: Natural language query describing the vehicle
                   (e.g., "2020 Honda CBR600", "Arctic Cat ATV 2018")
            top_k: Maximum number of results to return

        Returns:
            List of matching fitments with metadata:
            - chrome_model: Battery model name
            - chrome_sku: SKU for Shopify lookup
            - make, model, year: Vehicle details
            - distance: Similarity score (lower = better match)
        """
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where={"type": "vehicle_to_battery"},
                include=["documents", "metadatas", "distances"]
            )

            # Format results
            formatted = []
            if results and results.get("ids") and results["ids"][0]:
                for i, doc_id in enumerate(results["ids"][0]):
                    metadata = results["metadatas"][0][i] if results.get("metadatas") else {}
                    document = results["documents"][0][i] if results.get("documents") else ""
                    distance = results["distances"][0][i] if results.get("distances") else None

                    formatted.append({
                        "id": doc_id,
                        "document": document,
                        "chrome_model": metadata.get("chrome_model", ""),
                        "chrome_sku": metadata.get("chrome_sku", ""),
                        "make": metadata.get("make", ""),
                        "model": metadata.get("model", ""),
                        "year": metadata.get("year", ""),
                        "yuasa_model": metadata.get("yuasa_model", ""),
                        "distance": distance
                    })

            logger.info(f"Found {len(formatted)} battery matches for query: {query[:50]}...")
            return formatted

        except Exception as e:
            logger.error(f"Error searching batteries for vehicle: {e}")
            return []

    def search_vehicles_for_battery(
        self,
        battery_model: str,
        top_k: int = 20
    ) -> List[Dict[str, Any]]:
        """Search for vehicles compatible with a battery model.

        Args:
            battery_model: Battery model name (e.g., "YTZ7S", "YTX14-BS")
            top_k: Maximum number of results to return

        Returns:
            List of matching vehicles with metadata:
            - make, model, year: Vehicle details
            - document: Full description text
            - distance: Similarity score (lower = better match)
        """
        try:
            # Search for documents that mention this battery model
            query = f"{battery_model} battery fits"

            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where={"type": "battery_to_vehicle"},
                include=["documents", "metadatas", "distances"]
            )

            # Format results, filtering for exact battery model match
            formatted = []
            if results and results.get("ids") and results["ids"][0]:
                for i, doc_id in enumerate(results["ids"][0]):
                    metadata = results["metadatas"][0][i] if results.get("metadatas") else {}
                    document = results["documents"][0][i] if results.get("documents") else ""
                    distance = results["distances"][0][i] if results.get("distances") else None

                    # Only include if chrome_model matches (case-insensitive)
                    if metadata.get("chrome_model", "").upper() == battery_model.upper():
                        formatted.append({
                            "id": doc_id,
                            "document": document,
                            "make": metadata.get("make", ""),
                            "model": metadata.get("model", ""),
                            "year": metadata.get("year", ""),
                            "chrome_model": metadata.get("chrome_model", ""),
                            "chrome_sku": metadata.get("chrome_sku", ""),
                            "distance": distance
                        })

            logger.info(f"Found {len(formatted)} vehicle matches for battery: {battery_model}")
            return formatted

        except Exception as e:
            logger.error(f"Error searching vehicles for battery: {e}")
            return []

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics for debugging.

        Returns:
            Dictionary with collection count and metadata
        """
        try:
            count = self.collection.count()
            return {
                "collection_name": COLLECTION_NAME,
                "document_count": count
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}
