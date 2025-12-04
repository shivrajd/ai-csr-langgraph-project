"""Qdrant Cloud retriever for vehicle-battery fitment queries.

This module provides semantic search functionality for the chrome_fitments collection
stored in Qdrant Cloud, supporting bidirectional queries:
- Vehicle → Battery: "What battery fits my 2020 Honda CBR600?"
- Battery → Vehicle: "What vehicles use battery YTZ7S?"

Uses OpenAI's text-embedding-ada-002 for query embeddings (1536 dimensions).
"""

import os
import logging
from typing import List, Dict, Any

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from openai import OpenAI

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Constants
COLLECTION_NAME = "chrome_fitments"
EMBEDDING_MODEL = "text-embedding-ada-002"


class QdrantFitmentsRetriever:
    """Qdrant Cloud retriever for vehicle-battery fitment lookups.

    Uses semantic search with metadata filtering to find:
    - Batteries that fit a specific vehicle (make/model/year)
    - Vehicles compatible with a specific battery model
    """

    def __init__(self):
        """Initialize Qdrant Cloud connection and OpenAI client."""
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        openai_api_key = os.getenv("OPENAI_API_KEY")

        if not all([qdrant_url, qdrant_api_key]):
            raise ValueError(
                "Missing Qdrant Cloud credentials. "
                "Required: QDRANT_URL, QDRANT_API_KEY"
            )

        if not openai_api_key:
            raise ValueError(
                "Missing OpenAI API key. Required: OPENAI_API_KEY"
            )

        # Remove quotes if present (from .env file)
        qdrant_url = qdrant_url.strip('"').strip("'")
        qdrant_api_key = qdrant_api_key.strip('"').strip("'")

        self.qdrant_client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key
        )
        self.openai_client = OpenAI(api_key=openai_api_key)

        logger.info(f"Connected to Qdrant Cloud collection: {COLLECTION_NAME}")

    def _embed_query(self, query: str) -> List[float]:
        """Generate embedding for a query using OpenAI.

        Args:
            query: The search query text

        Returns:
            List of floats representing the embedding vector
        """
        response = self.openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=query
        )
        return response.data[0].embedding

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
            - score: Similarity score (higher = better match)
        """
        try:
            # Generate query embedding
            query_vector = self._embed_query(query)

            # Search Qdrant with filter for vehicle_to_battery type
            results = self.qdrant_client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                query_filter=Filter(
                    must=[
                        FieldCondition(
                            key="type",
                            match=MatchValue(value="vehicle_to_battery")
                        )
                    ]
                ),
                limit=top_k,
                with_payload=True
            )

            # Format results
            formatted = []
            for point in results.points:
                payload = point.payload or {}
                formatted.append({
                    "id": point.id,
                    "document": payload.get("document", ""),
                    "chrome_model": payload.get("chrome_model", ""),
                    "chrome_sku": payload.get("chrome_sku", ""),
                    "make": payload.get("make", ""),
                    "model": payload.get("model", ""),
                    "year": payload.get("year", ""),
                    "yuasa_model": payload.get("yuasa_model", ""),
                    "score": point.score
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
            - score: Similarity score (higher = better match)
        """
        try:
            # Search for documents that mention this battery model
            query = f"{battery_model} battery fits"
            query_vector = self._embed_query(query)

            # Search Qdrant with filter for battery_to_vehicle type
            results = self.qdrant_client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                query_filter=Filter(
                    must=[
                        FieldCondition(
                            key="type",
                            match=MatchValue(value="battery_to_vehicle")
                        )
                    ]
                ),
                limit=top_k,
                with_payload=True
            )

            # Format results, filtering for battery model match
            # Supports both exact ("YTZ10S-BS") and partial ("YTZ10S") matches
            formatted = []
            search_term = battery_model.upper().replace("-BS", "").replace("-", "")

            for point in results.points:
                payload = point.payload or {}
                chrome_model = payload.get("chrome_model", "").upper()
                # Normalize for comparison (remove -BS suffix and dashes)
                normalized = chrome_model.replace("-BS", "").replace("-", "")

                # Match if search term matches or is contained in the model
                if search_term == normalized or search_term in normalized:
                    formatted.append({
                        "id": point.id,
                        "document": payload.get("document", ""),
                        "make": payload.get("make", ""),
                        "model": payload.get("model", ""),
                        "year": payload.get("year", ""),
                        "chrome_model": payload.get("chrome_model", ""),
                        "chrome_sku": payload.get("chrome_sku", ""),
                        "score": point.score
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
            info = self.qdrant_client.get_collection(collection_name=COLLECTION_NAME)
            return {
                "collection_name": COLLECTION_NAME,
                "document_count": info.points_count,
                "vector_size": info.config.params.vectors.size if info.config.params.vectors else None
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}
