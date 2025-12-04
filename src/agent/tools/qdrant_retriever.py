"""Qdrant Cloud retriever for vehicle-battery fitment queries.

This module provides semantic search functionality for the chrome_fitments collection
stored in Qdrant Cloud, supporting bidirectional queries:
- Vehicle → Battery: "What battery fits my 2020 Honda CBR600?"
- Battery → Vehicle: "What vehicles use battery YTZ7S?"

Uses OpenAI's text-embedding-ada-002 for query embeddings (1536 dimensions).
"""

import os
import logging
import re
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from openai import OpenAI
from supabase import create_client, Client

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

        # Initialize Supabase client for fallback queries
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")
        if supabase_url and supabase_key:
            self.supabase: Optional[Client] = create_client(supabase_url, supabase_key)
        else:
            self.supabase = None
            logger.warning("Supabase credentials not found - fallback queries disabled")

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

    def _extract_search_terms(self, query: str) -> Dict[str, Any]:
        """Extract and normalize search terms from a vehicle query.

        Handles various query formats:
        - "CX-Sport 100" (model only)
        - "2020 Honda CBR600" (year + make + model)
        - "Arctic Cat ATV 2018" (make + type + year)
        - "Aeon Cobra" (make + model)

        Args:
            query: The user's vehicle search query

        Returns:
            Dict with normalized terms:
            - 'year': Extracted year (4-digit) or None
            - 'make': Potential make name or None
            - 'model_terms': List of model-related terms (normalized)
            - 'all_terms': All normalized terms for matching
            - 'original_query': Original query string
        """
        import re

        # Normalize query: lowercase, strip whitespace
        normalized = query.lower().strip()

        # Extract 4-digit year if present (1900-2099)
        year_match = re.search(r'\b(19\d{2}|20\d{2})\b', normalized)
        year = year_match.group(1) if year_match else None

        # Remove year from query for further processing
        query_without_year = re.sub(r'\b(19\d{2}|20\d{2})\b', '', normalized).strip()

        # Known vehicle types to filter out (not useful for matching)
        vehicle_types = {'motorcycle', 'atv', 'scooter', 'snowmobile', 'utv',
                         'pwc', 'jet ski', 'side-by-side', 'dirt bike', 'bike'}

        # Tokenize and filter - split on spaces, hyphens, slashes, commas
        tokens = re.split(r'[\s\-/,]+', query_without_year)
        tokens = [t for t in tokens if t and len(t) > 0]

        # Filter out vehicle types
        significant_tokens = [t for t in tokens if t not in vehicle_types]

        # Known makes for identification (common powersports manufacturers)
        known_makes = {
            'honda', 'yamaha', 'kawasaki', 'suzuki', 'bmw', 'ducati', 'ktm',
            'harley', 'triumph', 'aprilia', 'indian', 'victory', 'moto guzzi',
            'arctic cat', 'polaris', 'can-am', 'sea-doo', 'ski-doo', 'bombardier',
            'aeon', 'benzai', 'kymco', 'sym', 'piaggio', 'vespa', 'peugeot',
            'husqvarna', 'beta', 'gas gas', 'sherco', 'tm', 'royal enfield',
            'cfmoto', 'linhai', 'hisun', 'massimo', 'argo', 'textron'
        }

        # Check if first token is a known make
        make = None
        model_terms = significant_tokens.copy()
        if significant_tokens:
            first_token = significant_tokens[0]
            if first_token in known_makes:
                make = first_token
                model_terms = significant_tokens[1:]  # Remaining tokens are model
            # Handle compound makes like "Arctic Cat" or "Can-Am"
            elif len(significant_tokens) >= 2:
                compound = f"{significant_tokens[0]} {significant_tokens[1]}"
                if compound in known_makes:
                    make = compound
                    model_terms = significant_tokens[2:]

        return {
            'year': year,
            'make': make,
            'model_terms': model_terms,
            'all_terms': significant_tokens,
            'original_query': query
        }

    def _validate_vehicle_match(
        self,
        result: Dict[str, Any],
        search_terms: Dict[str, Any]
    ) -> tuple:
        """Validate if a search result matches the user's vehicle query.

        Uses multi-factor matching with weighted scoring:
        - Model name substring matching (highest weight)
        - Make matching (medium weight)
        - Year matching (lower weight, but important for exact matches)

        Args:
            result: Search result with make, model, year metadata
            search_terms: Extracted terms from _extract_search_terms()

        Returns:
            Tuple of (is_valid, confidence_score)
            - is_valid: True if result should be included
            - confidence_score: 0.0-1.0 indicating match quality
        """
        # Extract result metadata (normalize to lowercase)
        result_make = (result.get('make', '') or '').lower()
        result_model = (result.get('model', '') or '').lower()
        result_year = str(result.get('year', '') or '')

        # Normalize result model for comparison
        # Replace slashes, hyphens with spaces for token matching
        result_model_normalized = result_model.replace('/', ' ').replace('-', ' ')
        result_model_tokens = set(result_model_normalized.split())

        score = 0.0

        # 1. Model term matching (CRITICAL - highest weight: 0.6)
        model_terms = search_terms.get('model_terms', [])
        if model_terms:
            model_matches = 0
            for term in model_terms:
                # For numeric terms (like "100"), require exact token match
                # This prevents "100" from matching "gl1000" as a substring
                is_numeric = term.isdigit()

                if is_numeric:
                    # Numeric terms: only exact token match
                    if term in result_model_tokens:
                        model_matches += 1
                else:
                    # Non-numeric terms: check exact token match first
                    if term in result_model_tokens:
                        model_matches += 1
                    # Then check substring match in model (e.g., "sport" in "cx sport")
                    elif term in result_model_normalized:
                        model_matches += 1
                    # Finally check partial token match (e.g., "cbr" in "cbr1000rr")
                    elif any(term in token or token in term for token in result_model_tokens):
                        model_matches += 0.5

            model_score = min(model_matches / len(model_terms), 1.0)
            score += model_score * 0.6

        # 2. Make matching (weight: 0.25)
        query_make = search_terms.get('make')
        if query_make:
            if query_make in result_make or result_make in query_make:
                score += 0.25
            # Handle partial make matches (e.g., "aeon" matches "aeon (benzai)")
            elif query_make.split()[0] in result_make:
                score += 0.15

        # 3. Year matching (weight: 0.15)
        query_year = search_terms.get('year')
        if query_year:
            if query_year == result_year:
                score += 0.15

        # 4. All-terms fallback matching (for queries without clear make/model)
        # Check if any query term appears in result
        if not model_terms and not query_make:
            all_terms = search_terms.get('all_terms', [])
            if all_terms:
                term_matches = sum(
                    1 for term in all_terms
                    if term in result_model_normalized or term in result_make
                )
                fallback_score = min(term_matches / len(all_terms), 1.0)
                score = max(score, fallback_score * 0.5)

        # Determine if result is valid (threshold: 0.3)
        # This threshold allows partial matches while filtering out unrelated results
        is_valid = score >= 0.3

        return (is_valid, score)

    def _supabase_fallback_search(
        self,
        search_terms: Dict[str, Any],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Fallback to Supabase keyword search when semantic search fails.

        Uses ILIKE queries on the model field to find vehicles that match
        the search terms. This handles cases where semantic similarity fails
        (e.g., "CX-Sport 100" not finding "Cobra/CX-Sport 100").

        Args:
            search_terms: Extracted terms from _extract_search_terms()
            top_k: Maximum number of results

        Returns:
            List of matching fitments from Supabase
        """
        if not self.supabase:
            return []

        try:
            # Build search pattern from model terms
            model_terms = search_terms.get('model_terms', [])
            all_terms = search_terms.get('all_terms', [])
            query_year = search_terms.get('year')
            query_make = search_terms.get('make')

            # Use model terms if available, otherwise all terms
            search_list = model_terms if model_terms else all_terms
            if not search_list:
                return []

            # Start with base query
            query = self.supabase.table('chrome_fitments').select(
                'id, vehicle_type, make, model, year, cc, chrome_model, chrome_sku, yuasa_model'
            )

            # Build ILIKE conditions for each term
            # Using 'or' filters to find any match
            for term in search_list:
                if len(term) >= 2:  # Skip very short terms
                    query = query.ilike('model', f'%{term}%')

            # Add make filter if specified
            if query_make:
                query = query.ilike('make', f'%{query_make}%')

            # Add year filter if specified
            if query_year:
                query = query.eq('year', query_year)

            # Execute with limit
            result = query.limit(top_k * 3).execute()

            if not result.data:
                return []

            # Format results
            formatted = []
            seen_batteries = set()
            for row in result.data:
                chrome_model = row.get('chrome_model', '')
                # Deduplicate by battery model
                if chrome_model and chrome_model not in seen_batteries:
                    seen_batteries.add(chrome_model)
                    formatted.append({
                        "id": row.get('id'),
                        "document": f"Fallback: {row.get('make')} {row.get('model')} {row.get('year')}",
                        "chrome_model": chrome_model,
                        "chrome_sku": row.get('chrome_sku', ''),
                        "make": row.get('make', ''),
                        "model": row.get('model', ''),
                        "year": row.get('year', ''),
                        "yuasa_model": row.get('yuasa_model', ''),
                        "score": 0.8,  # Fixed score for fallback results
                        "source": "supabase_fallback"
                    })

                if len(formatted) >= top_k:
                    break

            logger.info(f"Supabase fallback found {len(formatted)} results")
            return formatted

        except Exception as e:
            logger.error(f"Supabase fallback search error: {e}")
            return []

    def search_battery_for_vehicle(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search for batteries that fit a vehicle.

        Uses a hybrid approach combining semantic search with post-retrieval validation
        to ensure accurate results. This prevents issues where semantically similar but
        incorrect vehicles are returned (e.g., "CX-Sport 100" matching "Gold Wing 1000").

        Args:
            query: Natural language query describing the vehicle
                   (e.g., "2020 Honda CBR600", "Arctic Cat ATV 2018")
            top_k: Maximum number of results to return

        Returns:
            List of matching fitments with metadata:
            - chrome_model: Battery model name
            - chrome_sku: SKU for Shopify lookup
            - make, model, year: Vehicle details
            - score: Combined similarity and validation score (higher = better match)
        """
        try:
            # Extract search terms for validation
            search_terms = self._extract_search_terms(query)

            # Generate query embedding
            query_vector = self._embed_query(query)

            # Search Qdrant with filter for vehicle_to_battery type
            # Request significantly more results to ensure correct matches aren't missed
            # during validation filtering (semantic search may rank correct match lower)
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
                limit=max(50, top_k * 10),  # Get many results to find correct matches
                with_payload=True
            )

            # Format and validate results
            validated_results = []
            for point in results.points:
                payload = point.payload or {}
                result_dict = {
                    "id": point.id,
                    "document": payload.get("document", ""),
                    "chrome_model": payload.get("chrome_model", ""),
                    "chrome_sku": payload.get("chrome_sku", ""),
                    "make": payload.get("make", ""),
                    "model": payload.get("model", ""),
                    "year": payload.get("year", ""),
                    "yuasa_model": payload.get("yuasa_model", ""),
                    "semantic_score": point.score
                }

                # Validate the result matches the query
                is_valid, match_score = self._validate_vehicle_match(result_dict, search_terms)

                if is_valid:
                    # Combine semantic score with validation score
                    # Weighted: 40% semantic + 60% validation
                    result_dict["score"] = (point.score * 0.4) + (match_score * 0.6)
                    result_dict["match_score"] = match_score
                    validated_results.append(result_dict)

            # Sort by combined score (highest first)
            validated_results.sort(key=lambda x: x["score"], reverse=True)

            # Limit to requested top_k
            validated_results = validated_results[:top_k]

            # If no validated results, try Supabase fallback (keyword search)
            # This handles cases where semantic search fails to find the vehicle
            # (e.g., "CX-Sport 100" not semantically similar to "Cobra/CX-Sport 100")
            if not validated_results:
                logger.info(f"Semantic search returned no validated results, trying Supabase fallback...")
                validated_results = self._supabase_fallback_search(search_terms, top_k)

            logger.info(
                f"Found {len(validated_results)} validated battery matches for query: {query[:50]}..."
            )
            return validated_results

        except Exception as e:
            logger.error(f"Error searching batteries for vehicle: {e}")
            return []

    def search_vehicles_for_battery(
        self,
        battery_model: str,
        top_k: int = 20
    ) -> List[Dict[str, Any]]:
        """Search for vehicles compatible with a battery model.

        Uses a two-stage search strategy:
        1. First, try exact metadata match on chrome_model field (fastest, most accurate)
        2. Fall back to semantic search if exact match fails

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
            # Normalize battery model to expected DB formats
            # Users may search "YTZ7S" but DB stores "YTZ7S-BS"
            normalized = battery_model.upper().strip()

            # Generate both variants: with and without -BS suffix
            if normalized.endswith("-BS"):
                model_with_bs = normalized
                model_without_bs = normalized.replace("-BS", "")
            else:
                model_without_bs = normalized.replace("-", "")  # Also handle "YTZ-7S" -> "YTZ7S"
                model_with_bs = f"{model_without_bs}-BS"

            # Build query text with both variants for better semantic matching
            query = f"{model_without_bs} {model_with_bs} battery fits vehicles compatible"
            query_vector = self._embed_query(query)

            # Search Qdrant with filter for battery_to_vehicle type
            # Get more results than needed to account for filtering
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
                limit=top_k * 5,  # Get more results to filter from
                with_payload=True
            )

            # Format results, filtering for battery model match
            # Match against both normalized forms for flexibility
            formatted = []
            search_variants = {
                model_without_bs,
                model_with_bs,
                model_without_bs.replace("-", ""),  # Handle any remaining dashes
            }

            for point in results.points:
                payload = point.payload or {}
                chrome_model = payload.get("chrome_model", "").upper()

                # Normalize stored model for comparison
                stored_normalized = chrome_model.replace("-BS", "").replace("-", "")
                stored_with_bs = chrome_model

                # Match if any variant matches
                is_match = (
                    stored_normalized in search_variants or
                    stored_with_bs in search_variants or
                    model_without_bs == stored_normalized or
                    model_without_bs in stored_normalized or
                    stored_normalized in model_without_bs
                )

                if is_match:
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

                    # Stop once we have enough results
                    if len(formatted) >= top_k:
                        break

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
