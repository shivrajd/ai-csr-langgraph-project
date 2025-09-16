"""Pinecone retriever for RAG-based knowledge queries using modern Pinecone SDK."""

import os
import time
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
import logging

logger = logging.getLogger(__name__)


class PineconeRetriever:
    """Pinecone-based retriever for customer service knowledge base using modern Pinecone SDK."""

    def __init__(
        self,
        index_name: str = "rag-knowledge-base",
        namespace: str = "default",
        embedding_model: str = "text-embedding-ada-002",  # Must match indexed documents!
        top_k: int = 3
    ):
        """Initialize Pinecone retriever."""
        self.index_name = index_name
        self.namespace = namespace
        self.top_k = top_k

        # Initialize Pinecone using modern SDK
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY not found in environment")

        # Initialize modern Pinecone client
        self.pc = Pinecone(api_key=api_key)

        # Get index
        self.index = self.pc.Index(self.index_name)

        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(model=embedding_model)

    def retrieve(
        self,
        query: str,
        filter_dict: Optional[Dict[str, Any]] = None,
        top_k: Optional[int] = None
    ) -> List[Document]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Search query
            filter_dict: Metadata filters
            top_k: Number of documents to retrieve

        Returns:
            List of relevant documents
        """
        start_time = time.time()

        try:
            # Use direct Pinecone query
            k = top_k or self.top_k

            # Embed the query
            query_embedding = self.embeddings.embed_query(query)

            # Query Pinecone directly
            results = self.index.query(
                vector=query_embedding,
                top_k=k,
                include_metadata=True,
                namespace=self.namespace,
                filter=filter_dict
            )

            # Convert to LangChain Document format
            documents = []
            for match in results.get("matches", []):
                metadata = match.get("metadata", {})
                text = metadata.get("text", "")

                # Remove 'text' from metadata since it's the page_content
                doc_metadata = {k: v for k, v in metadata.items() if k != "text"}
                doc_metadata["score"] = match.get("score", 0)

                documents.append(Document(
                    page_content=text,
                    metadata=doc_metadata
                ))

            retrieval_time = (time.time() - start_time) * 1000
            logger.info(f"Retrieved {len(documents)} documents in {retrieval_time:.2f}ms")

            return documents

        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []

    def add_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Add documents to the vector store.

        Args:
            documents: Documents to add
            ids: Optional document IDs

        Returns:
            List of document IDs
        """
        try:
            # This would require implementing upsert logic
            # For now, just return empty list as this is read-only retrieval
            logger.warning("add_documents not implemented for pinecone-client approach")
            return []
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return []