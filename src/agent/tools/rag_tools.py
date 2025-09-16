"""
RAG Tool Implementation for knowledge base retrieval.
"""

import logging
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def retrieve_knowledge(query: str) -> str:
    """
    Search the comprehensive company knowledge base for authoritative information.

    USE FOR: Product information, policies, procedures, technical questions, FAQ content
    PRIORITY: Always use this tool FIRST for any informational queries

    This tool searches the company's authoritative knowledge base containing:
    - Product specifications and details
    - Shipping, return, and warranty policies
    - Technical documentation and troubleshooting
    - Company procedures and guidelines
    - Frequently asked questions and answers

    Args:
        query: Customer's question or information need (be specific for better results)

    Returns:
        Authoritative information from knowledge base or indication if info unavailable
    """
    try:
        logger.info(f"Knowledge retrieval for: {query}")

        # Try to use the Pinecone retriever
        try:
            from src.agent.tools.retriever import PineconeRetriever

            retriever = PineconeRetriever(
                index_name="rag-knowledge-base",
                top_k=3
            )
            documents = retriever.retrieve(query)

            if documents:
                # Format documents for the LLM
                formatted_content = []
                for doc in documents:
                    content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                    # Clean up content
                    content = content.strip()
                    if content:
                        formatted_content.append(content)

                if formatted_content:
                    result = "\n\n".join(formatted_content)
                    logger.info(f"Retrieved {len(formatted_content)} relevant documents")
                    return result

            # No relevant documents found
            return "I don't have specific information about this topic in my knowledge base. I can still try to help based on my general knowledge, or you may want to contact support for more detailed assistance."

        except ImportError as e:
            logger.warning(f"Retriever not available: {e}")
            return "Knowledge base search is currently unavailable. I'll do my best to help based on my general knowledge."
        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            return "I encountered an issue searching the knowledge base. I'll try to help based on my general knowledge."

    except Exception as e:
        logger.error(f"Tool error: {e}")
        return "I'm having trouble accessing information right now. Please try rephrasing your question or contact support for assistance."


# List of available tools for the agent
available_tools = [retrieve_knowledge]

__all__ = ["retrieve_knowledge", "available_tools"]