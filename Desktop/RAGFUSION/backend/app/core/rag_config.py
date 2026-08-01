"""RAG-specific configuration settings.

This module contains configuration for RAG components including:
- Embedding models
- Vector stores
- Retrieval parameters
- Reranking settings
- LLM parameters
"""

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class RAGConfig(BaseSettings):
    """RAG pipeline configuration."""

    # Model Configuration
    embedding_model: str = Field(
        default="text-embedding-3-small", description="OpenAI embedding model"
    )
    llm_model: str = Field(default="gpt-4", description="LLM model for generation")
    rerank_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Cross-encoder model for reranking",
    )

    # Vector Store Configuration
    vector_store_path: Path = Field(
        default_factory=lambda: Path("./data/vectorstore"),
        description="Path to vector store data",
    )

    # Retrieval Configuration
    top_k_retrieval: int = Field(
        default=10,
        description="Number of documents to retrieve initially",
        ge=1,
        le=100,
    )
    top_k_rerank: int = Field(
        default=5, description="Number of documents after reranking", ge=1, le=50
    )

    # Embedding Configuration
    embedding_dimension: int = Field(
        default=1536, description="Embedding dimension for text-embedding-3-small"
    )
    chunk_size: int = Field(
        default=500, description="Size of text chunks", ge=100, le=2000
    )
    chunk_overlap: int = Field(
        default=50, description="Overlap between chunks", ge=0, le=500
    )

    # LLM Configuration
    temperature: float = Field(
        default=0.0, description="LLM temperature", ge=0.0, le=2.0
    )
    max_tokens: int = Field(
        default=1000, description="Maximum tokens for generation", ge=1, le=4000
    )

    # Prompt Configuration
    max_context_length: int = Field(
        default=4000,
        description="Maximum context length in characters",
        ge=1000,
        le=16000,
    )

    # Reranking Configuration
    retrieval_weight: float = Field(
        default=0.3,
        description="Weight for retrieval scores in hybrid reranking",
        ge=0.0,
        le=1.0,
    )
    rerank_weight: float = Field(
        default=0.7,
        description="Weight for reranking scores in hybrid reranking",
        ge=0.0,
        le=1.0,
    )

    # Conversation Memory
    max_conversation_history: int = Field(
        default=5, description="Maximum conversation history to include", ge=0, le=20
    )

    class Config:
        env_prefix = "RAG_"
        case_sensitive = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure vector store path exists
        self.vector_store_path.mkdir(parents=True, exist_ok=True)


# Global configuration instance
_rag_config: Optional[RAGConfig] = None


def get_rag_config() -> RAGConfig:
    """Get the global RAG configuration instance.

    Returns:
        RAGConfig instance
    """
    global _rag_config
    if _rag_config is None:
        _rag_config = RAGConfig()
    return _rag_config


def reset_rag_config():
    """Reset the global configuration (useful for testing)."""
    global _rag_config
    _rag_config = None


# For backwards compatibility with old config.py
def get_config_value(key: str, default=None):
    """Get a configuration value by key.

    Args:
        key: Configuration key
        default: Default value if key not found

    Returns:
        Configuration value
    """
    config = get_rag_config()
    return getattr(config, key.lower(), default)


# Convenience accessors for commonly used config values
EMBEDDING_MODEL = lambda: get_rag_config().embedding_model
LLM_MODEL = lambda: get_rag_config().llm_model
RERANK_MODEL = lambda: get_rag_config().rerank_model
VECTOR_STORE_PATH = lambda: get_rag_config().vector_store_path
TOP_K_RETRIEVAL = lambda: get_rag_config().top_k_retrieval
TOP_K_RERANK = lambda: get_rag_config().top_k_rerank
EMBEDDING_DIMENSION = lambda: get_rag_config().embedding_dimension
CHUNK_SIZE = lambda: get_rag_config().chunk_size
CHUNK_OVERLAP = lambda: get_rag_config().chunk_overlap
TEMPERATURE = lambda: get_rag_config().temperature
MAX_TOKENS = lambda: get_rag_config().max_tokens
