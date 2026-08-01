"""Embedding configuration for different providers and models."""

from typing import Dict

from app.services.embedding_service import EmbeddingConfig

# Predefined configurations for popular models
EMBEDDING_CONFIGS: Dict[str, EmbeddingConfig] = {
    # BAAI/BGE models
    "bge-small-en-v1.5": EmbeddingConfig(
        provider="baai",
        model_name="BAAI/bge-small-en-v1.5",
        dimension=384,
        batch_size=32,
        max_length=512,
        normalize=True,
        cache_enabled=True,
        device="cpu",
    ),
    "bge-base-en-v1.5": EmbeddingConfig(
        provider="baai",
        model_name="BAAI/bge-base-en-v1.5",
        dimension=768,
        batch_size=32,
        max_length=512,
        normalize=True,
        cache_enabled=True,
        device="cpu",
    ),
    "bge-large-en-v1.5": EmbeddingConfig(
        provider="baai",
        model_name="BAAI/bge-large-en-v1.5",
        dimension=1024,
        batch_size=16,
        max_length=512,
        normalize=True,
        cache_enabled=True,
        device="cpu",
    ),
    # Sentence-Transformers models
    "all-minilm-l6-v2": EmbeddingConfig(
        provider="sentence-transformers",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        dimension=384,
        batch_size=32,
        max_length=256,
        normalize=True,
        cache_enabled=True,
        device="cpu",
    ),
    "all-mpnet-base-v2": EmbeddingConfig(
        provider="sentence-transformers",
        model_name="sentence-transformers/all-mpnet-base-v2",
        dimension=768,
        batch_size=32,
        max_length=384,
        normalize=True,
        cache_enabled=True,
        device="cpu",
    ),
    "multi-qa-mpnet-base-dot-v1": EmbeddingConfig(
        provider="sentence-transformers",
        model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1",
        dimension=768,
        batch_size=32,
        max_length=512,
        normalize=True,
        cache_enabled=True,
        device="cpu",
    ),
    # NVIDIA NIM models (API-based)
    "nvidia-embed-qa": EmbeddingConfig(
        provider="nvidia",
        model_name="nvidia/embed-qa-4",
        dimension=1024,
        batch_size=100,
        max_length=512,
        normalize=False,  # NVIDIA handles normalization
        cache_enabled=True,
        device="cpu",  # Not used for API-based
    ),
    "nvidia-nv-embed-v1": EmbeddingConfig(
        provider="nvidia",
        model_name="nvidia/nv-embed-v1",
        dimension=4096,
        batch_size=50,
        max_length=512,
        normalize=False,
        cache_enabled=True,
        device="cpu",
    ),
}


def get_embedding_config(model_key: str) -> EmbeddingConfig:
    """
    Get embedding configuration by model key.

    Args:
        model_key: Key from EMBEDDING_CONFIGS

    Returns:
        EmbeddingConfig instance

    Raises:
        ValueError: If model_key not found
    """
    if model_key not in EMBEDDING_CONFIGS:
        available = ", ".join(EMBEDDING_CONFIGS.keys())
        raise ValueError(
            f"Unknown embedding model '{model_key}'. Available: {available}"
        )

    return EMBEDDING_CONFIGS[model_key]
