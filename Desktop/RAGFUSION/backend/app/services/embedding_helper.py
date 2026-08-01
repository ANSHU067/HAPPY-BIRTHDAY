"""Consolidated embedding generation helper for all ingestion services."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


async def generate_embeddings(
    chunks: list[Any],
    model_name: str,
    dimensions: int | None = None,
) -> list[list[float]]:
    """
    Generate embeddings for text chunks.

    This is a centralized embedding generation function used by all ingestion services.

    TODO: Integrate with embedding_service.py for actual embedding generation
    Currently returns placeholder embeddings until full integration is complete.

    Args:
        chunks: List of chunk objects (DocumentChunk, WebsiteChunk, etc.)
        model_name: Embedding model name
        dimensions: Embedding dimensions (default: 1536)

    Returns:
        List of embedding vectors as lists of floats
    """
    dim = dimensions or 1536

    # TODO: Integration steps:
    # 1. Import from app.services.embedding_service import get_embedding_service
    # 2. Get provider: service = get_embedding_service()
    # 3. Extract text from chunks: texts = [chunk.content for chunk in chunks]
    # 4. Generate embeddings: embeddings = await service.embed_batch(texts, provider_name="default")
    # 5. Convert numpy arrays to lists: return [emb.tolist() for emb in embeddings]

    logger.warning(
        f"Using placeholder embeddings. Generate {len(chunks)} embeddings "
        f"with model '{model_name}' and dimension {dim}"
    )

    return [[0.0] * dim for _ in chunks]
