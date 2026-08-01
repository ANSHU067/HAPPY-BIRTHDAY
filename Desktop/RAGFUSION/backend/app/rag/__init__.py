"""RAG (Retrieval-Augmented Generation) module.

This module contains all RAG-related components including embeddings,
retrievers, rerankers, pipelines, prompts, and utilities.

Note: Import components directly from their submodules to avoid circular dependencies.
"""

__all__ = [
    "EmbeddingManager",
    "RetrieverManager",
    "RerankerManager",
    "PromptBuilder",
    "RAGPipeline",
]


def __getattr__(name):
    """Lazy import to avoid circular dependencies."""
    if name == "EmbeddingManager":
        from app.rag.embeddings.embedding_manager import EmbeddingManager

        return EmbeddingManager
    elif name == "RetrieverManager":
        from app.rag.retrievers.retriever_manager import RetrieverManager

        return RetrieverManager
    elif name == "RerankerManager":
        from app.rag.rerankers.reranker_manager import RerankerManager

        return RerankerManager
    elif name == "PromptBuilder":
        from app.rag.prompts.prompt_builder import PromptBuilder

        return PromptBuilder
    elif name == "RAGPipeline":
        from app.rag.pipelines.rag_pipeline import RAGPipeline

        return RAGPipeline
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
