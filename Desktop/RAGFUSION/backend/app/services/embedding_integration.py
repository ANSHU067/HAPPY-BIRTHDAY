"""Integration guide for embedding service with existing RAG pipeline."""

from typing import List, Optional

import numpy as np

from app.config.embedding_config import get_embedding_config
from app.services.embedding_service import get_embedding_service


class DocumentEmbeddingIntegration:
    """
    Integration layer for embedding service with document processing.

    This class bridges the embedding service with the existing document
    ingestion and retrieval pipeline.
    """

    def __init__(
        self, provider_name: str = "default", model_key: str = "all-mpnet-base-v2"
    ):
        """
        Initialize the embedding integration.

        Args:
            provider_name: Name to register the provider under
            model_key: Key from embedding_config.EMBEDDING_CONFIGS
        """
        self.provider_name = provider_name
        self.service = get_embedding_service()

        # Register provider if not already registered
        try:
            self.service.get_provider(provider_name)
        except ValueError:
            config = get_embedding_config(model_key)
            self.service.register_provider(provider_name, config)

    async def embed_chunks(self, chunks: List[str]) -> np.ndarray:
        """
        Generate embeddings for document chunks.

        Args:
            chunks: List of text chunks from document processing

        Returns:
            Array of embeddings, shape (len(chunks), dimension)
        """
        if not chunks:
            return np.array([])

        return await self.service.embed_batch(chunks, provider_name=self.provider_name)

    async def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a search query.

        Args:
            query: User's search query

        Returns:
            Query embedding vector
        """
        return await self.service.embed_text(query, provider_name=self.provider_name)

    def get_dimension(self) -> int:
        """Get the embedding dimension for this provider."""
        provider = self.service.get_provider(self.provider_name)
        return provider.config.dimension


# Example: Integration with chunking service
async def embed_document_chunks(
    document_id: str,
    chunks: List[str],
    embedding_integration: DocumentEmbeddingIntegration,
) -> List[dict]:
    """
    Generate embeddings for document chunks.

    Args:
        document_id: Document identifier
        chunks: List of text chunks
        embedding_integration: Embedding integration instance

    Returns:
        List of chunk data with embeddings
    """
    embeddings = await embedding_integration.embed_chunks(chunks)

    chunk_data = []
    for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        chunk_data.append(
            {
                "document_id": document_id,
                "chunk_index": idx,
                "text": chunk,
                "embedding": embedding.tolist(),  # Convert to list for JSON serialization
            }
        )

    return chunk_data


# Example: Integration with vector store (ChromaDB)
async def store_embeddings_in_chromadb(
    collection_name: str,
    chunks: List[str],
    embeddings: np.ndarray,
    metadata: Optional[List[dict]] = None,
):
    """
    Store embeddings in ChromaDB.

    Args:
        collection_name: Name of the ChromaDB collection
        chunks: Text chunks
        embeddings: Pre-computed embeddings
        metadata: Optional metadata for each chunk
    """
    import chromadb

    client = chromadb.Client()
    collection = client.get_or_create_collection(name=collection_name)

    ids = [f"chunk_{i}" for i in range(len(chunks))]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings.tolist(),
        metadatas=metadata or [{}] * len(chunks),
    )


# Example: Similarity search
async def search_similar_chunks(
    query: str,
    embedding_integration: DocumentEmbeddingIntegration,
    collection_name: str,
    top_k: int = 5,
) -> List[dict]:
    """
    Search for similar chunks using embeddings.

    Args:
        query: Search query
        embedding_integration: Embedding integration instance
        collection_name: ChromaDB collection name
        top_k: Number of results to return

    Returns:
        List of similar chunks with scores
    """
    import chromadb

    # Generate query embedding
    query_embedding = await embedding_integration.embed_query(query)

    # Search in ChromaDB
    client = chromadb.Client()
    collection = client.get_collection(name=collection_name)

    results = collection.query(
        query_embeddings=[query_embedding.tolist()], n_results=top_k
    )

    # Format results
    similar_chunks = []
    for i in range(len(results["ids"][0])):
        similar_chunks.append(
            {
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "distance": results["distances"][0][i],
                "metadata": results["metadatas"][0][i],
            }
        )

    return similar_chunks


# Example: Usage in FastAPI endpoint
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

router = APIRouter()
embedding_integration = DocumentEmbeddingIntegration()


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class SearchResponse(BaseModel):
    results: List[dict]


@router.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    '''Search documents using semantic similarity.'''
    try:
        results = await search_similar_chunks(
            query=request.query,
            embedding_integration=embedding_integration,
            collection_name="documents",
            top_k=request.top_k
        )
        return SearchResponse(results=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class EmbedRequest(BaseModel):
    texts: List[str]


class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    dimension: int


@router.post("/embed", response_model=EmbedResponse)
async def embed_texts(request: EmbedRequest):
    '''Generate embeddings for texts.'''
    try:
        embeddings = await embedding_integration.embed_chunks(request.texts)
        return EmbedResponse(
            embeddings=embeddings.tolist(),
            dimension=embedding_integration.get_dimension()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
"""
