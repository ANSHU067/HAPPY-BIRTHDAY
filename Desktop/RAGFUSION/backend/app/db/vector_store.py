"""Vector database layer for storing and querying embeddings with metadata."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class VectorDocument:
    """A document with vector embedding and metadata."""

    id: str
    vector: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Ensure vector is a numpy array."""
        if not isinstance(self.vector, np.ndarray):
            self.vector = np.asarray(self.vector, dtype=np.float32)
        else:
            self.vector = self.vector.astype(np.float32)


class VectorStore:
    """In-memory vector database with similarity search and metadata filtering."""

    def __init__(self, dimension: int = 384):
        """Initialize the vector store.

        Args:
            dimension: Expected dimension of vectors (default: 384 for all-MiniLM-L6-v2)
        """
        self.dimension = dimension
        self._documents: dict[str, VectorDocument] = {}

    def add_vector(
        self,
        vector: np.ndarray | list[float],
        metadata: Optional[dict[str, Any]] = None,
        id: Optional[str] = None,
    ) -> str:
        """Add a vector to the store.

        Args:
            vector: The embedding vector
            metadata: Optional metadata dictionary
            id: Optional ID (generated if not provided)

        Returns:
            The document ID

        Raises:
            ValueError: If vector dimension doesn't match store dimension
        """
        if id is None:
            id = str(uuid.uuid4())

        vector_array = np.asarray(vector, dtype=np.float32)

        if vector_array.shape[0] != self.dimension:
            raise ValueError(
                f"Vector dimension {vector_array.shape[0]} doesn't match "
                f"store dimension {self.dimension}"
            )

        doc = VectorDocument(id=id, vector=vector_array, metadata=metadata or {})

        self._documents[id] = doc
        return id

    def add_vectors(
        self,
        vectors: list[np.ndarray | list[float]],
        metadatas: Optional[list[dict[str, Any]]] = None,
        ids: Optional[list[str]] = None,
    ) -> list[str]:
        """Add multiple vectors to the store.

        Args:
            vectors: List of embedding vectors
            metadatas: Optional list of metadata dictionaries
            ids: Optional list of IDs (generated if not provided)

        Returns:
            List of document IDs
        """
        if metadatas is None:
            metadatas = [{}] * len(vectors)

        if ids is None:
            ids = [str(uuid.uuid4()) for _ in vectors]

        if len(vectors) != len(metadatas) or len(vectors) != len(ids):
            raise ValueError("vectors, metadatas, and ids must have the same length")

        result_ids = []
        for vector, metadata, id in zip(vectors, metadatas, ids):
            result_ids.append(self.add_vector(vector, metadata, id))

        return result_ids

    def remove_vector(self, id: str) -> bool:
        """Remove a vector by ID.

        Args:
            id: Document ID to remove

        Returns:
            True if document was removed, False if not found
        """
        if id in self._documents:
            del self._documents[id]
            return True
        return False

    def remove_vectors(self, ids: list[str]) -> int:
        """Remove multiple vectors by ID.

        Args:
            ids: List of document IDs to remove

        Returns:
            Number of documents actually removed
        """
        count = 0
        for id in ids:
            if self.remove_vector(id):
                count += 1
        return count

    def update_vector(
        self,
        id: str,
        vector: Optional[np.ndarray | list[float]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> bool:
        """Update a vector and/or its metadata.

        Args:
            id: Document ID to update
            vector: New vector (if None, keeps existing)
            metadata: New metadata (if None, keeps existing)

        Returns:
            True if document was updated, False if not found
        """
        if id not in self._documents:
            return False

        doc = self._documents[id]

        if vector is not None:
            vector_array = np.asarray(vector, dtype=np.float32)
            if vector_array.shape[0] != self.dimension:
                raise ValueError(
                    f"Vector dimension {vector_array.shape[0]} doesn't match "
                    f"store dimension {self.dimension}"
                )
            doc.vector = vector_array

        if metadata is not None:
            doc.metadata = metadata

        return True

    def get_vector(self, id: str) -> Optional[VectorDocument]:
        """Get a document by ID.

        Args:
            id: Document ID

        Returns:
            VectorDocument if found, None otherwise
        """
        return self._documents.get(id)

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors.

        Args:
            a: First vector
            b: Second vector

        Returns:
            Cosine similarity score (0 to 1, higher is more similar)
        """
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(dot_product / (norm_a * norm_b))

    @staticmethod
    def _matches_filter(metadata: dict[str, Any], filter_dict: dict[str, Any]) -> bool:
        """Check if metadata matches all filter criteria.

        Args:
            metadata: Document metadata
            filter_dict: Filter criteria

        Returns:
            True if all filters match
        """
        for key, value in filter_dict.items():
            if key not in metadata:
                return False
            if metadata[key] != value:
                return False
        return True

    def similarity_search(
        self,
        query_vector: np.ndarray | list[float],
        k: int = 5,
        filter: Optional[dict[str, Any]] = None,
        min_score: float = 0.0,
    ) -> list[tuple[VectorDocument, float]]:
        """Search for similar vectors.

        Args:
            query_vector: Query embedding vector
            k: Number of results to return
            filter: Optional metadata filter (all key-value pairs must match)
            min_score: Minimum similarity score threshold (0 to 1)

        Returns:
            List of (document, similarity_score) tuples, sorted by score descending
        """
        query_array = np.asarray(query_vector, dtype=np.float32)

        if query_array.shape[0] != self.dimension:
            raise ValueError(
                f"Query vector dimension {query_array.shape[0]} doesn't match "
                f"store dimension {self.dimension}"
            )

        # Compute similarities
        results = []
        for doc in self._documents.values():
            # Apply metadata filter
            if filter and not self._matches_filter(doc.metadata, filter):
                continue

            similarity = self._cosine_similarity(query_array, doc.vector)

            # Apply minimum score threshold
            if similarity >= min_score:
                results.append((doc, similarity))

        # Sort by similarity descending and return top k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def clear(self) -> None:
        """Remove all documents from the store."""
        self._documents.clear()

    def count(self) -> int:
        """Return the number of documents in the store."""
        return len(self._documents)

    def list_ids(self) -> list[str]:
        """Return all document IDs in the store."""
        return list(self._documents.keys())
