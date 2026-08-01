"""Embedding service with support for multiple providers and caching."""

import asyncio
import hashlib
import logging
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Dict, List, Optional

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class EmbeddingConfig(BaseModel):
    """Configuration for embedding models."""

    provider: str = Field(
        ..., description="Embedding provider: baai, sentence-transformers, nvidia"
    )
    model_name: str = Field(..., description="Model name/identifier")
    dimension: int = Field(..., description="Expected embedding dimension")
    batch_size: int = Field(default=32, description="Batch size for processing")
    max_length: int = Field(default=512, description="Maximum token length")
    normalize: bool = Field(
        default=True, description="Normalize embeddings to unit length"
    )
    cache_enabled: bool = Field(default=True, description="Enable embedding caching")
    device: str = Field(default="cpu", description="Device to use: cpu, cuda, mps")


class EmbeddingCache:
    """LRU cache for embeddings with hash-based lookup."""

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._cache: Dict[str, np.ndarray] = {}
        self._access_order: List[str] = []

    def _hash_text(self, text: str, model_name: str) -> str:
        """Generate cache key from text and model name."""
        content = f"{model_name}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()

    def get(self, text: str, model_name: str) -> Optional[np.ndarray]:
        """Get cached embedding if available."""
        key = self._hash_text(text, model_name)
        if key in self._cache:
            # Update access order (LRU)
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        return None

    def put(self, text: str, model_name: str, embedding: np.ndarray) -> None:
        """Store embedding in cache."""
        key = self._hash_text(text, model_name)

        # Evict oldest if at capacity
        if len(self._cache) >= self.max_size and key not in self._cache:
            oldest = self._access_order.pop(0)
            del self._cache[oldest]

        self._cache[key] = embedding
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)

    def clear(self) -> None:
        """Clear all cached embeddings."""
        self._cache.clear()
        self._access_order.clear()

    def size(self) -> int:
        """Return current cache size."""
        return len(self._cache)


class BaseEmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self.cache = EmbeddingCache() if config.cache_enabled else None

    @abstractmethod
    async def _generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings without caching. Must be implemented by subclasses."""
        pass

    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize embeddings to unit length."""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / (norms + 1e-10)

    async def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts with caching support.

        Args:
            texts: List of texts to embed

        Returns:
            numpy array of shape (len(texts), dimension)
        """
        if not texts:
            return np.array([]).reshape(0, self.config.dimension)

        # Check cache for each text
        embeddings = []
        texts_to_generate = []
        indices_to_generate = []

        for i, text in enumerate(texts):
            if self.cache:
                cached = self.cache.get(text, self.config.model_name)
                if cached is not None:
                    embeddings.append((i, cached))
                    continue

            texts_to_generate.append(text)
            indices_to_generate.append(i)

        # Generate embeddings for uncached texts
        if texts_to_generate:
            new_embeddings = await self._generate_embeddings(texts_to_generate)

            # Normalize if configured
            if self.config.normalize:
                new_embeddings = self._normalize_embeddings(new_embeddings)

            # Cache new embeddings
            if self.cache:
                for text, embedding in zip(texts_to_generate, new_embeddings):
                    self.cache.put(text, self.config.model_name, embedding)

            # Add to results with correct indices
            for idx, embedding in zip(indices_to_generate, new_embeddings):
                embeddings.append((idx, embedding))

        # Sort by original index and extract embeddings
        embeddings.sort(key=lambda x: x[0])
        result = np.array([emb for _, emb in embeddings])

        # Validate dimensions
        if result.shape[1] != self.config.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.config.dimension}, "
                f"got {result.shape[1]}"
            )

        return result

    async def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        result = await self.embed_texts([text])
        return result[0]

    async def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings in batches for memory efficiency.

        Args:
            texts: List of texts to embed

        Returns:
            numpy array of shape (len(texts), dimension)
        """
        if not texts:
            return np.array([]).reshape(0, self.config.dimension)

        all_embeddings = []

        # Process in batches
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i : i + self.config.batch_size]
            batch_embeddings = await self.embed_texts(batch)
            all_embeddings.append(batch_embeddings)

        return np.vstack(all_embeddings)


class BAAIEmbeddingProvider(BaseEmbeddingProvider):
    """BAAI (FlagEmbedding) embedding provider."""

    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        self._model = None
        self._lock = asyncio.Lock()

    async def _load_model(self):
        """Load the BAAI model lazily."""
        if self._model is None:
            async with self._lock:
                if self._model is None:
                    try:
                        from FlagEmbedding import FlagModel

                        logger.info(f"Loading BAAI model: {self.config.model_name}")
                        # Run in thread pool to avoid blocking
                        loop = asyncio.get_event_loop()
                        self._model = await loop.run_in_executor(
                            None,
                            lambda: FlagModel(
                                self.config.model_name,
                                use_fp16=(self.config.device != "cpu"),
                                device=self.config.device,
                            ),
                        )
                        logger.info("BAAI model loaded successfully")
                    except ImportError:
                        raise ImportError(
                            "FlagEmbedding not installed. Install with: "
                            "pip install FlagEmbedding"
                        )

    async def _generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using BAAI model."""
        await self._load_model()

        # Run inference in thread pool
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None, lambda: self._model.encode(texts, batch_size=self.config.batch_size)
        )

        return np.array(embeddings)


class SentenceTransformerProvider(BaseEmbeddingProvider):
    """Sentence-Transformers embedding provider."""

    def __init__(self, config: EmbeddingConfig):
        super().__init__(config)
        self._model = None
        self._lock = asyncio.Lock()

    async def _load_model(self):
        """Load the sentence-transformers model lazily."""
        if self._model is None:
            async with self._lock:
                if self._model is None:
                    try:
                        from sentence_transformers import SentenceTransformer

                        logger.info(
                            f"Loading SentenceTransformer: {self.config.model_name}"
                        )
                        loop = asyncio.get_event_loop()
                        self._model = await loop.run_in_executor(
                            None,
                            lambda: SentenceTransformer(
                                self.config.model_name, device=self.config.device
                            ),
                        )
                        logger.info("SentenceTransformer loaded successfully")
                    except ImportError:
                        raise ImportError(
                            "sentence-transformers not installed. Install with: "
                            "pip install sentence-transformers"
                        )

    async def _generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using sentence-transformers."""
        await self._load_model()

        # Run inference in thread pool
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: self._model.encode(
                texts,
                batch_size=self.config.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            ),
        )

        return np.array(embeddings)


class NVIDIAEmbeddingProvider(BaseEmbeddingProvider):
    """NVIDIA NIM embedding provider (API-based)."""

    def __init__(self, config: EmbeddingConfig, api_key: Optional[str] = None):
        super().__init__(config)
        self.api_key = api_key
        self._session = None

    async def _get_session(self):
        """Get or create aiohttp session."""
        if self._session is None:
            import aiohttp

            self._session = aiohttp.ClientSession()
        return self._session

    async def _generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using NVIDIA NIM API."""
        if not self.api_key:
            raise ValueError("NVIDIA API key is required")

        try:
            import aiohttp
        except ImportError:
            raise ImportError(
                "aiohttp not installed. Install with: pip install aiohttp"
            )

        session = await self._get_session()

        # NVIDIA NIM endpoint
        url = "https://integrate.api.nvidia.com/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.config.model_name,
            "input": texts,
            "input_type": "passage",
        }

        async with session.post(url, json=payload, headers=headers) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(
                    f"NVIDIA API error (status {response.status}): {error_text}"
                )

            result = await response.json()
            embeddings = [item["embedding"] for item in result["data"]]
            return np.array(embeddings)

    async def close(self):
        """Close aiohttp session."""
        if self._session:
            await self._session.close()
            self._session = None


class EmbeddingService:
    """Main embedding service with multi-provider support."""

    def __init__(self):
        self._providers: Dict[str, BaseEmbeddingProvider] = {}

    def register_provider(
        self, name: str, config: EmbeddingConfig, api_key: Optional[str] = None
    ) -> None:
        """
        Register an embedding provider.

        Args:
            name: Unique name for this provider instance
            config: Configuration for the provider
            api_key: API key (for NVIDIA provider)
        """
        provider_map = {
            "baai": BAAIEmbeddingProvider,
            "sentence-transformers": SentenceTransformerProvider,
            "nvidia": NVIDIAEmbeddingProvider,
        }

        provider_class = provider_map.get(config.provider)
        if not provider_class:
            raise ValueError(
                f"Unknown provider: {config.provider}. "
                f"Available: {list(provider_map.keys())}"
            )

        if config.provider == "nvidia":
            self._providers[name] = provider_class(config, api_key=api_key)
        else:
            self._providers[name] = provider_class(config)

        logger.info(f"Registered provider '{name}' with {config.provider}")

    def get_provider(self, name: str) -> BaseEmbeddingProvider:
        """Get a registered provider by name."""
        if name not in self._providers:
            raise ValueError(f"Provider '{name}' not registered")
        return self._providers[name]

    async def embed_texts(
        self, texts: List[str], provider_name: str = "default"
    ) -> np.ndarray:
        """Generate embeddings using specified provider."""
        provider = self.get_provider(provider_name)
        return await provider.embed_texts(texts)

    async def embed_text(self, text: str, provider_name: str = "default") -> np.ndarray:
        """Generate embedding for single text using specified provider."""
        provider = self.get_provider(provider_name)
        return await provider.embed_text(text)

    async def embed_batch(
        self, texts: List[str], provider_name: str = "default"
    ) -> np.ndarray:
        """Generate embeddings in batches using specified provider."""
        provider = self.get_provider(provider_name)
        return await provider.embed_batch(texts)

    async def close_all(self):
        """Close all providers and release resources."""
        for provider in self._providers.values():
            if hasattr(provider, "close"):
                await provider.close()


# Global service instance
_embedding_service: Optional[EmbeddingService] = None


@lru_cache
def get_embedding_service() -> EmbeddingService:
    """Get or create the global embedding service instance."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
