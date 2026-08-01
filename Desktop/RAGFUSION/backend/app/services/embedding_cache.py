"""Redis-backed persistent cache for embeddings."""

import logging
from typing import Optional

import numpy as np
import redis.asyncio as redis

logger = logging.getLogger(__name__)


class RedisCacheBackend:
    """Redis backend for persistent embedding cache."""

    def __init__(self, redis_url: str, ttl_seconds: int = 86400 * 7):
        """
        Initialize Redis cache backend.

        Args:
            redis_url: Redis connection URL
            ttl_seconds: Time-to-live for cache entries (default: 7 days)
        """
        self.redis_url = redis_url
        self.ttl_seconds = ttl_seconds
        self._client: Optional[redis.Redis] = None

    async def connect(self):
        """Connect to Redis."""
        if self._client is None:
            self._client = await redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=False,  # We'll handle binary data
            )
            logger.info("Connected to Redis cache backend")

    async def disconnect(self):
        """Disconnect from Redis."""
        if self._client:
            await self._client.close()
            self._client = None
            logger.info("Disconnected from Redis cache backend")

    def _make_key(self, text_hash: str, model_name: str) -> str:
        """Generate Redis key."""
        return f"embedding:{model_name}:{text_hash}"

    async def get(self, text_hash: str, model_name: str) -> Optional[np.ndarray]:
        """
        Retrieve embedding from cache.

        Args:
            text_hash: Hash of the text
            model_name: Name of the embedding model

        Returns:
            Cached embedding or None
        """
        if not self._client:
            await self.connect()

        try:
            key = self._make_key(text_hash, model_name)
            data = await self._client.get(key)

            if data:
                # Deserialize numpy array
                embedding = np.frombuffer(data, dtype=np.float32)
                return embedding
        except Exception as e:
            logger.error(f"Redis cache get error: {e}")

        return None

    async def put(self, text_hash: str, model_name: str, embedding: np.ndarray) -> None:
        """
        Store embedding in cache.

        Args:
            text_hash: Hash of the text
            model_name: Name of the embedding model
            embedding: Embedding vector to cache
        """
        if not self._client:
            await self.connect()

        try:
            key = self._make_key(text_hash, model_name)
            # Serialize numpy array to bytes
            data = embedding.astype(np.float32).tobytes()
            await self._client.setex(key, self.ttl_seconds, data)
        except Exception as e:
            logger.error(f"Redis cache put error: {e}")

    async def delete(self, text_hash: str, model_name: str) -> None:
        """Delete embedding from cache."""
        if not self._client:
            await self.connect()

        try:
            key = self._make_key(text_hash, model_name)
            await self._client.delete(key)
        except Exception as e:
            logger.error(f"Redis cache delete error: {e}")

    async def clear_model(self, model_name: str) -> int:
        """
        Clear all embeddings for a specific model.

        Args:
            model_name: Name of the embedding model

        Returns:
            Number of keys deleted
        """
        if not self._client:
            await self.connect()

        try:
            pattern = f"embedding:{model_name}:*"
            cursor = 0
            deleted = 0

            while True:
                cursor, keys = await self._client.scan(cursor, match=pattern, count=100)
                if keys:
                    deleted += await self._client.delete(*keys)
                if cursor == 0:
                    break

            logger.info(f"Cleared {deleted} cached embeddings for model {model_name}")
            return deleted
        except Exception as e:
            logger.error(f"Redis cache clear error: {e}")
            return 0

    async def get_stats(self) -> dict:
        """Get cache statistics."""
        if not self._client:
            await self.connect()

        try:
            info = await self._client.info("stats")
            return {
                "total_keys": await self._client.dbsize(),
                "hits": info.get("keyspace_hits", 0),
                "misses": info.get("keyspace_misses", 0),
                "evicted_keys": info.get("evicted_keys", 0),
            }
        except Exception as e:
            logger.error(f"Redis stats error: {e}")
            return {}
