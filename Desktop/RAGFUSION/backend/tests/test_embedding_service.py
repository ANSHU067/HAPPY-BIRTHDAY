"""Tests for embedding service."""

import numpy as np
import pytest

from app.services.embedding_service import (BaseEmbeddingProvider,
                                            EmbeddingCache, EmbeddingConfig,
                                            EmbeddingService)


class TestEmbeddingCache:
    """Test embedding cache functionality."""

    def test_cache_put_and_get(self):
        """Test basic cache operations."""
        cache = EmbeddingCache(max_size=100)

        text = "Hello world"
        model = "test-model"
        embedding = np.array([0.1, 0.2, 0.3])

        # Put and retrieve
        cache.put(text, model, embedding)
        result = cache.get(text, model)

        assert result is not None
        np.testing.assert_array_almost_equal(result, embedding)

    def test_cache_miss(self):
        """Test cache miss returns None."""
        cache = EmbeddingCache(max_size=100)

        result = cache.get("nonexistent", "model")
        assert result is None

    def test_cache_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = EmbeddingCache(max_size=3)

        # Fill cache
        for i in range(3):
            cache.put(f"text{i}", "model", np.array([i]))

        # Verify all are cached
        assert cache.size() == 3

        # Add one more - should evict oldest (text0)
        cache.put("text3", "model", np.array([3]))

        assert cache.size() == 3
        assert cache.get("text0", "model") is None
        assert cache.get("text3", "model") is not None

    def test_cache_access_updates_lru(self):
        """Test that accessing an item updates LRU order."""
        cache = EmbeddingCache(max_size=3)

        # Fill cache
        for i in range(3):
            cache.put(f"text{i}", "model", np.array([i]))

        # Access text0 to make it most recent
        cache.get("text0", "model")

        # Add new item - should evict text1 (oldest)
        cache.put("text3", "model", np.array([3]))

        assert cache.get("text0", "model") is not None
        assert cache.get("text1", "model") is None
        assert cache.get("text2", "model") is not None
        assert cache.get("text3", "model") is not None

    def test_cache_different_models(self):
        """Test that same text with different models are cached separately."""
        cache = EmbeddingCache(max_size=100)

        text = "Hello world"
        embedding1 = np.array([0.1, 0.2, 0.3])
        embedding2 = np.array([0.4, 0.5, 0.6])

        cache.put(text, "model1", embedding1)
        cache.put(text, "model2", embedding2)

        result1 = cache.get(text, "model1")
        result2 = cache.get(text, "model2")

        np.testing.assert_array_almost_equal(result1, embedding1)
        np.testing.assert_array_almost_equal(result2, embedding2)

    def test_cache_clear(self):
        """Test cache clearing."""
        cache = EmbeddingCache(max_size=100)

        cache.put("text1", "model", np.array([1, 2, 3]))
        cache.put("text2", "model", np.array([4, 5, 6]))

        assert cache.size() == 2

        cache.clear()

        assert cache.size() == 0
        assert cache.get("text1", "model") is None


class MockEmbeddingProvider(BaseEmbeddingProvider):
    """Mock provider for testing that properly inherits from base."""

    def __init__(
        self, dimension: int = 384, expected_dim: int = 384, cache_enabled: bool = True
    ):
        config = EmbeddingConfig(
            provider="mock",
            model_name="mock-model",
            dimension=expected_dim,
            batch_size=32,
            normalize=False,
            cache_enabled=cache_enabled,
        )
        super().__init__(config)
        self.actual_dimension = dimension
        self.call_count = 0

    async def _generate_embeddings(self, texts: list) -> np.ndarray:
        """Generate random embeddings."""
        self.call_count += 1
        return np.random.rand(len(texts), self.actual_dimension)


@pytest.mark.asyncio
class TestEmbeddingProviders:
    """Test embedding provider implementations."""

    async def test_dimension_validation(self):
        """Test that dimension mismatches are caught."""
        # Config expects 768, but mock outputs 384
        provider = MockEmbeddingProvider(
            dimension=384, expected_dim=768, cache_enabled=False
        )

        # Because we inherit from BaseEmbeddingProvider, calling embed_texts
        # naturally triggers the dimension check logic!
        with pytest.raises(ValueError, match="dimension mismatch"):
            await provider.embed_texts(["test string"])

    async def test_batch_processing(self):
        """Test batch processing splits large inputs correctly."""
        provider = MockEmbeddingProvider(
            dimension=384, expected_dim=384, cache_enabled=False
        )
        provider.config.batch_size = 3  # Small batch size

        texts = [f"text {i}" for i in range(10)]

        # Call the actual embed_batch method from the base class
        embeddings = await provider.embed_batch(texts)

        assert embeddings.shape == (10, 384)
        # Should have called _generate_embeddings 4 times (10 texts / batch_size 3)
        assert provider.call_count == 4

    async def test_normalization(self):
        """Test embedding normalization."""
        config = EmbeddingConfig(
            provider="mock",
            model_name="test",
            dimension=2,
            batch_size=32,
            normalize=True,
            cache_enabled=False,
        )

        # Create embeddings with known values
        embeddings = np.array(
            [
                [3.0, 4.0],  # Length = 5
                [5.0, 12.0],  # Length = 13
            ]
        )

        # Mock provider override
        class TestProvider(BaseEmbeddingProvider):
            async def _generate_embeddings(self, texts):
                return embeddings

        provider = TestProvider(config)
        normalized = provider._normalize_embeddings(embeddings)

        # Check that vectors have unit length
        norms = np.linalg.norm(normalized, axis=1)
        np.testing.assert_array_almost_equal(norms, [1.0, 1.0])

        # Check that direction is preserved
        np.testing.assert_array_almost_equal(normalized[0], [3.0 / 5.0, 4.0 / 5.0])

    async def test_empty_input(self):
        """Test handling of empty input."""
        provider = MockEmbeddingProvider(dimension=384, expected_dim=384)

        result = await provider.embed_texts([])

        assert result.shape == (0, 384)

    async def test_single_text_embedding(self):
        """Test single text embedding returns correct shape."""
        provider = MockEmbeddingProvider(dimension=384, expected_dim=384)

        result = await provider.embed_text("single text")

        assert result.shape == (384,)


@pytest.mark.asyncio
class TestEmbeddingService:
    """Test the main embedding service."""

    async def test_register_and_get_provider(self):
        """Test provider registration and retrieval."""
        service = EmbeddingService()

        config = EmbeddingConfig(
            provider="sentence-transformers",
            model_name="test-model",
            dimension=384,
            batch_size=32,
            normalize=True,
            cache_enabled=True,
        )

        service.register_provider("test", config)

        provider = service.get_provider("test")
        assert provider is not None
        assert provider.config.model_name == "test-model"

    async def test_get_nonexistent_provider(self):
        """Test that getting nonexistent provider raises error."""
        service = EmbeddingService()

        with pytest.raises(ValueError, match="not registered"):
            service.get_provider("nonexistent")

    async def test_register_invalid_provider(self):
        """Test that registering invalid provider type raises error."""
        service = EmbeddingService()

        config = EmbeddingConfig(
            provider="invalid-provider",
            model_name="test",
            dimension=384,
            batch_size=32,
            normalize=True,
            cache_enabled=True,
        )

        with pytest.raises(ValueError, match="Unknown provider"):
            service.register_provider("test", config)


@pytest.mark.asyncio
class TestCacheIntegration:
    """Test caching behavior with providers."""

    async def test_cache_reduces_compute(self):
        """Test that cache reduces redundant computations."""
        provider = MockEmbeddingProvider(
            dimension=384, expected_dim=384, cache_enabled=True
        )
        texts = ["text1", "text2", "text1"]  # text1 appears twice

        # First run - should cache everything
        await provider.embed_texts(texts)
        initial_calls = provider.call_count

        # Second run - should hit cache
        await provider.embed_texts(texts)

        # Call count shouldn't have gone up because all texts were cached
        assert provider.call_count == initial_calls

    async def test_cache_disabled(self):
        """Test that disabling cache works correctly."""
        provider = MockEmbeddingProvider(
            dimension=384, expected_dim=384, cache_enabled=False
        )

        # Should work without cache
        embeddings = await provider.embed_texts(["text"])
        assert embeddings.shape == (1, 384)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
