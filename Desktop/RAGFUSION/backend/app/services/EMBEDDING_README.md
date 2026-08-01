"""
Embedding Service Documentation
================================

The embedding service provides a unified interface for generating text embeddings
using multiple providers: BAAI/FlagEmbedding, sentence-transformers, and NVIDIA NIM.

Features
--------
- Multi-provider support (BAAI, sentence-transformers, NVIDIA)
- Automatic batch processing for large datasets
- Two-tier caching (in-memory LRU + Redis persistent)
- Async/await support for high concurrency
- Dimension validation
- L2 normalization support

Installation
-----------
```bash
pip install -r requirements/embeddings.txt
```

Quick Start
-----------
```python
from app.services.embedding_service import get_embedding_service
from app.config.embedding_config import get_embedding_config

# Initialize service
service = get_embedding_service()

# Register a provider
config = get_embedding_config("all-minilm-l6-v2")
service.register_provider("default", config)

# Generate embeddings
texts = ["Hello world", "How are you?"]
embeddings = await service.embed_texts(texts, provider_name="default")
```

Available Models
---------------

### BAAI/BGE Models (Local)
- `bge-small-en-v1.5`: 384 dimensions, fast
- `bge-base-en-v1.5`: 768 dimensions, balanced
- `bge-large-en-v1.5`: 1024 dimensions, highest quality

### Sentence-Transformers (Local)
- `all-minilm-l6-v2`: 384 dimensions, very fast
- `all-mpnet-base-v2`: 768 dimensions, high quality
- `multi-qa-mpnet-base-dot-v1`: 768 dimensions, optimized for Q&A

### NVIDIA NIM (API-based)
- `nvidia-embed-qa`: 1024 dimensions, optimized for retrieval
- `nvidia-nv-embed-v1`: 4096 dimensions, state-of-the-art

Configuration
------------

### Basic Configuration
```python
from app.services.embedding_service import EmbeddingConfig

config = EmbeddingConfig(
    provider="sentence-transformers",
    model_name="sentence-transformers/all-mpnet-base-v2",
    dimension=768,
    batch_size=32,
    max_length=384,
    normalize=True,
    cache_enabled=True,
    device="cpu"  # or "cuda", "mps"
)
```

### GPU Acceleration
```python
config = get_embedding_config("bge-large-en-v1.5")
config.device = "cuda"  # Use GPU
service.register_provider("gpu-provider", config)
```

### Custom Batch Size
```python
config = get_embedding_config("all-mpnet-base-v2")
config.batch_size = 64  # Larger batches for throughput
service.register_provider("batch-provider", config)
```

Caching
-------

### In-Memory Cache (LRU)
Automatically enabled by default. Stores up to 10,000 embeddings in memory.

```python
config.cache_enabled = True  # Enable (default)
```

### Redis Cache (Persistent)
For persistent caching across restarts:

```python
from app.services.embedding_cache import RedisCacheBackend

# Initialize Redis backend
cache_backend = RedisCacheBackend(
    redis_url="redis://localhost:6379/0",
    ttl_seconds=86400 * 7  # 7 days
)
await cache_backend.connect()

# Use with provider
# (Note: You'll need to integrate this with the provider's cache)
```

### Cache Management
```python
# Clear cache for specific model
await cache_backend.clear_model("all-mpnet-base-v2")

# Get cache statistics
stats = await cache_backend.get_stats()
print(f"Cache hits: {stats['hits']}")
print(f"Cache misses: {stats['misses']}")
```

Usage Patterns
-------------

### Single Text Embedding
```python
text = "What is machine learning?"
embedding = await service.embed_text(text, provider_name="default")
# Returns: numpy array of shape (dimension,)
```

### Batch Embedding
```python
texts = ["text1", "text2", "text3"]
embeddings = await service.embed_texts(texts, provider_name="default")
# Returns: numpy array of shape (len(texts), dimension)
```

### Large Dataset Processing
```python
# Automatically batched for memory efficiency
large_dataset = [f"document {i}" for i in range(10000)]
embeddings = await service.embed_batch(large_dataset, provider_name="default")
```

### Multiple Providers
```python
# Register multiple models
service.register_provider("fast", get_embedding_config("all-minilm-l6-v2"))
service.register_provider("quality", get_embedding_config("bge-large-en-v1.5"))

# Use different models for different purposes
query_emb = await service.embed_text(query, provider_name="fast")
doc_embs = await service.embed_texts(documents, provider_name="quality")
```

### NVIDIA API Usage
```python
import os

config = get_embedding_config("nvidia-embed-qa")
api_key = os.getenv("NVIDIA_API_KEY")

service.register_provider("nvidia", config, api_key=api_key)
embeddings = await service.embed_texts(texts, provider_name="nvidia")
```

Similarity Search
----------------

```python
# Generate embeddings
doc_embeddings = await service.embed_texts(documents, provider_name="default")
query_embedding = await service.embed_text(query, provider_name="default")

# Compute cosine similarity (embeddings are normalized)
similarities = np.dot(doc_embeddings, query_embedding)

# Get top-k results
top_k_indices = np.argsort(similarities)[::-1][:5]
top_k_docs = [documents[i] for i in top_k_indices]
```

Performance Tips
---------------

1. **Use appropriate batch sizes**: Larger batches improve throughput but use more memory
2. **Enable GPU**: 3-10x faster for large batches
3. **Enable caching**: Avoid recomputing embeddings for repeated texts
4. **Choose the right model**: Smaller models are faster, larger models are more accurate
5. **Use Redis cache**: Share cache across multiple workers/instances

Testing
-------

Run the test suite:
```bash
cd backend
pytest tests/test_embedding_service.py -v
```

Run specific test:
```bash
pytest tests/test_embedding_service.py::TestEmbeddingCache::test_cache_lru_eviction -v
```

Examples
--------

See `examples/embedding_examples.py` for complete working examples:
```bash
python examples/embedding_examples.py
```

API Reference
------------

### EmbeddingService

#### Methods
- `register_provider(name, config, api_key=None)`: Register a new embedding provider
- `get_provider(name)`: Get a registered provider
- `embed_text(text, provider_name)`: Embed single text
- `embed_texts(texts, provider_name)`: Embed list of texts
- `embed_batch(texts, provider_name)`: Embed with automatic batching
- `close_all()`: Close all providers and release resources

### EmbeddingConfig

#### Fields
- `provider`: Provider type ("baai", "sentence-transformers", "nvidia")
- `model_name`: Model identifier
- `dimension`: Expected embedding dimension
- `batch_size`: Batch size for processing (default: 32)
- `max_length`: Maximum token length (default: 512)
- `normalize`: Normalize embeddings to unit length (default: True)
- `cache_enabled`: Enable caching (default: True)
- `device`: Device to use ("cpu", "cuda", "mps")

Error Handling
-------------

```python
try:
    embeddings = await service.embed_texts(texts, provider_name="default")
except ValueError as e:
    # Handle configuration errors (dimension mismatch, invalid provider, etc.)
    print(f"Configuration error: {e}")
except ImportError as e:
    # Handle missing dependencies
    print(f"Missing dependency: {e}")
except RuntimeError as e:
    # Handle API errors (NVIDIA)
    print(f"API error: {e}")
```

Troubleshooting
--------------

### Issue: Dimension mismatch error
**Solution**: Verify that `config.dimension` matches the model's actual output dimension

### Issue: Out of memory
**Solution**: Reduce `batch_size` in config or switch to a smaller model

### Issue: Slow embedding generation
**Solution**: Enable GPU, increase batch size, or use a smaller/faster model

### Issue: ImportError for FlagEmbedding
**Solution**: `pip install FlagEmbedding`

### Issue: NVIDIA API authentication failed
**Solution**: Set `NVIDIA_API_KEY` environment variable with valid API key
"""
