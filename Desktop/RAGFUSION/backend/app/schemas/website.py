"""Website ingestion schemas."""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class WebsiteStatus(str, enum.Enum):
    """Website processing status."""

    pending = "pending"
    validating = "validating"
    crawling = "crawling"
    extracting = "extracting"
    chunking = "chunking"
    embedding = "embedding"
    storing = "storing"
    ready = "ready"
    failed = "failed"


class CrawlDepth(str, enum.Enum):
    """Crawl depth options."""

    single = "single"
    shallow = "shallow"
    deep = "deep"


class WebsiteIngestRequest(BaseModel):
    """Request to ingest a website."""

    url: HttpUrl = Field(..., description="Website URL to ingest")
    crawl_depth: CrawlDepth = Field(
        default=CrawlDepth.single, description="How deep to crawl"
    )
    max_pages: int = Field(
        default=10, ge=1, le=1000, description="Maximum pages to crawl"
    )
    follow_redirects: bool = Field(default=True, description="Follow HTTP redirects")
    timeout_seconds: int = Field(
        default=30, ge=5, le=300, description="Request timeout in seconds"
    )
    extract_metadata: bool = Field(default=True, description="Extract page metadata")
    clean_content: bool = Field(
        default=True, description="Clean extracted HTML content"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "url": "https://example.com",
                "crawl_depth": "single",
                "max_pages": 10,
                "follow_redirects": True,
                "timeout_seconds": 30,
                "extract_metadata": True,
                "clean_content": True,
            }
        }
    )


class WebsiteCrawlRequest(BaseModel):
    """Request to crawl a website (discover URLs only)."""

    url: HttpUrl = Field(..., description="Website URL to crawl")
    crawl_depth: CrawlDepth = Field(
        default=CrawlDepth.shallow, description="How deep to crawl"
    )
    max_pages: int = Field(
        default=50, ge=1, le=5000, description="Maximum pages to crawl"
    )
    follow_redirects: bool = Field(default=True, description="Follow HTTP redirects")
    timeout_seconds: int = Field(
        default=30, ge=5, le=300, description="Request timeout in seconds"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "url": "https://example.com",
                "crawl_depth": "shallow",
                "max_pages": 50,
                "follow_redirects": True,
                "timeout_seconds": 30,
            }
        }
    )


class WebsiteIngestResponse(BaseModel):
    """Response after website ingestion initiation."""

    website_id: uuid.UUID
    url: str
    status: WebsiteStatus
    message: str

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "website_id": "550e8400-e29b-41d4-a716-446655440000",
                "url": "https://example.com",
                "status": "pending",
                "message": "Website ingestion started. Processing...",
            }
        }
    )


class WebsiteCrawlResponse(BaseModel):
    """Response after website crawl."""

    website_id: uuid.UUID
    url: str
    discovered_urls: list[str]
    status: WebsiteStatus
    message: str

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "website_id": "550e8400-e29b-41d4-a716-446655440000",
                "url": "https://example.com",
                "discovered_urls": ["https://example.com", "https://example.com/about"],
                "status": "ready",
                "message": "Crawl completed. Found 2 URLs.",
            }
        }
    )


class WebsiteMetadata(BaseModel):
    """Website/page metadata extracted during processing."""

    url: str
    title: str | None = None
    description: str | None = None
    author: str | None = None
    published_date: datetime | None = None
    modified_date: datetime | None = None
    language: str | None = None
    canonical_url: str | None = None
    og_type: str | None = None
    og_title: str | None = None
    og_description: str | None = None
    og_image: str | None = None
    twitter_card: str | None = None
    twitter_title: str | None = None
    twitter_description: str | None = None
    twitter_image: str | None = None
    schema_org: dict[str, Any] = Field(default_factory=dict)
    custom_properties: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "url": "https://example.com/article",
                "title": "Example Article",
                "description": "This is an example article",
                "author": "John Doe",
                "published_date": "2024-01-15T10:30:00Z",
                "language": "en",
                "og_type": "article",
                "og_title": "Example Article",
            }
        }
    )


class CrawledPage(BaseModel):
    """A single crawled page with extracted content."""

    url: str
    status_code: int
    content_type: str | None = None
    text_content: str
    metadata: WebsiteMetadata | None = None
    word_count: int = 0
    character_count: int = 0
    links: list[str] = Field(default_factory=list)
    error: str | None = None

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "url": "https://example.com",
                "status_code": 200,
                "content_type": "text/html",
                "text_content": "Example Domain\n\nThis domain is for use in illustrative examples...",
                "metadata": {"title": "Example Domain"},
                "word_count": 150,
                "character_count": 950,
                "links": ["https://example.com/about"],
            }
        }
    )


class WebsiteChunk(BaseModel):
    """A single chunk of website content."""

    index: int = Field(..., ge=0)
    content: str = Field(..., min_length=1)
    token_count: int | None = Field(None, ge=0)
    source_url: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "index": 0,
                "content": "Example Domain This domain is for use in illustrative examples...",
                "token_count": 150,
                "source_url": "https://example.com",
                "metadata": {"section": "main"},
            }
        }
    )


class WebsiteProcessingResult(BaseModel):
    """Result of website processing pipeline."""

    website_id: uuid.UUID
    url: str
    total_pages: int
    total_chunks: int
    total_tokens: int
    processing_time_ms: int
    pages: list[CrawledPage]
    chunks: list[WebsiteChunk]
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "website_id": "550e8400-e29b-41d4-a716-446655440000",
                "url": "https://example.com",
                "total_pages": 3,
                "total_chunks": 15,
                "total_tokens": 7500,
                "processing_time_ms": 5000,
                "pages": [],
                "chunks": [],
                "metadata": {"crawl_depth": "single"},
            }
        }
    )


class WebsiteResponse(BaseModel):
    """Website response with full details."""

    id: uuid.UUID
    url: str
    title: str | None = None
    status: WebsiteStatus
    last_crawled_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "url": "https://example.com",
                "title": "Example Domain",
                "status": "ready",
                "last_crawled_at": "2024-01-15T10:30:00Z",
                "metadata": {"page_count": 5, "total_chunks": 25},
                "created_at": "2024-01-15T10:00:00Z",
                "updated_at": "2024-01-15T10:30:00Z",
            }
        },
    )


class WebsiteListResponse(BaseModel):
    """Paginated list of websites."""

    items: list[WebsiteResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


class WebsiteDeleteResponse(BaseModel):
    """Response after website deletion."""

    website_id: uuid.UUID
    message: str


class WebsiteChunkingConfig(BaseModel):
    """Configuration for website content chunking."""

    chunk_size: int = Field(
        default=1000, ge=100, le=10000, description="Target chunk size in tokens"
    )
    chunk_overlap: int = Field(
        default=200, ge=0, le=1000, description="Overlap between chunks in tokens"
    )
    strategy: str = Field(
        default="recursive", description="Chunking strategy: recursive, semantic, fixed"
    )
    preserve_code_blocks: bool = Field(
        default=True, description="Preserve code blocks in chunks"
    )
    preserve_tables: bool = Field(default=True, description="Preserve tables in chunks")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "strategy": "recursive",
                "preserve_code_blocks": True,
                "preserve_tables": True,
            }
        }
    )


class WebsiteEmbeddingConfig(BaseModel):
    """Configuration for website content embedding."""

    model_name: str = Field(
        default="text-embedding-3-small", description="Embedding model to use"
    )
    dimensions: int | None = Field(
        None, description="Embedding dimensions (if supported by model)"
    )
    batch_size: int = Field(
        default=100, ge=1, le=1000, description="Batch size for embedding generation"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model_name": "text-embedding-3-small",
                "dimensions": 1536,
                "batch_size": 100,
            }
        }
    )


class WebsiteProcessingConfig(BaseModel):
    """Complete processing configuration for website ingestion."""

    chunking: WebsiteChunkingConfig = Field(default_factory=WebsiteChunkingConfig)
    embedding: WebsiteEmbeddingConfig = Field(default_factory=WebsiteEmbeddingConfig)
    extract_metadata: bool = Field(default=True, description="Extract page metadata")
    clean_content: bool = Field(
        default=True, description="Clean extracted HTML content"
    )
    min_content_length: int = Field(
        default=100, ge=0, description="Minimum content length to process"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "chunking": {
                    "chunk_size": 1000,
                    "chunk_overlap": 200,
                    "strategy": "recursive",
                },
                "embedding": {
                    "model_name": "text-embedding-3-small",
                    "dimensions": 1536,
                },
                "extract_metadata": True,
                "clean_content": True,
                "min_content_length": 100,
            }
        }
    )
