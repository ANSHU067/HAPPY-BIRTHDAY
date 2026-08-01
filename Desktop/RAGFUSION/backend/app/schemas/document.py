"""Document ingestion schemas."""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DocumentFormat(str, enum.Enum):
    """Supported document formats."""

    pdf = "pdf"
    docx = "docx"
    txt = "txt"
    csv = "csv"
    markdown = "markdown"


class DocumentStatus(str, enum.Enum):
    """Document processing status."""

    pending = "pending"
    validating = "validating"
    cleaning = "cleaning"
    chunking = "chunking"
    embedding = "embedding"
    storing = "storing"
    ready = "ready"
    failed = "failed"


class DocumentUploadRequest(BaseModel):
    """Request to upload a document."""

    filename: str = Field(..., min_length=1, max_length=512)
    content_type: str = Field(..., description="MIME type of the uploaded file")
    size_bytes: int = Field(
        ..., gt=0, le=100_000_000, description="File size in bytes (max 100MB)"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "filename": "document.pdf",
                "content_type": "application/pdf",
                "size_bytes": 1024000,
            }
        }
    )


class DocumentUploadResponse(BaseModel):
    """Response after document upload initiation."""

    document_id: uuid.UUID
    filename: str
    status: DocumentStatus
    message: str

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "document_id": "550e8400-e29b-41d4-a716-446655440000",
                "filename": "document.pdf",
                "status": "pending",
                "message": "Document uploaded successfully. Processing started.",
            }
        }
    )


class DocumentChunk(BaseModel):
    """A single chunk of document content."""

    index: int = Field(..., ge=0)
    content: str = Field(..., min_length=1)
    token_count: int | None = Field(None, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "index": 0,
                "content": "This is the first chunk of the document...",
                "token_count": 150,
                "metadata": {"page": 1, "section": "introduction"},
            }
        }
    )


class DocumentProcessingResult(BaseModel):
    """Result of document processing pipeline."""

    document_id: uuid.UUID
    filename: str
    format: DocumentFormat
    total_chunks: int
    total_tokens: int
    processing_time_ms: int
    chunks: list[DocumentChunk]
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "document_id": "550e8400-e29b-41d4-a716-446655440000",
                "filename": "document.pdf",
                "format": "pdf",
                "total_chunks": 10,
                "total_tokens": 5000,
                "processing_time_ms": 1500,
                "chunks": [],
                "metadata": {"pages": 5, "author": "John Doe"},
            }
        }
    )


class DocumentMetadata(BaseModel):
    """Document metadata extracted during processing."""

    format: DocumentFormat
    page_count: int | None = None
    word_count: int | None = None
    character_count: int | None = None
    author: str | None = None
    title: str | None = None
    subject: str | None = None
    creator: str | None = None
    producer: str | None = None
    creation_date: datetime | None = None
    modification_date: datetime | None = None
    custom_properties: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "format": "pdf",
                "page_count": 10,
                "word_count": 5000,
                "character_count": 30000,
                "author": "John Doe",
                "title": "Sample Document",
                "subject": "Document Processing",
                "creator": "Microsoft Word",
                "producer": "pymupdf",
                "creation_date": "2024-01-15T10:30:00Z",
                "modification_date": "2024-01-15T11:00:00Z",
                "custom_properties": {},
            }
        }
    )


class DocumentResponse(BaseModel):
    """Document response with full details."""

    id: uuid.UUID
    filename: str
    storage_key: str
    mime_type: str | None
    size_bytes: int | None
    checksum: str | None
    status: DocumentStatus
    metadata: DocumentMetadata | None = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(
        from_attributes=True,
        json_schema_extra={
            "example": {
                "id": "550e8400-e29b-41d4-a716-446655440000",
                "filename": "document.pdf",
                "storage_key": "uploads/550e8400-e29b-41d4-a716-446655440000/document.pdf",
                "mime_type": "application/pdf",
                "size_bytes": 1024000,
                "checksum": "sha256:abc123...",
                "status": "ready",
                "metadata": {
                    "format": "pdf",
                    "page_count": 10,
                    "word_count": 5000,
                },
                "created_at": "2024-01-15T10:30:00Z",
                "updated_at": "2024-01-15T10:35:00Z",
            }
        },
    )


class DocumentListResponse(BaseModel):
    """Paginated list of documents."""

    items: list[DocumentResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


class DocumentDeleteResponse(BaseModel):
    """Response after document deletion."""

    document_id: uuid.UUID
    message: str


class ChunkingConfig(BaseModel):
    """Configuration for document chunking."""

    chunk_size: int = Field(
        default=1000, ge=100, le=10000, description="Target chunk size in tokens"
    )
    chunk_overlap: int = Field(
        default=200, ge=0, le=1000, description="Overlap between chunks in tokens"
    )
    strategy: str = Field(
        default="recursive", description="Chunking strategy: recursive, semantic, fixed"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "strategy": "recursive",
            }
        }
    )


class EmbeddingConfig(BaseModel):
    """Configuration for document embedding."""

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


class ProcessingConfig(BaseModel):
    """Complete processing configuration."""

    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    extract_metadata: bool = Field(
        default=True, description="Extract document metadata"
    )
    clean_text: bool = Field(default=True, description="Clean extracted text")
