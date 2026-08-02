"""Chat request and response schemas."""

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class Citation(BaseModel):
    """Citation information for a source."""

    source_id: UUID
    source_type: str = Field(..., description="Type: document, website, youtube")
    content: str = Field(..., description="Relevant content snippet")
    score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")
    metadata: dict[str, Any] = Field(default_factory=dict)


class MessageBase(BaseModel):
    """Base message schema."""

    role: str = Field(..., description="Role: system, user, or assistant")
    content: str = Field(..., min_length=1)


class MessageCreate(MessageBase):
    """Message creation schema."""

    pass


class MessageResponse(MessageBase):
    """Message response schema."""

    id: UUID
    chat_session_id: UUID
    citations: list[Citation] = Field(default_factory=list)
    token_count: int | None = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ChatRequest(BaseModel):
    """Chat request schema."""

    message: str = Field(..., min_length=1, max_length=10000)
    session_id: UUID | None = Field(
        default=None, description="Existing session ID or null for new session"
    )
    max_tokens: int | None = Field(default=2000, ge=100, le=4000)
    temperature: float | None = Field(default=0.7, ge=0.0, le=2.0)
    top_k: int | None = Field(
        default=5, ge=1, le=20, description="Number of documents to retrieve"
    )
    include_sources: bool = Field(default=True, description="Include source citations")
    stream: bool = Field(default=False, description="Enable streaming response")


class ChatResponse(BaseModel):
    """Chat response schema."""

    session_id: UUID
    message: MessageResponse
    sources: list[Citation] = Field(default_factory=list)
    token_usage: dict[str, int] | None = None
    processing_time_ms: float | None = None


class StreamChunk(BaseModel):
    """Streaming response chunk."""

    type: str = Field(
        ..., description="Chunk type: content, citation, metadata, done, error"
    )
    content: str | None = None
    citation: Citation | None = None
    metadata: dict[str, Any] | None = None
    error: str | None = None


class ChatSessionCreate(BaseModel):
    """Chat session creation schema."""

    title: str = Field(default="New chat", max_length=255)
    session_metadata: dict[str, Any] = Field(default_factory=dict)


class ChatSessionResponse(BaseModel):
    """Chat session response schema."""

    id: UUID
    user_id: UUID
    title: str
    session_metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime
    message_count: int = 0

    model_config = {"from_attributes": True}


class ChatSessionListResponse(BaseModel):
    """List of chat sessions."""

    sessions: list[ChatSessionResponse]
    total: int


class ChatHistoryResponse(BaseModel):
    """Chat history response schema."""

    session: ChatSessionResponse
    messages: list[MessageResponse]
