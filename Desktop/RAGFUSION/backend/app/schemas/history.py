"""History request and response schemas."""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class HistorySortField(str, Enum):
    """Sort field options for history listing."""

    created_at = "created_at"
    updated_at = "updated_at"
    title = "title"
    message_count = "message_count"


class HistorySortOrder(str, Enum):
    """Sort order options."""

    asc = "asc"
    desc = "desc"


class HistoryFilterStatus(str, Enum):
    """Filter status options."""

    active = "active"
    deleted = "deleted"
    all = "all"


class HistorySessionBase(BaseModel):
    """Base history session schema."""

    title: str = Field(..., max_length=255)
    session_metadata: dict[str, Any] = Field(default_factory=dict)


class HistorySessionResponse(HistorySessionBase):
    """History session response schema."""

    id: UUID
    user_id: UUID
    message_count: int = 0
    is_deleted: bool = False
    deleted_at: datetime | None = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class HistorySessionListResponse(BaseModel):
    """List of history sessions with pagination info."""

    sessions: list[HistorySessionResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


class HistorySearchRequest(BaseModel):
    """History search request schema."""

    query: str = Field(..., min_length=1, max_length=500, description="Search query")
    limit: int = Field(default=50, ge=1, le=100, description="Maximum results")
    offset: int = Field(default=0, ge=0, description="Number of results to skip")


class HistorySearchResponse(BaseModel):
    """History search response schema."""

    sessions: list[HistorySessionResponse]
    total: int
    query: str


class HistoryRenameRequest(BaseModel):
    """History rename request schema."""

    title: str = Field(..., min_length=1, max_length=255)


class HistoryRenameResponse(BaseModel):
    """History rename response schema."""

    id: UUID
    title: str
    updated_at: datetime

    model_config = {"from_attributes": True}


class HistoryDeleteResponse(BaseModel):
    """History delete response schema."""

    id: UUID
    is_deleted: bool
    deleted_at: datetime
    message: str = "Session moved to trash"


class HistoryRestoreResponse(BaseModel):
    """History restore response schema."""

    id: UUID
    is_deleted: bool
    restored_at: datetime
    message: str = "Session restored successfully"


class HistoryPaginationParams(BaseModel):
    """History pagination parameters."""

    page: int = Field(default=1, ge=1, description="Page number (1-indexed)")
    page_size: int = Field(default=20, ge=1, le=100, description="Items per page")


class HistorySortParams(BaseModel):
    """History sort parameters."""

    sort_by: HistorySortField = Field(default=HistorySortField.updated_at)
    sort_order: HistorySortOrder = Field(default=HistorySortOrder.desc)


class HistoryFilterParams(BaseModel):
    """History filter parameters."""

    status: HistoryFilterStatus = Field(default=HistoryFilterStatus.active)
    search: str | None = Field(default=None, max_length=500)
    date_from: datetime | None = None
    date_to: datetime | None = None
