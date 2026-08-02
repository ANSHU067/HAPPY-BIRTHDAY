"""History API endpoints."""

import logging
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.auth import get_current_user_id
from app.db.session import get_db_session
from app.schemas.auth import UserResponse
from app.schemas.history import (HistoryDeleteResponse, HistoryFilterParams,
                                 HistoryFilterStatus, HistoryPaginationParams,
                                 HistoryRenameRequest, HistoryRenameResponse,
                                 HistoryRestoreResponse, HistorySearchRequest,
                                 HistorySearchResponse,
                                 HistorySessionListResponse,
                                 HistorySessionResponse, HistorySortField,
                                 HistorySortOrder, HistorySortParams)
from app.services.history_service import HistoryService

router = APIRouter(prefix="/history", tags=["history"])

logger = logging.getLogger(__name__)


async def get_history_service(
    db: AsyncSession = Depends(get_db_session),
    current_user: UserResponse = Depends(get_current_user_id),
) -> HistoryService:
    """Dependency to get history service."""
    return HistoryService(db=db)


@router.get("", response_model=HistorySessionListResponse)
async def list_history(
    pagination: Annotated[HistoryPaginationParams, Depends()],
    sort: Annotated[HistorySortParams, Depends()],
    filter_params: Annotated[HistoryFilterParams, Depends()],
    current_user: UserResponse = Depends(get_current_user_id),
    history_service: HistoryService = Depends(get_history_service),
) -> HistorySessionListResponse:
    """List chat sessions with pagination, sorting, and filtering.

    Supports:
    - Pagination (page, page_size)
    - Sorting by created_at, updated_at, title, message_count
    - Filtering by status (active, deleted, all), search term, date range

    Args:
        pagination: Pagination parameters
        sort: Sort parameters
        filter_params: Filter parameters
        current_user: Authenticated user

    Returns:
        Paginated list of chat sessions
    """
    try:
        return await history_service.list_sessions(
            user_id=current_user.id,
            pagination=pagination,
            sort=sort,
            filter_params=filter_params,
        )

    except Exception as e:
        logger.error(f"Error listing history: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list history",
        )


@router.get("/search", response_model=HistorySearchResponse)
async def search_history(
    q: Annotated[
        str, Query(..., min_length=1, max_length=500, description="Search query")
    ],
    limit: Annotated[int, Query(ge=1, le=100, description="Maximum results")] = 50,
    offset: Annotated[int, Query(ge=0, description="Number of results to skip")] = 0,
    current_user: UserResponse = Depends(get_current_user_id),
    history_service: HistoryService = Depends(get_history_service),
) -> HistorySearchResponse:
    """Search chat sessions by title.

    Args:
        q: Search query
        limit: Maximum results
        offset: Number of results to skip
        current_user: Authenticated user

    Returns:
        List of matching sessions
    """
    try:
        search_request = HistorySearchRequest(query=q, limit=limit, offset=offset)
        sessions = await history_service.search_sessions(
            current_user.id, search_request
        )

        # Get total count for the search query
        from sqlalchemy import func, select

        from app.models.entities import ChatSession

        search_term = f"%{q}%"
        count_stmt = (
            select(func.count())
            .select_from(ChatSession)
            .where(
                ChatSession.user_id == current_user.id,
                ChatSession.title.ilike(search_term),
                ChatSession.deleted_at.is_(None),
            )
        )
        count_result = await history_service.db.execute(count_stmt)
        total = count_result.scalar_one()

        return HistorySearchResponse(
            sessions=sessions,
            total=total,
            query=q,
        )

    except Exception as e:
        logger.error(f"Error searching history: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to search history",
        )


@router.get("/{session_id}", response_model=HistorySessionResponse)
async def get_history_session(
    session_id: UUID,
    current_user: UserResponse = Depends(get_current_user_id),
    history_service: HistoryService = Depends(get_history_service),
) -> HistorySessionResponse:
    """Get a single chat session by ID.

    Args:
        session_id: Session ID
        current_user: Authenticated user

    Returns:
        Session details
    """
    try:
        session = await history_service.get_session(session_id, current_user.id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Session not found"
            )
        return session

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting history session: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get session",
        )


@router.patch("/{session_id}", response_model=HistoryRenameResponse)
async def rename_history_session(
    session_id: UUID,
    request: HistoryRenameRequest,
    current_user: UserResponse = Depends(get_current_user_id),
    history_service: HistoryService = Depends(get_history_service),
) -> HistoryRenameResponse:
    """Rename a chat session.

    Args:
        session_id: Session ID
        request: Rename request with new title
        current_user: Authenticated user

    Returns:
        Updated session with new title
    """
    try:
        session = await history_service.rename_session(
            session_id=session_id,
            user_id=current_user.id,
            new_title=request.title,
        )

        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Session not found"
            )

        return HistoryRenameResponse(
            id=session.id,
            title=session.title,
            updated_at=session.updated_at,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error renaming history session: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to rename session",
        )


@router.delete("/{session_id}", response_model=HistoryDeleteResponse)
async def delete_history_session(
    session_id: UUID,
    current_user: UserResponse = Depends(get_current_user_id),
    history_service: HistoryService = Depends(get_history_service),
) -> HistoryDeleteResponse:
    """Soft delete a chat session (move to trash).

    Args:
        session_id: Session ID
        current_user: Authenticated user

    Returns:
        Deleted session info
    """
    try:
        session = await history_service.delete_session(session_id, current_user.id)

        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Session not found"
            )

        return HistoryDeleteResponse(
            id=session.id,
            is_deleted=session.is_deleted,
            deleted_at=session.deleted_at,
            message="Session moved to trash",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting history session: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete session",
        )


@router.post("/{session_id}/restore", response_model=HistoryRestoreResponse)
async def restore_history_session(
    session_id: UUID,
    current_user: UserResponse = Depends(get_current_user_id),
    history_service: HistoryService = Depends(get_history_service),
) -> HistoryRestoreResponse:
    """Restore a soft-deleted chat session.

    Args:
        session_id: Session ID
        current_user: Authenticated user

    Returns:
        Restored session info
    """
    try:
        session = await history_service.restore_session(session_id, current_user.id)

        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Session not found"
            )

        return HistoryRestoreResponse(
            id=session.id,
            is_deleted=session.is_deleted,
            restored_at=session.updated_at,
            message="Session restored successfully",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error restoring history session: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to restore session",
        )
