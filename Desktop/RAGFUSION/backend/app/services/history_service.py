"""History service for managing chat session history with advanced features."""

import logging
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import desc, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.entities import ChatSession
from app.schemas.history import (HistoryFilterParams, HistoryFilterStatus,
                                 HistoryPaginationParams, HistorySearchRequest,
                                 HistorySessionListResponse,
                                 HistorySessionResponse, HistorySortField,
                                 HistorySortOrder, HistorySortParams)

logger = logging.getLogger(__name__)


class HistoryService:
    """Service for managing chat session history with pagination, sorting, filtering, search, rename, delete, and restore."""

    def __init__(self, db: AsyncSession):
        """Initialize history service.

        Args:
            db: Database session
        """
        self.db = db

    async def list_sessions(
        self,
        user_id: UUID,
        pagination: HistoryPaginationParams,
        sort: HistorySortParams,
        filter_params: HistoryFilterParams,
    ) -> HistorySessionListResponse:
        """List chat sessions with pagination, sorting, and filtering.

        Args:
            user_id: User ID
            pagination: Pagination parameters
            sort: Sort parameters
            filter_params: Filter parameters

        Returns:
            Paginated list of sessions
        """
        # Build base query
        stmt = select(ChatSession).where(ChatSession.user_id == user_id)

        # Apply filters
        if filter_params.status == HistoryFilterStatus.active:
            stmt = stmt.where(ChatSession.deleted_at.is_(None))
        elif filter_params.status == HistoryFilterStatus.deleted:
            stmt = stmt.where(ChatSession.deleted_at.is_not(None))
        # If status is "all", no additional filter

        if filter_params.search:
            search_term = f"%{filter_params.search}%"
            stmt = stmt.where(ChatSession.title.ilike(search_term))

        if filter_params.date_from:
            stmt = stmt.where(ChatSession.created_at >= filter_params.date_from)

        if filter_params.date_to:
            stmt = stmt.where(ChatSession.created_at <= filter_params.date_to)

        # Get total count before pagination
        count_stmt = select(func.count()).select_from(stmt.subquery())
        count_result = await self.db.execute(count_stmt)
        total = count_result.scalar_one()

        # Apply sorting
        sort_column = self._get_sort_column(sort.sort_by)
        if sort.sort_order == HistorySortOrder.desc:
            stmt = stmt.order_by(desc(sort_column))
        else:
            stmt = stmt.order_by(sort_column)

        # Apply pagination
        offset = (pagination.page - 1) * pagination.page_size
        stmt = stmt.limit(pagination.page_size).offset(offset)

        # Execute query
        result = await self.db.execute(stmt)
        sessions = list(result.scalars().all())

        # Convert to response with message counts
        session_responses = []
        for session in sessions:
            message_count = await self._get_message_count(session.id)
            session_response = HistorySessionResponse.model_validate(session)
            session_response.message_count = message_count
            session_response.is_deleted = session.deleted_at is not None
            session_responses.append(session_response)

        total_pages = (total + pagination.page_size - 1) // pagination.page_size

        return HistorySessionListResponse(
            sessions=session_responses,
            total=total,
            page=pagination.page,
            page_size=pagination.page_size,
            total_pages=total_pages,
        )

    async def get_session(
        self, session_id: UUID, user_id: UUID
    ) -> HistorySessionResponse | None:
        """Get a single chat session by ID.

        Args:
            session_id: Session ID
            user_id: User ID (for ownership verification)

        Returns:
            Session response or None if not found
        """
        stmt = select(ChatSession).where(
            ChatSession.id == session_id, ChatSession.user_id == user_id
        )
        result = await self.db.execute(stmt)
        session = result.scalar_one_or_none()

        if not session:
            return None

        message_count = await self._get_message_count(session_id)
        session_response = HistorySessionResponse.model_validate(session)
        session_response.message_count = message_count
        session_response.is_deleted = session.deleted_at is not None

        return session_response

    async def search_sessions(
        self, user_id: UUID, search_request: HistorySearchRequest
    ) -> list[HistorySessionResponse]:
        """Search chat sessions by title.

        Args:
            user_id: User ID
            search_request: Search parameters

        Returns:
            List of matching sessions
        """
        search_term = f"%{search_request.query}%"

        stmt = (
            select(ChatSession)
            .where(
                ChatSession.user_id == user_id,
                ChatSession.title.ilike(search_term),
                ChatSession.deleted_at.is_(None),  # Only search active sessions
            )
            .order_by(desc(ChatSession.updated_at))
            .limit(search_request.limit)
            .offset(search_request.offset)
        )

        result = await self.db.execute(stmt)
        sessions = list(result.scalars().all())

        session_responses = []
        for session in sessions:
            message_count = await self._get_message_count(session.id)
            session_response = HistorySessionResponse.model_validate(session)
            session_response.message_count = message_count
            session_response.is_deleted = False
            session_responses.append(session_response)

        return session_responses

    async def rename_session(
        self, session_id: UUID, user_id: UUID, new_title: str
    ) -> HistorySessionResponse | None:
        """Rename a chat session.

        Args:
            session_id: Session ID
            user_id: User ID (for ownership verification)
            new_title: New title

        Returns:
            Updated session response or None if not found
        """
        stmt = select(ChatSession).where(
            ChatSession.id == session_id, ChatSession.user_id == user_id
        )
        result = await self.db.execute(stmt)
        session = result.scalar_one_or_none()

        if not session:
            return None

        session.title = new_title
        await self.db.commit()
        await self.db.refresh(session)

        logger.info(f"Renamed session {session_id} to '{new_title}'")

        message_count = await self._get_message_count(session_id)
        session_response = HistorySessionResponse.model_validate(session)
        session_response.message_count = message_count
        session_response.is_deleted = session.deleted_at is not None

        return session_response

    async def delete_session(
        self, session_id: UUID, user_id: UUID
    ) -> HistorySessionResponse | None:
        """Soft delete a chat session (move to trash).

        Args:
            session_id: Session ID
            user_id: User ID (for ownership verification)

        Returns:
            Deleted session response or None if not found
        """
        stmt = select(ChatSession).where(
            ChatSession.id == session_id, ChatSession.user_id == user_id
        )
        result = await self.db.execute(stmt)
        session = result.scalar_one_or_none()

        if not session:
            return None

        if session.deleted_at is not None:
            # Already deleted
            message_count = await self._get_message_count(session_id)
            session_response = HistorySessionResponse.model_validate(session)
            session_response.message_count = message_count
            session_response.is_deleted = True
            return session_response

        session.deleted_at = datetime.utcnow()
        await self.db.commit()
        await self.db.refresh(session)

        logger.info(f"Deleted session {session_id} (moved to trash)")

        message_count = await self._get_message_count(session_id)
        session_response = HistorySessionResponse.model_validate(session)
        session_response.message_count = message_count
        session_response.is_deleted = True

        return session_response

    async def restore_session(
        self, session_id: UUID, user_id: UUID
    ) -> HistorySessionResponse | None:
        """Restore a soft-deleted chat session.

        Args:
            session_id: Session ID
            user_id: User ID (for ownership verification)

        Returns:
            Restored session response or None if not found
        """
        stmt = select(ChatSession).where(
            ChatSession.id == session_id, ChatSession.user_id == user_id
        )
        result = await self.db.execute(stmt)
        session = result.scalar_one_or_none()

        if not session:
            return None

        if session.deleted_at is None:
            # Not deleted
            message_count = await self._get_message_count(session_id)
            session_response = HistorySessionResponse.model_validate(session)
            session_response.message_count = message_count
            session_response.is_deleted = False
            return session_response

        session.deleted_at = None
        await self.db.commit()
        await self.db.refresh(session)

        logger.info(f"Restored session {session_id}")

        message_count = await self._get_message_count(session_id)
        session_response = HistorySessionResponse.model_validate(session)
        session_response.message_count = message_count
        session_response.is_deleted = False

        return session_response

    async def _get_message_count(self, session_id: UUID) -> int:
        """Get message count for a session.

        Args:
            session_id: Session ID

        Returns:
            Message count
        """
        from app.models.entities import Message

        stmt = (
            select(func.count())
            .select_from(Message)
            .where(Message.chat_session_id == session_id)
        )
        result = await self.db.execute(stmt)
        return result.scalar_one()

    def _get_sort_column(self, sort_field: HistorySortField):
        """Get the SQLAlchemy column for sorting.

        Args:
            sort_field: Sort field enum

        Returns:
            SQLAlchemy column
        """
        sort_columns = {
            HistorySortField.created_at: ChatSession.created_at,
            HistorySortField.updated_at: ChatSession.updated_at,
            HistorySortField.title: ChatSession.title,
            HistorySortField.message_count: ChatSession.updated_at,  # Fallback to updated_at
        }
        return sort_columns.get(sort_field, ChatSession.updated_at)
