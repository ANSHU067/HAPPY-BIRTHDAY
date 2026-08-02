"""Memory service for conversation persistence and context management."""

import logging
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.entities import ChatSession, Message, MessageRole

logger = logging.getLogger(__name__)


class MemoryService:
    """Service for managing conversation memory and context."""

    def __init__(
        self,
        db: AsyncSession,
        max_context_messages: int = 10,
        max_tokens: int = 4000,
    ):
        """Initialize memory service.

        Args:
            db: Database session
            max_context_messages: Maximum number of messages to keep in context
            max_tokens: Maximum tokens to keep in context window
        """
        self.db = db
        self.max_context_messages = max_context_messages
        self.max_tokens = max_tokens

    async def create_session(
        self,
        user_id: UUID,
        title: str = "New chat",
        session_metadata: dict[str, Any] | None = None,
    ) -> ChatSession:
        """Create a new chat session.

        Args:
            user_id: User ID
            title: Session title
            session_metadata: Additional session metadata

        Returns:
            Created chat session
        """
        session = ChatSession(
            user_id=user_id,
            title=title,
            session_metadata=session_metadata or {},
        )
        self.db.add(session)
        await self.db.commit()
        await self.db.refresh(session)

        logger.info(f"Created chat session {session.id} for user {user_id}")
        return session

    async def get_session(self, session_id: UUID, user_id: UUID) -> ChatSession | None:
        """Get a chat session by ID.

        Args:
            session_id: Session ID
            user_id: User ID (for ownership verification)

        Returns:
            Chat session or None if not found
        """
        stmt = (
            select(ChatSession)
            .where(ChatSession.id == session_id, ChatSession.user_id == user_id)
            .options(selectinload(ChatSession.messages))
        )
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def list_sessions(
        self, user_id: UUID, limit: int = 50, offset: int = 0
    ) -> tuple[list[ChatSession], int]:
        """List chat sessions for a user.

        Args:
            user_id: User ID
            limit: Maximum number of sessions to return
            offset: Number of sessions to skip

        Returns:
            Tuple of (sessions, total_count)
        """
        # Get sessions
        stmt = (
            select(ChatSession)
            .where(ChatSession.user_id == user_id)
            .order_by(desc(ChatSession.updated_at))
            .limit(limit)
            .offset(offset)
        )
        result = await self.db.execute(stmt)
        sessions = list(result.scalars().all())

        # Get total count
        count_stmt = (
            select(func.count())
            .select_from(ChatSession)
            .where(ChatSession.user_id == user_id)
        )
        count_result = await self.db.execute(count_stmt)
        total = count_result.scalar_one()

        return sessions, total

    async def delete_session(self, session_id: UUID, user_id: UUID) -> bool:
        """Delete a chat session.

        Args:
            session_id: Session ID
            user_id: User ID (for ownership verification)

        Returns:
            True if deleted, False if not found
        """
        session = await self.get_session(session_id, user_id)
        if not session:
            return False

        await self.db.delete(session)
        await self.db.commit()

        logger.info(f"Deleted chat session {session_id}")
        return True

    async def add_message(
        self,
        session_id: UUID,
        role: MessageRole,
        content: str,
        citations: list[dict[str, Any]] | None = None,
        token_count: int | None = None,
    ) -> Message:
        """Add a message to a chat session.

        Args:
            session_id: Session ID
            role: Message role
            content: Message content
            citations: Optional citations
            token_count: Optional token count

        Returns:
            Created message
        """
        message = Message(
            chat_session_id=session_id,
            role=role,
            content=content,
            citations=citations or [],
            token_count=token_count,
        )
        self.db.add(message)
        session = await self.db.get(ChatSession, session_id)
        if session:
            # History's message-count sort uses the session activity timestamp.
            session.updated_at = datetime.now(UTC)
        await self.db.commit()
        await self.db.refresh(message)

        logger.debug(f"Added {role.value} message to session {session_id}")
        return message

    async def get_messages(
        self, session_id: UUID, limit: int | None = None
    ) -> list[Message]:
        """Get messages from a chat session.

        Args:
            session_id: Session ID
            limit: Optional limit on number of messages

        Returns:
            List of messages ordered by creation time
        """
        stmt = (
            select(Message)
            .where(Message.chat_session_id == session_id)
            .order_by(Message.created_at)
        )

        if limit:
            stmt = stmt.limit(limit)

        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def get_context_messages(
        self, session_id: UUID, max_messages: int | None = None
    ) -> list[Message]:
        """Get recent messages for context window.

        Args:
            session_id: Session ID
            max_messages: Maximum number of messages (defaults to max_context_messages)

        Returns:
            List of recent messages
        """
        limit = max_messages or self.max_context_messages

        stmt = (
            select(Message)
            .where(Message.chat_session_id == session_id)
            .order_by(desc(Message.created_at))
            .limit(limit)
        )

        result = await self.db.execute(stmt)
        messages = list(result.scalars().all())

        # Return in chronological order
        return messages[::-1]

    async def get_context_with_token_limit(self, session_id: UUID) -> list[Message]:
        """Get messages that fit within token limit.

        Args:
            session_id: Session ID

        Returns:
            List of messages that fit in context window
        """
        messages = await self.get_context_messages(session_id)

        # If no token counts recorded, return all messages up to max_context_messages
        if not any(msg.token_count for msg in messages):
            return messages

        # Filter messages to fit token limit
        context_messages = []
        total_tokens = 0

        for message in reversed(messages):
            msg_tokens = message.token_count or 0
            if total_tokens + msg_tokens <= self.max_tokens:
                context_messages.insert(0, message)
                total_tokens += msg_tokens
            else:
                break

        return context_messages

    async def update_session_title(
        self, session_id: UUID, user_id: UUID, title: str
    ) -> ChatSession | None:
        """Update chat session title.

        Args:
            session_id: Session ID
            user_id: User ID (for ownership verification)
            title: New title

        Returns:
            Updated session or None if not found
        """
        session = await self.get_session(session_id, user_id)
        if not session:
            return None

        session.title = title
        await self.db.commit()
        await self.db.refresh(session)

        logger.info(f"Updated title for session {session_id}")
        return session

    async def get_message_count(self, session_id: UUID) -> int:
        """Get total message count for a session.

        Args:
            session_id: Session ID

        Returns:
            Message count
        """
        stmt = (
            select(func.count())
            .select_from(Message)
            .where(Message.chat_session_id == session_id)
        )
        result = await self.db.execute(stmt)
        return result.scalar_one()

    async def clear_session_messages(self, session_id: UUID, user_id: UUID) -> bool:
        """Clear all messages from a session.

        Args:
            session_id: Session ID
            user_id: User ID (for ownership verification)

        Returns:
            True if cleared, False if session not found
        """
        session = await self.get_session(session_id, user_id)
        if not session:
            return False

        # Delete all messages
        for message in session.messages:
            await self.db.delete(message)

        await self.db.commit()

        logger.info(f"Cleared messages from session {session_id}")
        return True

    def messages_to_llm_format(self, messages: list[Message]) -> list[dict[str, str]]:
        """Convert database messages to LLM format.

        Args:
            messages: List of database messages

        Returns:
            List of messages in LLM format [{"role": "user", "content": "..."}]
        """
        return [{"role": msg.role.value, "content": msg.content} for msg in messages]
