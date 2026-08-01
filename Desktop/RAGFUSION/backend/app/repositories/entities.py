"""Domain repositories with ownership-aware query helpers."""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import (ChatSession, Document, Embedding, Message, Setting,
                        User, Website, YouTubeSource)
from app.repositories.base import Repository


class UserRepository(Repository[User]):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, User)

    async def by_email(self, email: str) -> User | None:
        return await self.session.scalar(select(User).where(User.email == email))


class DocumentRepository(Repository[Document]):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, Document)

    async def for_user(self, user_id: uuid.UUID) -> list[Document]:
        return list(
            await self.session.scalars(
                select(Document)
                .where(Document.user_id == user_id)
                .order_by(Document.created_at.desc())
            )
        )


class WebsiteRepository(Repository[Website]):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, Website)


class YouTubeSourceRepository(Repository[YouTubeSource]):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, YouTubeSource)


class ChatSessionRepository(Repository[ChatSession]):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, ChatSession)

    async def for_user(self, user_id: uuid.UUID) -> list[ChatSession]:
        return list(
            await self.session.scalars(
                select(ChatSession)
                .where(ChatSession.user_id == user_id)
                .order_by(ChatSession.updated_at.desc())
            )
        )


class MessageRepository(Repository[Message]):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, Message)

    async def for_chat(self, chat_session_id: uuid.UUID) -> list[Message]:
        return list(
            await self.session.scalars(
                select(Message)
                .where(Message.chat_session_id == chat_session_id)
                .order_by(Message.created_at)
            )
        )


class SettingRepository(Repository[Setting]):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, Setting)

    async def get_value(self, user_id: uuid.UUID, key: str) -> Setting | None:
        return await self.session.scalar(
            select(Setting).where(Setting.user_id == user_id, Setting.key == key)
        )


class EmbeddingRepository(Repository[Embedding]):
    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session, Embedding)
