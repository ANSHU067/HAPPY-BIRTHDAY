"""Tests for memory service."""

import asyncio
from uuid import uuid4

import pytest

from app.models.entities import MessageRole
from app.services.memory_service import MemoryService


@pytest.mark.asyncio
class TestMemoryService:
    """Test suite for MemoryService."""

    async def test_create_session(self, db_session, test_user):
        """Test creating a chat session."""
        memory_service = MemoryService(db_session)

        session = await memory_service.create_session(
            user_id=test_user.id,
            title="Test Chat",
            session_metadata={"test": "data"},
        )

        assert session.id is not None
        assert session.user_id == test_user.id
        assert session.title == "Test Chat"
        assert session.session_metadata["test"] == "data"

    async def test_get_session(self, db_session, test_user):
        """Test retrieving a chat session."""
        memory_service = MemoryService(db_session)

        # Create session
        created_session = await memory_service.create_session(
            user_id=test_user.id, title="Test Chat"
        )

        # Retrieve session
        session = await memory_service.get_session(created_session.id, test_user.id)

        assert session is not None
        assert session.id == created_session.id
        assert session.user_id == test_user.id

    async def test_get_session_not_found(self, db_session, test_user):
        """Test retrieving non-existent session."""
        memory_service = MemoryService(db_session)

        session = await memory_service.get_session(uuid4(), test_user.id)

        assert session is None

    async def test_get_session_wrong_user(self, db_session, test_user):
        """Test retrieving session with wrong user ID."""
        memory_service = MemoryService(db_session)

        # Create session
        created_session = await memory_service.create_session(
            user_id=test_user.id, title="Test Chat"
        )

        # Try to retrieve with different user ID
        session = await memory_service.get_session(created_session.id, uuid4())

        assert session is None

    async def test_list_sessions(self, db_session, test_user):
        """Test listing chat sessions."""
        memory_service = MemoryService(db_session)

        # Create multiple sessions
        for i in range(5):
            await memory_service.create_session(user_id=test_user.id, title=f"Chat {i}")

        # List sessions
        sessions, total = await memory_service.list_sessions(test_user.id)

        assert len(sessions) == 5
        assert total == 5

    async def test_list_sessions_pagination(self, db_session, test_user):
        """Test listing sessions with pagination."""
        memory_service = MemoryService(db_session)

        # Create sessions
        for i in range(10):
            await memory_service.create_session(user_id=test_user.id, title=f"Chat {i}")

        # List first page
        sessions, total = await memory_service.list_sessions(
            test_user.id, limit=5, offset=0
        )

        assert len(sessions) == 5
        assert total == 10

        # List second page
        sessions, total = await memory_service.list_sessions(
            test_user.id, limit=5, offset=5
        )

        assert len(sessions) == 5
        assert total == 10

    async def test_delete_session(self, db_session, test_user):
        """Test deleting a chat session."""
        memory_service = MemoryService(db_session)

        # Create session
        session = await memory_service.create_session(
            user_id=test_user.id, title="Test Chat"
        )

        # Delete session
        deleted = await memory_service.delete_session(session.id, test_user.id)

        assert deleted is True

        # Verify deletion
        retrieved = await memory_service.get_session(session.id, test_user.id)
        assert retrieved is None

    async def test_add_message(self, db_session, test_user):
        """Test adding a message to a session."""
        memory_service = MemoryService(db_session)

        # Create session
        session = await memory_service.create_session(
            user_id=test_user.id, title="Test Chat"
        )

        # Add message
        message = await memory_service.add_message(
            session_id=session.id,
            role=MessageRole.user,
            content="Hello, world!",
            token_count=10,
        )

        assert message.id is not None
        assert message.chat_session_id == session.id
        assert message.role == MessageRole.user
        assert message.content == "Hello, world!"
        assert message.token_count == 10

    async def test_get_messages(self, db_session, test_user):
        """Test retrieving messages from a session."""
        memory_service = MemoryService(db_session)

        # Create session
        session = await memory_service.create_session(
            user_id=test_user.id, title="Test Chat"
        )

        # Add messages
        await memory_service.add_message(
            session_id=session.id,
            role=MessageRole.user,
            content="Message 1",
        )
        await asyncio.sleep(0.01) # Ensure chronological order

        await memory_service.add_message(
            session_id=session.id,
            role=MessageRole.assistant,
            content="Message 2",
        )

        # Retrieve messages
        messages = await memory_service.get_messages(session.id)

        assert len(messages) == 2
        assert messages[0].content == "Message 1"
        assert messages[1].content == "Message 2"

    async def test_get_context_messages(self, db_session, test_user):
        """Test retrieving context messages with limit."""
        memory_service = MemoryService(db_session, max_context_messages=3)

        # Create session
        session = await memory_service.create_session(
            user_id=test_user.id, title="Test Chat"
        )

        # Add many messages
        for i in range(10):
            await memory_service.add_message(
                session_id=session.id,
                role=MessageRole.user,
                content=f"Message {i}",
            )
            # Fix: Add a tiny delay so timestamps are distinct for sorting
            await asyncio.sleep(0.01)

        # Retrieve context
        messages = await memory_service.get_context_messages(session.id)

        assert len(messages) == 3
        # Should get the most recent messages
        assert messages[0].content == "Message 7"
        assert messages[2].content == "Message 9"

    async def test_get_context_with_token_limit(self, db_session, test_user):
        """Test retrieving context messages within token limit."""
        memory_service = MemoryService(db_session, max_tokens=100)

        # Create session
        session = await memory_service.create_session(
            user_id=test_user.id, title="Test Chat"
        )

        # Add messages with token counts
        await memory_service.add_message(
            session_id=session.id,
            role=MessageRole.user,
            content="Message 1",
            token_count=30,
        )
        await asyncio.sleep(0.01) # Fix: Add a tiny delay so timestamps are distinct

        await memory_service.add_message(
            session_id=session.id,
            role=MessageRole.assistant,
            content="Message 2",
            token_count=40,
        )
        await asyncio.sleep(0.01) # Fix: Add a tiny delay so timestamps are distinct

        await memory_service.add_message(
            session_id=session.id,
            role=MessageRole.user,
            content="Message 3",
            token_count=50,
        )

        # Retrieve context (should only get last 2 messages = 90 tokens)
        messages = await memory_service.get_context_with_token_limit(session.id)

        assert len(messages) == 2
        assert messages[0].content == "Message 2"
        assert messages[1].content == "Message 3"

    async def test_update_session_title(self, db_session, test_user):
        """Test updating session title."""
        memory_service = MemoryService(db_session)

        # Create session
        session = await memory_service.create_session(
            user_id=test_user.id, title="Old Title"
        )

        # Update title
        updated = await memory_service.update_session_title(
            session_id=session.id,
            user_id=test_user.id,
            title="New Title",
        )

        assert updated is not None
        assert updated.title == "New Title"

    async def test_get_message_count(self, db_session, test_user):
        """Test getting message count."""
        memory_service = MemoryService(db_session)

        # Create session
        session = await memory_service.create_session(
            user_id=test_user.id, title="Test Chat"
        )

        # Add messages
        for i in range(5):
            await memory_service.add_message(
                session_id=session.id,
                role=MessageRole.user,
                content=f"Message {i}",
            )

        # Get count
        count = await memory_service.get_message_count(session.id)

        assert count == 5

    async def test_clear_session_messages(self, db_session, test_user):
        """Test clearing messages from a session."""
        memory_service = MemoryService(db_session)

        # Create session with messages
        session = await memory_service.create_session(
            user_id=test_user.id, title="Test Chat"
        )
        for i in range(5):
            await memory_service.add_message(
                session_id=session.id,
                role=MessageRole.user,
                content=f"Message {i}",
            )

        # Clear messages
        cleared = await memory_service.clear_session_messages(session.id, test_user.id)

        assert cleared is True

        # Verify messages cleared
        count = await memory_service.get_message_count(session.id)
        assert count == 0

    async def test_messages_to_llm_format(self, db_session, test_user):
        """Test converting messages to LLM format."""
        memory_service = MemoryService(db_session)

        # Create session with messages
        session = await memory_service.create_session(
            user_id=test_user.id, title="Test Chat"
        )
        await memory_service.add_message(
            session_id=session.id,
            role=MessageRole.user,
            content="Hello",
        )
        await asyncio.sleep(0.01) # Ensure chronological order

        await memory_service.add_message(
            session_id=session.id,
            role=MessageRole.assistant,
            content="Hi there!",
        )

        # Get messages
        messages = await memory_service.get_messages(session.id)

        # Convert to LLM format
        llm_messages = memory_service.messages_to_llm_format(messages)

        assert len(llm_messages) == 2
        assert llm_messages[0] == {"role": "user", "content": "Hello"}
        assert llm_messages[1] == {"role": "assistant", "content": "Hi there!"}
