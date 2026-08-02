"""Tests for chat service."""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.services.chat_service import ChatService
from app.services.memory_service import MemoryService


@pytest.mark.asyncio
class TestChatService:
    """Test suite for ChatService."""

    @pytest.fixture
    def mock_rag_pipeline(self):
        """Mock RAG pipeline."""
        mock = MagicMock()
        mock.graph = MagicMock()
        mock.graph.invoke = MagicMock(
            return_value={
                "response": "This is a test response.",
                "reranked_docs": [
                    (
                        "Test content snippet",
                        {
                            "source_id": str(uuid4()),
                            "source_type": "document",
                        },
                        0.95,
                    )
                ],
                "metadata": {"token_usage": {"total_tokens": 100}},
            }
        )
        return mock

    async def test_chat_new_session(self, db_session, test_user, mock_rag_pipeline):
        """Test chat with new session creation."""
        chat_service = ChatService(
            db=db_session,
            user_id=test_user.id,
            rag_pipeline=mock_rag_pipeline,
        )

        result = await chat_service.chat(
            message="Hello, world!",
            session_id=None,
        )

        assert result["session_id"] is not None
        assert result["message"].content == "This is a test response."
        assert len(result["sources"]) > 0
        assert result["token_usage"] is not None
        assert result["processing_time_ms"] > 0

    async def test_chat_existing_session(
        self, db_session, test_user, mock_rag_pipeline
    ):
        """Test chat with existing session."""
        # Create session
        memory_service = MemoryService(db_session)
        session = await memory_service.create_session(
            user_id=test_user.id, title="Test Chat"
        )

        chat_service = ChatService(
            db=db_session,
            user_id=test_user.id,
            rag_pipeline=mock_rag_pipeline,
        )

        result = await chat_service.chat(
            message="Hello again!",
            session_id=session.id,
        )

        assert result["session_id"] == session.id
        assert result["message"].content == "This is a test response."

    async def test_chat_invalid_session(self, db_session, test_user, mock_rag_pipeline):
        """Test chat with invalid session ID."""
        chat_service = ChatService(
            db=db_session,
            user_id=test_user.id,
            rag_pipeline=mock_rag_pipeline,
        )

        with pytest.raises(ValueError):
            await chat_service.chat(
                message="Hello!",
                session_id=uuid4(),
            )

    async def test_chat_with_context(self, db_session, test_user, mock_rag_pipeline):
        """Test chat with conversation context."""
        # Create session with history
        memory_service = MemoryService(db_session)
        session = await memory_service.create_session(
            user_id=test_user.id, title="Test Chat"
        )

        from app.models.entities import MessageRole

        await memory_service.add_message(
            session_id=session.id,
            role=MessageRole.user,
            content="Previous message",
        )
        await memory_service.add_message(
            session_id=session.id,
            role=MessageRole.assistant,
            content="Previous response",
        )

        chat_service = ChatService(
            db=db_session,
            user_id=test_user.id,
            rag_pipeline=mock_rag_pipeline,
        )

        result = await chat_service.chat(
            message="Follow-up question",
            session_id=session.id,
        )

        assert result["session_id"] == session.id
        assert result["message"].content == "This is a test response."

        # Verify history is preserved
        messages = await memory_service.get_messages(session.id)
        assert len(messages) == 4  # 2 old + 2 new

    async def test_chat_timeout(self, db_session, test_user):
        """Test chat with timeout."""
        mock_pipeline = MagicMock()
        mock_pipeline.graph = MagicMock()

        # Simulate slow response
        # Simulate slow response synchronously
        def slow_invoke(*args, **kwargs):
            import time
            time.sleep(2)
            return {"response": "Too slow"}

        mock_pipeline.graph.invoke = slow_invoke

        chat_service = ChatService(
            db=db_session,
            user_id=test_user.id,
            rag_pipeline=mock_pipeline,
            timeout_seconds=1,
        )

        result = await chat_service.chat(
            message="Hello!",
            session_id=None,
        )

        # Should return timeout error message
        assert "timed out" in result["message"].content.lower()

    async def test_chat_error_handling(self, db_session, test_user):
        """Test chat with RAG pipeline error."""
        mock_pipeline = MagicMock()
        mock_pipeline.graph = MagicMock()
        mock_pipeline.graph.invoke = MagicMock(side_effect=Exception("Pipeline error"))

        chat_service = ChatService(
            db=db_session,
            user_id=test_user.id,
            rag_pipeline=mock_pipeline,
        )

        result = await chat_service.chat(
            message="Hello!",
            session_id=None,
        )

        # Should return error message
        assert "error" in result["message"].content.lower()

    async def test_chat_without_sources(self, db_session, test_user, mock_rag_pipeline):
        """Test chat without source citations."""
        chat_service = ChatService(
            db=db_session,
            user_id=test_user.id,
            rag_pipeline=mock_rag_pipeline,
        )

        result = await chat_service.chat(
            message="Hello!",
            session_id=None,
            include_sources=False,
        )

        assert len(result["sources"]) == 0

    async def test_chat_with_custom_params(
        self, db_session, test_user, mock_rag_pipeline
    ):
        """Test chat with custom parameters."""
        chat_service = ChatService(
            db=db_session,
            user_id=test_user.id,
            rag_pipeline=mock_rag_pipeline,
        )

        result = await chat_service.chat(
            message="Hello!",
            session_id=None,
            max_tokens=1000,
            temperature=0.5,
            top_k=10,
        )

        assert result["message"] is not None

    @pytest.mark.asyncio
    async def test_chat_stream(self, db_session, test_user, mock_rag_pipeline):
        """Test streaming chat response."""
        chat_service = ChatService(
            db=db_session,
            user_id=test_user.id,
            rag_pipeline=mock_rag_pipeline,
        )

        chunks = []
        async for chunk in chat_service.chat_stream(
            message="Hello!",
            session_id=None,
        ):
            chunks.append(chunk)

        # Check for different chunk types
        chunk_types = {chunk.type for chunk in chunks}
        assert "metadata" in chunk_types
        assert "content" in chunk_types
        assert "done" in chunk_types

    @pytest.mark.asyncio
    async def test_chat_stream_error(self, db_session, test_user):
        """Test streaming chat with error."""
        mock_pipeline = MagicMock()
        mock_pipeline.graph = MagicMock()
        mock_pipeline.graph.invoke = MagicMock(side_effect=Exception("Stream error"))

        chat_service = ChatService(
            db=db_session,
            user_id=test_user.id,
            rag_pipeline=mock_pipeline,
        )

        chunks = []
        async for chunk in chat_service.chat_stream(
            message="Hello!",
            session_id=None,
        ):
            chunks.append(chunk)

        # Should include error chunk
        error_chunks = [c for c in chunks if c.type == "error"]
        assert len(error_chunks) > 0

    async def test_convert_citations(self, db_session, test_user, mock_rag_pipeline):
        """Test citation conversion."""
        chat_service = ChatService(
            db=db_session,
            user_id=test_user.id,
            rag_pipeline=mock_rag_pipeline,
        )

        citations_data = [
            {
                "source_id": str(uuid4()),
                "source_type": "document",
                "content": "Test content",
                "score": 0.95,
                "metadata": {"page": 1},
            }
        ]

        citations = chat_service._convert_citations(citations_data)

        assert len(citations) == 1
        assert citations[0].content == "Test content"
        assert citations[0].score == 0.95
        assert citations[0].source_type == "document"
