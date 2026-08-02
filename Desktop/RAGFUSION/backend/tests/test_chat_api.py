"""Tests for chat API endpoints."""

import json
from uuid import uuid4

import pytest
from fastapi import status


@pytest.mark.asyncio
class TestChatAPI:
    """Test suite for chat API endpoints."""

    async def test_chat_endpoint_new_session(self, client, auth_headers):
        """Test POST /chat with new session."""
        response = client.post(
            "/api/v1/chat",
            json={
                "message": "Hello, world!",
                "session_id": None,
            },
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "session_id" in data
        assert "message" in data
        assert data["message"]["content"]

    async def test_chat_endpoint_existing_session(self, client, auth_headers):
        """Test POST /chat with existing session."""
        # Create session first
        create_response = client.post(
            "/api/v1/chat/sessions",
            json={"title": "Test Chat"},
            headers=auth_headers,
        )
        session_id = create_response.json()["id"]

        # Send chat message
        response = client.post(
            "/api/v1/chat",
            json={
                "message": "Hello!",
                "session_id": session_id,
            },
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["session_id"] == session_id

    async def test_chat_endpoint_invalid_session(self, client, auth_headers):
        """Test POST /chat with invalid session."""
        response = client.post(
            "/api/v1/chat",
            json={
                "message": "Hello!",
                "session_id": str(uuid4()),
            },
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_404_NOT_FOUND

    async def test_chat_endpoint_unauthorized(self, client):
        """Test POST /chat without authentication."""
        response = client.post(
            "/api/v1/chat",
            json={
                "message": "Hello!",
                "session_id": None,
            },
        )

        assert response.status_code in (
            status.HTTP_401_UNAUTHORIZED,
            status.HTTP_422_UNPROCESSABLE_ENTITY,
        )

    async def test_chat_endpoint_with_params(self, client, auth_headers):
        """Test POST /chat with custom parameters."""
        response = client.post(
            "/api/v1/chat",
            json={
                "message": "Hello!",
                "session_id": None,
                "max_tokens": 1000,
                "temperature": 0.5,
                "top_k": 10,
                "include_sources": True,
            },
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "sources" in data

    async def test_chat_endpoint_without_sources(self, client, auth_headers):
        """Test POST /chat without sources."""
        response = client.post(
            "/api/v1/chat",
            json={
                "message": "Hello!",
                "session_id": None,
                "include_sources": False,
            },
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["sources"] == []

    async def test_chat_stream_endpoint(self, client, auth_headers):
        """Test POST /chat/stream."""
        response = client.post(
            "/api/v1/chat/stream",
            json={
                "message": "Hello!",
                "session_id": None,
            },
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_200_OK
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

    async def test_create_session(self, client, auth_headers):
        """Test POST /chat/sessions."""
        response = client.post(
            "/api/v1/chat/sessions",
            json={
                "title": "New Chat Session",
                "session_metadata": {"key": "value"},
            },
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["title"] == "New Chat Session"
        assert data["session_metadata"]["key"] == "value"
        assert "id" in data

    async def test_list_sessions(self, client, auth_headers):
        """Test GET /chat/sessions."""
        # Create some sessions
        for i in range(3):
            client.post(
                "/api/v1/chat/sessions",
                json={"title": f"Chat {i}"},
                headers=auth_headers,
            )

        response = client.get(
            "/api/v1/chat/sessions",
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "sessions" in data
        assert "total" in data
        assert data["total"] >= 3

    async def test_list_sessions_pagination(self, client, auth_headers):
        """Test GET /chat/sessions with pagination."""
        response = client.get(
            "/api/v1/chat/sessions?limit=2&offset=0",
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["sessions"]) <= 2

    async def test_get_session_history(self, client, auth_headers):
        """Test GET /chat/sessions/{session_id}."""
        # Create session
        create_response = client.post(
            "/api/v1/chat/sessions",
            json={"title": "Test Chat"},
            headers=auth_headers,
        )
        session_id = create_response.json()["id"]

        # Send some messages
        client.post(
            "/api/v1/chat",
            json={
                "message": "Hello!",
                "session_id": session_id,
            },
            headers=auth_headers,
        )

        # Get history
        response = client.get(
            f"/api/v1/chat/sessions/{session_id}",
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "session" in data
        assert "messages" in data
        assert len(data["messages"]) > 0

    async def test_get_session_history_not_found(self, client, auth_headers):
        """Test GET /chat/sessions/{session_id} with invalid ID."""
        response = client.get(
            f"/api/v1/chat/sessions/{uuid4()}",
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_404_NOT_FOUND

    async def test_update_session(self, client, auth_headers):
        """Test PATCH /chat/sessions/{session_id}."""
        # Create session
        create_response = client.post(
            "/api/v1/chat/sessions",
            json={"title": "Old Title"},
            headers=auth_headers,
        )
        session_id = create_response.json()["id"]

        # Update session
        response = client.patch(
            f"/api/v1/chat/sessions/{session_id}",
            json={"title": "New Title"},
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["title"] == "New Title"

    async def test_delete_session(self, client, auth_headers):
        """Test DELETE /chat/sessions/{session_id}."""
        # Create session
        create_response = client.post(
            "/api/v1/chat/sessions",
            json={"title": "Test Chat"},
            headers=auth_headers,
        )
        session_id = create_response.json()["id"]

        # Delete session
        response = client.delete(
            f"/api/v1/chat/sessions/{session_id}",
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_204_NO_CONTENT

        # Verify deletion
        get_response = client.get(
            f"/api/v1/chat/sessions/{session_id}",
            headers=auth_headers,
        )
        assert get_response.status_code == status.HTTP_404_NOT_FOUND

    async def test_clear_session_messages(self, client, auth_headers):
        """Test DELETE /chat/sessions/{session_id}/messages."""
        # Create session with messages
        create_response = client.post(
            "/api/v1/chat/sessions",
            json={"title": "Test Chat"},
            headers=auth_headers,
        )
        session_id = create_response.json()["id"]

        client.post(
            "/api/v1/chat",
            json={
                "message": "Hello!",
                "session_id": session_id,
            },
            headers=auth_headers,
        )

        # Clear messages
        response = client.delete(
            f"/api/v1/chat/sessions/{session_id}/messages",
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_204_NO_CONTENT

        # Verify messages cleared
        history_response = client.get(
            f"/api/v1/chat/sessions/{session_id}",
            headers=auth_headers,
        )
        data = history_response.json()
        assert len(data["messages"]) == 0

    async def test_concurrent_chat_requests(self, client, auth_headers):
        """Test concurrent chat requests."""
        # Create session
        create_response = client.post(
            "/api/v1/chat/sessions",
            json={"title": "Concurrent Test"},
            headers=auth_headers,
        )
        session_id = create_response.json()["id"]

        # Send multiple requests sequentially (TestClient is synchronous)
        responses = []
        for i in range(5):
            response = client.post(
                "/api/v1/chat",
                json={
                    "message": f"Message {i}",
                    "session_id": session_id,
                },
                headers=auth_headers,
            )
            responses.append(response)

        # All should succeed
        for response in responses:
            assert response.status_code == status.HTTP_200_OK

        # Verify all messages saved
        history_response = client.get(
            f"/api/v1/chat/sessions/{session_id}",
            headers=auth_headers,
        )
        data = history_response.json()
        # Should have 10 messages (5 user + 5 assistant)
        assert len(data["messages"]) == 10

    async def test_chat_validation_empty_message(self, client, auth_headers):
        """Test chat with empty message."""
        response = client.post(
            "/api/v1/chat",
            json={
                "message": "",
                "session_id": None,
            },
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    async def test_chat_validation_invalid_params(self, client, auth_headers):
        """Test chat with invalid parameters."""
        response = client.post(
            "/api/v1/chat",
            json={
                "message": "Hello!",
                "session_id": None,
                "temperature": 5.0,  # Invalid: should be <= 2.0
            },
            headers=auth_headers,
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
