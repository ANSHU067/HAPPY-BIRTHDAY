"""Tests for history API endpoints."""

from datetime import datetime, timedelta
from uuid import UUID, uuid4

import pytest
from fastapi import status

from app.models.entities import MessageRole
from app.services.memory_service import MemoryService


@pytest.mark.asyncio
class TestHistoryAPI:
    """Test suite for history API endpoints."""

    def create_test_sessions(self, client, auth_headers, count=5):
        """Helper to create test sessions."""
        session_ids = []
        for i in range(count):
            response = client.post(
                "/api/v1/chat/sessions",
                json={"title": f"Test Session {i}"},
                headers=auth_headers,
            )
            assert response.status_code == status.HTTP_201_CREATED
            session_ids.append(response.json()["id"])
        return session_ids

    async def test_list_history_basic(self, client, auth_headers):
        """Test GET /history - basic listing."""
        # Create some sessions
        self.create_test_sessions(client, auth_headers, 3)

        response = client.get("/api/v1/history", headers=auth_headers)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "sessions" in data
        assert "total" in data
        assert "page" in data
        assert "page_size" in data
        assert "total_pages" in data
        assert data["total"] >= 3
        assert data["page"] == 1
        assert data["page_size"] == 20

    async def test_list_history_pagination(self, client, auth_headers):
        """Test GET /history with pagination."""
        self.create_test_sessions(client, auth_headers, 5)

        # Test first page
        response = client.get("/api/v1/history?page=1&page_size=2", headers=auth_headers)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["sessions"]) == 2
        assert data["page"] == 1
        assert data["page_size"] == 2
        assert data["total_pages"] == 3

        # Test second page
        response = client.get("/api/v1/history?page=2&page_size=2", headers=auth_headers)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["sessions"]) == 2
        assert data["page"] == 2

        # Test third page
        response = client.get("/api/v1/history?page=3&page_size=2", headers=auth_headers)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["sessions"]) == 1
        assert data["page"] == 3

    async def test_list_history_sorting(self, client, auth_headers):
        """Test GET /history with sorting."""
        self.create_test_sessions(client, auth_headers, 3)

        # Sort by title ascending
        response = client.get(
            "/api/v1/history?sort_by=title&sort_order=asc",
            headers=auth_headers,
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        titles = [s["title"] for s in data["sessions"]]
        assert titles == sorted(titles)

        # Sort by title descending
        response = client.get(
            "/api/v1/history?sort_by=title&sort_order=desc",
            headers=auth_headers,
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        titles = [s["title"] for s in data["sessions"]]
        assert titles == sorted(titles, reverse=True)

        # Sort by updated_at descending (default)
        response = client.get(
            "/api/v1/history?sort_by=updated_at&sort_order=desc",
            headers=auth_headers,
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["sessions"]) >= 3

    async def test_list_history_filtering_status(self, client, auth_headers):
        """Test GET /history with status filtering."""
        self.create_test_sessions(client, auth_headers, 3)

        # Filter active (default)
        response = client.get("/api/v1/history?status=active", headers=auth_headers)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        for session in data["sessions"]:
            assert session["is_deleted"] is False

        # Filter deleted (should be empty initially)
        response = client.get("/api/v1/history?status=deleted", headers=auth_headers)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total"] == 0

        # Filter all
        response = client.get("/api/v1/history?status=all", headers=auth_headers)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total"] >= 3

    async def test_list_history_filtering_date_range(self, client, auth_headers):
        """Test GET /history with date range filtering."""
        self.create_test_sessions(client, auth_headers, 2)

        # Filter by date_from (today)
        today = datetime.utcnow().date().isoformat()
        response = client.get(
            f"/api/v1/history?date_from={today}",
            headers=auth_headers,
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total"] >= 2

        # Filter by date_to (tomorrow)
        tomorrow = (datetime.utcnow() + timedelta(days=1)).date().isoformat()
        response = client.get(
            f"/api/v1/history?date_to={tomorrow}",
            headers=auth_headers,
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total"] >= 2

    async def test_search_history(self, client, auth_headers):
        """Test GET /history/search."""
        # Create sessions with specific titles
        client.post(
            "/api/v1/chat/sessions",
            json={"title": "Python Programming"},
            headers=auth_headers,
        )
        client.post(
            "/api/v1/chat/sessions",
            json={"title": "JavaScript Basics"},
            headers=auth_headers,
        )
        client.post(
            "/api/v1/chat/sessions",
            json={"title": "Python Advanced"},
            headers=auth_headers,
        )

        # Search for "Python"
        response = client.get("/api/v1/history/search?q=Python", headers=auth_headers)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["query"] == "Python"
        assert data["total"] == 2
        assert len(data["sessions"]) == 2
        for session in data["sessions"]:
            assert "Python" in session["title"]

        # Search with limit
        response = client.get(
            "/api/v1/history/search?q=Python&limit=1", headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["sessions"]) == 1

        # Search with offset
        response = client.get(
            "/api/v1/history/search?q=Python&offset=1", headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert len(data["sessions"]) == 1

        # Search non-existent
        response = client.get(
            "/api/v1/history/search?q=NonExistent", headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total"] == 0
        assert len(data["sessions"]) == 0

    async def test_get_history_session(self, client, auth_headers):
        """Test GET /history/{session_id}."""
        # Create session
        create_response = client.post(
            "/api/v1/chat/sessions",
            json={"title": "Test Session"},
            headers=auth_headers,
        )
        session_id = create_response.json()["id"]

        # Get session
        response = client.get(f"/api/v1/history/{session_id}", headers=auth_headers)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["id"] == session_id
        assert data["title"] == "Test Session"
        assert "message_count" in data
        assert "is_deleted" in data
        assert "created_at" in data
        assert "updated_at" in data

    async def test_get_history_session_not_found(self, client, auth_headers):
        """Test GET /history/{session_id} with invalid ID."""
        response = client.get(f"/api/v1/history/{uuid4()}", headers=auth_headers)
        assert response.status_code == status.HTTP_404_NOT_FOUND

    async def test_rename_history_session(self, client, auth_headers):
        """Test PATCH /history/{session_id} - rename."""
        # Create session
        create_response = client.post(
            "/api/v1/chat/sessions",
            json={"title": "Old Title"},
            headers=auth_headers,
        )
        session_id = create_response.json()["id"]

        # Rename session
        response = client.patch(
            f"/api/v1/history/{session_id}",
            json={"title": "New Title"},
            headers=auth_headers,
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["id"] == session_id
        assert data["title"] == "New Title"
        assert "updated_at" in data

        # Verify the change
        get_response = client.get(f"/api/v1/history/{session_id}", headers=auth_headers)
        assert get_response.json()["title"] == "New Title"

    async def test_rename_history_session_not_found(self, client, auth_headers):
        """Test PATCH /history/{session_id} with invalid ID."""
        response = client.patch(
            f"/api/v1/history/{uuid4()}",
            json={"title": "New Title"},
            headers=auth_headers,
        )
        assert response.status_code == status.HTTP_404_NOT_FOUND

    async def test_rename_history_session_validation(self, client, auth_headers):
        """Test PATCH /history/{session_id} with invalid title."""
        create_response = client.post(
            "/api/v1/chat/sessions",
            json={"title": "Test"},
            headers=auth_headers,
        )
        session_id = create_response.json()["id"]

        # Empty title
        response = client.patch(
            f"/api/v1/history/{session_id}",
            json={"title": ""},
            headers=auth_headers,
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

        # Title too long
        response = client.patch(
            f"/api/v1/history/{session_id}",
            json={"title": "x" * 256},
            headers=auth_headers,
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    async def test_delete_history_session(self, client, auth_headers):
        """Test DELETE /history/{session_id} - soft delete."""
        # Create session
        create_response = client.post(
            "/api/v1/chat/sessions",
            json={"title": "To Delete"},
            headers=auth_headers,
        )
        session_id = create_response.json()["id"]

        # Delete session
        response = client.delete(f"/api/v1/history/{session_id}", headers=auth_headers)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["id"] == session_id
        assert data["is_deleted"] is True
        assert data["deleted_at"] is not None
        assert data["message"] == "Session moved to trash"

        # Verify it's in deleted list
        list_response = client.get(
            "/api/v1/history?status=deleted", headers=auth_headers
        )
        deleted_sessions = list_response.json()["sessions"]
        assert any(s["id"] == session_id for s in deleted_sessions)

        # Verify it's not in active list
        list_response = client.get("/api/v1/history?status=active", headers=auth_headers)
        active_sessions = list_response.json()["sessions"]
        assert not any(s["id"] == session_id for s in active_sessions)

    async def test_delete_history_session_not_found(self, client, auth_headers):
        """Test DELETE /history/{session_id} with invalid ID."""
        response = client.delete(f"/api/v1/history/{uuid4()}", headers=auth_headers)
        assert response.status_code == status.HTTP_404_NOT_FOUND

    async def test_delete_already_deleted_session(self, client, auth_headers):
        """Test DELETE on already deleted session."""
        # Create and delete
        create_response = client.post(
            "/api/v1/chat/sessions",
            json={"title": "Test"},
            headers=auth_headers,
        )
        session_id = create_response.json()["id"]

        client.delete(f"/api/v1/history/{session_id}", headers=auth_headers)

        # Delete again
        response = client.delete(f"/api/v1/history/{session_id}", headers=auth_headers)
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["is_deleted"] is True

    async def test_restore_history_session(self, client, auth_headers):
        """Test POST /history/{session_id}/restore."""
        # Create and delete
        create_response = client.post(
            "/api/v1/chat/sessions",
            json={"title": "To Restore"},
            headers=auth_headers,
        )
        session_id = create_response.json()["id"]

        client.delete(f"/api/v1/history/{session_id}", headers=auth_headers)

        # Restore session
        response = client.post(
            f"/api/v1/history/{session_id}/restore", headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["id"] == session_id
        assert data["is_deleted"] is False
        assert "restored_at" in data
        assert data["message"] == "Session restored successfully"

        # Verify it's in active list
        list_response = client.get("/api/v1/history?status=active", headers=auth_headers)
        active_sessions = list_response.json()["sessions"]
        assert any(s["id"] == session_id for s in active_sessions)

        # Verify it's not in deleted list
        list_response = client.get(
            "/api/v1/history?status=deleted", headers=auth_headers
        )
        deleted_sessions = list_response.json()["sessions"]
        assert not any(s["id"] == session_id for s in deleted_sessions)

    async def test_restore_history_session_not_found(self, client, auth_headers):
        """Test POST /history/{session_id}/restore with invalid ID."""
        response = client.post(
            f"/api/v1/history/{uuid4()}/restore", headers=auth_headers
        )
        assert response.status_code == status.HTTP_404_NOT_FOUND

    async def test_restore_already_active_session(self, client, auth_headers):
        """Test POST /history/{session_id}/restore on active session."""
        create_response = client.post(
            "/api/v1/chat/sessions",
            json={"title": "Active"},
            headers=auth_headers,
        )
        session_id = create_response.json()["id"]

        response = client.post(
            f"/api/v1/history/{session_id}/restore", headers=auth_headers
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["is_deleted"] is False

    async def test_history_unauthorized(self, client):
        """Test history endpoints without authentication."""
        endpoints = [
            ("GET", "/api/v1/history"),
            ("GET", "/api/v1/history/search?q=test"),
            ("GET", f"/api/v1/history/{uuid4()}"),
            ("PATCH", f"/api/v1/history/{uuid4()}"),
            ("DELETE", f"/api/v1/history/{uuid4()}"),
            ("POST", f"/api/v1/history/{uuid4()}/restore"),
        ]

        for method, endpoint in endpoints:
            if method == "GET":
                response = client.get(endpoint)
            elif method == "PATCH":
                response = client.patch(endpoint, json={"title": "New"})
            elif method == "DELETE":
                response = client.delete(endpoint)
            elif method == "POST":
                response = client.post(endpoint)

            assert response.status_code in (
                status.HTTP_401_UNAUTHORIZED,
                status.HTTP_422_UNPROCESSABLE_ENTITY,
            )

    async def test_history_isolation_between_users(
        self, client, auth_headers, test_user, db_session
    ):
        """Test that users can only see their own history."""
        from app.models.entities import User, UserRole
        from app.services.auth import create_access_token, hash_password

        # Create another user
        other_user = User(
            id=uuid4(),
            email="other@example.com",
            display_name="Other User",
            password_hash=hash_password("password"),
            role=UserRole.user,
            is_active=True,
        )
        db_session.add(other_user)
        await db_session.commit()

        other_token = create_access_token(other_user.id, role=other_user.role)
        other_headers = {"Authorization": f"Bearer {other_token}"}

        # Create sessions for both users
        client.post(
            "/api/v1/chat/sessions", json={"title": "User 1 Session"}, headers=auth_headers
        )
        client.post(
            "/api/v1/chat/sessions", json={"title": "User 2 Session"}, headers=other_headers
        )

        # User 1 should only see their session
        response = client.get("/api/v1/history", headers=auth_headers)
        data = response.json()
        assert data["total"] == 1
        assert data["sessions"][0]["title"] == "User 1 Session"

        # User 2 should only see their session
        response = client.get("/api/v1/history", headers=other_headers)
        data = response.json()
        assert data["total"] == 1
        assert data["sessions"][0]["title"] == "User 2 Session"

    async def test_concurrent_history_operations(self, client, auth_headers):
        """Test concurrent history operations."""
        import asyncio

        # Create session
        create_response = client.post(
            "/api/v1/chat/sessions",
            json={"title": "Concurrent Test"},
            headers=auth_headers,
        )
        session_id = create_response.json()["id"]

        # Concurrent rename and delete
        async def rename():
            return client.patch(
                f"/api/v1/history/{session_id}",
                json={"title": "Renamed"},
                headers=auth_headers,
            )

        async def delete():
            return client.delete(f"/api/v1/history/{session_id}", headers=auth_headers)

        # Run concurrently
        results = await asyncio.gather(rename(), delete(), return_exceptions=True)

        # At least one should succeed
        success_count = sum(
            1 for r in results if not isinstance(r, Exception) and r.status_code < 400
        )
        assert success_count >= 1

    async def test_history_with_messages(self, client, auth_headers):
        """Test history with sessions that have messages."""
        # Create session
        create_response = client.post(
            "/api/v1/chat/sessions",
            json={"title": "With Messages"},
            headers=auth_headers,
        )
        session_id = create_response.json()["id"]

        # Send some messages via chat
        client.post(
            "/api/v1/chat",
            json={"message": "Hello", "session_id": session_id},
            headers=auth_headers,
        )
        client.post(
            "/api/v1/chat",
            json={"message": "World", "session_id": session_id},
            headers=auth_headers,
        )

        # Check history shows message count
        response = client.get("/api/v1/history", headers=auth_headers)
        data = response.json()
        session = next(s for s in data["sessions"] if s["id"] == session_id)
        assert session["message_count"] >= 2  # user + assistant messages

    async def test_history_sort_by_message_count(
        self, client, auth_headers, db_session
    ):
        """Test sorting by message count."""
        # Create sessions with different message counts
        session1 = client.post(
            "/api/v1/chat/sessions", json={"title": "Session 1"}, headers=auth_headers
        )
        session2 = client.post(
            "/api/v1/chat/sessions", json={"title": "Session 2"}, headers=auth_headers
        )

        sid1 = session1.json()["id"]
        sid2 = session2.json()["id"]

        # Add messages directly so this history test does not depend on the LLM.
        memory_service = MemoryService(db_session)
        for _ in range(3):
            await memory_service.add_message(
                session_id=UUID(sid1),
                role=MessageRole.user,
                content="Hi",
            )

        # Sort by message_count (using updated_at as proxy)
        response = client.get(
            "/api/v1/history?sort_by=message_count&sort_order=desc",
            headers=auth_headers,
        )
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        # Session with more messages should appear first (due to updated_at)
        sessions = data["sessions"]
        # Find our test sessions
        test_sessions = [s for s in sessions if s["id"] in (sid1, sid2)]
        # The one with more messages should have more recent updated_at
        if len(test_sessions) == 2:
            assert (
                test_sessions[0]["id"] == sid1
            )  # More messages = more recent updated_at


@pytest.mark.asyncio
class TestHistoryEdgeCases:
    """Test edge cases for history API."""

    async def test_history_pagination_boundary(self, client, auth_headers):
        """Test pagination at boundaries."""
        # Create exactly 20 sessions (default page size)
        for i in range(20):
            client.post(
                "/api/v1/chat/sessions", json={"title": f"Session {i}"}, headers=auth_headers
            )

        # Page 1 should have 20
        response = client.get(
            "/api/v1/history?page=1&page_size=20", headers=auth_headers
        )
        data = response.json()
        assert len(data["sessions"]) == 20
        assert data["total_pages"] == 1

        # Page 2 should be empty
        response = client.get(
            "/api/v1/history?page=2&page_size=20", headers=auth_headers
        )
        data = response.json()
        assert len(data["sessions"]) == 0

    async def test_history_invalid_page_size(self, client, auth_headers):
        """Test invalid page_size parameter."""
        # Too large
        response = client.get("/api/v1/history?page_size=101", headers=auth_headers)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

        # Too small
        response = client.get("/api/v1/history?page_size=0", headers=auth_headers)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    async def test_history_invalid_page(self, client, auth_headers):
        """Test invalid page parameter."""
        response = client.get("/api/v1/history?page=0", headers=auth_headers)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    async def test_history_invalid_sort_field(self, client, auth_headers):
        """Test invalid sort_by parameter."""
        response = client.get(
            "/api/v1/history?sort_by=invalid_field", headers=auth_headers
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    async def test_history_invalid_sort_order(self, client, auth_headers):
        """Test invalid sort_order parameter."""
        response = client.get("/api/v1/history?sort_order=invalid", headers=auth_headers)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    async def test_history_invalid_status(self, client, auth_headers):
        """Test invalid status parameter."""
        response = client.get("/api/v1/history?status=invalid", headers=auth_headers)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    async def test_search_history_empty_query(self, client, auth_headers):
        """Test search with empty query."""
        response = client.get("/api/v1/history/search?q=", headers=auth_headers)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    async def test_search_history_query_too_long(self, client, auth_headers):
        """Test search with query too long."""
        response = client.get(
            "/api/v1/history/search?q=" + "x" * 501, headers=auth_headers
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
