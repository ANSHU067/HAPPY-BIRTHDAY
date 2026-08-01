"""Authentication endpoint tests."""

import pytest
from fastapi.testclient import TestClient
from jose import jwt
from sqlalchemy import select

from app.config.settings import get_settings
from app.db.session import get_async_engine, get_db_session
from app.models.base import Base
from app.models.entities import User
from main import create_app


@pytest.fixture
def settings():
    """Get test settings with SQLite database."""
    test_settings = get_settings()
    test_settings.database_url = "sqlite+aiosqlite:///:memory:"
    return test_settings


@pytest.fixture
async def test_engine(settings):
    """Create test engine with SQLite in-memory database."""
    engine = get_async_engine(settings.database_url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture
async def test_session_factory(test_engine):
    """Create test session factory."""
    from sqlalchemy.ext.asyncio import async_sessionmaker

    return async_sessionmaker(test_engine, expire_on_commit=False, autoflush=False)


@pytest.fixture
async def db_session(test_session_factory):
    """Create a test database session."""
    async with test_session_factory() as session:
        yield session
        await session.rollback()


@pytest.fixture
def test_app(settings, test_session_factory):
    """Create test FastAPI app with SQLite database."""
    app = create_app()

    async def override_get_db_session():
        async with test_session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    app.dependency_overrides[get_db_session] = override_get_db_session
    return app


@pytest.fixture
def client(test_app):
    """Create test client."""
    return TestClient(test_app)


@pytest.fixture
def valid_signup_data():
    """Valid signup request data."""
    return {
        "email": "test@example.com",
        "password": "SecurePass123",
        "display_name": "Test User",
    }


@pytest.fixture
def valid_login_data():
    """Valid login request data."""
    return {
        "email": "test@example.com",
        "password": "SecurePass123",
    }


@pytest.fixture
async def created_user(client, valid_signup_data, valid_login_data):
    """Create a user and return login data."""
    client.post("/api/v1/auth/signup", json=valid_signup_data)
    return valid_login_data


class TestSignup:
    """Tests for POST /auth/signup."""

    def test_signup_success(self, client, valid_signup_data, settings):
        """Test successful user signup."""
        response = client.post("/api/v1/auth/signup", json=valid_signup_data)

        assert response.status_code == 201
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
        assert data["expires_in"] == settings.access_token_expire_minutes * 60

    def test_signup_duplicate_email(self, client, valid_signup_data):
        """Test signup with duplicate email fails."""
        client.post("/api/v1/auth/signup", json=valid_signup_data)
        response = client.post("/api/v1/auth/signup", json=valid_signup_data)

        assert response.status_code == 409
        assert "already registered" in response.json()["error"]["message"].lower()

    def test_signup_invalid_email(self, client):
        """Test signup with invalid email format."""
        data = {
            "email": "not-an-email",
            "password": "SecurePass123",
            "display_name": "Test User",
        }
        response = client.post("/api/v1/auth/signup", json=data)

        assert response.status_code == 422

    def test_signup_short_password(self, client):
        """Test signup with password too short."""
        data = {
            "email": "test@example.com",
            "password": "short",
            "display_name": "Test User",
        }
        response = client.post("/api/v1/auth/signup", json=data)

        assert response.status_code == 422

    def test_signup_missing_fields(self, client):
        """Test signup with missing required fields."""
        data = {"email": "test@example.com"}
        response = client.post("/api/v1/auth/signup", json=data)

        assert response.status_code == 422


class TestLogin:
    """Tests for POST /auth/login."""

    def test_login_success(self, client, valid_signup_data, valid_login_data):
        """Test successful user login."""
        client.post("/api/v1/auth/signup", json=valid_signup_data)
        response = client.post("/api/v1/auth/login", json=valid_login_data)

        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"

    def test_login_invalid_password(self, client, valid_signup_data):
        """Test login with incorrect password."""
        client.post("/api/v1/auth/signup", json=valid_signup_data)
        response = client.post(
            "/api/v1/auth/login",
            json={"email": "test@example.com", "password": "WrongPass123"},
        )

        assert response.status_code == 401
        assert "invalid credentials" in response.json()["error"]["message"].lower()

    def test_login_nonexistent_user(self, client):
        """Test login with non-existent user."""
        response = client.post(
            "/api/v1/auth/login",
            json={"email": "nonexistent@example.com", "password": "SecurePass123"},
        )

        assert response.status_code == 401
        assert "invalid credentials" in response.json()["error"]["message"].lower()

    @pytest.mark.asyncio
    async def test_login_inactive_user(
        self, client, valid_signup_data, valid_login_data, db_session
    ):
        """Test login with deactivated user."""
        client.post("/api/v1/auth/signup", json=valid_signup_data)

        result = await db_session.execute(
            select(User).where(User.email == "test@example.com")
        )
        user = result.scalar_one()
        user.is_active = False
        await db_session.commit()

        response = client.post("/api/v1/auth/login", json=valid_login_data)
        assert response.status_code == 401
        assert "deactivated" in response.json()["error"]["message"].lower()


class TestRefresh:
    """Tests for POST /auth/refresh."""

    def test_refresh_success(self, client, created_user):
        """Test successful token refresh."""
        login_response = client.post("/api/v1/auth/login", json=created_user)
        refresh_token = login_response.json()["refresh_token"]

        response = client.post(
            "/api/v1/auth/refresh", json={"refresh_token": refresh_token}
        )

        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"

    def test_refresh_expired_token(self, client, settings):
        """Test refresh with expired refresh token."""
        expired_token = jwt.encode(
            {
                "sub": "00000000-0000-0000-0000-000000000000",
                "type": "refresh",
                "exp": 0,
            },
            settings.jwt_secret_key,
            algorithm=settings.jwt_algorithm,
        )

        response = client.post(
            "/api/v1/auth/refresh", json={"refresh_token": expired_token}
        )

        assert response.status_code == 401

    def test_refresh_invalid_token(self, client):
        """Test refresh with invalid token."""
        response = client.post(
            "/api/v1/auth/refresh", json={"refresh_token": "invalid.token.here"}
        )

        assert response.status_code == 401

    def test_refresh_access_token_instead_of_refresh(self, client, created_user):
        """Test refresh using access token instead of refresh token."""
        login_response = client.post("/api/v1/auth/login", json=created_user)
        access_token = login_response.json()["access_token"]

        response = client.post(
            "/api/v1/auth/refresh", json={"refresh_token": access_token}
        )

        assert response.status_code == 401
        assert "invalid token type" in response.json()["error"]["message"].lower()

    def test_refresh_nonexistent_user(self, client, settings):
        """Test refresh for non-existent user."""
        token = jwt.encode(
            {"sub": "00000000-0000-0000-0000-000000000000", "type": "refresh"},
            settings.jwt_secret_key,
            algorithm=settings.jwt_algorithm,
        )

        response = client.post("/api/v1/auth/refresh", json={"refresh_token": token})

        assert response.status_code == 401


class TestLogout:
    """Tests for POST /auth/logout."""

    def test_logout_success(self, client, created_user):
        """Test successful logout."""
        login_response = client.post("/api/v1/auth/login", json=created_user)
        access_token = login_response.json()["access_token"]

        response = client.post(
            "/api/v1/auth/logout",
            headers={"Authorization": f"Bearer {access_token}"},
        )

        assert response.status_code == 200
        assert response.json()["message"] == "Successfully logged out"

    def test_logout_invalid_token(self, client):
        """Test logout with invalid token."""
        response = client.post(
            "/api/v1/auth/logout",
            headers={"Authorization": "Bearer invalid.token.here"},
        )

        assert response.status_code == 401

    def test_logout_missing_token(self, client):
        """Test logout without authorization header."""
        response = client.post("/api/v1/auth/logout")

        assert response.status_code == 401


class TestGetMe:
    """Tests for GET /auth/me."""

    def test_get_me_success(self, client, created_user):
        """Test getting current user info."""
        login_response = client.post("/api/v1/auth/login", json=created_user)
        access_token = login_response.json()["access_token"]

        response = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {access_token}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["email"] == "test@example.com"
        assert data["display_name"] == "Test User"
        assert data["role"] == "user"
        assert data["is_active"] is True
        assert "id" in data
        assert "created_at" in data
        assert "updated_at" in data

    def test_get_me_invalid_token(self, client):
        """Test getting current user with invalid token."""
        response = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": "Bearer invalid.token.here"},
        )

        assert response.status_code == 401


class TestTokenPayload:
    """Tests for JWT token payload contents."""

    def test_access_token_contains_user_id_and_role(
        self, client, created_user, settings
    ):
        """Test access token contains user ID and role."""
        login_response = client.post("/api/v1/auth/login", json=created_user)
        access_token = login_response.json()["access_token"]

        payload = jwt.decode(
            access_token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm]
        )

        assert payload["type"] == "access"
        assert "sub" in payload
        assert payload["role"] == "user"
        assert "exp" in payload
        assert "iat" in payload

    def test_refresh_token_contains_user_id(self, client, created_user, settings):
        """Test refresh token contains user ID."""
        login_response = client.post("/api/v1/auth/login", json=created_user)
        refresh_token = login_response.json()["refresh_token"]

        payload = jwt.decode(
            refresh_token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm]
        )

        assert payload["type"] == "refresh"
        assert "sub" in payload
        assert "role" not in payload
        assert "exp" in payload
        assert "iat" in payload
