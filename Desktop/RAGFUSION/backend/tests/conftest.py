"""Pytest configuration for backend tests."""

import sys
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import (AsyncSession, async_sessionmaker,
                                    create_async_engine)


# ---------------------------------------------------------------------
# Add the backend directory to the Python path
# ---------------------------------------------------------------------

BACKEND_DIR = Path(__file__).resolve().parents[1]

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# ---------------------------------------------------------------------
# Import application objects
# ---------------------------------------------------------------------

# IMPORTANT
import app.services.ingestion_service as ingestion_service
from app.db import session as db_module
from app.db.session import get_db_session
from app.models.base import Base
from app.models.entities import User, UserRole
from main import app

# ---------------------------------------------------------------------
# Database fixture
# ---------------------------------------------------------------------


@pytest.fixture
async def test_db(tmp_path):
    """Create a temporary SQLite database."""

    database_url = f"sqlite+aiosqlite:///{tmp_path / 'test.db'}"

    engine = create_async_engine(
        database_url,
        future=True,
        echo=False,
    )

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    # -----------------------------------------------------------------
    # Remove cached PostgreSQL connections
    # -----------------------------------------------------------------
# -----------------------------------------------------------------
        # Remove cached PostgreSQL connections
        # -----------------------------------------------------------------

    try:
        db_module.get_async_engine.cache_clear()
        db_module.get_session_factory.cache_clear()
    except Exception:
        pass

    original_get_session_factory = db_module.get_session_factory
    original_ingestion_session_factory = ingestion_service.get_session_factory

    patched_factory = lambda database_url=None: session_factory

    # Patch both locations
    db_session.get_session_factory = patched_factory
    ingestion_service.get_session_factory = patched_factory

    async def override_get_db():
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_db_session] = override_get_db

    yield session_factory

    # -----------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------

    app.dependency_overrides.clear()

    db_session.get_session_factory = original_get_session_factory
    ingestion_service.get_session_factory = original_ingestion_session_factory

    await engine.dispose()


# ---------------------------------------------------------------------
# Test client fixture
# ---------------------------------------------------------------------


@pytest.fixture
def client(test_db):
    """Create a FastAPI test client."""

    with TestClient(app) as test_client:
        yield test_client


# ---------------------------------------------------------------------
# Database session fixture
# ---------------------------------------------------------------------


@pytest.fixture
async def db_session(test_db):
    """Create a database session for testing."""

    async with test_db() as session:
        yield session


# ---------------------------------------------------------------------
# Test user fixture
# ---------------------------------------------------------------------


@pytest.fixture
async def test_user(db_session):
    """Create a test user in the database."""
    from app.services.auth import hash_password

    user = User(
        id=uuid4(),
        email="test@example.com",
        display_name="Test User",
        password_hash=hash_password("testpassword"),
        role=UserRole.user,
        is_active=True,
    )

    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)

    return user


# ---------------------------------------------------------------------
# Auth headers fixture
# ---------------------------------------------------------------------


@pytest.fixture
async def auth_headers(test_user):
    """Create authentication headers for testing."""
    from app.services.auth import create_access_token
        
    # Add the 'role' argument here!
    token = create_access_token(test_user.id, role=test_user.role)
    return {"Authorization": f"Bearer {token}"}

# ---------------------------------------------------------------------
# Mock user fixture (for unit tests without DB)
# ---------------------------------------------------------------------


@pytest.fixture
def mock_user():
    """Create a mock authenticated user."""

    return User(
        id=uuid4(),
        email="test@example.com",
        display_name="Test User",
        role=UserRole.user,
        is_active=True,
    )
