"""Async SQLAlchemy engine, sessions, and transaction helpers."""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from functools import lru_cache

from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import (AsyncEngine, AsyncSession,
                                    async_sessionmaker, create_async_engine)

from app.config.settings import get_settings


def to_async_database_url(database_url: str) -> str:
    """Convert common synchronous SQLAlchemy URLs to their async driver form."""

    url = make_url(database_url)
    driver = url.drivername
    replacements = {
        "postgresql": "postgresql+asyncpg",
        "postgresql+psycopg": "postgresql+asyncpg",
        "sqlite": "sqlite+aiosqlite",
        "mysql": "mysql+aiomysql",
    }
    return str(url.set(drivername=replacements.get(driver, driver)))


@lru_cache
def get_async_engine(database_url: str | None = None) -> AsyncEngine:
    """Create the process-wide async engine, or an engine for an explicit URL."""

    url = to_async_database_url(database_url or get_settings().database_url)
    connect_args = {"check_same_thread": False} if url.startswith("sqlite+") else {}
    return create_async_engine(url, pool_pre_ping=True, connect_args=connect_args)


@lru_cache
def get_session_factory(
    database_url: str | None = None,
) -> async_sessionmaker[AsyncSession]:
    """Return a non-expiring async session factory."""

    return async_sessionmaker(
        get_async_engine(database_url), expire_on_commit=False, autoflush=False
    )


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that commits successful requests and rolls back failures."""

    async with get_session_factory()() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


@asynccontextmanager
async def transactional_session(
    database_url: str | None = None,
) -> AsyncIterator[AsyncSession]:
    """Yield a session with an atomic commit/rollback boundary for jobs and scripts."""

    async with get_session_factory(database_url)() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def check_database_connection(database_url: str | None = None) -> bool:
    """Return whether the configured database accepts a minimal async query."""

    try:
        async with get_async_engine(database_url).connect() as connection:
            await connection.execute(text("SELECT 1"))
    except Exception:
        return False
    return True


async def dispose_engines() -> None:
    """Dispose cached engines; useful during application shutdown and tests."""

    # Explicitly supplied engines are owned by their caller. Dispose the application engine.
    default_engine = get_async_engine()
    await default_engine.dispose()
    get_session_factory.cache_clear()
    get_async_engine.cache_clear()
