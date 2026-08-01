"""SQLAlchemy engine and session dependencies."""

from collections.abc import Generator
from functools import lru_cache

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from app.config.settings import get_settings


@lru_cache
def get_engine() -> Engine:
    """Create the process-wide SQLAlchemy engine on first use."""

    return create_engine(get_settings().database_url, pool_pre_ping=True)


@lru_cache
def get_session_factory() -> sessionmaker[Session]:
    """Create the process-wide session factory on first use."""

    return sessionmaker(bind=get_engine(), autocommit=False, autoflush=False)


def get_db_session() -> Generator[Session, None, None]:
    """Yield a transactional SQLAlchemy session for a request."""

    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()


def check_database_connection() -> bool:
    """Return whether PostgreSQL accepts a minimal query."""

    try:
        with get_engine().connect() as connection:
            connection.execute(text("SELECT 1"))
    except Exception:
        return False
    return True
