"""Database primitives."""

from app.db.session import (check_database_connection, get_db_session,
                            get_session_factory, transactional_session)

__all__ = [
    "check_database_connection",
    "get_db_session",
    "get_session_factory",
    "transactional_session",
    "VectorDocument",
    "VectorStore",
]

# Import vector store components if available
try:
    from app.db.vector_store import VectorDocument, VectorStore
except ImportError:
    pass
