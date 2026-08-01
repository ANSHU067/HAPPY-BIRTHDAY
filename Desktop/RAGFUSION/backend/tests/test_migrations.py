"""Verify the initial Alembic revision creates and removes every table."""

from __future__ import annotations

from sqlalchemy import create_engine, inspect

from alembic import command
from alembic.config import Config

EXPECTED_TABLES = {
    "alembic_version",
    "users",
    "documents",
    "websites",
    "youtube_sources",
    "chat_sessions",
    "messages",
    "settings",
    "embeddings",
}


def test_initial_migration_upgrade_and_downgrade(tmp_path) -> None:
    database_url = f"sqlite:///{tmp_path / 'migrations.db'}"
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", database_url)

    command.upgrade(config, "head")
    engine = create_engine(database_url)
    assert EXPECTED_TABLES <= set(inspect(engine).get_table_names())
    command.downgrade(config, "base")
    # Alembic retains its version bookkeeping table after a downgrade.
    assert set(inspect(engine).get_table_names()) == {"alembic_version"}
    engine.dispose()
