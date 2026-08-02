"""Rename the chat-session metadata column.

Revision ID: 20260802_0002
Revises: 20260801_0001
Create Date: 2026-08-02
"""

from alembic import op


revision = "20260802_0002"
down_revision = "20260801_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column("chat_sessions", "metadata", new_column_name="session_metadata")


def downgrade() -> None:
    op.alter_column("chat_sessions", "session_metadata", new_column_name="metadata")
