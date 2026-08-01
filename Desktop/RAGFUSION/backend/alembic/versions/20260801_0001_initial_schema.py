"""Create the RAGFUSION database schema.

Revision ID: 20260801_0001
Revises:
Create Date: 2026-08-01
"""

import sqlalchemy as sa

from alembic import op

revision = "20260801_0001"
down_revision = None
branch_labels = None
depends_on = None


source_status = sa.Enum(
    "pending", "processing", "ready", "failed", name="sourcestatus", native_enum=False
)
user_role = sa.Enum("user", "admin", name="userrole", native_enum=False)
message_role = sa.Enum(
    "system", "user", "assistant", name="messagerole", native_enum=False
)


def id_column() -> sa.Column:
    return sa.Column("id", sa.Uuid(), primary_key=True, nullable=False)


def timestamps() -> list[sa.Column]:
    return [
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("CURRENT_TIMESTAMP"),
            nullable=False,
        ),
    ]


def upgrade() -> None:
    op.create_table(
        "users",
        id_column(),
        sa.Column("email", sa.String(320), nullable=False),
        sa.Column("display_name", sa.String(120)),
        sa.Column("password_hash", sa.String(255)),
        sa.Column("role", user_role, server_default="user", nullable=False),
        sa.Column("is_active", sa.Boolean(), server_default=sa.true(), nullable=False),
        *timestamps(),
        sa.UniqueConstraint("email"),
    )
    op.create_index("ix_users_email", "users", ["email"])
    op.create_table(
        "documents",
        id_column(),
        sa.Column(
            "user_id",
            sa.Uuid(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("filename", sa.String(512), nullable=False),
        sa.Column("storage_key", sa.String(1024), nullable=False),
        sa.Column("mime_type", sa.String(255)),
        sa.Column("size_bytes", sa.Integer()),
        sa.Column("checksum", sa.String(128)),
        sa.Column("status", source_status, nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
        *timestamps(),
        sa.UniqueConstraint("storage_key"),
    )
    op.create_index("ix_documents_user_id", "documents", ["user_id"])
    op.create_index("ix_documents_checksum", "documents", ["checksum"])
    op.create_table(
        "websites",
        id_column(),
        sa.Column(
            "user_id",
            sa.Uuid(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("url", sa.String(2048), nullable=False),
        sa.Column("title", sa.String(512)),
        sa.Column("status", source_status, nullable=False),
        sa.Column("last_crawled_at", sa.DateTime(timezone=True)),
        sa.Column("metadata", sa.JSON(), nullable=False),
        *timestamps(),
        sa.UniqueConstraint("user_id", "url", name="uq_websites_user_url"),
    )
    op.create_index("ix_websites_user_id", "websites", ["user_id"])
    op.create_table(
        "youtube_sources",
        id_column(),
        sa.Column(
            "user_id",
            sa.Uuid(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("video_id", sa.String(64), nullable=False),
        sa.Column("url", sa.String(2048), nullable=False),
        sa.Column("title", sa.String(512)),
        sa.Column("channel_name", sa.String(255)),
        sa.Column("duration_seconds", sa.Integer()),
        sa.Column("status", source_status, nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
        *timestamps(),
        sa.UniqueConstraint(
            "user_id", "video_id", name="uq_youtube_sources_user_video"
        ),
    )
    op.create_index("ix_youtube_sources_user_id", "youtube_sources", ["user_id"])
    op.create_table(
        "chat_sessions",
        id_column(),
        sa.Column(
            "user_id",
            sa.Uuid(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("title", sa.String(255), nullable=False),
        sa.Column("metadata", sa.JSON(), nullable=False),
        *timestamps(),
    )
    op.create_index("ix_chat_sessions_user_id", "chat_sessions", ["user_id"])
    op.create_table(
        "messages",
        id_column(),
        sa.Column(
            "chat_session_id",
            sa.Uuid(),
            sa.ForeignKey("chat_sessions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("role", message_role, nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("citations", sa.JSON(), nullable=False),
        sa.Column("token_count", sa.Integer()),
        *timestamps(),
    )
    op.create_index("ix_messages_chat_session_id", "messages", ["chat_session_id"])
    op.create_table(
        "settings",
        id_column(),
        sa.Column(
            "user_id",
            sa.Uuid(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("key", sa.String(128), nullable=False),
        sa.Column("value", sa.JSON(), nullable=False),
        *timestamps(),
        sa.UniqueConstraint("user_id", "key", name="uq_settings_user_key"),
    )
    op.create_index("ix_settings_user_id", "settings", ["user_id"])
    op.create_table(
        "embeddings",
        id_column(),
        sa.Column(
            "document_id", sa.Uuid(), sa.ForeignKey("documents.id", ondelete="CASCADE")
        ),
        sa.Column(
            "website_id", sa.Uuid(), sa.ForeignKey("websites.id", ondelete="CASCADE")
        ),
        sa.Column(
            "youtube_source_id",
            sa.Uuid(),
            sa.ForeignKey("youtube_sources.id", ondelete="CASCADE"),
        ),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("vector", sa.JSON(), nullable=False),
        sa.Column("model_name", sa.String(255), nullable=False),
        sa.Column("token_count", sa.Integer()),
        sa.Column("score", sa.Float()),
        sa.Column("metadata", sa.JSON(), nullable=False),
        *timestamps(),
    )
    op.create_index(
        "ix_embeddings_document_chunk", "embeddings", ["document_id", "chunk_index"]
    )
    op.create_index(
        "ix_embeddings_website_chunk", "embeddings", ["website_id", "chunk_index"]
    )
    op.create_index(
        "ix_embeddings_youtube_chunk",
        "embeddings",
        ["youtube_source_id", "chunk_index"],
    )


def downgrade() -> None:
    for table in (
        "embeddings",
        "settings",
        "messages",
        "chat_sessions",
        "youtube_sources",
        "websites",
        "documents",
        "users",
    ):
        op.drop_table(table)
