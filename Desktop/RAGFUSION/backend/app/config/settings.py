"""Typed configuration loaded from the environment."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for the API and its backing services."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="DOCPRO_",
        extra="ignore",
    )

    app_name: str = "DOCPRO V2 API"
    environment: str = "development"
    debug: bool = False
    api_v1_prefix: str = "/api/v1"
    database_url: str = "postgresql+psycopg://docpro:docpro@postgres:5432/docpro"
    redis_url: str = "redis://redis:6379/0"
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    """Return the cached application configuration."""

    return Settings()
