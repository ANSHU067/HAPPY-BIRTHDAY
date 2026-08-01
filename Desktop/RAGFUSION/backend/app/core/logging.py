"""Logging configuration for the API process."""

import logging

from app.config.settings import get_settings


def configure_logging() -> None:
    """Configure a consistent, idempotent process logger."""

    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        force=True,
    )
