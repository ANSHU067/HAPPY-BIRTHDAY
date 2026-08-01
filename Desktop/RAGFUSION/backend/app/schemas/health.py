"""Health endpoint response schemas."""

from typing import Literal

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Availability of the API and its required infrastructure."""

    status: Literal["ok", "degraded"]
    database: bool
    redis: bool
