"""Top-level HTTP routes."""

from fastapi import APIRouter

from app.schemas.health import HealthResponse
from app.services.health import get_health_status


router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Report API dependency readiness."""

    return HealthResponse.model_validate(get_health_status())
