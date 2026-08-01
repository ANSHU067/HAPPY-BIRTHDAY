"""Infrastructure health checks."""

from app.config.settings import get_settings
from app.db.session import check_database_connection


def check_redis_connection() -> bool:
    """Return whether Redis responds to a ping without breaking API startup."""

    try:
        from redis import Redis

        return bool(Redis.from_url(get_settings().redis_url).ping())
    except Exception:
        return False


async def get_health_status() -> dict[str, bool | str]:
    """Collect the API dependency statuses."""

    database = await check_database_connection()
    redis = check_redis_connection()
    return {
        "status": "ok" if database and redis else "degraded",
        "database": database,
        "redis": redis,
    }
