"""FastAPI startup and HTTP contract tests."""

from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def test_docs_are_available() -> None:
    response = client.get("/docs")

    assert response.status_code == 200


def test_health_reports_dependencies(monkeypatch) -> None:
    monkeypatch.setattr("app.services.health.check_database_connection", lambda: True)
    monkeypatch.setattr("app.services.health.check_redis_connection", lambda: True)

    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "database": True, "redis": True}
    assert response.headers["X-Request-ID"]
