# DOCPRO V2 Backend

## Local development

Use Python 3.11 and install the dependencies:

```bash
pip install -r requirements/base.txt
cp .env.example .env
alembic upgrade head
python -m uvicorn main:app --reload
```

Open Swagger UI at `http://localhost:8000/docs`. The health endpoint is
available at `http://localhost:8000/api/v1/health` and reports `degraded`
until PostgreSQL and Redis are reachable.

## Docker

```bash
docker compose up --build
```

The Compose stack starts the API, PostgreSQL, and Redis. It uses `.env` for
the API configuration and waits for the database and cache health checks.

## Verification

```bash
python -m compileall .
pytest
```

## Database architecture

The application uses async SQLAlchemy sessions and repositories. PostgreSQL is
the production database; SQLite with `aiosqlite` is supported for integration
tests. Apply schema changes with `alembic upgrade head`; generate a reviewed
revision with `alembic revision --autogenerate -m "description"`.
