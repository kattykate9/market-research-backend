import os
import secrets

import psycopg
from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.concurrency import run_in_threadpool

app = FastAPI(
    title="Market Research Backend",
    version="0.1.0",
)

bearer = HTTPBearer(auto_error=False)


def require_service_key(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
):
    expected = os.getenv("SERVICE_API_KEY", "")

    if not expected:
        raise HTTPException(
            status_code=503,
            detail="SERVICE_AUTH_NOT_CONFIGURED",
        )

    if (
        credentials is None
        or credentials.scheme.lower() != "bearer"
        or not secrets.compare_digest(credentials.credentials, expected)
    ):
        raise HTTPException(
            status_code=401,
            detail="UNAUTHORIZED",
            headers={"WWW-Authenticate": "Bearer"},
        )


def check_database():
    names = ("PGHOST", "PGPORT", "PGDATABASE", "PGUSER", "PGPASSWORD")

    if not all(os.getenv(name) for name in names):
        return "not_configured"

    try:
        with psycopg.connect(
            host=os.environ["PGHOST"],
            port=int(os.environ["PGPORT"]),
            dbname=os.environ["PGDATABASE"],
            user=os.environ["PGUSER"],
            password=os.environ["PGPASSWORD"],
            sslmode="require",
            connect_timeout=5,
            options="-c statement_timeout=5000",
            autocommit=True,
        ) as connection:
            row = connection.execute("SELECT 1").fetchone()

        return "connected" if row == (1,) else "unavailable"

    except (psycopg.Error, ValueError):
        return "unavailable"


@app.get("/health")
def health():
    # Проверка процесса, без обращения к базе.
    return {"status": "ok", "service": "market-research-backend"}


@app.get("/v1/health", dependencies=[Depends(require_service_key)])
async def integration_health():
    database = await run_in_threadpool(check_database)

    if database != "connected":
        raise HTTPException(
            status_code=503,
            detail={"status": "not_ready", "database": database},
        )

    return {
        "status": "ok",
        "database": "connected",
        "analysis_enabled": False,
        "openai_enabled": False,
    }
