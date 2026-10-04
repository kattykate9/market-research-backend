import asyncio
import hashlib
import json
import os
import secrets
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Literal

import httpx
import psycopg
from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from starlette.concurrency import run_in_threadpool

app = FastAPI(
    title="Market Research Backend",
    version="0.2.0",
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


MARKET_ORIGIN = "https://data-api.binance.vision"

INTERVAL_MS = {
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}


def utc_iso(milliseconds):
    return datetime.fromtimestamp(
        milliseconds / 1000,
        timezone.utc,
    ).isoformat()


async def market_json(client, path, params=None):
    response = await client.get(path, params=params)

    if response.status_code in (418, 429):
        raise HTTPException(503, "MARKET_SOURCE_RATE_LIMITED")

    if response.status_code in (403, 451):
        raise HTTPException(503, "MARKET_SOURCE_ACCESS_DENIED")

    if response.status_code != 200:
        raise HTTPException(502, "MARKET_SOURCE_UNAVAILABLE")

    try:
        return response.json()
    except ValueError:
        raise HTTPException(
            502,
            "MARKET_SOURCE_INVALID_RESPONSE",
        ) from None


def validate_candles(rows, step, cutoff):
    if not isinstance(rows, list) or not 1 <= len(rows) <= 500:
        raise ValueError("Invalid rows")

    candles = []
    gaps = 0
    previous = None

    for row in rows:
        if not isinstance(row, list) or len(row) != 12:
            raise ValueError("Invalid candle")

        opened, closed = row[0], row[6]

        if type(opened) is not int or type(closed) is not int:
            raise ValueError("Invalid timestamp")

        if (
            opened < 0
            or opened % step
            or closed != opened + step - 1
        ):
            raise ValueError("Invalid candle boundaries")

        if closed > cutoff:
            raise ValueError("Unclosed candle")

        if previous is not None:
            if opened <= previous:
                raise ValueError("Unsorted or duplicate candles")

            gaps += (opened - previous) // step - 1

        previous = opened
        values = []

        for index in (1, 2, 3, 4, 5, 7):
            if (
                not isinstance(row[index], str)
                or len(row[index]) > 64
            ):
                raise ValueError("Invalid number")

            value = Decimal(row[index])

            if not value.is_finite() or value < 0:
                raise ValueError("Invalid number")

            values.append(value)

        op, hi, lo, cl, volume, quote_volume = values

        if (
            min(op, hi, lo, cl) <= 0
            or not lo <= min(op, cl) <= max(op, cl) <= hi
        ):
            raise ValueError("Invalid OHLC")

        candles.append({
            "open_time": utc_iso(opened),
            "close_time": utc_iso(closed),
            "open": str(op),
            "high": str(hi),
            "low": str(lo),
            "close": str(cl),
            "volume": str(volume),
            "quote_volume": str(quote_volume),
        })

    return candles, gaps


@app.get(
    "/v1/market-snapshot",
    dependencies=[Depends(require_service_key)],
)
async def market_snapshot(
    asset: Literal["BTC", "ETH"] = "BTC",
    timeframe: Literal["1h", "4h", "1d"] = "1h",
):
    symbol = asset + "USDT"
    step = INTERVAL_MS[timeframe]

    try:
        async with asyncio.timeout(12):
            async with httpx.AsyncClient(
                base_url=MARKET_ORIGIN,
                timeout=5,
                follow_redirects=False,
            ) as client:
                clock = await market_json(client, "/api/v3/time")

                if (
                    not isinstance(clock, dict)
                    or type(clock.get("serverTime")) is not int
                ):
                    raise ValueError("Invalid source time")

                source_ms = clock["serverTime"]
                local_ms = int(
                    datetime.now(timezone.utc).timestamp() * 1000
                )

                if (
                    source_ms <= 0
                    or abs(source_ms - local_ms) > 60_000
                ):
                    raise HTTPException(
                        503,
                        "MARKET_SOURCE_CLOCK_MISMATCH",
                    )

                cutoff = source_ms // step * step - 1

                info = await market_json(
                    client,
                    "/api/v3/exchangeInfo",
                    {"symbol": symbol},
                )

                if (
                    not isinstance(info, dict)
                    or not isinstance(info.get("symbols"), list)
                ):
                    raise ValueError("Invalid pair response")

                pairs = info["symbols"]

                if len(pairs) != 1 or not isinstance(pairs[0], dict):
                    raise ValueError("Invalid pair response")

                pair = pairs[0]

                if (
                    pair.get("symbol") != symbol
                    or pair.get("baseAsset") != asset
                    or pair.get("quoteAsset") != "USDT"
                ):
                    raise ValueError("Wrong pair")

                if (
                    pair.get("status") != "TRADING"
                    or pair.get("isSpotTradingAllowed") is not True
                ):
                    raise HTTPException(
                        503,
                        "MARKET_PAIR_UNAVAILABLE",
                    )

                rows = await market_json(
                    client,
                    "/api/v3/klines",
                    {
                        "symbol": symbol,
                        "interval": timeframe,
                        "endTime": cutoff,
                        "limit": 500,
                        "timeZone": "0",
                    },
                )

                candles, gaps = validate_candles(
                    rows,
                    step,
                    cutoff,
                )

    except (httpx.TimeoutException, TimeoutError):
        raise HTTPException(504, "MARKET_SOURCE_TIMEOUT") from None

    except httpx.RequestError:
        raise HTTPException(
            502,
            "MARKET_SOURCE_UNAVAILABLE",
        ) from None

    except (ValueError, InvalidOperation, OverflowError, OSError):
        raise HTTPException(
            502,
            "MARKET_SOURCE_INVALID_RESPONSE",
        ) from None

    issues = []

    if gaps:
        issues.append("MISSING_CANDLES")

    if len(candles) < 500:
        issues.append("SHORT_HISTORY")

    if rows[-1][6] != cutoff:
        issues.append("STALE_CANDLES")

    snapshot_content = {
        "schema_version": "market-snapshot.v1",
        "symbol": symbol,
        "market": "spot",
        "timeframe": timeframe,
        "candles": candles,
    }

    digest = hashlib.sha256(
        json.dumps(
            snapshot_content,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()

    return {
        **snapshot_content,
        "asset": asset,
        "quote_currency": "USDT",
        "data_mode": "real",
        "source": "binance_public_spot",
        "source_url": MARKET_ORIGIN + "/api/v3/klines",
        "source_time": utc_iso(source_ms),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "as_of": candles[-1]["close_time"],
        "expected_last_close": utc_iso(cutoff),
        "data_snapshot_id": digest,
        "persisted": False,
        "candle_count": len(candles),
        "closed_candles_only": True,
        "quality": {
            "status": "ok" if not issues else "insufficient_data",
            "issues": issues,
            "missing_candle_count": gaps,
        },
        "analysis_enabled": False,
        "openai_enabled": False,
    }
