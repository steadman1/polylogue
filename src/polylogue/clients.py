from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Request
from redis.asyncio import ConnectionPool, Redis

# exposes a fastapi app and a function for creating a redis client


def get_redis_client() -> tuple[ConnectionPool, Redis]:
    pool = ConnectionPool.from_url(
        "redis://localhost:6379/0",
        max_connections=20,
        decode_responses=True,
    )
    client = Redis(connection_pool=pool)
    return (pool, client)


# written by Gemini
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    # Initialize connection pool on startup
    pool, client = get_redis_client()

    # Attach client to application state
    app.state.db_client = client

    yield

    # Clean teardown on shutdown
    await client.aclose()
    await pool.disconnect()


# init global app var for other files to import
app = FastAPI(lifespan=lifespan)


def get_db(request: Request) -> Redis:
    """Provides the shared database client instance to endpoints."""
    return request.app.state.db_client


# Type alias for cleaner endpoint signatures
DBClient = Annotated[Redis, Depends(get_db)]
