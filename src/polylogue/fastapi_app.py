from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Annotated

import redis.asyncio as aioredis
from fastapi import Depends, FastAPI, Request


# written by Gemini
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    # Initialize connection pool on startup
    pool = aioredis.ConnectionPool.from_url(
        "redis://localhost:6379/0",
        max_connections=20,
        decode_responses=True,
    )
    client = aioredis.Redis(connection_pool=pool)

    # Attach client to application state
    app.state.db_client = client

    yield

    # Clean teardown on shutdown
    await client.aclose()
    await pool.disconnect()


# init global app var for other files to import
app = FastAPI(lifespan=lifespan)


def get_db(request: Request) -> aioredis.Redis:
    """Provides the shared database client instance to endpoints."""
    return request.app.state.db_client


# Type alias for cleaner endpoint signatures
DBClient = Annotated[aioredis.Redis, Depends(get_db)]
