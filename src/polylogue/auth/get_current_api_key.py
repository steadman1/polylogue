from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from polylogue.clients import RedisClientDep  # Your Annotated Redis/DB dependency
from polylogue.db.api_key_manager import APIKeyManager, APIKeyRecord

bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_api_key(
    auth: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
    db_client: RedisClientDep,
) -> APIKeyRecord:
    if auth is None or not auth.credentials:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing Bearer API key in Authorization header.",
            headers={"Authorization": "Bearer"},
        )

    manager = APIKeyManager(db_client)
    record = await manager.verify_raw_key(auth.credentials)

    if record is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or revoked API key.",
            headers={"Authorization": "Bearer"},
        )

    return record


# Type alias for cleaner endpoint signatures
AuthenticatedKey = Annotated[APIKeyRecord, Depends(get_current_api_key)]
