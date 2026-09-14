from datetime import datetime

from pydantic import BaseModel, Field


class APIKeyRecord(BaseModel):
    key_id: str
    owner_id: str
    name: str = "default"
    secret_hash: str
    rate_limit: int = 100
    is_active: bool = True
    created_at: int = Field(default_factory=lambda: int(datetime.now().timestamp()))
    last_used_at: int | None = None


class APIKeyCreateResult(BaseModel):
    raw_key: str
    record: APIKeyRecord
