from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel


class ModelSchema(BaseModel):
    model_path: Path
    description: str | None = None


class ModelRepository(Protocol):
    async def get(self, model_id: str) -> ModelSchema | None: ...

    async def list_all(self) -> Sequence[ModelSchema]: ...

    async def save(self, record: ModelSchema) -> None: ...

    async def delete(self, model_id: str) -> bool: ...
