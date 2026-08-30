from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from polylogue.db.model_record import ModelRecord


class RecordManager(Protocol):
    async def get(self, model_id: str) -> ModelRecord | None: ...

    async def list_all(self) -> Sequence[ModelRecord]: ...

    async def save(self, record: ModelRecord) -> None: ...

    async def delete(self, model_id: str) -> bool: ...
