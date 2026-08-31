from __future__ import annotations

from collections.abc import Sequence
from typing import final

from redis.asyncio.client import Redis

from polylogue.constants import DB_MODELS_NAMESPACE
from polylogue.db.model_record import ModelRecord
from polylogue.db.protocols.db_client import DBClient


@final
class ModelRecordManager:
    def __init__(
        self, client: DBClient[str], namespace: str = DB_MODELS_NAMESPACE
    ) -> None:
        self.namespace = namespace
        self.client: DBClient[str] = client

    async def get(self, model_id: str) -> ModelRecord:
        json_data = await self.client.hget(self.namespace, model_id)
        if not json_data:
            raise ValueError(f"No model record {model_id} was found")

        return ModelRecord.model_validate_json(json_data)

    async def list_all(self) -> Sequence[ModelRecord]:
        json_data_mapping = await self.client.hgetall(self.namespace)
        if not json_data_mapping:
            return []

        return [
            ModelRecord.model_validate_json(json_data_single)
            for json_data_single in json_data_mapping.values()
        ]

    async def save(self, record: ModelRecord) -> None:
        _ = await self.client.hset(self.namespace, record.id, record.model_dump_json())

    async def delete(self, model_id: str) -> bool:
        return bool(await self.client.hdel(self.namespace, model_id))
