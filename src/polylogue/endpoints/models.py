from __future__ import annotations

from polylogue.clients import DBClient, app
from polylogue.constants import Endpoint
from polylogue.db.model_record import ModelRecord
from polylogue.db.model_record_manager import ModelRecordManager
from polylogue.helpers.build_prefix import build_prefix


# need redis hooked up to query model
@app.get(build_prefix(Endpoint.MODELS))
async def list_models(db_client: DBClient) -> list[ModelRecord]:
    db = ModelRecordManager(db_client)
    return list(await db.list_all())


@app.get(build_prefix(Endpoint.MODELS) + "/{model}")
async def get_model(model: str, db_client: DBClient) -> ModelRecord:
    db = ModelRecordManager(db_client)
    return await db.get(model)
