from __future__ import annotations

from openai.types import Model

from polylogue.clients import DBClient, app
from polylogue.constants import Endpoint
from polylogue.db.model_record_manager import ModelRecordManager
from polylogue.helpers.build_prefix import build_prefix

# return Model instead of ModelRecord to hide the path


@app.get(build_prefix(Endpoint.MODELS))
async def list_models_with_version(db_client: DBClient) -> list[Model]:
    db = ModelRecordManager(db_client)
    return list(await db.list_all())

# expose models without the api version number for wider api endpoint support
@app.get(build_prefix(Endpoint.MODELS, without_version=True))
async def list_models_without_version(db_client: DBClient) -> dict[str, list[Model]]:
    db = ModelRecordManager(db_client)
    return {
        "data": list(await db.list_all())
    }


@app.get(build_prefix(Endpoint.MODELS) + "/{model}")
@app.get(build_prefix(Endpoint.MODELS, without_version=True) + "/{model}")
async def get_model(model: str, db_client: DBClient) -> Model:
    db = ModelRecordManager(db_client)
    return await db.get(model)
