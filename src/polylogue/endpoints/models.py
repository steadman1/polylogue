from __future__ import annotations

from openai.types import Model

from polylogue.constants import Endpoint
from polylogue.fastapi_app import app
from polylogue.helpers.build_prefix import build_prefix


# need redis hooked up to query model
@app.get(build_prefix(Endpoint.MODELS))
async def list_models() -> list[Model]:
    return []


@app.get(build_prefix(Endpoint.MODELS) + "/{model}")
async def get_model(model: str) -> Model:
    return Model(
        id=model,
        created=0,
        object="model",
        owned_by="unimplemented",
    )
