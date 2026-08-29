from __future__ import annotations

from fastapi import APIRouter
from openai.types import Model
from polylogue.build_prefix import build_prefix

from polylogue.constants import Endpoint

router = APIRouter(prefix=build_prefix(Endpoint.MODELS))


# need redis hooked up to query model
@router.get("")
async def list_models() -> list[Model]:
    return []


@router.get("/{model}")
async def get_model(model: str) -> Model:
    model: str = request.get_or_422("model")

    raise Exception("unimplemented")
