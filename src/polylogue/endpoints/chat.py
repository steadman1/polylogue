from __future__ import annotations

from pathlib import Path

from fastapi import Request
from fastapi.responses import StreamingResponse
from openai.types.chat.chat_completion import ChatCompletion

from polylogue.auth.get_current_api_key import AuthenticatedKey
from polylogue.clients import RedisClientDep, app
from polylogue.constants import Endpoint
from polylogue.db.model_record import ModelRecord
from polylogue.db.model_record_manager import ModelRecordManager
from polylogue.helpers.build_prefix import build_prefix
from polylogue.helpers.get_or_HTTPException import *
from polylogue.inference.model_cache_manager import ModelCacheManager
from polylogue.inference.text_to_text_engine import TextToTextEngine
from polylogue.inference.text_to_text_factory import TextToTextFactory

# only want to store completions for 3 (?) days if "store=true"


# want to hand-off each completion to be handled by the client
# and store as few on-server as possible, ideally none at all
@app.get(build_prefix(Endpoint.CHAT), response_model=list[ChatCompletion])
@app.get(
    build_prefix(Endpoint.CHAT, without_version=True),
    response_model=list[ChatCompletion],
)
async def list_chat_completions(api_key: AuthenticatedKey) -> list[ChatCompletion]:
    # return a list of stored completions
    return []


@app.post(build_prefix(Endpoint.CHAT), response_model=None)
@app.get(build_prefix(Endpoint.CHAT, without_version=True), response_model=None)
async def create_chat_completion(
    request: Request,
    create_params: ValidatedCompletionCreateParams,
    db_client: RedisClientDep,
    api_key: AuthenticatedKey,
) -> ChatCompletion | StreamingResponse:
    # owner = api_key.owner_id
    # limit = api_key.rate_limit

    # check required request body parameters are provided
    model_id: str = create_params.get_or_422("model")
    messages: list[Any] = create_params.get_or_422("messages")
    tools: list[Any] = create_params.get_or_422("tools")
    stream = create_params.get("stream", False)

    db = ModelRecordManager(db_client)
    model_record: ModelRecord = await db.get(model_id)

    # manager: ModelCacheManager = request.app.state.model_manager
    # engine: TextToTextEngine = manager.get_or_load(model_record)

    model = TextToTextFactory.from_record(model_record)
    engine = TextToTextEngine(model, model_id)

    if stream:
        # should be text/event-stream since were yielding json encoded ChatCompletionChunks
        return StreamingResponse(
            engine.stream_generate(messages, tools), media_type="text/event-stream"
        )

    return engine.generate(messages, tools)
