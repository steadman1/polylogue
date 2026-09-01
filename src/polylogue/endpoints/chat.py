from __future__ import annotations

from fastapi.responses import StreamingResponse
from openai.types.chat.chat_completion import ChatCompletion

from polylogue.clients import DBClient, app
from polylogue.constants import Endpoint
from polylogue.db.model_record import ModelRecord
from polylogue.db.model_record_manager import ModelRecordManager
from polylogue.helpers.build_prefix import build_prefix
from polylogue.helpers.get_or_HTTPException import *
from polylogue.inference.protocols.inference_model import InferenceModel
from polylogue.inference.text_to_text_engine import TextToTextEngine
from polylogue.inference.text_to_text_factory import TextToTextFactory

# only want to store completions for 3 (?) days if "store=true"


# want to hand-off each completion to be handled by the client
# and store as few on-server as possible, ideally none at all
@app.get(build_prefix(Endpoint.CHAT), response_model=list[ChatCompletion])
@app.get(build_prefix(Endpoint.CHAT, without_version=True), response_model=list[ChatCompletion])
async def list_chat_completions() -> list[ChatCompletion]:
    # return a list of stored completions
    return []


@app.post(build_prefix(Endpoint.CHAT), response_model=None)
@app.get(build_prefix(Endpoint.CHAT, without_version=True), response_model=None)
async def create_chat_completion(
    create_params: ValidatedCompletionCreateParams, db_client: DBClient
) -> ChatCompletion | StreamingResponse:
    # check required request body parameters are provided
    model_id: str = create_params.get_or_422("model")
    messages: list[Any] = create_params.get_or_422("messages")
    stream = create_params.get("stream", False)

    db = ModelRecordManager(db_client)
    model_record: ModelRecord = await db.get(model_id)

    model: InferenceModel = TextToTextFactory.from_path(model_record.path)
    engine: TextToTextEngine = TextToTextEngine(model, model_id)

    if stream:
        # should be text/event-stream since were yielding json encoded ChatCompletionChunks
        return StreamingResponse(
            engine.stream_generate(messages), media_type="text/event-stream"
        )

    return engine.generate(messages)
