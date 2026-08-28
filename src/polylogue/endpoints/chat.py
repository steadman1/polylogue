from __future__ import annotations

from polylogue.build_url import build_url
from polylogue.fastapi_app import app
from polylogue.constants import Endpoint
from polylogue.objects.text_to_text_engine import TextToTextEngine
from polylogue.constants import TEST_PROMPT
from polylogue.functions.get_or_HTTPException import *

from pathlib import Path
from fastapi.responses import StreamingResponse
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat import CompletionCreateParams

# only want to store completions for 3 (?) days if "store=true"

# want to hand-off each completion to be handled by the client
# and store as few on-server as possible, ideally none at all
@app.get(build_url(Endpoint.CHAT))
async def list_chat_completions() -> list[ChatCompletion]:
    # return a list of stored completions
    return []



@app.post(build_url(Endpoint.CHAT), response_model=None)
async def create_chat_completion(request: ValidatedCompletionCreateParams) -> ChatCompletion | StreamingResponse:
    # check required request body parameters are provided
    model: str = request.get_or_422("model")
    messages: list[Any] = request.get_or_422("messages")
    prompt: str = request.get_or_422("prompt")
    stream = request.get("stream", False)

    # need some mapping from model names to model paths
    model_path = Path("/Users/spencersteadman/Models/lil-bard/")
    engine = TextToTextEngine(model_path)

    if stream:
        # should be text/event-stream since were yielding json encoded ChatCompletionChunks
        return StreamingResponse(engine.stream_generate(TEST_PROMPT), media_type="text/event-stream")

    return engine.generate(TEST_PROMPT)
