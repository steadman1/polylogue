from __future__ import annotations

from polylogue.build_url import build_url
from polylogue.fastapi_app import app
from polylogue.constants import Endpoint
from polylogue.objects.text_to_text_engine import TextToTextEngine

from pathlib import Path
from fastapi.responses import StreamingResponse
from openai.types import Completion
from openai.types.chat import CompletionCreateParams

# only want to store completions for 3 (?) days if "store=true"

# want to hand-off each completion to be handled by the client
# and store as few on-server as possible, ideally none at all
@app.get(build_url(Endpoint.CHAT))
async def list_chat_completions() -> list[Completion]:
    # return a list of stored completions
    return []



@app.post(build_url(Endpoint.CHAT), response_model=None)
async def create_chat_completion(request: CompletionCreateParams) -> Completion | StreamingResponse:
    stream = request.get("stream", False)

    model_path = Path("/Users/spencersteadman/Models/qwen3.8-27b/Qwen3.8-27B-UD-Q6_K_L.gguf")
    engine = TextToTextEngine(model_path)

    # if stream:
    #     return StreamingResponse(model_runner.stream_response(), media_type="text/plain")

    return engine.generate()
