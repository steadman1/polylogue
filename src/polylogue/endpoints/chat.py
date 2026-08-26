from polylogue.build_url import build_url
from polylogue.fastapi_app import app
from polylogue.constants import API_VERSION, EndpointType

from fastapi.responses import StreamingResponse

from openai.types import Completion
from openai.types.chat import CompletionCreateParams

# only want to store completions for 3 (?) days if "store=true"

# want to hand-off each completion to be handled by the client
# and store as few on-server as possible, ideally none at all
@app.get(build_url(EndpointType.CHAT))
async def list_chat_completions() -> list[Completion]:
    # return a list of stored completions
    return []



@app.post(build_url(EndpointType.CHAT), response_model=None)
async def create_chat_completion(request: CompletionCreateParams) -> None:
    stream = request.get("stream", False)

    if stream:
        return None # StreamingResponse( ... )

    return None
