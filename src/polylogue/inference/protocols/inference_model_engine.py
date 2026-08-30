from __future__ import annotations

from collections.abc import Generator
from typing import Protocol, runtime_checkable

from polylogue.inference.protocols.inference_model import InferenceModel


@runtime_checkable
class InferenceModelEngine[SomeResponse](Protocol):
    model: InferenceModel
    model_id: str

    def destroy(self) -> None: ...

    def generate(self) -> SomeResponse: ...

    # fastapi's StreamingResponse expects yield results to be bytes or a string
    def stream_generate(self) -> Generator[bytes | str, None, None]: ...
