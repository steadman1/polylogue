from __future__ import annotations

from collections.abc import Generator
from typing import Protocol, runtime_checkable

from polylogue.models.protocols.model import Model


@runtime_checkable
class ModelEngine[SomeResponse](Protocol):
    model: Model
    model_id: str

    def destroy(self) -> None: ...

    def generate(self) -> SomeResponse: ...

    # fastapi's StreamingResponse expects yield results to be bytes or a string
    def stream_generate(self) -> Generator[bytes | str, None, None]: ...
