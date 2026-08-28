from __future__ import annotations
from collections.abc import Generator
from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class ModelEngine[SomeResponse](Protocol):

    model_path: Path
    model_name: str

    def _choose_model_type(self):
        ...

    def destroy(self) -> None:
        ...

    def generate(self) -> SomeResponse:
        ...

    # fastapi's StreamingResponse expects yield results to be bytes or a string
    def stream_generate(self) -> Generator[bytes | str, None, None]:
        ...
