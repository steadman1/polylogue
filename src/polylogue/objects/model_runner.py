from pathlib import Path
from typing import Protocol, runtime_checkable
from collections.abc import Generator

# generic response
@runtime_checkable
class ModelRunner[SomeResponse](Protocol):

    model_id: Path

    def _load(self, model_dir) -> bool:
        ...

    def destroy(self) -> None:
        ...

    def await_response(self) -> SomeResponse:
        ...

    def stream_response(self) -> Generator[SomeResponse, None, None]:
        ...
