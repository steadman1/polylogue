from pathlib import Path
from typing import Protocol, runtime_checkable
from collections.abc import AsyncIterator
from polylogue.objects.model_engine import ModelEngine

# generic response
@runtime_checkable
class ModelRunner[SomeResponse](Protocol):

    model_path: Path
    model_engine: ModelEngine

    def _choose_model(self) -> None:
        ...

    def _load(self, model_path) -> bool:
        ...

    def destroy(self) -> None:
        ...

    def await_response(self) -> SomeResponse:
        ...

    def stream_response(self) -> AsyncIterator[SomeResponse]:
        ...
