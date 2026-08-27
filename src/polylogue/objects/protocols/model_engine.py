from collections.abc import Generator
from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class ModelEngine[SomeResponse](Protocol):

    model_path: Path
    model_name: str

    def load(self) -> bool:
        ...

    def destroy(self) -> None:
        ...

    def generate(self) -> SomeResponse:
        ...

    def stream_generate(self) -> Generator[SomeResponse, None, None]:
        ...
