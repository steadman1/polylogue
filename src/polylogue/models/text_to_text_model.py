from collections.abc import AsyncIterator
from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class TextToTextModel(Protocol):
    def load(self): ...

    def destroy(self) -> None: ...

    def generate(self) -> dict[str, str]: ...

    def stream_generate(self) -> AsyncIterator[dict[str, str]]: ...
