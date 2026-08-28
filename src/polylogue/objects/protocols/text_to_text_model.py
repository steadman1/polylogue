from typing import Protocol, runtime_checkable
from collections.abc import Generator

@runtime_checkable
class TextToTextModel(Protocol):
    def load(self) -> None:
        ...

    def destroy(self) -> None:
        ...

    def generate(self, prompt: str) -> str:
        ...

    def stream_generate(self, prompt: str) -> Generator[str, None, None]:
        ...
